#!/usr/bin/env python3
"""AutoDL runtime gate and guarded promotion of CAARMA commits to PSC."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


BASE_URL = "https://api.autodl.com"
DEFAULT_ENV = Path.home() / ".codex" / "caarma-cloud.env"
RECEIPT_ROOT = Path.home() / ".codex" / "caarma-cloud-gates" / "receipts"
PSC_RUNNER = Path.home() / ".codex" / "skills" / "psc-caarma-remote-runner" / "scripts" / "psc_remote.py"
STOPPED = {"stopped", "shutdown"}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
EER_RE = re.compile(r"cosine EER:\s*([0-9.]+)%")
DCF2_RE = re.compile(r"cosine minDCF\(10-2\):\s*([0-9.]+)")
DCF3_RE = re.compile(r"cosine minDCF\(10-3\):\s*([0-9.]+)")
FATAL_RE = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|NCCL[^\n]*(?:error|timeout)|"
    r"ModuleNotFoundError|ImportError|Segmentation fault|DataLoader worker.*exited",
    re.IGNORECASE,
)


class GateError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def config(args: argparse.Namespace) -> dict[str, str]:
    path = Path(args.env_file).expanduser() if args.env_file else DEFAULT_ENV
    file_values = load_env(path)
    keys = (
        "AUTODL_TOKEN", "AUTODL_INSTANCE_UUID", "AUTODL_SSH_KEY",
        "AUTODL_SSH_USER",
    )
    values = {key: os.environ.get(key) or file_values.get(key, "") for key in keys}
    values["AUTODL_SSH_USER"] = values["AUTODL_SSH_USER"] or "root"
    if path.exists() and path.stat().st_mode & 0o077:
        raise PermissionError(f"AutoDL env file must be chmod 600: {path}")
    return values


def api_request(
    command: str,
    token: str,
    instance_uuid: str,
    *,
    timeout: int = 30,
    start_command: str | None = None,
) -> dict[str, Any]:
    routes = {
        "status": ("GET", "/api/v1/dev/instance/pro/status"),
        "snapshot": ("GET", "/api/v1/dev/instance/pro/snapshot"),
        "power-on": ("POST", "/api/v1/dev/instance/pro/power_on"),
        "power-off": ("POST", "/api/v1/dev/instance/pro/power_off"),
    }
    method, path = routes[command]
    payload: dict[str, Any] = {"instance_uuid": instance_uuid}
    if command == "power-on":
        payload["payload"] = "gpu"
        if start_command:
            payload["start_command"] = start_command
    url = BASE_URL + path
    body = None
    if method == "GET":
        url += "?" + urllib.parse.urlencode(payload)
    else:
        body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Authorization": token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError) as error:
        raise GateError(f"AutoDL API {command} failed: {error}") from error
    if result.get("code") != "Success":
        raise GateError(f"AutoDL API {command} returned {result.get('code')}: {result.get('msg')}")
    return result


def require_credentials(values: dict[str, str]) -> tuple[str, str, Path, str]:
    token = values.get("AUTODL_TOKEN", "")
    instance = values.get("AUTODL_INSTANCE_UUID", "")
    key = Path(values.get("AUTODL_SSH_KEY", "")).expanduser()
    user = values.get("AUTODL_SSH_USER", "root")
    if not token:
        raise GateError("Missing AUTODL_TOKEN")
    if not instance.startswith("pro-"):
        raise GateError("Missing or invalid AUTODL_INSTANCE_UUID")
    if not key.is_file():
        raise GateError(f"AutoDL SSH key does not exist: {key}")
    return token, instance, key, user


def wait_status(token: str, instance: str, targets: set[str], timeout: int) -> str:
    deadline = time.time() + timeout
    while True:
        state = str(api_request("status", token, instance)["data"])
        if state in targets:
            return state
        if time.time() >= deadline:
            raise GateError(f"Timed out waiting for {sorted(targets)}; last state={state}")
        time.sleep(15)


def connection(token: str, instance: str) -> tuple[str, int]:
    snapshot = api_request("snapshot", token, instance).get("data") or {}
    host, port = snapshot.get("proxy_host"), snapshot.get("ssh_port")
    if not host or not port:
        raise GateError("AutoDL snapshot did not contain proxy_host/ssh_port")
    return str(host), int(port)


def ssh_command(host: str, port: int, key: Path, user: str, remote: str) -> list[str]:
    return [
        "ssh", "-i", str(key), "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(port), f"{user}@{host}", remote,
    ]


def ssh_run(
    host: str, port: int, key: Path, user: str, remote: str,
    *, timeout: int = 60, check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ssh_command(host, port, key, user, remote),
            text=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise GateError("AutoDL SSH timed out") from error
    if check and result.returncode != 0:
        raise GateError(f"AutoDL SSH failed ({result.returncode}): {(result.stderr or result.stdout).strip()}")
    return result


def stage_commit(
    local_repo: Path, commit: str, remote_dir: str,
    host: str, port: int, key: Path, user: str, timeout: int,
) -> None:
    remote = (
        f"set -euo pipefail; target={shlex.quote(remote_dir)}; "
        "test ! -e \"$target\"; mkdir -p \"$target\"; tar -xf - -C \"$target\"; "
        f"printf '%s\\n' {shlex.quote(commit)} > \"$target/.caarma_commit\""
    )
    archive = subprocess.Popen(
        ["git", "-C", str(local_repo), "archive", "--format=tar", commit],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    try:
        transfer = subprocess.run(
            ssh_command(host, port, key, user, remote),
            stdin=archive.stdout, capture_output=True, timeout=timeout,
        )
    finally:
        archive.stdout.close()
    archive_code = archive.wait()
    if archive_code or transfer.returncode:
        raise GateError(
            "AutoDL staging failed: "
            + transfer.stderr.decode(errors="replace").strip()
        )


def start_gate(
    host: str, port: int, key: Path, user: str,
    remote_dir: str, session: str, train_command: str,
) -> None:
    encoded = base64.b64encode(train_command.encode()).decode()
    remote = f'''set -euo pipefail
cd {shlex.quote(remote_dir)}
printf '%s' {shlex.quote(encoded)} | base64 -d > .gate_command.sh
chmod 700 .gate_command.sh
tmux new-session -d -s {shlex.quote(session)} "bash -lc 'bash .gate_command.sh > gate.log 2>&1; echo \\$? > gate.exit'"
echo STARTED
'''
    ssh_run(host, port, key, user, "bash -lc " + shlex.quote(remote))


def probe_gate(
    host: str, port: int, key: Path, user: str, remote_dir: str, session: str,
) -> tuple[bool, str, str | None]:
    remote = (
        f"cd {shlex.quote(remote_dir)}; "
        f"tmux has-session -t {shlex.quote(session)} 2>/dev/null && echo SESSION=1 || echo SESSION=0; "
        "test -f gate.exit && echo EXIT=$(cat gate.exit) || true; "
        "test -f gate.log && tail -c 300000 gate.log || true"
    )
    result = ssh_run(host, port, key, user, remote, check=False)
    text = ANSI_RE.sub("", result.stdout).replace("\r", "\n")
    running = "SESSION=1" in text
    match = re.search(r"^EXIT=([0-9]+)$", text, re.MULTILINE)
    return running, text, match.group(1) if match else None


def stop_gate(host: str, port: int, key: Path, user: str, session: str) -> None:
    remote = f'''
if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then
  tmux send-keys -t {shlex.quote(session)} C-c
  sleep 20
  tmux kill-session -t {shlex.quote(session)} 2>/dev/null || true
fi
'''
    ssh_run(host, port, key, user, "bash -lc " + shlex.quote(remote), timeout=45)


def gpu_apps(host: str, port: int, key: Path, user: str) -> list[str]:
    result = ssh_run(
        host, port, key, user,
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def write_receipt(payload: dict[str, Any]) -> Path:
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    target = RECEIPT_ROOT / f"{payload['name']}-{payload['commit'][:12]}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def cmd_doctor(args: argparse.Namespace) -> int:
    values = config(args)
    token, instance, key, user = require_credentials(values)
    state = str(api_request("status", token, instance)["data"])
    payload: dict[str, Any] = {
        "instance_uuid": instance,
        "status": state,
        "ssh_key": str(key),
        "ssh_user": user,
        "ok": True,
    }
    if state == "running":
        host, port = connection(token, instance)
        probe = ssh_run(host, port, key, user, "hostname; nvidia-smi -L", check=False)
        payload["ssh_ok"] = probe.returncode == 0
        payload["proxy_host"] = host
        payload["ssh_port"] = port
        payload["ok"] = payload["ssh_ok"]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


def cmd_status(args: argparse.Namespace) -> int:
    values = config(args)
    token, instance, _, _ = require_credentials(values)
    state = api_request("status", token, instance)["data"]
    print(json.dumps({"instance_uuid": instance, "status": state}, indent=2))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    if not args.yes:
        raise PermissionError("gate requires --yes")
    values = config(args)
    token, instance, key, user = require_credentials(values)
    local_repo = Path(args.local_repo).expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(local_repo), "rev-parse", f"{args.ref}^{{commit}}"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", args.name)
    run_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    remote_dir = f"/root/autodl-tmp/caarma_gates/{safe_name}-{commit[:12]}-{run_stamp}"
    session = f"gate-{safe_name[:24]}-{commit[:7]}"
    payload: dict[str, Any] = {
        "name": safe_name,
        "commit": commit,
        "instance_uuid": instance,
        "remote_dir": remote_dir,
        "session": session,
        "started_at_utc": utc_now(),
        "status": "FAIL",
    }
    host = ""
    port = 0
    log = ""
    try:
        state = str(api_request("status", token, instance)["data"])
        if state != "running":
            api_request("power-on", token, instance, start_command="sleep 1")
            wait_status(token, instance, {"running"}, args.startup_timeout)
        host, port = connection(token, instance)
        ssh_run(host, port, key, user, "hostname; nvidia-smi -L", timeout=60)
        stage_commit(local_repo, commit, remote_dir, host, port, key, user, args.ssh_timeout)
        start_gate(host, port, key, user, remote_dir, session, args.train_command)
        deadline = time.time() + args.max_seconds
        while True:
            running, log, exit_code = probe_gate(host, port, key, user, remote_dir, session)
            fatal = FATAL_RE.search(log)
            passed = (
                "All distributed processes registered" in log
                and re.search(r"Epoch 0:\s*100%", log) is not None
                and EER_RE.search(log) is not None
                and DCF2_RE.search(log) is not None
                and DCF3_RE.search(log) is not None
                and re.search(r"Epoch 1:", log) is not None
            )
            if fatal:
                raise GateError(f"Fatal training log: {fatal.group(0)}")
            if passed:
                payload["status"] = "PASS"
                break
            if not running and exit_code is not None:
                raise GateError(f"Gate process exited before PASS: {exit_code}")
            if time.time() >= deadline:
                raise GateError("Gate timed out before validation/Epoch 1")
            time.sleep(args.poll_interval)
    except Exception as error:
        payload["failure"] = str(error)
    finally:
        if host and port:
            try:
                stop_gate(host, port, key, user, session)
                time.sleep(5)
                apps = gpu_apps(host, port, key, user)
                payload["remaining_gpu_apps"] = apps
                if apps:
                    payload["status"] = "FAIL"
                    payload["failure"] = "GPU compute remained after stopping owned gate session"
                else:
                    api_request("power-off", token, instance)
                    payload["shutdown_state"] = wait_status(
                        token, instance, STOPPED, args.shutdown_timeout
                    )
            except Exception as cleanup_error:
                payload["status"] = "FAIL"
                payload["cleanup_failure"] = str(cleanup_error)

    clean_log = ANSI_RE.sub("", log).replace("\r", "\n")
    eers = [float(value) for value in EER_RE.findall(clean_log)]
    dcf2 = [float(value) for value in DCF2_RE.findall(clean_log)]
    dcf3 = [float(value) for value in DCF3_RE.findall(clean_log)]
    payload.update({
        "best_eer": min(eers) if eers else None,
        "latest_min_dcf_1e2": dcf2[-1] if dcf2 else None,
        "latest_min_dcf_1e3": dcf3[-1] if dcf3 else None,
        "finished_at_utc": utc_now(),
    })
    receipt = write_receipt(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"RECEIPT={receipt}")
    return 0 if payload["status"] == "PASS" else 2


def run_cli(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise GateError((result.stderr or result.stdout).strip())
    start = result.stdout.find("{")
    end = result.stdout.rfind("}")
    return json.loads(result.stdout[start:end + 1])


def cmd_promote(args: argparse.Namespace) -> int:
    if not args.yes or not args.confirm_production:
        raise PermissionError("promote requires --yes and --confirm-production")
    receipt = json.loads(Path(args.receipt).expanduser().read_text())
    if receipt.get("status") != "PASS" or receipt.get("shutdown_state") not in STOPPED:
        raise GateError("Receipt is not PASS with confirmed AutoDL shutdown")
    local_repo = Path(args.local_repo).expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(local_repo), "rev-parse", f"{args.ref}^{{commit}}"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if commit != receipt.get("commit"):
        raise GateError("PSC promotion commit does not match AutoDL PASS receipt")
    if not PSC_RUNNER.is_file():
        raise GateError(f"PSC runner missing: {PSC_RUNNER}")
    base = [sys.executable, str(PSC_RUNNER), "--host", args.psc_host]
    stage = run_cli(base + [
        "stage", "--local-repo", str(local_repo), "--ref", commit,
        "--remote-dir", args.psc_remote_dir, "--yes",
    ])
    submit_base = base + [
        "submit", "--repo", args.psc_remote_dir, "--script", args.psc_script,
        "--expected-commit", commit, "--confirm-production",
    ]
    if args.psc_afterok:
        submit_base += ["--afterok", args.psc_afterok]
    dry = run_cli(submit_base + ["--dry-run"])
    submitted = run_cli(submit_base)
    print(json.dumps({"stage": stage, "dry_run": dry, "submitted": submitted}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--env-file")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    sub.add_parser("status").set_defaults(func=cmd_status)

    gate = sub.add_parser("gate")
    gate.add_argument("--local-repo", required=True)
    gate.add_argument("--ref", default="HEAD")
    gate.add_argument("--name", required=True)
    gate.add_argument("--train-command", required=True)
    gate.add_argument("--max-seconds", type=int, default=1800)
    gate.add_argument("--poll-interval", type=int, default=30)
    gate.add_argument("--startup-timeout", type=int, default=900)
    gate.add_argument("--shutdown-timeout", type=int, default=600)
    gate.add_argument("--ssh-timeout", type=int, default=180)
    gate.add_argument("--yes", action="store_true")
    gate.set_defaults(func=cmd_gate)

    promote = sub.add_parser("promote")
    promote.add_argument("--receipt", required=True)
    promote.add_argument("--local-repo", required=True)
    promote.add_argument("--ref", default="HEAD")
    promote.add_argument("--psc-host", default="bridges2")
    promote.add_argument("--psc-remote-dir", required=True)
    promote.add_argument("--psc-script", required=True)
    promote.add_argument("--psc-afterok")
    promote.add_argument("--confirm-production", action="store_true")
    promote.add_argument("--yes", action="store_true")
    promote.set_defaults(func=cmd_promote)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except (GateError, PermissionError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
