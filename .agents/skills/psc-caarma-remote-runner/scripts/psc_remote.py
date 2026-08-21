#!/usr/bin/env python3
"""Guarded Mac-to-PSC SSH orchestration for CAARMA Slurm jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


DEFAULT_HOST = os.environ.get("PSC_SSH_HOST", "bridges2")
STATE_ROOT = Path.home() / ".codex" / "psc-caarma-remote" / "jobs"
JOB_ID_RE = re.compile(r"^[0-9]+$")
SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+:-]+$")
SAFE_REL_RE = re.compile(r"^[A-Za-z0-9._/+-]+$")
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_job_id(value: str) -> str:
    if not JOB_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid Slurm JobID: {value!r}")
    return value


def validate_remote_path(value: str, *, relative: bool = False) -> str:
    pattern = SAFE_REL_RE if relative else SAFE_PATH_RE
    if not pattern.fullmatch(value) or ".." in Path(value).parts:
        raise ValueError(f"Unsafe remote path: {value!r}")
    return value


def ssh_run(
    host: str,
    script: str,
    args: list[str] | None = None,
    *,
    timeout: int = 45,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        host, "bash", "-s", "--", *(args or []),
    ]
    try:
        result = subprocess.run(
            command,
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "PSC SSH timed out. Register a public key or establish an "
            "interactive ControlMaster connection first."
        ) from error
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"PSC SSH command failed ({result.returncode}): {detail}")
    return result


def parse_kv(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            payload[key.strip()] = value.strip()
    return payload


def state_path(job_id: str) -> Path:
    return STATE_ROOT / f"{job_id}.json"


def save_state(payload: dict[str, Any]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    payload["updated_at_utc"] = utc_now()
    target = state_path(str(payload["job_id"]))
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def load_state(job_id: str) -> dict[str, Any]:
    path = state_path(job_id)
    if not path.is_file():
        raise FileNotFoundError(f"No local PSC record for JobID {job_id}: {path}")
    return json.loads(path.read_text())


DOCTOR_SCRIPT = r'''
set -euo pipefail
echo "HOSTNAME=$(hostname)"
echo "USER=$(id -un)"
echo "SBATCH=$(command -v sbatch || true)"
echo "SQUEUE=$(command -v squeue || true)"
echo "SACCT=$(command -v sacct || true)"
echo "SCANCEL=$(command -v scancel || true)"
echo "ACCOUNT=$(groups | tr ' ' '\n' | grep -x cis220031p || true)"
test -d /ocean/projects/cis220031p/sge2
echo "USER_PROJECT_ROOT=OK"
'''


SUBMIT_SCRIPT = r'''
set -euo pipefail
repo=$1
script_rel=$2
confirm=$3
allow_dirty=$4
expected_commit=$5
afterok=$6
dry_run=$7

case "$repo" in
  /jet/home/sge2/*|/ocean/projects/cis220031p/sge2/*) ;;
  *) echo "ERROR=repo outside approved scope: $repo" >&2; exit 20 ;;
esac
case "$script_rel" in scripts/psc/*) ;; *) exit 21 ;; esac

cd "$repo"
script=$(realpath -m "$repo/$script_rel")
case "$script" in "$repo"/scripts/psc/*) ;; *) exit 22 ;; esac
test -f "$script"

commit=$(git rev-parse --short HEAD)
full_commit=$(git rev-parse HEAD)
if [[ -n "$expected_commit" && "$full_commit" != "$expected_commit"* ]]; then
  echo "ERROR=expected commit $expected_commit, got $full_commit" >&2
  exit 23
fi

dirty=$(git status --porcelain --untracked-files=no)
if [[ -n "$dirty" && "$allow_dirty" != yes ]]; then
  echo "ERROR=tracked dirty files require --allow-launcher-dirty" >&2
  echo "$dirty" >&2
  exit 24
fi

grep -Eq '^#SBATCH[[:space:]]+--account(=|[[:space:]]+)cis220031p$' "$script"
grep -Eq '^#SBATCH[[:space:]]+--partition(=|[[:space:]]+)GPU-shared$' "$script"

directive() {
  local key=$1
  sed -nE "s/^#SBATCH[[:space:]]+--${key}(=|[[:space:]]+)(.*)$/\\2/p" "$script" | head -1
}

gpus=$(directive gpus)
gpu_count=${gpus##*:}
job_name=$(directive job-name)
output=$(directive output)
error=$(directive error)
walltime=$(directive time)

if [[ "$gpu_count" =~ ^[0-9]+$ ]] && (( gpu_count >= 4 )) && [[ "$confirm" != yes ]]; then
  echo "ERROR=four-GPU job requires --confirm-production" >&2
  exit 25
fi
if [[ -n "$afterok" && ! "$afterok" =~ ^[0-9]+$ ]]; then exit 26; fi

echo "REPO=$repo"
echo "COMMIT=$full_commit"
echo "DIRTY=${dirty//$'\n'/;}"
echo "SCRIPT=$script_rel"
echo "JOB_NAME=$job_name"
echo "GPUS=$gpus"
echo "WALLTIME=$walltime"
echo "DEPENDENCY=$afterok"

if [[ "$dry_run" == yes ]]; then echo "DRY_RUN=yes"; exit 0; fi

args=(--parsable --export="ALL,CAARMA_JOB_REPO=$repo,CAARMA_METHOD_COMMIT=$full_commit")
if [[ -n "$afterok" ]]; then args+=(--dependency="afterok:$afterok"); fi
job_raw=$(sbatch "${args[@]}" "$script")
job_id=${job_raw%%;*}
echo "JOB_ID=$job_id"
echo "OUTPUT=${output//%x/$job_name}"
echo "ERROR_LOG=${error//%x/$job_name}"
'''


STATUS_SCRIPT = r'''
set -euo pipefail
job_id=$1
active=$(squeue -h -j "$job_id" -o '%i|%j|%T|%M|%R' || true)
if [[ -n "$active" ]]; then echo "ACTIVE=$active"; fi
sacct -n -P -j "$job_id" --format=JobIDRaw,JobName,State,Elapsed,Start,End,ExitCode \
  | awk -F'|' -v id="$job_id" '$1 == id {print "ACCOUNTING=" $0; exit}'
'''


TAIL_SCRIPT = r'''
set -euo pipefail
path=$1
lines=$2
case "$path" in /ocean/projects/cis220031p/sge2/logs/*) ;; *) exit 31 ;; esac
if [[ -f "$path" ]]; then tail -n "$lines" "$path"; else echo "LOG_NOT_CREATED=$path"; fi
'''


def cmd_doctor(args: argparse.Namespace) -> int:
    result = ssh_run(args.host, DOCTOR_SCRIPT, timeout=args.timeout)
    payload = parse_kv(result.stdout)
    payload["ssh_host"] = args.host
    payload["ok"] = all(payload.get(key) for key in (
        "SBATCH", "SQUEUE", "SACCT", "SCANCEL", "ACCOUNT"
    )) and payload.get("USER_PROJECT_ROOT") == "OK"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


def cmd_submit(args: argparse.Namespace) -> int:
    repo = validate_remote_path(args.repo)
    script = validate_remote_path(args.script, relative=True)
    expected = args.expected_commit or ""
    if expected and not re.fullmatch(r"[0-9a-fA-F]{7,40}", expected):
        raise ValueError("--expected-commit must be a 7-40 character hex SHA")
    afterok = validate_job_id(args.afterok) if args.afterok else ""
    result = ssh_run(args.host, SUBMIT_SCRIPT, [
        repo, script,
        "yes" if args.confirm_production else "no",
        "yes" if args.allow_launcher_dirty else "no",
        expected, afterok,
        "yes" if args.dry_run else "no",
    ], timeout=args.timeout)
    payload = parse_kv(result.stdout)
    payload["ssh_host"] = args.host
    payload["submitted_at_utc"] = utc_now()
    if "JOB_ID" in payload:
        job_id = validate_job_id(payload["JOB_ID"])
        for key in ("OUTPUT", "ERROR_LOG"):
            if key in payload:
                payload[key] = payload[key].replace("%j", job_id).replace("%A", job_id)
        payload["job_id"] = job_id
        save_state(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def query_status(host: str, job_id: str, timeout: int) -> dict[str, Any]:
    result = ssh_run(host, STATUS_SCRIPT, [job_id], timeout=timeout)
    raw = parse_kv(result.stdout)
    payload: dict[str, Any] = {"job_id": job_id, "ssh_host": host}
    if "ACTIVE" in raw:
        fields = raw["ACTIVE"].split("|", 4)
        if len(fields) == 5:
            payload.update(zip(("job_id", "job_name", "state", "elapsed", "reason"), fields))
    if "ACCOUNTING" in raw:
        fields = raw["ACCOUNTING"].split("|")
        if len(fields) >= 7:
            payload.update({
                "job_id": fields[0], "job_name": fields[1], "state": fields[2],
                "elapsed": fields[3], "start": fields[4], "end": fields[5],
                "exit_code": fields[6],
            })
    payload.setdefault("state", "UNKNOWN")
    return payload


def cmd_status(args: argparse.Namespace) -> int:
    payload = query_status(args.host, validate_job_id(args.job_id), args.timeout)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    job_id = validate_job_id(args.job_id)
    deadline = time.time() + args.max_seconds
    while True:
        payload = query_status(args.host, job_id, args.timeout)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if payload["state"].split()[0] in TERMINAL_STATES or time.time() >= deadline:
            return 0
        time.sleep(args.interval)


def remote_tail(host: str, path: str, lines: int, timeout: int) -> str:
    validate_remote_path(path)
    raw = ssh_run(host, TAIL_SCRIPT, [path, str(lines)], timeout=timeout).stdout
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    logical_lines = clean.splitlines()
    return "\n".join(logical_lines[-lines:]) + ("\n" if logical_lines else "")


def cmd_logs(args: argparse.Namespace) -> int:
    job_id = validate_job_id(args.job_id)
    try:
        state = load_state(job_id)
    except FileNotFoundError:
        if not args.output and not args.error:
            raise
        state = {}
    for label, path in (
        ("STDOUT", args.output or state.get("OUTPUT")),
        ("STDERR", args.error or state.get("ERROR_LOG")),
    ):
        print(f"===== {label} =====")
        print(remote_tail(args.host, path, args.lines, args.timeout), end="") if path else print("No path")
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    job_id = validate_job_id(args.job_id)
    repo = validate_remote_path(args.repo)
    output = validate_remote_path(args.output)
    error = validate_remote_path(args.error)
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.commit):
        raise ValueError("--commit must be a 7-40 character hex SHA")
    payload = {
        "job_id": job_id,
        "ssh_host": args.host,
        "REPO": repo,
        "COMMIT": args.commit,
        "JOB_NAME": args.job_name,
        "OUTPUT": output,
        "ERROR_LOG": error,
        "adopted_at_utc": utc_now(),
    }
    payload["status_at_adoption"] = query_status(args.host, job_id, args.timeout)
    save_state(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


METRIC_RE = re.compile(r"cosine EER:\s*([0-9.]+)%")
DCF2_RE = re.compile(r"cosine minDCF\(10-2\):\s*([0-9.]+)")
DCF3_RE = re.compile(r"cosine minDCF\(10-3\):\s*([0-9.]+)")


def cmd_collect(args: argparse.Namespace) -> int:
    job_id = validate_job_id(args.job_id)
    state = load_state(job_id)
    status = query_status(args.host, job_id, args.timeout)
    output_path = state.get("OUTPUT")
    log = remote_tail(args.host, output_path, args.lines, args.timeout) if output_path else ""
    eers = [float(value) for value in METRIC_RE.findall(log)]
    dcf2 = [float(value) for value in DCF2_RE.findall(log)]
    dcf3 = [float(value) for value in DCF3_RE.findall(log)]
    summary = {
        **state, "status": status,
        "best_eer_in_tail": min(eers) if eers else None,
        "latest_min_dcf_1e2_in_tail": dcf2[-1] if dcf2 else None,
        "latest_min_dcf_1e3_in_tail": dcf3[-1] if dcf3 else None,
        "collected_at_utc": utc_now(),
    }
    target = STATE_ROOT / "summaries" / f"{job_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY_PATH={target}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    job_id = validate_job_id(args.job_id)
    if not args.yes:
        raise PermissionError("Cancellation requires --yes after explicit user approval")
    script = 'set -euo pipefail\njob_id=$1\nscancel "$job_id"\necho "CANCELLED=$job_id"\n'
    print(ssh_run(args.host, script, [job_id], timeout=args.timeout).stdout, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--timeout", type=int, default=45)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").set_defaults(func=cmd_doctor)

    adopt = sub.add_parser("adopt")
    adopt.add_argument("--job-id", required=True)
    adopt.add_argument("--repo", required=True)
    adopt.add_argument("--commit", required=True)
    adopt.add_argument("--job-name", required=True)
    adopt.add_argument("--output", required=True)
    adopt.add_argument("--error", required=True)
    adopt.set_defaults(func=cmd_adopt)

    submit = sub.add_parser("submit")
    submit.add_argument("--repo", required=True)
    submit.add_argument("--script", required=True)
    submit.add_argument("--expected-commit")
    submit.add_argument("--afterok")
    submit.add_argument("--confirm-production", action="store_true")
    submit.add_argument("--allow-launcher-dirty", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status")
    status.add_argument("--job-id", required=True)
    status.set_defaults(func=cmd_status)

    watch = sub.add_parser("watch")
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--max-seconds", type=int, default=900)
    watch.add_argument("--interval", type=int, default=30)
    watch.set_defaults(func=cmd_watch)

    logs = sub.add_parser("logs")
    logs.add_argument("--job-id", required=True)
    logs.add_argument("--lines", type=int, default=100)
    logs.add_argument("--output")
    logs.add_argument("--error")
    logs.set_defaults(func=cmd_logs)

    collect = sub.add_parser("collect")
    collect.add_argument("--job-id", required=True)
    collect.add_argument("--lines", type=int, default=20000)
    collect.set_defaults(func=cmd_collect)

    cancel = sub.add_parser("cancel")
    cancel.add_argument("--job-id", required=True)
    cancel.add_argument("--yes", action="store_true")
    cancel.set_defaults(func=cmd_cancel)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
