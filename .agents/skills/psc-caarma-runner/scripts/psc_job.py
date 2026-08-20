#!/usr/bin/env python3
"""Safe Slurm lifecycle manager for CAARMA jobs on PSC Bridges-2."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_ROOT = Path("/ocean/projects/cis220031p/sge2/caarma_runs")
ALLOWED_ACCOUNT = "cis220031p"
REQUIRED_BRANCH = "exp/psc-bridges2-caarma-reproduction"
ALLOWED_PARTITIONS = {"GPU-shared"}
ALLOWED_PROJECT_PREFIXES = (
    "/ocean/projects/cis220031p/sge2/",
    "/ocean/projects/cis220031p/shared/raw/data",
    "/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1",
)
FORBIDDEN_TEXT = (
    "/ocean/projects/cis220031p/mbaali",
    "recover_original_assets",
    "VoxCeleb2",
)
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def state_root() -> Path:
    root = Path(os.environ.get("CAARMA_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
    path = root / "job_state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_path(job_id: str) -> Path:
    return state_root() / f"{job_id}.json"


def load_state(job_id: str) -> dict[str, Any]:
    path = state_path(job_id)
    if not path.is_file():
        raise FileNotFoundError(f"No recorded state for Slurm job {job_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(payload: dict[str, Any]) -> None:
    payload["updated_at_utc"] = now_utc()
    path = state_path(str(payload["job_id"]))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str:
    result = run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    return result.stdout.strip()


def git_branch() -> str:
    return run(["git", "-C", str(REPO_ROOT), "branch", "--show-current"]).stdout.strip()


def git_dirty_paths() -> list[str]:
    output = run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]).stdout
    return [line for line in output.splitlines() if line.strip()]


def parse_directives(script: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    pattern = re.compile(r"^\s*#SBATCH\s+(-{1,2}[\w-]+)(?:[=\s]+(.+?))?\s*$")
    for line in script.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key = match.group(1).lstrip("-")
        value = (match.group(2) or "true").strip().strip('"\'')
        aliases = {"A": "account", "p": "partition", "t": "time", "J": "job-name"}
        directives[aliases.get(key, key)] = value
    return directives


def parse_hours(value: str) -> float:
    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        days = int(day_part)
    fields = [int(part) for part in value.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, fields[0], fields[1]
    else:
        hours, minutes, seconds = fields[0], 0, 0
    return days * 24 + hours + minutes / 60 + seconds / 3600


def parse_gpu_request(value: str | None) -> tuple[str | None, int]:
    if not value:
        return None, 0
    fields = value.split(":")
    if len(fields) < 2:
        raise ValueError(f"Unrecognized GPU request: {value}")
    return ":".join(fields[:-1]), int(fields[-1])


def estimate_su(gpu_type: str | None, count: int, hours: float) -> float:
    if not gpu_type or count == 0:
        return 0.0
    multiplier = 2.0 if gpu_type.startswith("h100") else 1.0
    return count * hours * multiplier


def expand_log_path(template: str | None, job_name: str, job_id: str) -> str | None:
    if not template:
        return None
    return template.replace("%x", job_name).replace("%j", job_id).replace("%A", job_id)


def validate_script(script: Path, confirm_production: bool) -> dict[str, Any]:
    resolved = script.expanduser().resolve()
    allowed_root = (REPO_ROOT / "scripts" / "psc").resolve()
    if allowed_root not in resolved.parents:
        raise ValueError(f"Job script must be inside {allowed_root}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    text = resolved.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_TEXT:
        if forbidden in text:
            raise ValueError(f"Forbidden path/dataset token in job script: {forbidden}")
    for candidate in re.findall(r"/ocean/projects/cis220031p/[^\s\"']+", text):
        candidate = candidate.rstrip("/)};,")
        if not any(candidate.startswith(prefix) for prefix in ALLOWED_PROJECT_PREFIXES):
            raise ValueError(f"Project path is outside the approved PSC scope: {candidate}")

    directives = parse_directives(resolved)
    account = directives.get("account")
    partition = directives.get("partition")
    if account != ALLOWED_ACCOUNT:
        raise ValueError(f"Job account must be {ALLOWED_ACCOUNT}, got {account!r}")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"Partition must be one of {sorted(ALLOWED_PARTITIONS)}, got {partition!r}")

    gpu_type, gpu_count = parse_gpu_request(directives.get("gpus") or directives.get("gres"))
    hours = parse_hours(directives.get("time", "1:00:00"))
    tasks = int(directives.get("ntasks", directives.get("ntasks-per-node", "1")))
    cpus_per_task = int(directives.get("cpus-per-task", "1"))
    cpu_count = tasks * cpus_per_task
    if partition == "GPU-shared" and gpu_count > 4:
        raise ValueError("GPU-shared jobs may request at most four GPUs")
    if hours > 48:
        raise ValueError("PSC reproduction jobs may not request more than 48 hours")
    production = gpu_count >= 4 or "train_vox1" in resolved.name
    if production and not confirm_production:
        raise PermissionError(
            "Production job requires --confirm-production after smoke and protocol gates pass"
        )

    job_name = directives.get("job-name", resolved.stem)
    return {
        "script": str(resolved),
        "directives": directives,
        "job_name": job_name,
        "partition": partition,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "cpu_count": cpu_count,
        "requested_hours": hours,
        "maximum_su": (
            estimate_su(gpu_type, gpu_count, hours)
            if gpu_count else cpu_count * hours
        ),
        "production": production,
    }


def validate_env_file(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"Environment file must be chmod 600: {resolved} mode={oct(mode)}")
    text = resolved.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_TEXT:
        if forbidden in text:
            raise ValueError(f"Forbidden path/dataset token in environment file: {forbidden}")


def submit_command(env_file: Path | None, sbatch_args: list[str]) -> list[str]:
    if env_file is None:
        return ["sbatch", "--parsable", *sbatch_args]
    shell = 'set -a; source "$1"; set +a; shift; exec sbatch --parsable "$@"'
    return ["bash", "-lc", shell, "psc-caarma-runner", str(env_file.resolve()), *sbatch_args]


def cmd_preflight(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    commands = {name: command_exists(name) for name in ("sbatch", "squeue", "sacct", "scancel")}
    shared_root = Path("/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1")
    payload = {
        "hostname": hostname,
        "is_bridges2": "bridges2" in hostname or hostname.startswith("br"),
        "commands": commands,
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty_paths": git_dirty_paths(),
        "shared_voxceleb1_exists": shared_root.is_dir(),
        "run_root": str(state_root().parent),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["is_bridges2"] and all(commands.values()) else 2


def cmd_submit(args: argparse.Namespace) -> int:
    branch = git_branch()
    dirty_paths = git_dirty_paths()
    if branch != REQUIRED_BRANCH:
        raise PermissionError(f"Required branch is {REQUIRED_BRANCH}, got {branch!r}")
    if dirty_paths:
        raise PermissionError(
            "Repository must be clean before submission: " + "; ".join(dirty_paths[:20])
        )
    env_file = Path(args.env_file) if args.env_file else None
    validate_env_file(env_file)
    metadata = validate_script(Path(args.script), args.confirm_production)
    completed_gates = []
    for gate_job_id in args.require_completed or []:
        gate_state = load_state(gate_job_id)
        gate_status = query_status(gate_job_id).get("state", "UNKNOWN").split()[0]
        if gate_status != "COMPLETED":
            raise PermissionError(
                f"Required gate job {gate_job_id} is not COMPLETED: {gate_status}"
            )
        completed_gates.append({
            "job_id": gate_job_id,
            "label": gate_state.get("label"),
            "status": gate_status,
        })
    sbatch_args = []
    if args.afterok:
        sbatch_args.append(f"--dependency=afterok:{args.afterok}")
    sbatch_args.append(metadata["script"])

    preview = {
        **metadata,
        "dependency_afterok": args.afterok,
        "env_file": args.env_file,
        "completed_gates": completed_gates,
        "production_gate_satisfied": (
            not metadata["production"] or bool(completed_gates)
        ),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    if metadata["production"] and not completed_gates:
        raise PermissionError(
            "Production submission requires --require-completed SMOKE_JOB_ID"
        )

    result = run(submit_command(env_file, sbatch_args))
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id: {result.stdout!r}")
    output_path = expand_log_path(
        metadata["directives"].get("output"), metadata["job_name"], job_id
    )
    error_path = expand_log_path(
        metadata["directives"].get("error"), metadata["job_name"], job_id
    )
    state = {
        **preview,
        "job_id": job_id,
        "label": args.label or metadata["job_name"],
        "git_commit": git_commit(),
        "git_branch": branch,
        "submitted_at_utc": now_utc(),
        "status": "SUBMITTED",
        "output_path": output_path,
        "error_path": error_path,
    }
    save_state(state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def query_status(job_id: str) -> dict[str, str]:
    queue = run(
        ["squeue", "-h", "-j", job_id, "-o", "%i|%T|%M|%l|%R"], check=False
    ).stdout.strip()
    if queue:
        fields = queue.splitlines()[0].split("|", 4)
        return dict(zip(("job_id", "state", "elapsed", "limit", "reason"), fields))

    accounting = run(
        ["sacct", "-n", "-P", "-j", job_id,
         "--format=JobIDRaw,State,Elapsed,ExitCode,Start,End"],
        check=False,
    ).stdout.strip()
    for line in accounting.splitlines():
        fields = line.split("|")
        if fields and fields[0] == job_id:
            return dict(zip(
                ("job_id", "state", "elapsed", "exit_code", "start", "end"), fields
            ))
    return {"job_id": job_id, "state": "UNKNOWN"}


def cmd_status(args: argparse.Namespace) -> int:
    state = load_state(args.job_id)
    status = query_status(args.job_id)
    state["status"] = status.get("state", "UNKNOWN").split()[0]
    state["slurm"] = status
    save_state(state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def tail_text(path_value: str | None, max_bytes: int = 256_000_000) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace").replace("\r", "\n")


def parse_metrics(text: str) -> dict[str, Any]:
    current_epoch = None
    evaluations = []
    for line in text.splitlines():
        epoch_match = re.search(r"Epoch\s+(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        eer_match = re.search(r"cosine EER:\s*([0-9.]+)%", line)
        if eer_match:
            evaluations.append({"epoch_zero_based": current_epoch, "eer_percent": float(eer_match.group(1))})
            continue
        dcf2 = re.search(r"cosine minDCF\(10-2\):\s*([0-9.]+)", line)
        if dcf2 and evaluations:
            evaluations[-1]["min_dcf_1e2"] = float(dcf2.group(1))
            continue
        dcf3 = re.search(r"cosine minDCF\(10-3\):\s*([0-9.]+)", line)
        if dcf3 and evaluations:
            evaluations[-1]["min_dcf_1e3"] = float(dcf3.group(1))

    best = min(evaluations, key=lambda row: row["eer_percent"]) if evaluations else None
    error_lines = []
    error_pattern = re.compile(
        r"Traceback|RuntimeError|NCCL|CUDA out of memory|OutOfMemory|"
        r"Segmentation fault|Killed|DataLoader worker.*exited",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        if error_pattern.search(line):
            error_lines.append(line[-500:])
    smoke_complete = bool(re.search(
        r"Trainer\.fit stopped.*max_steps\s*=\s*12.*reached|"
        r"max_steps\s*=\s*12.*reached",
        text,
        flags=re.IGNORECASE,
    ))
    return {
        "evaluations": evaluations,
        "best": best,
        "error_lines": error_lines[-40:],
        "smoke_steps_complete": smoke_complete,
    }


def write_summary(state: dict[str, Any], metrics: dict[str, Any]) -> tuple[Path, Path]:
    summary_dir = state_root() / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(state["job_id"])
    payload = {
        "job_id": job_id,
        "label": state.get("label"),
        "status": state.get("status"),
        "git_commit": state.get("git_commit"),
        "script": state.get("script"),
        "slurm": state.get("slurm"),
        "resources": {
            "partition": state.get("partition"),
            "gpu_type": state.get("gpu_type"),
            "gpu_count": state.get("gpu_count"),
            "cpu_count": state.get("cpu_count"),
            "requested_hours": state.get("requested_hours"),
            "maximum_su": state.get("maximum_su"),
        },
        **metrics,
    }
    json_path = summary_dir / f"{job_id}.json"
    md_path = summary_dir / f"{job_id}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best = metrics.get("best") or {}
    lines = [
        f"# PSC Job {job_id}", "",
        f"- Label: `{state.get('label')}`",
        f"- State: `{state.get('status')}`",
        f"- Commit: `{state.get('git_commit')}`",
        f"- Script: `{state.get('script')}`",
        f"- Best EER: `{best.get('eer_percent', 'N/A')}`",
        f"- Best epoch (zero-based): `{best.get('epoch_zero_based', 'N/A')}`",
        f"- minDCF 1e-2: `{best.get('min_dcf_1e2', 'N/A')}`",
        f"- minDCF 1e-3: `{best.get('min_dcf_1e3', 'N/A')}`",
        f"- Error signals: `{len(metrics.get('error_lines', []))}`",
        f"- 12-step smoke complete: `{metrics.get('smoke_steps_complete', False)}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def cmd_collect(args: argparse.Namespace) -> int:
    state = load_state(args.job_id)
    status = query_status(args.job_id)
    state["status"] = status.get("state", "UNKNOWN").split()[0]
    state["slurm"] = status
    output = tail_text(state.get("output_path"))
    error = tail_text(state.get("error_path"))
    metrics = parse_metrics(output + "\n" + error)
    json_path, md_path = write_summary(state, metrics)
    state["summary_json"] = str(json_path)
    state["summary_markdown"] = str(md_path)
    save_state(state)
    print(json.dumps({"state": state, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.max_seconds
    seen_size = 0
    while True:
        state = load_state(args.job_id)
        status = query_status(args.job_id)
        current = status.get("state", "UNKNOWN").split()[0]
        print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {args.job_id} {current}")
        output_path = state.get("output_path")
        if output_path and Path(output_path).is_file():
            with Path(output_path).open(encoding="utf-8", errors="replace") as handle:
                handle.seek(seen_size)
                chunk = handle.read()
                seen_size = handle.tell()
            if chunk:
                print(chunk.replace("\r", "\n"), end="")
        if current in TERMINAL_STATES:
            collect_args = argparse.Namespace(job_id=args.job_id)
            return cmd_collect(collect_args)
        if time.monotonic() >= deadline:
            return 0
        time.sleep(args.interval)


def cmd_cancel(args: argparse.Namespace) -> int:
    if not args.yes:
        raise PermissionError("Cancellation requires --yes after explicit user approval")
    state = load_state(args.job_id)
    current = query_status(args.job_id).get("state", "UNKNOWN").split()[0]
    if current in TERMINAL_STATES:
        print(f"Job {args.job_id} is already terminal: {current}")
        return 0
    run(["scancel", args.job_id])
    state["status"] = "CANCEL_REQUESTED"
    state["cancel_requested_at_utc"] = now_utc()
    save_state(state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for path in sorted(state_root().glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append({
            "job_id": state.get("job_id"),
            "label": state.get("label"),
            "status": state.get("status"),
            "script": state.get("script"),
            "submitted_at_utc": state.get("submitted_at_utc"),
        })
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=cmd_preflight)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--script", required=True)
    submit.add_argument("--env-file")
    submit.add_argument("--label")
    submit.add_argument("--afterok")
    submit.add_argument("--require-completed", action="append")
    submit.add_argument("--confirm-production", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    for name, function in (("status", cmd_status), ("collect", cmd_collect)):
        command = subparsers.add_parser(name)
        command.add_argument("--job-id", required=True)
        command.set_defaults(func=function)

    watch = subparsers.add_parser("watch")
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--interval", type=int, default=30)
    watch.add_argument("--max-seconds", type=int, default=3600)
    watch.set_defaults(func=cmd_watch)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--job-id", required=True)
    cancel.add_argument("--yes", action="store_true")
    cancel.set_defaults(func=cmd_cancel)

    list_command = subparsers.add_parser("list")
    list_command.set_defaults(func=cmd_list)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
