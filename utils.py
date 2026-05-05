"""Utilities for running flashinfer-bench evaluations through Slurm."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel


REPO_TOP_PATH = os.path.abspath(os.path.dirname(__file__))
REPO_PARENT_PATH = os.path.abspath(os.path.join(REPO_TOP_PATH, ".."))

DATASET_ROOT_CANDIDATES = [
    os.path.join(REPO_TOP_PATH, "datasets", "mlsys26-contest"),
    os.path.join(REPO_TOP_PATH, "mlsys26-contest"),
    os.path.join(REPO_PARENT_PATH, "datasets", "mlsys26-contest"),
    os.path.join(REPO_PARENT_PATH, "mlsys26-contest"),
]

KERNEL_LANGUAGE_EXTENSIONS = {
    "triton": ".py",
    "cuda": ".cu",
}


class EvalResult(BaseModel):
    """Result of evaluating one implementation."""

    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    latency_ms: float | None = None
    task_id: str = ""
    error: str | None = None
    stats: dict | None = None


def get_dataset_root() -> str:
    """Return the mlsys26-contest dataset root path."""
    for candidate in DATASET_ROOT_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Dataset root for 'mlsys26-contest' not found. Checked: "
        + ", ".join(DATASET_ROOT_CANDIDATES)
    )


def normalize_kernel_language(kernel_language: str) -> str:
    """Return a normalized kernel language or raise for unsupported values."""
    normalized = (kernel_language or "triton").lower()
    if normalized not in KERNEL_LANGUAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported kernel_language: {kernel_language}, expected one of "
            f"{list(KERNEL_LANGUAGE_EXTENSIONS)}"
        )
    return normalized


def _get_solution_build_spec(backend: str):
    """Return language-specific source metadata for flashinfer-bench."""
    from flashinfer_bench.data import SupportedLanguages

    kernel_language = normalize_kernel_language(backend)
    if kernel_language == "triton":
        return {
            "language": SupportedLanguages.TRITON,
            "source_path": "main.py",
            "entry_point": "main.py::run",
            "destination_passing_style": False,
        }
    return {
        "language": SupportedLanguages.CUDA,
        "source_path": "main.cu",
        "entry_point": "main.cu::run",
        "destination_passing_style": True,
    }


def eval_kernel(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    backend: str = "triton",
    timeout: int = 60,
) -> EvalResult:
    """Evaluate a kernel against the flashinfer-bench reference."""
    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import (
        BuildSpec,
        EvaluationStatus,
        Solution,
        SourceFile,
        TraceSet,
    )

    trace_set = TraceSet.from_path(dataset_root)
    solution_name = f"slurm_eval_{uuid.uuid4().hex[:8]}"
    build_spec = _get_solution_build_spec(backend)

    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="slurm_eval",
        spec=BuildSpec(
            language=build_spec["language"],
            target_hardware=["cuda"],
            entry_point=build_spec["entry_point"],
            dependencies=[],
            destination_passing_style=build_spec["destination_passing_style"],
        ),
        sources=[SourceFile(path=build_spec["source_path"], content=kernel_code)],
    )
    trace_set.solutions.setdefault(task_id, []).append(solution)
    trace_set._solution_by_name[solution_name] = solution

    config = BenchmarkConfig(
        warmup_runs=3,
        iterations=5,
        num_trials=1,
        definitions=[task_id],
        solutions=[solution_name],
        timeout_seconds=timeout,
    )

    benchmark = Benchmark(trace_set, config)
    try:
        result_ts = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()

    traces = result_ts.traces.get(task_id, [])
    error_statuses = {
        EvaluationStatus.COMPILE_ERROR,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.TIMEOUT,
    }
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status in error_statuses:
            return EvalResult(
                compiled=(ev.status != EvaluationStatus.COMPILE_ERROR),
                task_id=task_id,
                error=f"{ev.status.value}: {ev.log}",
            )

    latencies, ref_latencies, speedups = [], [], []
    rel_errors, abs_errors = [], []
    for trace in traces:
        ev = trace.evaluation
        if ev and ev.status == EvaluationStatus.PASSED:
            latencies.append(ev.performance.latency_ms)
            ref_latencies.append(ev.performance.reference_latency_ms)
            speedups.append(ev.performance.speedup_factor)
            rel_errors.append(ev.correctness.max_relative_error)
            abs_errors.append(ev.correctness.max_absolute_error)

    if not latencies:
        return EvalResult(task_id=task_id, error="No evaluation results")

    n = len(latencies)
    return EvalResult(
        compiled=True,
        correct=True,
        speedup=sum(speedups) / n,
        latency_ms=sum(latencies) / n,
        task_id=task_id,
        stats={
            "reference_latency_ms": sum(ref_latencies) / n,
            "max_relative_error": sum(rel_errors) / n,
            "max_absolute_error": sum(abs_errors) / n,
            "total_workloads": n,
        },
    )


def _split_sbatch_args(
    value: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if str(v).strip())
    return tuple(shlex.split(str(value)))


@dataclass(frozen=True)
class SlurmEvalConfig:
    """Configuration for one-GPU Slurm evaluation jobs."""

    partition: str | None = "instruction"
    gres: str | None = "gpu:1"
    cpus_per_task: int = 1
    time_limit: str = "30"
    cuda_module: str | None = "nvidia/cuda/13.0.0"
    python: str = sys.executable
    setup: str | None = None
    work_dir: str = os.path.join(REPO_TOP_PATH, "outputs", "slurm_eval_jobs")
    poll_interval: float = 2.0
    job_timeout: int = 900
    keep_jobs: bool = False
    extra_sbatch_args: tuple[str, ...] = ()

    @classmethod
    def from_args(cls, args) -> "SlurmEvalConfig":
        return cls(
            partition=getattr(args, "slurm_partition", "instruction") or None,
            gres=getattr(args, "slurm_gres", "gpu:1") or None,
            cpus_per_task=int(getattr(args, "slurm_cpus_per_task", 1)),
            time_limit=str(getattr(args, "slurm_time", "30")),
            cuda_module=getattr(args, "slurm_cuda_module", "nvidia/cuda/13.0.0")
            or None,
            python=getattr(args, "slurm_python", sys.executable) or sys.executable,
            setup=getattr(args, "slurm_setup", None) or None,
            work_dir=getattr(args, "slurm_work_dir", cls.work_dir),
            poll_interval=float(getattr(args, "slurm_poll_interval", 2.0)),
            job_timeout=int(getattr(args, "slurm_job_timeout", 900)),
            keep_jobs=bool(getattr(args, "slurm_keep_jobs", False)),
            extra_sbatch_args=_split_sbatch_args(
                getattr(args, "slurm_sbatch_args", None)
            ),
        )


def create_slurm_eval_fn(
    config: SlurmEvalConfig,
) -> Callable[[str, str, str, str, int], EvalResult]:
    """Return an eval function with the same signature as eval_kernel()."""

    def _slurm_eval(
        kernel_code: str,
        task_id: str,
        dataset_root: str,
        backend: str = "triton",
        timeout: int = 60,
    ) -> EvalResult:
        return eval_kernel_slurm(
            kernel_code=kernel_code,
            task_id=task_id,
            dataset_root=dataset_root,
            backend=backend,
            timeout=timeout,
            config=config,
        )

    return _slurm_eval


def eval_kernel_slurm(
    kernel_code: str,
    task_id: str,
    dataset_root: str,
    backend: str,
    timeout: int,
    config: SlurmEvalConfig,
) -> EvalResult:
    """Submit one kernel evaluation to Slurm and return the parsed result."""
    job_dir = _create_job_dir(config.work_dir, task_id)
    payload_path = job_dir / "payload.json"
    result_path = job_dir / "result.json"
    script_path = job_dir / "eval.sbatch"

    payload = {
        "kernel_code": kernel_code,
        "task_id": task_id,
        "dataset_root": dataset_root,
        "backend": backend,
        "timeout": timeout,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    (job_dir / "worker.py").write_text(_worker_source(), encoding="utf-8")
    script_path.write_text(_job_script(config, _job_name(task_id)), encoding="utf-8")

    try:
        submit = subprocess.run(
            ["sbatch", "--parsable", script_path.name],
            cwd=job_dir,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError:
        _cleanup_after_submission_failure(job_dir, config)
        return EvalResult(
            task_id=task_id,
            error="Slurm backend requested, but 'sbatch' was not found on PATH.",
        )
    except subprocess.TimeoutExpired:
        _cleanup_after_submission_failure(job_dir, config)
        return EvalResult(task_id=task_id, error="Timed out while submitting sbatch job.")

    if submit.returncode != 0:
        _cleanup_after_submission_failure(job_dir, config)
        return EvalResult(
            task_id=task_id,
            error=(
                "sbatch submission failed: "
                + _join_nonempty(submit.stdout, submit.stderr)
            ),
        )

    job_id = _parse_job_id(submit.stdout)
    if not job_id:
        _cleanup_after_submission_failure(job_dir, config)
        return EvalResult(
            task_id=task_id,
            error=f"Could not parse Slurm job id from sbatch output: {submit.stdout!r}",
        )

    result = _wait_for_result(job_id, result_path, job_dir, task_id, config)
    if (
        not config.keep_jobs
        and result_path.exists()
        and not _is_worker_failure(result)
    ):
        shutil.rmtree(job_dir, ignore_errors=True)
    return result


def _worker_source() -> str:
    repo_root = json.dumps(REPO_TOP_PATH)
    return f"""#!/usr/bin/env python3
import json
import multiprocessing
import os
import sys
import traceback

REPO_ROOT = {repo_root}
sys.path.insert(0, REPO_ROOT)


def write_result(payload):
    tmp_path = "result.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, "result.json")


def main():
    payload = {{}}
    try:
        with open("payload.json", "r") as f:
            payload = json.load(f)

        from utils import eval_kernel

        result = eval_kernel(
            kernel_code=payload["kernel_code"],
            task_id=payload["task_id"],
            dataset_root=payload["dataset_root"],
            backend=payload["backend"],
            timeout=int(payload["timeout"]),
        )
        if hasattr(result, "model_dump"):
            result_payload = result.model_dump()
        else:
            result_payload = result.dict()
        write_result(result_payload)
    except BaseException:
        write_result(
            {{
                "compiled": False,
                "correct": False,
                "speedup": 0.0,
                "task_id": payload.get("task_id", ""),
                "error": "Slurm worker failed:\\n" + traceback.format_exc(),
            }}
        )
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
"""


def _create_job_dir(work_dir: str, task_id: str) -> Path:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))[:80] or "task"
    for _ in range(10):
        candidate = root / f"{safe_task}_{os.urandom(4).hex()}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique Slurm eval directory under {root}")


def _cleanup_after_submission_failure(job_dir: Path, config: SlurmEvalConfig):
    if not config.keep_jobs:
        shutil.rmtree(job_dir, ignore_errors=True)


def _job_name(task_id: str) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))[:40] or "task"
    return f"ka_{safe_task}"


def _job_script(config: SlurmEvalConfig, job_name: str) -> str:
    lines = [
        "#!/usr/bin/env zsh",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=job.out",
        "#SBATCH --error=job.err",
    ]
    if config.partition:
        lines.append(f"#SBATCH --partition={config.partition}")
    if config.time_limit:
        lines.append(f"#SBATCH --time={config.time_limit}")
    if config.gres:
        lines.append(f"#SBATCH --gres={config.gres}")
    if config.cpus_per_task:
        lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")
    for arg in config.extra_sbatch_args:
        lines.append(f"#SBATCH {arg}")

    lines.extend(["", "set -eu"])
    if config.cuda_module:
        lines.append(f"module load {shlex.quote(config.cuda_module)}")
    lines.extend(
        [
            'cd "$SLURM_SUBMIT_DIR"',
            f"export PYTHONPATH={shlex.quote(REPO_TOP_PATH)}:${{PYTHONPATH:-}}",
        ]
    )
    if config.setup:
        lines.extend(["", config.setup])
    lines.extend(["", f"{shlex.quote(config.python)} worker.py"])
    return "\n".join(lines) + "\n"


def _wait_for_result(
    job_id: str,
    result_path: Path,
    job_dir: Path,
    task_id: str,
    config: SlurmEvalConfig,
) -> EvalResult:
    start = time.monotonic()
    deadline = time.monotonic() + max(config.job_timeout, 1)
    poll_interval = max(config.poll_interval, 0.2)
    seen_active = False

    while time.monotonic() < deadline:
        if result_path.exists():
            return _read_result(result_path, task_id, job_dir)

        if _job_is_active(job_id):
            seen_active = True
        else:
            if not seen_active and time.monotonic() - start < 10:
                time.sleep(min(poll_interval, 1.0))
                continue
            for _ in range(5):
                if result_path.exists():
                    return _read_result(result_path, task_id, job_dir)
                time.sleep(0.5)
            break

        time.sleep(poll_interval)

    if time.monotonic() >= deadline:
        _cancel_job(job_id)
        return EvalResult(
            task_id=task_id,
            error=(
                f"Slurm job {job_id} timed out after {config.job_timeout}s. "
                f"Job directory kept at {job_dir}."
            ),
        )

    return EvalResult(
        task_id=task_id,
        error=(
            f"Slurm job {job_id} finished without result.json. "
            + _diagnostics(job_id, job_dir)
        ),
    )


def _read_result(result_path: Path, task_id: str, job_dir: Path) -> EvalResult:
    try:
        with result_path.open("r") as f:
            return EvalResult(**json.load(f))
    except Exception as exc:
        return EvalResult(
            task_id=task_id,
            error=f"Failed to read Slurm result {result_path}: {exc}. Job dir: {job_dir}",
        )


def _job_is_active(job_id: str) -> bool:
    try:
        proc = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    if proc.returncode != 0:
        return True
    return bool(proc.stdout.strip())


def _cancel_job(job_id: str):
    try:
        subprocess.run(
            ["scancel", job_id],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _parse_job_id(output: str) -> str | None:
    first_line = (output or "").strip().splitlines()[0] if output.strip() else ""
    candidate = first_line.split(";", 1)[0].strip()
    if candidate and candidate[0].isdigit():
        return candidate
    match = re.search(r"\d+", output or "")
    return match.group(0) if match else None


def _is_worker_failure(result: EvalResult) -> bool:
    return bool(
        result.error
        and (
            result.error.startswith("Slurm worker failed:")
            or result.error.startswith("Failed to read Slurm result")
        )
    )


def _diagnostics(job_id: str, job_dir: Path) -> str:
    parts = [f"Job directory kept at {job_dir}."]
    for name in ("job.out", "job.err"):
        tail = _tail_file(job_dir / name)
        if tail:
            parts.append(f"{name} tail:\n{tail}")
    sacct = _sacct_summary(job_id)
    if sacct:
        parts.append(f"sacct:\n{sacct}")
    return "\n".join(parts)


def _tail_file(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def _sacct_summary(job_id: str) -> str:
    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode",
                "--noheader",
                "--parsable2",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _join_nonempty(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())
