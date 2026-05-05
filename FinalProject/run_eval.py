"""Run baseline and solution evaluations on a Slurm GPU cluster.

This script is intended to be launched on the Euler login node, not on a local
machine. It submits one Slurm job per implementation and writes a combined
metrics JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import (
    REPO_TOP_PATH,
    SlurmEvalConfig,
    create_slurm_eval_fn,
    get_dataset_root,
)


REPO_ROOT = Path(REPO_TOP_PATH)
DEFAULT_BASELINE = REPO_ROOT / "baselines" / "attention.py"
DEFAULT_SOLUTION = REPO_ROOT / "results" / "solution.py"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "slurm_eval_metrics.json"
DEFAULT_TASK_ID = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit baseline and solution implementations to Slurm and collect "
            "flashinfer-bench metrics."
        )
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help=(
            "flashinfer-bench definition name. Defaults to results/metrics.json "
            f"if present, otherwise {DEFAULT_TASK_ID}."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Explicit dataset root path on Euler. Use this if auto-discovery "
            "cannot find it."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("triton", "cuda"),
        default="triton",
        help="Kernel backend passed to flashinfer-bench.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-solution flashinfer-bench timeout in seconds.",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Path to the baseline implementation.",
    )
    parser.add_argument(
        "--solution",
        default=str(DEFAULT_SOLUTION),
        help="Path to the solution implementation.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the combined metrics JSON.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit nonzero if either implementation fails to compile or pass correctness.",
    )

    parser.add_argument("--slurm-partition", default="instruction")
    parser.add_argument("--slurm-gres", default="gpu:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=1)
    parser.add_argument("--slurm-time", default="30")
    parser.add_argument("--slurm-cuda-module", default="nvidia/cuda/13.0.0")
    parser.add_argument("--slurm-python", default=sys.executable)
    parser.add_argument(
        "--slurm-setup",
        default=None,
        help="Shell setup snippet inserted before worker.py, for example venv activation.",
    )
    parser.add_argument(
        "--slurm-work-dir",
        default=str(REPO_ROOT / "outputs" / "slurm_eval_jobs"),
    )
    parser.add_argument("--slurm-poll-interval", type=float, default=2.0)
    parser.add_argument("--slurm-job-timeout", type=int, default=900)
    parser.add_argument("--slurm-keep-jobs", action="store_true")
    parser.add_argument(
        "--slurm-sbatch-arg",
        dest="slurm_sbatch_args",
        action="append",
        default=(),
        help=(
            "Additional #SBATCH argument, e.g. --slurm-sbatch-arg='--constraint=a100'."
        ),
    )

    args = parser.parse_args(argv)
    args.task_id = args.task_id or _default_task_id()
    args.dataset_root = _resolve_dataset_root(args.dataset_root, parser)
    args.baseline = _resolve_existing_file(args.baseline, parser)
    args.solution = _resolve_existing_file(args.solution, parser)
    args.output = Path(args.output).expanduser().resolve()
    args.slurm_work_dir = str(Path(args.slurm_work_dir).expanduser().resolve())
    return args


def _default_task_id() -> str:
    metrics_path = REPO_ROOT / "results" / "metrics.json"
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_TASK_ID
    return str(data.get("task_id") or DEFAULT_TASK_ID)


def _resolve_dataset_root(
    dataset_root: str | None, parser: argparse.ArgumentParser
) -> str:
    if dataset_root:
        path = Path(dataset_root).expanduser().resolve()
        if not path.is_dir():
            parser.error(f"--dataset-root does not exist or is not a directory: {path}")
        return str(path)

    try:
        return get_dataset_root()
    except FileNotFoundError as exc:
        parser.error(f"{exc}. Pass --dataset-root with the dataset path on Euler.")
    raise AssertionError("parser.error should have exited")


def _resolve_existing_file(path_value: str, parser: argparse.ArgumentParser) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        parser.error(f"Implementation file does not exist: {path}")
    return path


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return dict(result)


def _evaluate_one(
    eval_fn, name: str, path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    print(f"Submitting {name}: {path}", flush=True)
    result = eval_fn(
        path.read_text(encoding="utf-8"),
        args.task_id,
        args.dataset_root,
        args.backend,
        args.timeout,
    )
    result_dict = _result_to_dict(result)
    status = (
        "passed"
        if result_dict.get("compiled") and result_dict.get("correct")
        else "failed"
    )
    speedup = float(result_dict.get("speedup") or 0.0)
    print(
        (
            f"{name} {status}: speedup={speedup:.4f}, "
            f"latency_ms={result_dict.get('latency_ms')}"
        ),
        flush=True,
    )
    if result_dict.get("error"):
        print(f"{name} error: {result_dict['error']}", flush=True)
    return {"path": str(path), "result": result_dict}


def _comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = results["baseline"]["result"]
    solution = results["solution"]["result"]
    comparison: dict[str, Any] = {
        "baseline_correct": bool(baseline.get("compiled") and baseline.get("correct")),
        "solution_correct": bool(solution.get("compiled") and solution.get("correct")),
        "speedup_delta": float(solution.get("speedup") or 0.0)
        - float(baseline.get("speedup") or 0.0),
    }

    baseline_latency = baseline.get("latency_ms")
    solution_latency = solution.get("latency_ms")
    if baseline_latency and solution_latency:
        comparison["solution_vs_baseline_latency_speedup"] = float(
            baseline_latency
        ) / float(solution_latency)

    return comparison


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    slurm_config = SlurmEvalConfig.from_args(args)
    eval_fn = create_slurm_eval_fn(slurm_config)

    results = {
        "baseline": _evaluate_one(eval_fn, "baseline", args.baseline, args),
        "solution": _evaluate_one(eval_fn, "solution", args.solution, args),
    }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "dataset_root": args.dataset_root,
        "backend": args.backend,
        "timeout": args.timeout,
        "slurm": asdict(slurm_config),
        "implementations": results,
        "comparison": _comparison(results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metrics: {args.output}", flush=True)

    if args.fail_on_error:
        failures = [
            name
            for name, item in results.items()
            if not (item["result"].get("compiled") and item["result"].get("correct"))
        ]
        if failures:
            print(f"Failed implementations: {', '.join(failures)}", flush=True)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
