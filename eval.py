"""
Kernel evaluation using flashinfer-bench Python API directly.
"""

import json
import logging
import uuid

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    BuildSpec,
    EvaluationStatus,
    Solution,
    SourceFile,
    SupportedLanguages,
    TraceSet,
)
from pydantic import BaseModel

from agent.utils import normalize_kernel_language

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    """Result of evaluating a single kernel."""

    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    latency_ms: float | None = None
    task_id: str = ""
    error: str | None = None
    stats: dict | None = None


def calculate_score(metric: EvalResult):
    """Return (compiled, correct, speedup) tuple for ranking."""
    if metric is None:
        return (0, 0, 0)
    if not metric.compiled:
        return (0, 0, 0)
    if not metric.correct:
        return (1, 0, 0)
    return (1, 1, metric.speedup)


def read_metrics(metrics_path: str, full: bool = False):
    """
    Read metrics from a JSON file.

    Returns:
        If full: EvalResult object
        Otherwise: tuple (correct: bool, speedup: float)
    """
    with open(metrics_path, "r") as f:
        data = json.load(f)

    if full:
        return EvalResult(**data)

    if data.get("compiled") and data.get("correct"):
        return (True, data.get("speedup", 0.0))
    return (False, 0.0)


def create_eval_fn(
    backend: str = "local", dataset_name: str = "mlsys26-contest", remote_fn=None
):
    """Factory to create eval function based on backend.

    Args:
        backend: "local" for local GPU, "modal" for Modal remote GPU.
        dataset_name: Dataset subdirectory name (used by modal backend).
        remote_fn: Modal remote function (required when backend="modal").

    Returns:
        Callable with same signature as eval_kernel.
    """
    if backend == "local":
        return eval_kernel
    elif backend == "modal":
        if remote_fn is None:
            raise ValueError("remote_fn is required for modal backend")

        def _modal_eval(
            kernel_code, task_id, dataset_root, backend="triton", timeout=60
        ):
            result_dict = remote_fn.remote(
                kernel_code, task_id, dataset_name, backend, timeout
            )
            return EvalResult(**result_dict)

        return _modal_eval
    else:
        raise ValueError(f"Unknown eval backend: {backend}")


def _get_solution_build_spec(backend: str):
    """Return language-specific source metadata for flashinfer-bench."""
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
    """
    Evaluate a kernel against the reference using flashinfer-bench API.

    Args:
        kernel_code: Source code of the kernel to evaluate.
        task_id: Definition/problem name (e.g. "moe_fp8_block_scale_...").
        dataset_root: Path to the dataset root directory.
        backend: "triton" or "cuda".
        timeout: Timeout in seconds per solution evaluation.

    Returns:
        EvalResult with compiled, correct, speedup, latency_ms, etc.
    """
    trace_set = TraceSet.from_path(dataset_root)

    solution_name = f"agent_{uuid.uuid4().hex[:8]}"
    build_spec = _get_solution_build_spec(backend)

    solution = Solution(
        name=solution_name,
        definition=task_id,
        author="agent",
        spec=BuildSpec(
            language=build_spec["language"],
            target_hardware=["cuda"],
            entry_point=build_spec["entry_point"],
            dependencies=[],
            destination_passing_style=build_spec["destination_passing_style"],
        ),
        sources=[SourceFile(path=build_spec["source_path"], content=kernel_code)],
    )

    # Inject solution into in-memory trace set
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

    # Find first error
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

    # Aggregate PASSED results
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
        return EvalResult(
            task_id=task_id,
            error="No evaluation results",
        )

    n = len(latencies)
    avg_speedup = sum(speedups) / n
    return EvalResult(
        compiled=True,
        correct=True,
        speedup=avg_speedup,
        latency_ms=sum(latencies) / n,
        task_id=task_id,
        stats={
            "reference_latency_ms": sum(ref_latencies) / n,
            "max_relative_error": sum(rel_errors) / n,
            "max_absolute_error": sum(abs_errors) / n,
            "total_workloads": n,
        },
    )
