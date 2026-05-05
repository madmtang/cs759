"""
Remote kernel evaluation on Modal GPU.

Usage: set eval_backend="modal" in config or --eval_backend modal on CLI.
The Modal app is started programmatically via app.run() in main.py,
no `modal deploy` or `modal run` CLI commands needed.
"""

import logging
import os

import modal

logger = logging.getLogger(__name__)

MODAL_APP_NAME = "flashinfer-bench-agent"
VOLUME_NAME = "flashinfer-trace"
DATASET_PATH = "/data"
MODAL_PIP_PACKAGES = (
    "flashinfer-bench==0.1.2",
    "apache-tvm-ffi==0.1.10",
    "torch==2.11.0",
    "triton==3.6.0",
    "pydantic==2.13.1",
)


def create_modal_app(gpu_type: str = "B200"):
    """Create Modal app, remote eval function, and dataset volume.

    Args:
        gpu_type: GPU type string for Modal (e.g. "B200", "A100", "H100").

    Returns:
        Tuple of (app, remote_eval_kernel, dataset_vol).
    """
    app = modal.App(MODAL_APP_NAME)

    image = modal.Image.from_registry(
        "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    ).pip_install(*MODAL_PIP_PACKAGES)

    dataset_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu=gpu_type,
        volumes={DATASET_PATH: dataset_vol},
        timeout=300,
        serialized=True,
    )
    def remote_eval_kernel(
        kernel_code: str,
        task_id: str,
        dataset_name: str,
        backend: str = "triton",
        timeout: int = 60,
    ) -> dict:
        """Evaluate a kernel on Modal GPU. Mirrors eval.eval_kernel() logic."""
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

        dataset_root = os.path.join(DATASET_PATH, dataset_name)
        trace_set = TraceSet.from_path(dataset_root)

        solution_name = f"agent_{uuid.uuid4().hex[:8]}"
        kernel_language = (backend or "triton").lower()
        if kernel_language == "triton":
            language = SupportedLanguages.TRITON
            source_path = "main.py"
            entry_point = "main.py::run"
            destination_passing_style = False
        elif kernel_language == "cuda":
            language = SupportedLanguages.CUDA
            source_path = "main.cu"
            entry_point = "main.cu::run"
            destination_passing_style = True
        else:
            raise ValueError(
                f"Unsupported kernel language: {backend}, expected 'triton' or 'cuda'"
            )

        solution = Solution(
            name=solution_name,
            definition=task_id,
            author="agent",
            spec=BuildSpec(
                language=language,
                target_hardware=["cuda"],
                entry_point=entry_point,
                dependencies=[],
                destination_passing_style=destination_passing_style,
            ),
            sources=[SourceFile(path=source_path, content=kernel_code)],
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
                return {
                    "compiled": ev.status != EvaluationStatus.COMPILE_ERROR,
                    "task_id": task_id,
                    "error": f"{ev.status.value}: {ev.log}",
                }

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
            return {"task_id": task_id, "error": "No evaluation results"}

        n = len(latencies)
        return {
            "compiled": True,
            "correct": True,
            "speedup": sum(speedups) / n,
            "latency_ms": sum(latencies) / n,
            "task_id": task_id,
            "stats": {
                "reference_latency_ms": sum(ref_latencies) / n,
                "max_relative_error": sum(rel_errors) / n,
                "max_absolute_error": sum(abs_errors) / n,
                "total_workloads": n,
            },
        }

    return app, remote_eval_kernel, dataset_vol


def ensure_dataset_synced(
    dataset_vol: modal.Volume, local_dataset_root: str, dataset_name: str
):
    """Ensure local dataset is uploaded to Modal Volume via batch_upload."""
    if not os.path.isdir(local_dataset_root):
        raise FileNotFoundError(
            f"Local dataset root does not exist or is not a directory: {local_dataset_root}"
        )

    try:
        if dataset_vol.listdir(f"/{dataset_name}"):
            logger.info(
                f"Dataset '{dataset_name}' found in Modal Volume, skipping upload"
            )
            return
    except Exception:
        pass

    logger.info(f"Uploading dataset '{dataset_name}' to Modal Volume...")
    with dataset_vol.batch_upload() as batch:
        batch.put_directory(local_dataset_root, f"/{dataset_name}")
    logger.info(f"Dataset uploaded to Modal Volume '{VOLUME_NAME}'")
