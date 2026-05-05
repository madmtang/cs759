# CS759 Evaluation

This repository evaluates two implementations of the sparse attention task:

- Baseline: `baselines/attention.py`
- Solution: `results/solution.py`

Evaluation is launched from `run_eval.py`. The script is intended to run on the
Euler cluster login node and submits one Slurm GPU job per implementation. Do
not run the benchmark workload on a local machine.

## 0. Install Dependencies

From the repository root, install the Python dependencies:

```bash
pip install -r requirements.txt
```

The evaluation path depends on `flashinfer-bench`, `pydantic`, `torch`, and
`triton`. The Euler environment must also provide Slurm commands such as
`sbatch`, `squeue`, and `scancel`.

## 1. Download the Dataset

Download the contest dataset into `datasets/mlsys26-contest`:

```bash
mkdir datasets
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest datasets/mlsys26-contest
```

The final dataset path should contain a `definitions/` directory. If
`definitions/` is missing, the dataset was not cloned to the expected location
or Git LFS did not fetch the required files.

## 2. Check the Files Being Evaluated

By default, `run_eval.py` evaluates:

```text
baselines/attention.py
results/solution.py
```

The default task id is read from `results/metrics.json` when available. If that
file is missing or does not contain a task id, the script falls back to:

```text
dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
```

## 3. Run the Evaluation

The quickest start:

```bash
python run_eval.py --dataset-root datasets/mlsys26-contest/
```

The script submits each implementation as a separate Slurm job with these
defaults:

```text
partition: instruction
gres: gpu:1
cpus-per-task: 1
time: 30
cuda module: nvidia/cuda/13.0.0
```

If Euler requires different settings, override them on the command line:

```bash
python run_eval.py \
  --dataset-root datasets/mlsys26-contest/ \
  --slurm-partition instruction \
  --slurm-gres gpu:1 \
  --slurm-time 30
```

If you use a virtual environment or need extra module setup inside each Slurm
job, pass a shell snippet with `--slurm-setup`:

```bash
python run_eval.py \
  --dataset-root datasets/mlsys26-contest/ \
  --slurm-setup "source .venv/bin/activate"
```

## 4. Outputs

The script prints submission and result status for both implementations:

```text
Submitting baseline: ...
baseline passed: speedup=..., latency_ms=...
Submitting solution: ...
solution passed: speedup=..., latency_ms=...
```

It writes the full JSON report to:

```text
results/slurm_eval_metrics.json
```

That report includes:

- The task id and dataset root used for the run
- Slurm configuration
- Baseline compile/correctness/speedup/latency
- Solution compile/correctness/speedup/latency
- Solution-vs-baseline comparison when both latencies are available

## 5. Useful Options

Use `--fail-on-error` to make the script exit nonzero if either implementation
fails compilation or correctness:

```bash
python run_eval.py --dataset-root datasets/mlsys26-contest/ --fail-on-error
```

Use `--slurm-keep-jobs` to keep generated Slurm job directories for debugging:

```bash
python run_eval.py --dataset-root datasets/mlsys26-contest/ --slurm-keep-jobs
```

Kept job directories are written under:

```text
outputs/slurm_eval_jobs/
```

Each job directory contains the generated `eval.sbatch`, `worker.py`,
`payload.json`, and Slurm stdout/stderr files.

## 6. Troubleshooting

If both baseline and solution fail with `No evaluation results`, first check
that `--dataset-root` points to the trace-set root, not just the parent
`datasets/` directory. The correct path should normally be:

```text
datasets/mlsys26-contest/
```

If `sbatch` is not found, run the script from the Euler cluster environment
rather than locally.

If CUDA, Torch, Triton, or `flashinfer-bench` imports fail inside the Slurm job,
use `--slurm-setup` or `--slurm-python` so the worker uses the same Python
environment where dependencies were installed.
