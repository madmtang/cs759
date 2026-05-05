# CS759 Final Project

This repository contains a CS759 final project for evaluating sparse attention
implementations with `flashinfer-bench` on a Slurm GPU cluster.

The project code lives in `FinalProject/`:

- `baselines/`: reference sparse attention implementation and indexing helpers.
- `results/`: submitted solution code and recorded evaluation metrics.
- `run_eval.py`: Slurm-based evaluation driver for comparing the baseline and
  solution implementations.
- `utils.py`: shared helpers for dataset discovery and Slurm job execution.

For setup, dataset download, and evaluation instructions, see
[`FinalProject/README.md`](FinalProject/README.md).
