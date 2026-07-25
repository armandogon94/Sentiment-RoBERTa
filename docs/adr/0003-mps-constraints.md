# ADR 0003 — One device per process, and no `MultiheadAttention` on MPS

**Status:** accepted · **Date:** 2026-07-25

## Context

This repo trains on Apple Silicon through PyTorch's Metal backend (`torch 2.13.0`, MPS available).
Two failure modes on this exact build are known and reproduced, and both are the kind that present
as a hang rather than an exception — so they cost hours before they cost a stack trace.

1. **Mixing CPU and MPS tensor workloads in one process can deadlock.** A CPU transformer loop was
   observed sitting at 0% CPU indefinitely after an MPS matmul had run earlier in the same process.
   No error, no traceback, no progress.
2. **`torch.nn.MultiheadAttention` hangs outright on MPS.** HuggingFace's RoBERTa implements its own
   attention and does not use it, so this repo is not currently exposed — but any hand-rolled
   attention module would be.

A third, milder constraint shapes what can be claimed: MPS gives roughly a **1.9×** speedup over CPU
on dense fp32 matmul on this machine (8.27 ms/op vs 16.02 ms/op at 2048³), not the 5–10× that CUDA
comparisons in papers might lead a reader to expect. There is no mixed precision and no
`torch.compile` on this path.

## Decision

1. **One device per process.** `utils.device.resolve_device` is called exactly once, at the top of
   `train.py`, and the resulting `torch.device` is threaded through everything downstream. No module
   selects a device on its own; `datasets/torch_dataset.py` builds CPU tensors and the training loop
   is the only place that calls `.to(device)`. There is no code path where a CPU stage and an MPS
   stage run in the same interpreter.
2. **Never construct `torch.nn.MultiheadAttention`.** If custom attention is ever needed, use
   `torch.nn.functional.scaled_dot_product_attention`.
3. **Every published timing names its device and its macOS power mode.** `utils.device.low_power_mode`
   shells out to `pmset -g`, and the result goes into `run_meta.json` and into the `Train time`
   column of the results table. On this hardware the mode materially changes the number, so a timing
   without it is not reproducible.
4. **`DEVICE: auto` must degrade to CPU cleanly.** CI runs on `ubuntu-latest` where MPS does not
   exist. `tests/test_models.py::test_resolve_device_degrades_to_cpu_where_mps_is_absent` covers it,
   and `cfg/smoke.yaml` pins `DEVICE: cpu` explicitly rather than relying on the fallback.

## Consequences

- Figure generation reloads the run's saved checkpoint onto the run's own device rather than
  evaluating on CPU "because it is only a few examples". That would be exactly the mixed-device
  process this ADR forbids.
- Timings in this repo are not comparable to CUDA figures in the literature, and the README says so
  rather than leaving the reader to assume.
- The `Train time` column is wider than it would otherwise be. That is the correct trade: a bare
  "18m 42s" is not a reproducible measurement.

## Alternatives considered

- **Force CPU everywhere for determinism.** Rejected: it roughly doubles every training run for no
  gain in the numbers, and the 45-minute cap is the binding constraint on what can be published.
- **Detect and warn instead of forbidding.** Rejected: the failure is a silent hang, so a warning
  would be read after the hang rather than before it. Structure is a better guard than a log line.
