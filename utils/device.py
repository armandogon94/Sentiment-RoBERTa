"""Device selection and a capability report, with the MPS constraints stated in code.

Hard rule for this repo, recorded in ``docs/adr/0003-mps-constraints.md``: **one device per
process**. A verified failure on this exact torch build is that a CPU transformer loop
deadlocks at 0% CPU if an MPS matmul ran earlier in the same process. So the device is
resolved once, at the top of ``train.py``, and nothing later moves a tensor to a different
one. ``torch.nn.MultiheadAttention`` is separately known to hang on MPS here; this repo
never constructs one (HuggingFace RoBERTa uses its own attention), and if custom attention
were ever needed the replacement is ``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Literal

import torch

DeviceName = Literal["auto", "mps", "cpu"]


def resolve_device(requested: DeviceName = "auto") -> torch.device:
    """Resolve the configured device to a concrete one, degrading to CPU cleanly.

    CI runs on ``ubuntu-latest`` where MPS does not exist, so ``auto`` must not raise
    there — ``tests/test_device.py`` covers exactly that path.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("DEVICE: mps was requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def low_power_mode() -> bool | None:
    """macOS Low Power Mode state, or ``None`` where it cannot be determined.

    Recorded in ``run_meta.json`` because on this hardware it materially changes timings —
    every timing this repo publishes names the mode it was measured under.
    """
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["/usr/bin/pmset", "-g"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform dependent
        return None
    for line in out.stdout.splitlines():
        if "lowpowermode" in line.replace(" ", "").lower():
            return line.strip().split()[-1] == "1"
    return None


def capability_report(device: torch.device) -> dict[str, Any]:
    """Everything about the machine that could change a number. Goes into ``run_meta.json``."""
    return {
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": int(torch.get_num_threads()),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "low_power_mode": low_power_mode(),
        "loadavg_1m": round(os.getloadavg()[0], 2),
    }


def power_mode_label() -> str:
    """Human-readable power-mode suffix for a published timing, e.g. ``Low Power Mode ON``."""
    state = low_power_mode()
    if state is None:
        return "power mode unknown"
    return "Low Power Mode ON" if state else "Low Power Mode OFF"
