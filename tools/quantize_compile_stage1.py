#!/usr/bin/env python3
"""Quantize and compile stage1 HARs for multiple DeiM-V2 variants."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from hailo_sdk_client.runner.client_runner import ClientRunner


CALIB_ROOT = Path("/local/shared_with_docker/calib")
VARIANTS: Dict[str, Dict[str, Path]] = {
    "atto": {
        "har": Path("dst_models/deimv2_atto_stage1.har"),
        "calib": CALIB_ROOT / "calib_set_a",
    },
    "femto": {
        "har": Path("dst_models/deimv2_femto_stage1.har"),
        "calib": CALIB_ROOT / "calib_set_f",
    },
    "pico": {
        "har": Path("dst_models/deimv2_pico_stage1.har"),
        "calib": CALIB_ROOT / "calib_set_p",
    },
    "n": {
        "har": Path("dst_models/deimv2_n_stage1.har"),
        "calib": CALIB_ROOT / "calib_set_n",
    },
}


def load_calibration_stack(calib_dir: Path, max_samples: int | None) -> np.ndarray:
    files = sorted(glob.glob(str(calib_dir / "*.npy")))
    if max_samples is not None:
        files = files[:max_samples]
    arrays = []
    for f in files:
        arr = np.load(f).astype(np.float32)
        if arr.ndim != 4:
            raise RuntimeError(f"Unexpected shape {arr.shape} in {f}")
        if arr.shape[1] in (1, 3):  # assume NCHW -> convert to NHWC
            arr = np.transpose(arr, (0, 2, 3, 1))
        arrays.append(arr)
    if not arrays:
        raise RuntimeError(f"No calibration .npy files found in {calib_dir}")
    stacked = np.concatenate(arrays, axis=0)
    return stacked


def quantize_and_compile(variant: str, calib_limit: int | None) -> Tuple[Path, Path]:
    cfg = VARIANTS[variant]
    har_path = cfg["har"]
    if not har_path.exists():
        raise FileNotFoundError(har_path)

    calib = load_calibration_stack(cfg["calib"], calib_limit)

    runner = ClientRunner(hw_arch="hailo8")
    runner.load_har(str(har_path))

    runner.optimize(calib)

    quantized_har = har_path.with_name(f"{har_path.stem}_quantized.har")
    runner.save_har(str(quantized_har))

    hef = runner.compile()
    hef_path = har_path.with_suffix(".hef")
    hef_path.write_bytes(hef)
    return quantized_har, hef_path


def main(max_samples: int | None) -> None:
    for variant in VARIANTS:
        print(f"[INFO] Processing {variant}")
        quant_har, hef = quantize_and_compile(variant, max_samples)
        print(f"  Quantized HAR: {quant_har}")
        print(f"  HEF: {hef}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize & compile deimv2 stage1 HARs for selected variants."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum number of calibration samples per variant (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.max_samples)
