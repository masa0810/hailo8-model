#!/usr/bin/env python3
"""Utility to split DeiM-V2 ONNX models into Hailo-ready Stage1/Stage2 graphs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import helper, shape_inference, utils


DEFAULT_VARIANTS = ["atto", "femto", "pico", "n", "s", "x1", "x2", "x3"]
STAGE1_OUTPUT_NAME = "hailo_stage1_feat"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _update_stage1(stage1_path: Path) -> None:
    stage1 = onnx.load(stage1_path)
    # remove existing outputs
    while stage1.graph.output:
        stage1.graph.output.pop()
    # add identity to set deterministic output name
    stage1.graph.node.extend(
        [
            helper.make_node(
                "Identity",
                ["/model/decoder/Mul_output_0"],
                [STAGE1_OUTPUT_NAME],
                name="hailo_stage1_identity_feat",
            )
        ]
    )
    stage1 = shape_inference.infer_shapes(stage1)
    value_infos = {
        vi.name: vi
        for vi in list(stage1.graph.value_info) + list(stage1.graph.output)
    }
    vi = value_infos.get(STAGE1_OUTPUT_NAME)
    if vi is None:
        vi = helper.make_tensor_value_info(
            STAGE1_OUTPUT_NAME, onnx.TensorProto.FLOAT, None
        )
    stage1.graph.output.extend([vi])
    onnx.save(stage1, stage1_path)


def _update_stage2(stage2_path: Path) -> None:
    stage2 = onnx.load(stage2_path)
    for tensor in stage2.graph.input:
        if tensor.name == "/model/decoder/Mul_output_0":
            tensor.name = STAGE1_OUTPUT_NAME
    for node in stage2.graph.node:
        node.input[:] = [
            STAGE1_OUTPUT_NAME if inp == "/model/decoder/Mul_output_0" else inp
            for inp in node.input
        ]
    onnx.save(stage2, stage2_path)


def _check_equivalence(src_path: Path, stage1_path: Path, stage2_path: Path) -> float:
    import onnxruntime as ort

    sess_src = ort.InferenceSession(str(src_path), providers=["CPUExecutionProvider"])
    sess_stage1 = ort.InferenceSession(
        str(stage1_path), providers=["CPUExecutionProvider"]
    )
    sess_stage2 = ort.InferenceSession(
        str(stage2_path), providers=["CPUExecutionProvider"]
    )

    input_name = sess_src.get_inputs()[0].name
    dims = []
    for dim in sess_src.get_inputs()[0].shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            dims.append(1)
    dummy = np.random.rand(*dims).astype(np.float32)
    src_out = sess_src.run(None, {input_name: dummy})[0]
    feat = sess_stage1.run(None, {"images": dummy})[0]
    stage2_inputs = {STAGE1_OUTPUT_NAME: feat}
    if "images" in [i.name for i in sess_stage2.get_inputs()]:
        stage2_inputs["images"] = dummy
    stage2_out = sess_stage2.run(None, stage2_inputs)[0]
    return float(np.max(np.abs(src_out - stage2_out)))


def prepare_variant(variant: str, check: bool = False) -> float | None:
    src = Path("src_models") / f"deimv2_{variant}.onnx"
    if not src.exists():
        raise FileNotFoundError(src)
    stage1_path = Path("src_models") / f"deimv2_{variant}_stage1.onnx"
    stage2_path = Path("src_models") / f"deimv2_{variant}_stage2.onnx"

    _ensure_dir(stage1_path)
    utils.extract_model(
        str(src),
        str(stage1_path),
        ["images"],
        ["/model/decoder/Mul_output_0"],
        check_model=False,
    )
    utils.extract_model(
        str(src),
        str(stage2_path),
        ["/model/decoder/Mul_output_0", "images"],
        ["label_xyxy_score"],
        check_model=False,
    )
    _update_stage1(stage1_path)
    _update_stage2(stage2_path)
    onnx.checker.check_model(str(stage1_path))
    onnx.checker.check_model(str(stage2_path))

    if check:
        return _check_equivalence(src, stage1_path, stage2_path)
    return None


def main(variants: Iterable[str], check: bool) -> None:
    for variant in variants:
        diff = prepare_variant(variant, check=check)
        msg = f"Prepared stage models for deimv2_{variant}"
        if diff is not None:
            msg += f" (max diff {diff:.3e})"
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split DeiM-V2 ONNX into Stage1/Stage2 graphs."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="Variant suffixes to process (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run ONNX Runtime equivalence check (requires onnxruntime).",
    )
    args = parser.parse_args()
    main(args.variants, args.check)
