"""
Stage2 runner for deimv2_atto pipeline.

This module executes the host-side portion (GridSample / TopK / GatherElements 等)
using ONNX Runtime. The Stage1 HAR compiled for Hailo should produce the tensor
`hailo_stage1_feat`, which becomes the input to this runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "onnxruntime が見つかりません。`pip install onnxruntime` でインストールしてください。"
    ) from exc


class Stage2Runner:
    """Load and execute the Stage2 ONNX graph on host CPU."""

    def __init__(self, onnx_path: str | Path, providers: Optional[list[str]] = None) -> None:
        self.onnx_path = Path(onnx_path)
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, stage1_feat: np.ndarray) -> np.ndarray:
        """Run Stage2 with a single Stage1 feature tensor."""
        outputs = self.session.run([self.output_name], {self.input_name: stage1_feat})
        return outputs[0]


def demo() -> None:
    """Small demonstration using random input and CPU sessions."""
    stage1 = ort.InferenceSession("src_models/deimv2_stage1.onnx", providers=["CPUExecutionProvider"])
    runner = Stage2Runner("src_models/deimv2_stage2.onnx")

    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    feat = stage1.run(None, {"images": dummy})[0]
    result = runner(feat)
    print("Stage2 output shape:", result.shape)


if __name__ == "__main__":
    demo()
