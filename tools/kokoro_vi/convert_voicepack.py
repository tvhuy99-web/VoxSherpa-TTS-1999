#!/usr/bin/env python3
"""Convert a Kokoro voice tensor to the float32 little-endian layout used by Android."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
EXPECTED_SHAPE = (510, 1, 256)
EXPECTED_BYTES = 510 * 256 * 4

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    tensor = torch.load(args.source, map_location="cpu", weights_only=True)
    if tuple(tensor.shape) != EXPECTED_SHAPE or tensor.dtype != torch.float32:
        raise ValueError(f"Unexpected voice tensor: {tensor.dtype} {tuple(tensor.shape)}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    values = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
    args.destination.write_bytes(values.tobytes(order="C"))
    if args.destination.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"Wrong voicepack byte count: {args.destination.stat().st_size}")
    print(f"source_sha256={sha256(args.source)}")
    print(f"output_sha256={sha256(args.destination)}")
    print(f"output_bytes={args.destination.stat().st_size}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
