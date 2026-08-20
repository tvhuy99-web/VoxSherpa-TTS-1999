#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime.quantization as ortq


def _node_label(node: onnx.NodeProto, idx: int) -> str:
    return node.name or f"__node_{idx}_{node.op_type}"


def _manual_dynamic_matmul_quantize(
    model_input,
    model_output,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    weight_type=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    extra_options=None,
    **kwargs,
):
    """Quantize selected TOP-LEVEL constant-B MatMul nodes without ORT's graph rewrite.

    Uses standard ONNX DynamicQuantizeLinear + MatMulInteger. Weights are symmetric
    signed int8 per output channel. The function intentionally never descends into
    Loop/If subgraphs, which avoids ORT 1.17.1's recursive Gemm rewrite bug on the
    Kokoro Loop/Sequence body.
    """
    model = onnx.load(str(model_input), load_external_data=True)
    initializer_by_name = {x.name: x for x in model.graph.initializer}
    targets = set(nodes_to_quantize or [])
    excluded = set(nodes_to_exclude or [])
    allowed = set(op_types_to_quantize or ["MatMul"])

    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    activation_cache: dict[str, tuple[str, str, str]] = {}
    converted = 0
    skipped = []
    qmax = 63.0 if reduce_range else 127.0

    for idx, node in enumerate(model.graph.node):
        name = _node_label(node, idx)
        if (
            node.op_type != "MatMul"
            or "MatMul" not in allowed
            or (targets and name not in targets)
            or name in excluded
            or len(node.input) < 2
            or node.input[1] not in initializer_by_name
        ):
            new_nodes.append(node)
            continue

        weight = numpy_helper.to_array(initializer_by_name[node.input[1]])
        if weight.ndim != 2 or weight.dtype.kind not in {"f"}:
            skipped.append({"name": name, "reason": f"unsupported weight shape/dtype {weight.shape} {weight.dtype}"})
            new_nodes.append(node)
            continue
        w = weight.astype(np.float32, copy=False)
        max_abs = np.max(np.abs(w), axis=0).astype(np.float32)
        scales = max_abs / np.float32(qmax)
        scales = np.where(scales > 1e-12, scales, np.float32(1.0)).astype(np.float32)
        qweight = np.rint(w / scales.reshape(1, -1))
        qweight = np.clip(qweight, -qmax, qmax).astype(np.int8)
        zero_points = np.zeros((qweight.shape[1],), dtype=np.int8)

        stem = name.replace(":", "_")
        qw_name = f"{stem}__weight_int8"
        ws_name = f"{stem}__weight_scale"
        wz_name = f"{stem}__weight_zero"
        new_initializers.extend([
            numpy_helper.from_array(np.ascontiguousarray(qweight), name=qw_name),
            numpy_helper.from_array(np.ascontiguousarray(scales), name=ws_name),
            numpy_helper.from_array(np.ascontiguousarray(zero_points), name=wz_name),
        ])

        a = node.input[0]
        if a not in activation_cache:
            aq = f"{a}__manual_dyn_u8_{converted}"
            a_scale = f"{a}__manual_dyn_scale_{converted}"
            a_zero = f"{a}__manual_dyn_zero_{converted}"
            new_nodes.append(onnx.helper.make_node(
                "DynamicQuantizeLinear",
                [a],
                [aq, a_scale, a_zero],
                name=f"{stem}__dynamic_quantize_activation",
            ))
            activation_cache[a] = (aq, a_scale, a_zero)
        aq, a_scale, a_zero = activation_cache[a]

        int_out = f"{node.output[0]}__manual_int32"
        float_out = f"{node.output[0]}__manual_float"
        scale_out = f"{node.output[0]}__manual_combined_scale"
        new_nodes.append(onnx.helper.make_node(
            "MatMulInteger",
            [aq, qw_name, a_zero, wz_name],
            [int_out],
            name=f"{stem}__MatMulInteger",
        ))
        new_nodes.append(onnx.helper.make_node(
            "Cast",
            [int_out],
            [float_out],
            name=f"{stem}__cast_int32_to_float",
            to=onnx.TensorProto.FLOAT,
        ))
        new_nodes.append(onnx.helper.make_node(
            "Mul",
            [a_scale, ws_name],
            [scale_out],
            name=f"{stem}__combine_scales",
        ))
        new_nodes.append(onnx.helper.make_node(
            "Mul",
            [float_out, scale_out],
            list(node.output),
            name=f"{stem}__dequantize_matmul",
        ))
        converted += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(model_output), save_as_external_data=bool(use_external_data_format))
    print(f"Manual top-level MatMulInteger quantization converted={converted}, skipped={len(skipped)}")
    if skipped:
        print("Manual quantizer skipped:", skipped[:20])
    if converted == 0:
        raise RuntimeError("Manual MatMul quantizer converted zero targets")


# Replace only the Python offline helper imported by the v2 optimizer. This does
# not modify ONNX Runtime itself or the resulting Android runtime.
ortq.quantize_dynamic = _manual_dynamic_matmul_quantize
print("Using manual top-level per-channel MatMulInteger quantizer; ORT recursive dynamic pass is bypassed")

script = Path(__file__).with_name("prepare_aggressive_kokoro_model_v2.py")
runpy.run_path(str(script), run_name="__main__")
