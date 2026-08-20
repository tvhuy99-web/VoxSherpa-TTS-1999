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
    model = onnx.load(str(model_input), load_external_data=True)
    inits = {x.name: x for x in model.graph.initializer}
    targets = set(nodes_to_quantize or [])
    excluded = set(nodes_to_exclude or [])
    new_nodes = []
    new_inits = []
    activation_cache = {}
    converted = 0
    qmax = 63.0 if reduce_range else 127.0

    for idx, node in enumerate(model.graph.node):
        name = _node_label(node, idx)
        if (
            node.op_type != "MatMul"
            or (targets and name not in targets)
            or name in excluded
            or len(node.input) < 2
            or node.input[1] not in inits
        ):
            new_nodes.append(node)
            continue
        w = numpy_helper.to_array(inits[node.input[1]])
        if w.ndim != 2 or w.dtype.kind != "f":
            new_nodes.append(node)
            continue
        w = w.astype(np.float32, copy=False)
        # Per-output-channel signed INT8 gives substantially better linear quality.
        scale = np.max(np.abs(w), axis=0).astype(np.float32) / np.float32(qmax)
        scale = np.where(scale > 1e-12, scale, np.float32(1.0)).astype(np.float32)
        qw = np.clip(np.rint(w / scale.reshape(1, -1)), -qmax, qmax).astype(np.int8)
        wz = np.zeros((qw.shape[1],), dtype=np.int8)
        stem = name.replace(":", "_")
        qw_n, ws_n, wz_n = stem + "__w_i8", stem + "__w_scale", stem + "__w_zero"
        new_inits.extend([
            numpy_helper.from_array(np.ascontiguousarray(qw), name=qw_n),
            numpy_helper.from_array(np.ascontiguousarray(scale), name=ws_n),
            numpy_helper.from_array(np.ascontiguousarray(wz), name=wz_n),
        ])
        a = node.input[0]
        if a not in activation_cache:
            aq, ascale, az = a + f"__dq_u8_{converted}", a + f"__dq_scale_{converted}", a + f"__dq_zero_{converted}"
            new_nodes.append(onnx.helper.make_node("DynamicQuantizeLinear", [a], [aq, ascale, az], name=stem + "__dyn_quant_a"))
            activation_cache[a] = (aq, ascale, az)
        aq, ascale, az = activation_cache[a]
        i32, f32, both_scale = node.output[0] + "__i32", node.output[0] + "__f32", node.output[0] + "__both_scale"
        new_nodes.extend([
            onnx.helper.make_node("MatMulInteger", [aq, qw_n, az, wz_n], [i32], name=stem + "__MatMulInteger"),
            onnx.helper.make_node("Cast", [i32], [f32], name=stem + "__cast", to=onnx.TensorProto.FLOAT),
            onnx.helper.make_node("Mul", [ascale, ws_n], [both_scale], name=stem + "__scale_mul"),
            onnx.helper.make_node("Mul", [f32, both_scale], list(node.output), name=stem + "__dequant"),
        ])
        converted += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(model_output), save_as_external_data=bool(use_external_data_format))
    print(f"manual MatMulInteger converted={converted}")
    if converted == 0:
        raise RuntimeError("manual MatMul quantizer converted zero targets")


def _manual_dynamic_conv_quantize(
    model_input,
    model_output,
    calibration_data_reader=None,
    quant_format=None,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=None,
    weight_type=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    calibrate_method=None,
    extra_options=None,
    **kwargs,
):
    """Top-level Conv -> DynamicQuantizeLinear + ConvInteger + dequant + bias.

    This is the same standard integer path used by ORT's ConvInteger quantizer,
    but it is emitted directly so no ORT model-rewrite pass can recurse into the
    Kokoro Loop/Sequence body. Conv weights use symmetric signed INT8 per tensor;
    keeping scale scalar matches ORT 1.17.1's dynamic ConvInteger path and avoids
    output-channel scale broadcasting ambiguity. Large-first candidate selection
    remains controlled by the caller.
    """
    model = onnx.load(str(model_input), load_external_data=True)
    inits = {x.name: x for x in model.graph.initializer}
    targets = set(nodes_to_quantize or [])
    excluded = set(nodes_to_exclude or [])
    new_nodes = []
    new_inits = []
    activation_cache = {}
    converted = 0
    qmax = 63.0 if reduce_range else 127.0

    for idx, node in enumerate(model.graph.node):
        name = _node_label(node, idx)
        if (
            node.op_type != "Conv"
            or (targets and name not in targets)
            or name in excluded
            or len(node.input) < 2
            or node.input[1] not in inits
        ):
            new_nodes.append(node)
            continue

        weight = numpy_helper.to_array(inits[node.input[1]])
        if weight.dtype.kind != "f" or weight.ndim < 3:
            new_nodes.append(node)
            continue
        if len(node.input) >= 3 and node.input[2] and node.input[2] not in inits:
            # Keep exact behavior when bias is dynamic; current Kokoro Conv biases
            # are constants, but refuse unsafe transformation if that changes.
            new_nodes.append(node)
            continue

        w = weight.astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(w)))
        wscale = np.float32(max_abs / qmax if max_abs > 1e-12 else 1.0)
        qw = np.clip(np.rint(w / wscale), -qmax, qmax).astype(np.int8)
        wz = np.asarray(0, dtype=np.int8)
        stem = name.replace(":", "_")
        qw_n, ws_n, wz_n = stem + "__conv_w_i8", stem + "__conv_w_scale", stem + "__conv_w_zero"
        new_inits.extend([
            numpy_helper.from_array(np.ascontiguousarray(qw), name=qw_n),
            numpy_helper.from_array(np.asarray(wscale, dtype=np.float32), name=ws_n),
            numpy_helper.from_array(wz, name=wz_n),
        ])

        a = node.input[0]
        if a not in activation_cache:
            aq = a + f"__conv_dq_u8_{converted}"
            ascale = a + f"__conv_dq_scale_{converted}"
            az = a + f"__conv_dq_zero_{converted}"
            new_nodes.append(onnx.helper.make_node(
                "DynamicQuantizeLinear", [a], [aq, ascale, az], name=stem + "__conv_dyn_quant_a"
            ))
            activation_cache[a] = (aq, ascale, az)
        aq, ascale, az = activation_cache[a]

        conv_i32 = node.output[0] + "__conv_i32"
        conv_f32 = node.output[0] + "__conv_f32"
        scales = node.output[0] + "__conv_scales"
        scaled = node.output[0] + "__conv_scaled" if len(node.input) >= 3 and node.input[2] else node.output[0]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        new_nodes.append(onnx.helper.make_node(
            "ConvInteger", [aq, qw_n, az, wz_n], [conv_i32], name=stem + "__ConvInteger", **attrs
        ))
        new_nodes.append(onnx.helper.make_node(
            "Cast", [conv_i32], [conv_f32], name=stem + "__conv_cast", to=onnx.TensorProto.FLOAT
        ))
        new_nodes.append(onnx.helper.make_node(
            "Mul", [ascale, ws_n], [scales], name=stem + "__conv_scale_mul"
        ))
        new_nodes.append(onnx.helper.make_node(
            "Mul", [conv_f32, scales], [scaled], name=stem + "__conv_dequant"
        ))

        if len(node.input) >= 3 and node.input[2]:
            # ORT's own ConvInteger quantizer reshapes bias to [1,C,1,...] before
            # adding, because a raw [C] tensor would broadcast on the wrong axis.
            shape = np.ones((weight.ndim,), dtype=np.int64)
            shape[1] = -1
            shape_n = stem + "__bias_shape"
            bias_r = node.output[0] + "__bias_reshaped"
            new_inits.append(numpy_helper.from_array(shape, name=shape_n))
            new_nodes.append(onnx.helper.make_node(
                "Reshape", [node.input[2], shape_n], [bias_r], name=stem + "__bias_reshape"
            ))
            new_nodes.append(onnx.helper.make_node(
                "Add", [scaled, bias_r], list(node.output), name=stem + "__bias_add"
            ))
        converted += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(model_output), save_as_external_data=bool(use_external_data_format))
    print(f"manual ConvInteger converted={converted}")
    if converted == 0:
        raise RuntimeError("manual Conv quantizer converted zero targets")


ortq.quantize_dynamic = _manual_dynamic_matmul_quantize
ortq.quantize_static = _manual_dynamic_conv_quantize
print("Using manual top-level MatMulInteger + ConvInteger quantizers; no ORT recursive graph rewrite passes")

script = Path(__file__).with_name("prepare_aggressive_kokoro_model_v2.py")
runpy.run_path(str(script), run_name="__main__")
