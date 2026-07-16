from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .fixtures import require_numpy
from .kernel_test_config import load_config
from .oracles import generate_oracle
from .required_tools import (
    require_ggml_hrx_run_loom_expected_buffer_tolerance,
    require_tool,
    resolve_tool,
)
from .routing.case_selection import select_cases
from .routing.v2.candidates import candidate_from_shape
from .routing.v2.matching import materialize_route_tensors, route_accepts_tensors
from .routing.v2.query import load_route_catalog, select_route
from .routing.v2.runtime import shape_for_case


SCHEMA = "ggml_hrx_kernel_bench.loom_execution_descriptor.v1"
DESCRIPTOR_MANIFEST_SCHEMA = "ggml_hrx_kernel_bench.loom_execution_descriptors.v1"
ROUTE_EXECUTION_ABI_SCHEMA = "ggml_hrx_kernel_bench.route_execution_abi.v1"


@dataclass(frozen=True)
class PreparedLoomExecution:
    descriptor_path: Path
    fixture_dir: Path
    output_path: Path
    command: list[str]


@dataclass(frozen=True)
class GeneratedDescriptorResult:
    status: str
    descriptor: dict[str, Any] | None = None
    reason: str | None = None


BINARY_F32_FAMILIES = {"add_f32", "mul_f32", "div_f32", "sub_f32"}
BINARY_F16_FAMILIES = {"add_f16", "mul_f16", "div_f16", "sub_f16"}
UNARY_F32_FAMILIES = {
    "abs_f32",
    "ceil_f32",
    "cos_f32",
    "elu_f32",
    "exp_f32",
    "expm1_f32",
    "floor_f32",
    "gelu_f32",
    "gelu_erf_f32",
    "gelu_quick_f32",
    "hardsigmoid_f32",
    "hardswish_f32",
    "leaky_relu_f32",
    "log_f32",
    "neg_f32",
    "relu_f32",
    "round_f32",
    "sgn_f32",
    "sigmoid_f32",
    "silu_f32",
    "sin_f32",
    "softcap_f32",
    "softplus_f32",
    "sqr_f32",
    "sqrt_f32",
    "step_f32",
    "tanh_f32",
    "trunc_f32",
    "xielu_f32",
}
UNARY_F16_FAMILIES = {
    "abs_f16",
    "ceil_f16",
    "cos_f16",
    "elu_f16",
    "exp_f16",
    "expm1_f16",
    "floor_f16",
    "gelu_f16",
    "gelu_erf_f16",
    "gelu_quick_f16",
    "hardsigmoid_f16",
    "hardswish_f16",
    "leaky_relu_f16",
    "log_f16",
    "neg_f16",
    "relu_f16",
    "round_f16",
    "sgn_f16",
    "sigmoid_f16",
    "silu_f16",
    "sin_f16",
    "softplus_f16",
    "sqr_f16",
    "sqrt_f16",
    "step_f16",
    "tanh_f16",
    "trunc_f16",
}
APPROXIMATE_UNARY_F16_FAMILIES = {
    "elu_f16",
    "expm1_f16",
    "gelu_f16",
    "hardswish_f16",
    "sigmoid_f16",
    "tanh_f16",
}
APPROXIMATE_UNARY_F16_TOLERANCE = {"atol": 1e-3, "rtol": 1e-3}
SCALAR_F32_FAMILIES = {"scale_f32", "clamp_f32"}
SCALAR_F16_FAMILIES = {"clamp_f16"}
NORMALIZATION_F32_FAMILIES = {"rms_norm_f32"}
GATED_ACTIVATION_F32_FAMILIES = {"swiglu_f32"}
SOFTMAX_F32_FAMILIES = {"soft_max_f32"}
FLASH_ATTN_FAMILIES = {"flash_attn_ext_f32_f16"}
ROPE_F32_FAMILIES = {"rope_f32", "rope_neox_f32"}
INDEX_F32_FAMILIES = {"get_rows_f32", "set_rows_f32", "cont_set_rows_f32"}
INDEX_QUANTIZED_FAMILIES = {
    "get_rows_q4_k_f32",
    "get_rows_q5_k_f32",
    "get_rows_q6_k_f32",
    "get_rows_q8_0_f32",
}
COPY_F32_FAMILIES = {"cont_f32"}
COPY_CAST_FAMILIES = {
    "copy_bf16_bf16",
    "copy_bf16_f16",
    "copy_bf16_f32",
    "copy_f16_bf16",
    "copy_f16_f16",
    "copy_f16_f32",
    "copy_f32_bf16",
    "copy_f32_f16",
    "copy_f32_f32",
    "copy_i32_i32",
}
MATMUL_FAMILIES = {
    "mul_mat_f16_f32_batched",
    "mul_mat_f16_f32_tiled_batched",
    "mul_mat_f32_f32",
    "mul_mat_q4_k_f32",
    "mul_mat_q5_k_f32",
    "mul_mat_q6_k_f32",
    "mul_mat_q8_0_f32",
}
SUPPORTED_F32_BUFFER_FAMILIES = (
    BINARY_F32_FAMILIES
    | UNARY_F32_FAMILIES
    | SCALAR_F32_FAMILIES
    | NORMALIZATION_F32_FAMILIES
    | GATED_ACTIVATION_F32_FAMILIES
    | SOFTMAX_F32_FAMILIES
    | FLASH_ATTN_FAMILIES
    | ROPE_F32_FAMILIES
    | COPY_F32_FAMILIES
)
SUPPORTED_BUFFER_FAMILIES = (
    SUPPORTED_F32_BUFFER_FAMILIES
    | BINARY_F16_FAMILIES
    | UNARY_F16_FAMILIES
    | SCALAR_F16_FAMILIES
    | INDEX_F32_FAMILIES
    | INDEX_QUANTIZED_FAMILIES
    | COPY_CAST_FAMILIES
    | MATMUL_FAMILIES
)
SUPPORTED_BUFFER_DTYPES = {"bf16", "f32", "f16", "i32", "q4_k", "q5_k", "q6_k", "q8_0"}
NPY_STORAGE_DTYPE_BY_DESCRIPTOR_DTYPE = {
    "bf16": "int16",
    "f32": "float32",
    "f16": "int16",
    "i32": "int32",
    "q4_k": "int8",
    "q5_k": "int8",
    "q6_k": "int8",
    "q8_0": "int8",
}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_value_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _runtime_dtype(dtype: str | None) -> str:
    return str(dtype or "").strip().lower()


def _resolve_descriptor_path(path: str | Path, *, descriptor_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (descriptor_dir / candidate).resolve()


def _resolve_kernel_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _safe_fixture_name(value: str, fallback: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    safe = safe.strip("._")
    return safe or fallback


def load_descriptor(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_descriptor(data)
    return data


def validate_descriptor(data: object) -> None:
    _expect(isinstance(data, dict), "descriptor must be a JSON object")
    _expect(data.get("schema") == SCHEMA, f"descriptor schema must be {SCHEMA!r}")
    _expect(isinstance(data.get("kernel"), str) and data["kernel"], "kernel must be a non-empty string")
    _expect(isinstance(data.get("root"), str) and data["root"], "root must be a non-empty string")
    _expect(isinstance(data.get("target"), str) and data["target"], "target must be a non-empty string")
    if "workgroup_count" in data:
        _validate_workgroup_count(data["workgroup_count"])
    configs = data.get("configs", {})
    _expect(isinstance(configs, dict), "configs must be an object when present")
    scalars = data.get("scalars", [])
    _expect(isinstance(scalars, list), "scalars must be an array when present")
    bindings = data.get("bindings")
    _expect(isinstance(bindings, list) and bindings, "bindings must be a non-empty array")

    seen_positions: set[int] = set()
    for index, scalar in enumerate(scalars):
        _expect(isinstance(scalar, dict), f"scalars[{index}] must be an object")
        position = scalar.get("position")
        _expect(isinstance(position, int) and position >= 0, f"scalars[{index}].position must be a non-negative integer")
        _expect(position not in seen_positions, f"scalars[{index}].position duplicates {position}")
        seen_positions.add(position)
        dtype = scalar.get("dtype")
        _expect(dtype == "f32", f"scalars[{index}].dtype must be f32")
        value = scalar.get("value")
        _expect(
            isinstance(value, (int, float, str)) and not isinstance(value, bool) and str(value),
            f"scalars[{index}].value must be a scalar value",
        )
    for index, binding in enumerate(bindings):
        _expect(isinstance(binding, dict), f"bindings[{index}] must be an object")
        position = binding.get("position")
        _expect(isinstance(position, int) and position >= 0, f"bindings[{index}].position must be a non-negative integer")
        _expect(position not in seen_positions, f"bindings[{index}].position duplicates {position}")
        seen_positions.add(position)
        kind = binding.get("kind")
        _expect(kind in ("input", "output"), f"bindings[{index}].kind must be input or output")
        dtype = binding.get("dtype")
        _expect(
            dtype in SUPPORTED_BUFFER_DTYPES,
            f"bindings[{index}].dtype must be one of {', '.join(sorted(SUPPORTED_BUFFER_DTYPES))}",
        )
        has_values = "values" in binding
        has_path = "path" in binding
        _expect(has_values != has_path, f"bindings[{index}] must provide exactly one of values or path")
        if has_values:
            _expect(dtype == "f32", f"bindings[{index}].values are only supported for f32")
            _validate_values(binding["values"], context=f"bindings[{index}].values")
        else:
            _expect(isinstance(binding["path"], str) and binding["path"], f"bindings[{index}].path must be a non-empty string")
        if kind == "output":
            expect = binding.get("expect")
            _expect(isinstance(expect, dict), f"bindings[{index}].expect must be an object for output bindings")
            _expect(expect.get("mode") == "close", f"bindings[{index}].expect.mode must be close")
            has_expect_values = "values" in expect
            has_expect_path = "path" in expect
            _expect(has_expect_values != has_expect_path, f"bindings[{index}].expect must provide exactly one of values or path")
            if has_expect_values:
                _expect(dtype == "f32", f"bindings[{index}].expect.values are only supported for f32")
                _validate_values(expect["values"], context=f"bindings[{index}].expect.values")
            else:
                _expect(isinstance(expect["path"], str) and expect["path"], f"bindings[{index}].expect.path must be a non-empty string")
            _expect(_is_number(expect.get("atol", 0.0)), f"bindings[{index}].expect.atol must be numeric")
            _expect(_is_number(expect.get("rtol", 0.0)), f"bindings[{index}].expect.rtol must be numeric")


def _validate_values(values: object, *, context: str) -> None:
    _expect(isinstance(values, list), f"{context} must be an array")
    _expect(values, f"{context} must not be empty")
    for index, value in enumerate(values):
        _expect(_is_number(value), f"{context}[{index}] must be numeric")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_workgroup_count(value: object) -> None:
    _expect(isinstance(value, list) and len(value) == 3, "workgroup_count must be an array of three integers")
    for index, part in enumerate(value):
        _expect(
            isinstance(part, int) and part > 0 and not isinstance(part, bool),
            f"workgroup_count[{index}] must be a positive integer",
        )


def _storage_elements(tensor: Any) -> int:
    if not tensor.dimensions:
        return 0
    max_offset = 0
    for dimension in tensor.dimensions:
        max_offset += (int(dimension.size) - 1) * int(dimension.stride)
    return max_offset + 1


def _descriptor_relative_path(path: Path, *, descriptor_dir: Path | None) -> str:
    resolved = path.resolve()
    if descriptor_dir is None:
        return str(resolved)
    try:
        return str(resolved.relative_to(descriptor_dir.resolve()))
    except ValueError:
        return str(resolved)


def _load_array_element_count(path: Path, *, allowed_dtypes: set[str]) -> int:
    np = require_numpy()
    array = np.load(path, allow_pickle=False)
    _expect(array.ndim == 1, f"{path} must be one-dimensional")
    actual_dtype = str(array.dtype)
    _expect(
        actual_dtype in allowed_dtypes,
        f"{path} must be one of {sorted(allowed_dtypes)} npy array dtypes, saw {actual_dtype}",
    )
    return int(array.shape[0])


def _oracle_arrays(metadata_path: Path) -> dict[str, Path]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = metadata.get("arrays")
    _expect(isinstance(arrays, dict), f"oracle metadata {metadata_path} must contain arrays")
    result: dict[str, Path] = {}
    for name, path in arrays.items():
        _expect(isinstance(name, str), f"oracle array name must be a string in {metadata_path}")
        _expect(isinstance(path, str) and path, f"oracle array {name} path must be a non-empty string")
        result[name] = Path(path)
    return result


def _execution_abi_entries(config_data: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    abi = config_data.get("execution_abi")
    if not isinstance(abi, dict):
        return None, "generated config is missing execution_abi"
    if abi.get("schema") != ROUTE_EXECUTION_ABI_SCHEMA:
        return None, f"execution_abi schema must be {ROUTE_EXECUTION_ABI_SCHEMA!r}"
    entries = abi.get("entries")
    if not isinstance(entries, list) or not entries:
        return None, "execution_abi.entries must be a non-empty list"
    result: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return None, f"execution_abi.entries[{index}] must be an object"
        position = entry.get("position")
        if not isinstance(position, int) or position < 0 or isinstance(position, bool):
            return None, f"execution_abi.entries[{index}].position must be a non-negative integer"
        if position in seen_positions:
            return None, f"execution_abi position {position} is duplicated"
        seen_positions.add(position)
        kind = entry.get("kind")
        if kind not in ("input", "output", "scalar"):
            return None, f"execution_abi.entries[{index}].kind must be input, output, or scalar for descriptor v1"
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            return None, f"execution_abi.entries[{index}].role must be a non-empty string"
        dtype = entry.get("dtype")
        if dtype not in SUPPORTED_BUFFER_DTYPES:
            return None, f"execution_abi.entries[{index}].dtype must be bf16, f32, f16, or i32 for descriptor v1"
        if kind == "scalar" and dtype != "f32":
            return None, f"execution_abi.entries[{index}].dtype must be f32 for scalar descriptor v1"
        if kind == "scalar":
            value = entry.get("value")
            if not isinstance(value, (int, float, str)) or isinstance(value, bool) or not str(value):
                return None, f"execution_abi.entries[{index}].value must be a scalar value"
        else:
            fixture = entry.get("fixture")
            if not isinstance(fixture, str) or not fixture:
                return None, f"execution_abi.entries[{index}].fixture must be a non-empty string"
        if kind == "output":
            expect = entry.get("expect")
            if not isinstance(expect, dict):
                return None, f"execution_abi.entries[{index}].expect must be an object for output entries"
            if expect.get("mode") != "close":
                return None, f"execution_abi.entries[{index}].expect.mode must be close"
            expect_fixture = expect.get("fixture")
            if not isinstance(expect_fixture, str) or not expect_fixture:
                return None, f"execution_abi.entries[{index}].expect.fixture must be a non-empty string"
        result.append(entry)
    return sorted(result, key=lambda current: current["position"]), None


def _required_abi_fixtures(entries: list[dict[str, Any]]) -> set[str]:
    required: set[str] = set()
    for entry in entries:
        if entry["kind"] == "scalar":
            continue
        required.add(str(entry["fixture"]))
        if entry["kind"] == "output":
            required.add(str(entry["expect"]["fixture"]))
    return required


def _uses_approximate_unary_f16_expected(family: str, entry: dict[str, Any]) -> bool:
    return family in APPROXIMATE_UNARY_F16_FAMILIES and entry["kind"] == "output" and str(entry["dtype"]) == "f16"


def _materialize_approximate_unary_f16_expected_arrays(
    *,
    entries: list[dict[str, Any]],
    arrays: dict[str, Path],
    family: str,
) -> dict[str, Path]:
    if family not in APPROXIMATE_UNARY_F16_FAMILIES:
        return arrays
    np = require_numpy()
    result = dict(arrays)
    for entry in entries:
        if not _uses_approximate_unary_f16_expected(family, entry):
            continue
        expect_fixture = str(entry["expect"]["fixture"])
        source_path = arrays[expect_fixture]
        expected_bits = np.load(source_path, allow_pickle=False)
        _expect(
            expected_bits.ndim == 1 and str(expected_bits.dtype) == "int16",
            f"{source_path} must be a one-dimensional int16 f16 storage npy array",
        )
        expected_values = expected_bits.view(np.float16)
        value_path = source_path.with_name(f"{source_path.stem}.f16-values.npy")
        np.save(value_path, expected_values, allow_pickle=False)
        result[expect_fixture] = value_path
    return result


def _descriptor_scalars_from_execution_abi(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scalars: list[dict[str, Any]] = []
    for entry in entries:
        if entry["kind"] != "scalar":
            continue
        scalars.append(
            {
                "name": entry["role"],
                "position": entry["position"],
                "dtype": entry["dtype"],
                "value": entry["value"],
            }
        )
    return scalars


def _descriptor_bindings_from_execution_abi(
    *,
    entries: list[dict[str, Any]],
    arrays: dict[str, Path],
    family: str,
    tolerance: dict[str, float],
    descriptor_dir: Path | None,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for entry in entries:
        if entry["kind"] == "scalar":
            continue
        fixture = str(entry["fixture"])
        binding: dict[str, Any] = {
            "name": entry["role"],
            "position": entry["position"],
            "kind": entry["kind"],
            "dtype": entry["dtype"],
            "path": _descriptor_relative_path(arrays[fixture], descriptor_dir=descriptor_dir),
        }
        if entry["kind"] == "output":
            expect = entry["expect"]
            expect_fixture = str(expect["fixture"])
            output_tolerance = tolerance
            if (
                family in APPROXIMATE_UNARY_F16_FAMILIES
                and str(entry["dtype"]) == "f16"
            ):
                output_tolerance = {
                    "atol": max(
                        float(tolerance.get("atol", 1e-5)),
                        APPROXIMATE_UNARY_F16_TOLERANCE["atol"],
                    ),
                    "rtol": max(
                        float(tolerance.get("rtol", 1e-5)),
                        APPROXIMATE_UNARY_F16_TOLERANCE["rtol"],
                    ),
                }
            binding["expect"] = {
                "mode": "close",
                "path": _descriptor_relative_path(arrays[expect_fixture], descriptor_dir=descriptor_dir),
                "atol": float(output_tolerance.get("atol", 1e-5)),
                "rtol": float(output_tolerance.get("rtol", 1e-5)),
            }
        bindings.append(binding)
    return bindings


def descriptor_from_generated_case(
    *,
    config_data: dict[str, Any],
    case_id: str,
    case_values: list[int],
    kernel_dir: Path,
    routing_dir: Path,
    target: str,
    max_elements: int = 65536,
    oracle_fixture_dir: Path | None = None,
    descriptor_dir: Path | None = None,
) -> GeneratedDescriptorResult:
    family = str(config_data.get("kernel") or "")
    if family not in SUPPORTED_BUFFER_FAMILIES:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"generated buffer descriptors are not enabled for family {family!r}",
        )
    abi_entries, abi_error = _execution_abi_entries(config_data)
    if abi_entries is None:
        return GeneratedDescriptorResult(status="unsupported", reason=abi_error)
    catalog = load_route_catalog(routing_dir)
    route = select_route(
        catalog,
        family=str(config_data["kernel"]),
        route_id=config_data.get("route_id"),
    )
    abi_roles = {str(entry["role"]) for entry in abi_entries if entry["kind"] != "scalar"}
    if not abi_roles.issubset(set(route.tensors)):
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"execution_abi references roles not present in route tensors: {sorted(abi_roles - set(route.tensors))}",
        )
    shape = shape_for_case(config_data, case_values)
    tensors = materialize_route_tensors(route, shape)
    if not route_accepts_tensors(route, tensors):
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"route {route.id!r} does not accept selected shape",
        )
    abi_dtype_by_role = {
        str(entry["role"]): str(entry["dtype"])
        for entry in abi_entries
        if entry["kind"] != "scalar"
    }
    if any(
        abi_dtype_by_role.get(name, _runtime_dtype(str(tensor.dtype))) not in SUPPORTED_BUFFER_DTYPES
        for name, tensor in tensors.items()
    ):
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"only {', '.join(sorted(SUPPORTED_BUFFER_DTYPES))} tensor descriptors are currently supported",
        )
    element_counts = {name: _storage_elements(tensor) for name, tensor in tensors.items()}
    largest = max(element_counts.values())
    if largest > max_elements:
        return GeneratedDescriptorResult(
            status="skipped",
            reason=f"largest generated fixture has {largest} elements, above max {max_elements}",
        )

    candidate = candidate_from_shape(kernel_dir=kernel_dir, route=route, shape=shape)
    if candidate.status != "planned":
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=candidate.message or f"candidate {candidate.id} is not planned",
        )
    scalar_values = {
        str(entry["role"]): entry["value"]
        for entry in abi_entries
        if entry["kind"] == "scalar"
    }
    if scalar_values:
        candidate = replace(candidate, values={**candidate.values, **scalar_values})
    if oracle_fixture_dir is None:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason="oracle_fixture_dir is required for oracle-backed descriptor emission",
        )
    oracle = generate_oracle(candidate, oracle_fixture_dir)
    if oracle.status != "fixtures_ready" or oracle.metadata_path is None:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=oracle.message or f"oracle generation failed with status {oracle.status}",
        )
    arrays = _oracle_arrays(oracle.metadata_path)
    required_arrays = _required_abi_fixtures(abi_entries)
    missing_arrays = sorted(required_arrays - set(arrays))
    if missing_arrays:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"oracle did not produce required arrays: {missing_arrays}",
        )
    descriptor_arrays = _materialize_approximate_unary_f16_expected_arrays(
        entries=abi_entries,
        arrays=arrays,
        family=str(candidate.family),
    )
    abi_fixture_allowed_dtypes: dict[str, set[str]] = {}
    for entry in abi_entries:
        if entry["kind"] == "scalar":
            continue
        abi_fixture_allowed_dtypes[str(entry["fixture"])] = {
            NPY_STORAGE_DTYPE_BY_DESCRIPTOR_DTYPE[str(entry["dtype"])]
        }
        if entry["kind"] == "output":
            expect_fixture = str(entry["expect"]["fixture"])
            if _uses_approximate_unary_f16_expected(str(candidate.family), entry):
                abi_fixture_allowed_dtypes[expect_fixture] = {"float16"}
            else:
                abi_fixture_allowed_dtypes[expect_fixture] = {
                    NPY_STORAGE_DTYPE_BY_DESCRIPTOR_DTYPE[str(entry["dtype"])]
                }
    array_element_counts = {
        name: _load_array_element_count(
            descriptor_arrays[name],
            allowed_dtypes=abi_fixture_allowed_dtypes[name],
        )
        for name in required_arrays
    }
    if array_element_counts:
        largest_array = max(array_element_counts.values())
        if largest_array > max_elements:
            return GeneratedDescriptorResult(
                status="skipped",
                reason=f"largest oracle fixture has {largest_array} elements, above max {max_elements}",
            )
    tolerance = oracle.tolerance or {"atol": 1e-5, "rtol": 1e-5}
    descriptor = {
        "schema": SCHEMA,
        "kernel": str(candidate.source_path),
        "root": candidate.root_symbol,
        "target": target,
        "workgroup_count": list(candidate.dispatch["workgroup_count"]),
        "configs": dict(sorted(candidate.config.items())),
        "scalars": _descriptor_scalars_from_execution_abi(abi_entries),
        "bindings": _descriptor_bindings_from_execution_abi(
            entries=abi_entries,
            arrays=descriptor_arrays,
            family=str(candidate.family),
            tolerance=tolerance,
            descriptor_dir=descriptor_dir,
        ),
        "metadata": {
            "source": "generated-kernel-tests",
            "case_id": case_id,
            "case_values": list(case_values),
            "shape": candidate.shape,
            "route_id": candidate.route_id,
            "candidate_id": candidate.id,
            "dispatch": candidate.dispatch,
            "element_counts": element_counts,
            "oracle": oracle.to_ledger(),
            "oracle_array_element_counts": array_element_counts,
            "execution_abi": config_data["execution_abi"],
        },
    }
    validate_descriptor(descriptor)
    return GeneratedDescriptorResult(status="emitted", descriptor=descriptor)


def _safe_name(value: str, *, max_length: int = 96) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "generated"
    if len(safe) <= max_length:
        return safe
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    prefix_length = max_length - len(digest) - 1
    if prefix_length <= 0:
        return digest[:max_length]
    return f"{safe[:prefix_length].rstrip('-')}-{digest}"


def write_generated_execution_descriptors(
    *,
    manifest_path: Path,
    output_dir: Path,
    kernel_dir: Path,
    routing_dir: Path,
    target: str,
    max_elements: int = 65536,
    limit: int | None = None,
    kernels: set[str] | None = None,
    route_ids: set[str] | None = None,
    case_selectors: list[str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), "generated manifest must be a JSON object")
    _expect(payload.get("schema") == "ggml_hrx_kernel_bench.generated_kernel_tests.v1", "unsupported generated manifest schema")
    entries = payload.get("entries")
    _expect(isinstance(entries, list), "generated manifest entries must be a list")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    emitted_count = 0
    skipped_count = 0
    unsupported_count = 0
    filtered_count = 0
    for entry_index, entry in enumerate(entries):
        _expect(isinstance(entry, dict), f"entries[{entry_index}] must be an object")
        entry_kernel = str(entry.get("kernel") or "")
        entry_route_id = str(entry.get("route_id") or "")
        if kernels is not None and entry_kernel not in kernels:
            filtered_count += 1
            continue
        if route_ids is not None and entry_route_id not in route_ids:
            filtered_count += 1
            continue
        config_path = Path(str(entry.get("config_path") or ""))
        _expect(config_path.is_file(), f"missing generated config {config_path}")
        config_data = load_config(config_path)
        for case_id, case_values in select_cases(config_data, case_selectors):
            if limit is not None and emitted_count >= limit:
                break
            name = _safe_name(f"{entry_index:03d}-{config_path.stem}-{case_id}")
            result = descriptor_from_generated_case(
                config_data=config_data,
                case_id=case_id,
                case_values=case_values,
                kernel_dir=kernel_dir,
                routing_dir=routing_dir,
                target=target,
                max_elements=max_elements,
                oracle_fixture_dir=output_dir / "fixtures" / name,
                descriptor_dir=output_dir,
            )
            descriptor_path = None
            if result.status == "emitted":
                assert result.descriptor is not None
                descriptor_path = output_dir / f"{name}.json"
                descriptor_path.write_text(
                    json.dumps(result.descriptor, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                emitted_count += 1
            elif result.status == "skipped":
                skipped_count += 1
            else:
                unsupported_count += 1
            manifest_entries.append(
                {
                    "status": result.status,
                    "reason": result.reason,
                    "descriptor_path": str(descriptor_path) if descriptor_path else None,
                    "config_path": str(config_path),
                    "config_name": config_path.name,
                    "kernel": config_data.get("kernel"),
                    "route_id": config_data.get("route_id"),
                    "case_id": case_id,
                    "case_values": list(case_values),
                }
            )
        if limit is not None and emitted_count >= limit:
            break

    descriptor_manifest = {
        "schema": DESCRIPTOR_MANIFEST_SCHEMA,
        "source_manifest_path": str(manifest_path),
        "target": target,
        "max_elements": max_elements,
        "entry_count": len(manifest_entries),
        "emitted_count": emitted_count,
        "skipped_count": skipped_count,
        "unsupported_count": unsupported_count,
        "filtered_count": filtered_count,
        "entries": manifest_entries,
    }
    (output_dir / "loom-execution-descriptors.json").write_text(
        json.dumps(descriptor_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return descriptor_manifest


def load_descriptor_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), "descriptor manifest must be a JSON object")
    _expect(payload.get("schema") == DESCRIPTOR_MANIFEST_SCHEMA, "unsupported descriptor manifest schema")
    entries = payload.get("entries")
    _expect(isinstance(entries, list), "descriptor manifest entries must be a list")
    return payload


def _resolve_manifest_path(path: str | Path, *, manifest_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (manifest_dir / candidate).resolve()


def _descriptor_entry_path(entry: dict[str, Any], *, manifest_dir: Path) -> Path | None:
    raw = entry.get("descriptor_path")
    if not isinstance(raw, str) or not raw:
        return None
    return _resolve_manifest_path(raw, manifest_dir=manifest_dir)


def _descriptor_execution_test_name(entry: dict[str, Any], descriptor_path: Path) -> str:
    route_id = entry.get("route_id")
    case_id = entry.get("case_id")
    suite = str(route_id) if isinstance(route_id, str) and route_id else descriptor_path.stem
    test = str(case_id) if isinstance(case_id, str) and case_id else descriptor_path.stem
    return f"{suite}.{test}"


def _emit_descriptor_execution_summary(
    *,
    progress: Callable[[str], None],
    executed_count: int,
    passed_count: int,
    failed_count: int,
    failed_test_names: Sequence[str],
) -> None:
    progress(f"[==========] {executed_count} tests ran.")
    progress(f"[  PASSED  ] {passed_count} tests.")
    if failed_count == 0:
        progress("[  FAILED  ] 0 tests.")
        return
    progress(f"[  FAILED  ] {failed_count} tests, listed below:")
    for test_name in failed_test_names:
        progress(f"[  FAILED  ] {test_name}")
    progress("")
    progress(f"{failed_count} FAILED TESTS")


def run_execution_descriptor_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    runner: Path | str,
    loom_link: Path | str | None,
    ggml_hrx_run_loom: Path | str | None,
    repo_root: Path,
    execute: bool = False,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_descriptor_manifest(manifest_path)
    run_entries: list[dict[str, Any]] = []
    prepared_count = 0
    executed_count = 0
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    failed_test_names: list[str] = []

    for index, entry in enumerate(payload["entries"]):
        _expect(isinstance(entry, dict), f"entries[{index}] must be an object")
        if entry.get("status") != "emitted":
            skipped_count += 1
            continue
        if limit is not None and prepared_count >= limit:
            break
        descriptor_path = _descriptor_entry_path(entry, manifest_dir=manifest_dir)
        if descriptor_path is None:
            skipped_count += 1
            continue
        case_id = str(entry.get("case_id") or descriptor_path.stem)
        case_dir = output_dir / f"{prepared_count:03d}-{_safe_name(case_id)}"
        prepared = prepare_execution(
            descriptor_path=descriptor_path,
            fixture_dir=case_dir / "fixtures",
            output_path=case_dir / "run.json",
            runner=runner,
            loom_link=loom_link,
            ggml_hrx_run_loom=ggml_hrx_run_loom,
            repo_root=repo_root,
            linked_kernel_output=case_dir / "linked.loom",
            execute_ggml_hrx_run_loom=execute,
        )
        prepared_count += 1
        run_entry: dict[str, Any] = {
            "descriptor_path": str(descriptor_path),
            "fixture_dir": str(prepared.fixture_dir),
            "output_path": str(prepared.output_path),
            "command": prepared.command,
            "case_id": entry.get("case_id"),
            "case_values": entry.get("case_values"),
            "kernel": entry.get("kernel"),
            "route_id": entry.get("route_id"),
            "status": "prepared",
        }
        if execute:
            executed_count += 1
            test_name = _descriptor_execution_test_name(entry, descriptor_path)
            if progress is not None:
                progress(f"[ RUN      ] {test_name}")
            result = execute_prepared(prepared)
            run_entry["process_returncode"] = result.returncode
            run_entry["stdout"] = result.stdout
            run_entry["stderr"] = result.stderr
            if result.returncode != 0:
                run_entry["status"] = "process_failed"
                failed_count += 1
                failed_test_names.append(test_name)
            elif prepared.output_path.is_file():
                result_payload = json.loads(prepared.output_path.read_text(encoding="utf-8"))
                run_status = str(result_payload.get("status") or "missing_status")
                run_entry["status"] = run_status
                if run_status == "run_passed":
                    passed_count += 1
                else:
                    failed_count += 1
                    failed_test_names.append(test_name)
            else:
                run_entry["status"] = "missing_output"
                failed_count += 1
                failed_test_names.append(test_name)
            if progress is not None:
                if run_entry["status"] == "run_passed":
                    progress(f"[       OK ] {test_name}")
                else:
                    progress(f"[  FAILED  ] {test_name}")
        run_entries.append(run_entry)

    if execute and progress is not None:
        _emit_descriptor_execution_summary(
            progress=progress,
            executed_count=executed_count,
            passed_count=passed_count,
            failed_count=failed_count,
            failed_test_names=failed_test_names,
        )

    run_manifest = {
        "schema": "ggml_hrx_kernel_bench.loom_execution_runs.v1",
        "descriptor_manifest_path": str(manifest_path),
        "execute": execute,
        "entry_count": len(run_entries),
        "prepared_count": prepared_count,
        "executed_count": executed_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "entries": run_entries,
    }
    (output_dir / "loom-execution-runs.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def _write_f32_npy(path: Path, values: Sequence[object]) -> None:
    np = require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    np.save(path, array, allow_pickle=False)


def _fixture_path(
    *,
    binding: dict[str, Any],
    fixture_dir: Path,
    suffix: str,
    fallback: str,
) -> Path:
    name = _safe_fixture_name(str(binding.get("name") or fallback), fallback)
    return fixture_dir / f"{name}{suffix}.npy"


def _materialize_binding_file(
    *,
    binding: dict[str, Any],
    descriptor_dir: Path,
    fixture_dir: Path,
    suffix: str,
    fallback: str,
) -> Path:
    if "path" in binding:
        return _resolve_descriptor_path(binding["path"], descriptor_dir=descriptor_dir)
    path = _fixture_path(binding=binding, fixture_dir=fixture_dir, suffix=suffix, fallback=fallback)
    _write_f32_npy(path, binding["values"])
    return path


def prepare_execution(
    *,
    descriptor_path: Path,
    fixture_dir: Path,
    output_path: Path,
    runner: Path | str,
    loom_link: Path | str | None,
    ggml_hrx_run_loom: Path | str | None,
    repo_root: Path,
    linked_kernel_output: Path | None = None,
    execute_ggml_hrx_run_loom: bool = False,
) -> PreparedLoomExecution:
    descriptor_path = descriptor_path.resolve()
    descriptor_dir = descriptor_path.parent
    fixture_dir = fixture_dir.resolve()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root.resolve()
    data = load_descriptor(descriptor_path)

    command = [
        str(runner),
        "--kernel",
        str(_resolve_kernel_path(data["kernel"], repo_root=repo_root)),
        "--root",
        str(data["root"]),
        "--target",
        str(data["target"]),
        "--output",
        str(output_path),
    ]
    if "workgroup_count" in data:
        workgroup_count = data["workgroup_count"]
        _validate_workgroup_count(workgroup_count)
        command.extend(
            [
                "--workgroup-count",
                ",".join(str(value) for value in workgroup_count),
            ]
        )

    configs = data.get("configs", {})
    for key, value in sorted(configs.items()):
        _expect(isinstance(key, str) and key, "config keys must be non-empty strings")
        _expect(
            isinstance(value, (str, int, float, bool)),
            f"config {key!r} must be a scalar JSON value",
        )
        command.extend(["--config", f"{key}={_json_value_to_text(value)}"])

    if configs:
        if loom_link is not None:
            command.extend(["--loom-link", str(loom_link)])
        linked_path = linked_kernel_output or (fixture_dir / "linked.loom")
        command.extend(["--linked-kernel-output", str(linked_path.resolve())])
    elif loom_link is not None:
        command.extend(["--loom-link", str(loom_link)])

    if ggml_hrx_run_loom is not None:
        command.extend(["--ggml-hrx-run-loom", str(ggml_hrx_run_loom)])

    for scalar in data.get("scalars", []):
        command.extend(
            [
                "--scalar",
                f"{scalar['position']}:{scalar['dtype']}:{_json_value_to_text(scalar['value'])}",
            ]
        )

    for index, binding in enumerate(data["bindings"]):
        binding_path = _materialize_binding_file(
            binding=binding,
            descriptor_dir=descriptor_dir,
            fixture_dir=fixture_dir,
            suffix="",
            fallback=f"binding{index}",
        )
        command.extend(
            [
                "--binding",
                f"{binding['position']}:{binding['kind']}:{binding['dtype']}:{_element_count(binding, binding_path)}:{binding_path}",
            ]
        )
        if binding["kind"] == "output":
            expect = dict(binding["expect"])
            expect_binding = {
                "name": binding.get("name", f"binding{index}"),
                **expect,
            }
            expect_path = _materialize_binding_file(
                binding=expect_binding,
                descriptor_dir=descriptor_dir,
                fixture_dir=fixture_dir,
                suffix="_expected",
                fallback=f"binding{index}_expected",
            )
            command.extend(
                [
                    "--expect",
                    f"{binding['position']}:close:{expect_path}:{expect.get('atol', 0.0)}:{expect.get('rtol', 0.0)}",
                ]
            )

    if execute_ggml_hrx_run_loom:
        _expect(ggml_hrx_run_loom is not None, "execute requires an explicit ggml-hrx-run-loom path")
        command.append("--execute-ggml-hrx-run-loom-command")

    return PreparedLoomExecution(
        descriptor_path=descriptor_path,
        fixture_dir=fixture_dir,
        output_path=output_path,
        command=command,
    )


def _element_count(binding: dict[str, Any], path: Path) -> int:
    if "values" in binding:
        return len(binding["values"])
    np = require_numpy()
    array = np.load(path, allow_pickle=False)
    _expect(array.ndim == 1, f"{path} must be one-dimensional")
    return int(array.shape[0])


def execute_prepared(prepared: PreparedLoomExecution) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        prepared.command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _default_output_path(descriptor_path: Path, fixture_dir: Path) -> Path:
    return fixture_dir / f"{descriptor_path.stem}-run.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact Loom execution descriptor.")
    parser.add_argument("descriptor_path", type=Path)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runner", default="ggml-hrx-run-loom-simple")
    parser.add_argument("--tool-dir", help="optional PATH-style search list containing loom-link and ggml-hrx-run-loom")
    parser.add_argument("--loom-link", type=Path)
    parser.add_argument("--ggml-hrx-run-loom", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--linked-kernel-output", type=Path)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    descriptor_path = args.descriptor_path.resolve()
    fixture_dir = args.fixture_dir.resolve()
    loom_link = args.loom_link
    ggml_hrx_run_loom = args.ggml_hrx_run_loom
    if loom_link is None:
        resolved = resolve_tool("loom-link", tool_dir=args.tool_dir)
        loom_link = Path(resolved) if resolved else None
    if ggml_hrx_run_loom is None:
        resolved = (
            require_tool("ggml-hrx-run-loom", tool_dir=args.tool_dir)
            if args.execute
            else resolve_tool("ggml-hrx-run-loom", tool_dir=args.tool_dir)
        )
        ggml_hrx_run_loom = Path(resolved) if resolved else None
    if ggml_hrx_run_loom is not None:
        require_ggml_hrx_run_loom_expected_buffer_tolerance(tool_path=ggml_hrx_run_loom)

    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=fixture_dir,
        output_path=(args.output or _default_output_path(descriptor_path, fixture_dir)),
        runner=args.runner,
        loom_link=loom_link,
        ggml_hrx_run_loom=ggml_hrx_run_loom,
        repo_root=args.repo_root,
        linked_kernel_output=args.linked_kernel_output,
        execute_ggml_hrx_run_loom=args.execute,
    )
    if args.print_command or not args.execute:
        print(json.dumps({"command": prepared.command}, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    result = execute_prepared(prepared)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode
