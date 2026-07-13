from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .fixtures import require_numpy
from .kernel_test_config import load_config
from .oracles import generate_oracle
from .required_tools import resolve_tool
from .routing.case_selection import select_cases
from .routing.v2.candidates import candidate_from_shape
from .routing.v2.matching import materialize_route_tensors, route_accepts_tensors
from .routing.v2.query import load_route_catalog, select_route
from .routing.v2.runtime import shape_for_case


SCHEMA = "ggml_hrx_kernel_bench.loom_execution_descriptor.v1"
DESCRIPTOR_MANIFEST_SCHEMA = "ggml_hrx_kernel_bench.loom_execution_descriptors.v1"


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


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_value_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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
    bindings = data.get("bindings")
    _expect(isinstance(bindings, list) and bindings, "bindings must be a non-empty array")

    seen_positions: set[int] = set()
    for index, binding in enumerate(bindings):
        _expect(isinstance(binding, dict), f"bindings[{index}] must be an object")
        position = binding.get("position")
        _expect(isinstance(position, int) and position >= 0, f"bindings[{index}].position must be a non-negative integer")
        _expect(position not in seen_positions, f"bindings[{index}].position duplicates {position}")
        seen_positions.add(position)
        kind = binding.get("kind")
        _expect(kind in ("input", "output"), f"bindings[{index}].kind must be input or output")
        dtype = binding.get("dtype")
        _expect(dtype == "f32", f"bindings[{index}].dtype must be f32")
        has_values = "values" in binding
        has_path = "path" in binding
        _expect(has_values != has_path, f"bindings[{index}] must provide exactly one of values or path")
        if has_values:
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


def _load_f32_array_element_count(path: Path) -> int:
    np = require_numpy()
    array = np.load(path, allow_pickle=False)
    _expect(array.ndim == 1, f"{path} must be one-dimensional")
    _expect(str(array.dtype) == "float32", f"{path} must be a float32 npy array")
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
    if family not in BINARY_F32_FAMILIES:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"only f32 binary generated descriptors are currently supported, saw {family!r}",
        )
    catalog = load_route_catalog(routing_dir)
    route = select_route(
        catalog,
        family=str(config_data["kernel"]),
        route_id=config_data.get("route_id"),
    )
    if set(route.tensors) != {"src0", "src1", "dst"}:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"add_f32 descriptor generation requires src0/src1/dst tensors, saw {sorted(route.tensors)}",
        )
    shape = shape_for_case(config_data, case_values)
    tensors = materialize_route_tensors(route, shape)
    if not route_accepts_tensors(route, tensors):
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"route {route.id!r} does not accept selected shape",
        )
    if any(str(tensor.dtype).upper() != "F32" for tensor in tensors.values()):
        return GeneratedDescriptorResult(
            status="unsupported",
            reason="only f32 tensor descriptors are currently supported",
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
    required_arrays = {"src0", "src1", "dst_init", "expected"}
    missing_arrays = sorted(required_arrays - set(arrays))
    if missing_arrays:
        return GeneratedDescriptorResult(
            status="unsupported",
            reason=f"oracle did not produce required arrays: {missing_arrays}",
        )
    array_element_counts = {
        name: _load_f32_array_element_count(arrays[name])
        for name in required_arrays
    }
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
        "bindings": [
            {
                "name": "src0",
                "position": 0,
                "kind": "input",
                "dtype": "f32",
                "path": _descriptor_relative_path(arrays["src0"], descriptor_dir=descriptor_dir),
            },
            {
                "name": "src1",
                "position": 1,
                "kind": "input",
                "dtype": "f32",
                "path": _descriptor_relative_path(arrays["src1"], descriptor_dir=descriptor_dir),
            },
            {
                "name": "dst",
                "position": 2,
                "kind": "output",
                "dtype": "f32",
                "path": _descriptor_relative_path(arrays["dst_init"], descriptor_dir=descriptor_dir),
                "expect": {
                    "mode": "close",
                    "path": _descriptor_relative_path(arrays["expected"], descriptor_dir=descriptor_dir),
                    "atol": float(tolerance.get("atol", 1e-5)),
                    "rtol": float(tolerance.get("rtol", 1e-5)),
                },
            },
        ],
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


def run_execution_descriptor_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    runner: Path | str,
    loom_link: Path | str | None,
    iree_run_loom: Path | str | None,
    repo_root: Path,
    execute: bool = False,
    limit: int | None = None,
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
            iree_run_loom=iree_run_loom,
            repo_root=repo_root,
            linked_kernel_output=case_dir / "linked.loom",
            execute_iree_run_loom=execute,
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
            result = execute_prepared(prepared)
            run_entry["process_returncode"] = result.returncode
            run_entry["stdout"] = result.stdout
            run_entry["stderr"] = result.stderr
            if result.returncode != 0:
                run_entry["status"] = "process_failed"
                failed_count += 1
            elif prepared.output_path.is_file():
                result_payload = json.loads(prepared.output_path.read_text(encoding="utf-8"))
                run_status = str(result_payload.get("status") or "missing_status")
                run_entry["status"] = run_status
                if run_status == "run_passed":
                    passed_count += 1
                else:
                    failed_count += 1
            else:
                run_entry["status"] = "missing_output"
                failed_count += 1
        run_entries.append(run_entry)

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
    iree_run_loom: Path | str | None,
    repo_root: Path,
    linked_kernel_output: Path | None = None,
    execute_iree_run_loom: bool = False,
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

    if iree_run_loom is not None:
        command.extend(["--iree-run-loom", str(iree_run_loom)])

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

    if execute_iree_run_loom:
        command.append("--execute-iree-run-loom-command")

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
    parser.add_argument("--tool-dir", help="optional PATH-style search list containing loom-link and iree-run-loom")
    parser.add_argument("--loom-link", type=Path)
    parser.add_argument("--iree-run-loom", type=Path)
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
    iree_run_loom = args.iree_run_loom
    if loom_link is None:
        resolved = resolve_tool("loom-link", tool_dir=args.tool_dir)
        loom_link = Path(resolved) if resolved else None
    if iree_run_loom is None:
        resolved = resolve_tool("iree-run-loom", tool_dir=args.tool_dir)
        iree_run_loom = Path(resolved) if resolved else None

    prepared = prepare_execution(
        descriptor_path=descriptor_path,
        fixture_dir=fixture_dir,
        output_path=(args.output or _default_output_path(descriptor_path, fixture_dir)),
        runner=args.runner,
        loom_link=loom_link,
        iree_run_loom=iree_run_loom,
        repo_root=args.repo_root,
        linked_kernel_output=args.linked_kernel_output,
        execute_iree_run_loom=args.execute,
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
