from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ggml_hrx_kernel_bench.fixtures import require_numpy

from .common import json_scalar as _json_scalar
from .common import safe_name as _safe_name
from .common import sha1_file as _sha1_file
from .discovery import (
    QUANTIZED_STORAGE_DTYPES,
    DescriptorCase,
    _binding_element_count,
    _expectation_element_count,
    _resolve_descriptor_file_path,
    _resolve_kernel_path,
)

def _dtype_element_type(dtype: str) -> str:
    if dtype in QUANTIZED_STORAGE_DTYPES:
        return "i8"
    return dtype


def _symbol_base(value: str) -> str:
    value = value[1:] if value.startswith("@") else value
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    return safe.strip("_") or "case"


def _copy_or_write_fixture(
    *,
    source: dict[str, Any],
    descriptor_dir: Path,
    fixture_dir: Path,
    name: str,
    descriptor_dtype: str,
) -> Path:
    output = fixture_dir / f"{_safe_name(name)}.npy"
    np = require_numpy()
    if isinstance(source.get("path"), str):
        fixture_path = _resolve_descriptor_file_path(source["path"], descriptor_dir=descriptor_dir)
        if descriptor_dtype == "f16":
            array = np.load(fixture_path, allow_pickle=False).reshape(-1)
            if array.dtype == np.int16 or array.dtype == np.uint16:
                array = array.view(np.float16)
            else:
                array = array.astype(np.float16)
            np.save(output, array.reshape(-1), allow_pickle=False)
            return output
        shutil.copy2(fixture_path, output)
        return output
    if "values" in source:
        output.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(source["values"]).reshape(-1)
        if descriptor_dtype == "f16":
            if array.dtype == np.int16 or array.dtype == np.uint16:
                array = array.view(np.float16)
            else:
                array = array.astype(np.float16)
        np.save(output, array, allow_pickle=False)
        return output
    raise RuntimeError(f"fixture source for {name!r} has neither path nor values")


def _rel_path(path: Path, *, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _write_descriptor_workbench(
    *,
    case: DescriptorCase,
    run_dir: Path,
    repo_root: Path,
    loom_link: str,
    kernel_source_override: Path | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    descriptor = case.descriptor
    descriptor_dir = case.descriptor_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = run_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    linked_path = run_dir / "linked.loom"
    workbench_path = run_dir / "benchmark.loom"

    descriptor_kernel_source = _resolve_kernel_path(descriptor["kernel"], repo_root=repo_root)
    kernel_source = kernel_source_override or descriptor_kernel_source
    link_command = [
        loom_link,
        str(kernel_source),
        "--mode=link",
        "--to=text",
        "--require-resolved-config",
        f"--root={descriptor['root']}",
        f"--output={linked_path}",
    ]
    for key, value in sorted((descriptor.get("configs") or {}).items()):
        link_command.append(f"--config={key}={_json_scalar(value)}")
    link_result = subprocess.run(
        link_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if link_result.returncode != 0:
        raise RuntimeError(f"loom-link failed for {case.descriptor_path}: {link_result.stderr}")

    case_symbol = f"@case_{_symbol_base(case.route_id)}_{case.execution_digest[:12]}"
    bench_symbol = f"@bench_{_symbol_base(case.route_id)}_{case.execution_digest[:12]}"
    values: dict[int, tuple[str, str]] = {}
    lines: list[str] = [linked_path.read_text(encoding="utf-8").rstrip(), "", f"check.case public {case_symbol} {{"]

    for scalar in sorted(descriptor.get("scalars") or [], key=lambda item: item["position"]):
        name = _symbol_base(str(scalar.get("name") or f"scalar{scalar['position']}"))
        var = f"%{name}"
        values[int(scalar["position"])] = (var, str(scalar["dtype"]))
        lines.append(f"  {var} = check.literal value({_json_scalar(scalar['value'])}) : {scalar['dtype']}")

    for binding in sorted(descriptor["bindings"], key=lambda item: item["position"]):
        name = _symbol_base(str(binding.get("name") or f"binding{binding['position']}"))
        var = f"%{name}"
        element_type = _dtype_element_type(str(binding["dtype"]))
        element_count = _binding_element_count(binding, descriptor, descriptor_dir)
        fixture_path = _copy_or_write_fixture(
            source=binding,
            descriptor_dir=descriptor_dir,
            fixture_dir=fixture_dir,
            name=name,
            descriptor_dtype=str(binding["dtype"]),
        )
        values[int(binding["position"])] = (var, f"tensor<{element_count}x{element_type}>")
        lines.append(
            f"  {var} = check.file.read.npy path(\"{_rel_path(fixture_path, base=run_dir)}\") : "
            f"tensor<{element_count}x{element_type}>"
        )

    call_values = [values[position] for position in sorted(values)]
    args_text = ", ".join(value[0] for value in call_values)
    types_text = ", ".join(value[1] for value in call_values)
    lines.append(f"  func.call {descriptor['root']}({args_text}) : ({types_text})")

    for binding in sorted(descriptor["bindings"], key=lambda item: item["position"]):
        if binding.get("kind") != "output":
            continue
        expect = binding.get("expect")
        if not isinstance(expect, dict):
            continue
        name = _symbol_base(str(binding.get("name") or f"binding{binding['position']}"))
        expected_name = f"expected_{name}"
        expected_var = f"%{expected_name}"
        element_type = _dtype_element_type(str(binding["dtype"]))
        element_count = _expectation_element_count(binding, descriptor, descriptor_dir)
        expected_path = _copy_or_write_fixture(
            source=expect,
            descriptor_dir=descriptor_dir,
            fixture_dir=fixture_dir,
            name=expected_name,
            descriptor_dtype=str(binding["dtype"]),
        )
        lines.append(
            f"  {expected_var} = check.file.read.npy path(\"{_rel_path(expected_path, base=run_dir)}\") : "
            f"tensor<{element_count}x{element_type}>"
        )
        mode = str(expect.get("mode") or "close")
        actual = values[int(binding["position"])][0]
        if mode == "close":
            atol = _json_scalar(expect.get("atol", 0.0))
            rtol = _json_scalar(expect.get("rtol", 0.0))
            lines.append(
                f"  check.expect.close actual({actual}) expected({expected_var}) "
                f"atol({atol}) rtol({rtol}) nan(same) : tensor<{element_count}x{element_type}>"
            )
        elif mode in {"equal", "bitwise"}:
            lines.append(
                f"  check.expect.{mode} actual({actual}) expected({expected_var}) : "
                f"tensor<{element_count}x{element_type}>"
            )
        else:
            raise RuntimeError(f"unsupported expectation mode {mode!r}")

    lines.extend(["  check.return", "}", "", f"check.benchmark<{case_symbol}> {bench_symbol}", ""])
    workbench_path.write_text("\n".join(lines), encoding="utf-8")
    return workbench_path, bench_symbol, {
        "link_command": link_command,
        "linked_path": str(linked_path),
        "workbench_path": str(workbench_path),
        "descriptor_kernel_source": str(descriptor_kernel_source),
        "descriptor_kernel_source_hash": _sha1_file(descriptor_kernel_source),
        "effective_kernel_source": str(kernel_source),
        "effective_kernel_source_hash": _sha1_file(kernel_source),
        "source_override_used": kernel_source_override is not None,
    }
