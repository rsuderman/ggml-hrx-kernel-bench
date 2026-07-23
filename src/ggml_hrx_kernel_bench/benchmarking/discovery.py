from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ggml_hrx_kernel_bench.fixtures import require_numpy
from ggml_hrx_kernel_bench.loom_execution_descriptor import load_descriptor

from .common import (
    FLOP_ESTIMATE_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    load_json as _load_json,
    sha1_file as _sha1_file,
)

QUANTIZED_STORAGE_DTYPES = {"q4_k", "q5_k", "q6_k", "q8_0"}

@dataclass(frozen=True)
class DescriptorCase:
    op: str
    run_manifest_path: Path
    descriptor_path: Path
    descriptor: dict[str, Any]
    prepared_entry: dict[str, Any]
    normalized_kernel_source: str
    source_content_hash: str | None
    implementation_id: str
    execution_digest: str

    @property
    def route_id(self) -> str:
        metadata = self.descriptor.get("metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("route_id"), str):
            return metadata["route_id"]
        route_id = self.prepared_entry.get("route_id")
        return route_id if isinstance(route_id, str) else ""

    @property
    def root(self) -> str:
        return str(self.descriptor["root"])

    @property
    def kernel_source(self) -> str:
        return str(self.descriptor["kernel"])

    @property
    def case_id(self) -> str:
        metadata = self.descriptor.get("metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("case_id"), str):
            return metadata["case_id"]
        case_id = self.prepared_entry.get("case_id")
        return case_id if isinstance(case_id, str) else self.descriptor_path.stem

    @property
    def family(self) -> str:
        family = self.prepared_entry.get("kernel")
        return family if isinstance(family, str) else ""


@dataclass
class BenchmarkBucket:
    implementation_id: str
    op: str
    route_id: str
    root: str
    kernel_source: str
    normalized_kernel_source: str
    source_content_hash: str | None
    cases: list[DescriptorCase] = field(default_factory=list)


def _resolve_manifest_path(path: str | Path, *, manifest_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (manifest_dir / candidate).resolve()


def _resolve_descriptor_path(entry: dict[str, Any], *, manifest_dir: Path) -> Path | None:
    raw_path = entry.get("descriptor_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return _resolve_manifest_path(raw_path, manifest_dir=manifest_dir)


def _resolve_descriptor_file_path(path: str | Path, *, descriptor_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (descriptor_dir / candidate).resolve()


def _resolve_kernel_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _normalize_kernel_source(path: str | Path, *, repo_root: Path, asset_root: Path | None) -> str:
    resolved = _resolve_kernel_path(path, repo_root=repo_root)
    roots: list[tuple[str, Path]] = []
    if asset_root is not None:
        roots.append(("asset", asset_root.resolve()))
    roots.append(("repo", repo_root.resolve()))
    for label, root in roots:
        try:
            return f"{label}:{resolved.relative_to(root)}"
        except ValueError:
            pass
    parts = resolved.parts
    if "generated" in parts:
        index = parts.index("generated")
        suffix = Path(*parts[index:])
        return f"generated:{suffix}"
    return str(resolved)


def _route_id_from_descriptor(descriptor: dict[str, Any], entry: dict[str, Any]) -> str:
    metadata = descriptor.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("route_id"), str):
        return metadata["route_id"]
    route_id = entry.get("route_id")
    return route_id if isinstance(route_id, str) else ""


def _binding_element_count(binding: dict[str, Any], descriptor: dict[str, Any], descriptor_dir: Path) -> int:
    name = str(binding.get("name") or "")
    dtype = str(binding.get("dtype") or "")
    if dtype in QUANTIZED_STORAGE_DTYPES and isinstance(binding.get("path"), str):
        np = require_numpy()
        path = _resolve_descriptor_file_path(binding["path"], descriptor_dir=descriptor_dir)
        return int(np.load(path, allow_pickle=False).reshape(-1).shape[0])
    metadata = descriptor.get("metadata", {})
    if isinstance(metadata, dict):
        element_counts = metadata.get("element_counts")
        if isinstance(element_counts, dict) and isinstance(element_counts.get(name), int):
            return int(element_counts[name])
    if "values" in binding:
        values = binding["values"]
        if isinstance(values, list):
            return len(values)
    if isinstance(binding.get("path"), str):
        np = require_numpy()
        path = _resolve_descriptor_file_path(binding["path"], descriptor_dir=descriptor_dir)
        return int(np.load(path, allow_pickle=False).reshape(-1).shape[0])
    raise RuntimeError(f"cannot determine element count for binding {name!r}")


def _expectation_element_count(
    binding: dict[str, Any],
    descriptor: dict[str, Any],
    descriptor_dir: Path,
) -> int:
    expect = binding.get("expect")
    if not isinstance(expect, dict):
        return _binding_element_count(binding, descriptor, descriptor_dir)
    if "values" in expect and isinstance(expect["values"], list):
        return len(expect["values"])
    metadata = descriptor.get("metadata", {})
    if isinstance(metadata, dict):
        counts = metadata.get("oracle_array_element_counts")
        if isinstance(counts, dict):
            raw_path = expect.get("path")
            if isinstance(raw_path, str):
                fixture_name = Path(raw_path).stem
                if isinstance(counts.get(fixture_name), int):
                    return int(counts[fixture_name])
            if isinstance(counts.get("expected"), int):
                return int(counts["expected"])
    if isinstance(expect.get("path"), str):
        np = require_numpy()
        path = _resolve_descriptor_file_path(expect["path"], descriptor_dir=descriptor_dir)
        return int(np.load(path, allow_pickle=False).reshape(-1).shape[0])
    raise RuntimeError("cannot determine expected element count")


def _fixture_digest(raw_path: object, *, descriptor_dir: Path) -> str | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return _sha1_file(_resolve_descriptor_file_path(raw_path, descriptor_dir=descriptor_dir))


def _canonical_binding(
    binding: dict[str, Any],
    *,
    descriptor: dict[str, Any],
    descriptor_dir: Path,
) -> dict[str, Any]:
    canonical: dict[str, Any] = {
        "position": binding.get("position"),
        "kind": binding.get("kind"),
        "dtype": binding.get("dtype"),
        "element_count": _binding_element_count(binding, descriptor, descriptor_dir),
        "fixture_digest": _fixture_digest(binding.get("path"), descriptor_dir=descriptor_dir),
    }
    expect = binding.get("expect")
    if isinstance(expect, dict):
        canonical["expect"] = {
            "mode": expect.get("mode", "close"),
            "atol": expect.get("atol"),
            "rtol": expect.get("rtol"),
            "element_count": _expectation_element_count(binding, descriptor, descriptor_dir),
            "fixture_digest": _fixture_digest(expect.get("path"), descriptor_dir=descriptor_dir),
        }
    return canonical


def descriptor_execution_digest(
    descriptor: dict[str, Any],
    *,
    descriptor_path: Path,
    normalized_kernel_source: str,
) -> str:
    descriptor_dir = descriptor_path.parent
    metadata = descriptor.get("metadata", {})
    route_id = metadata.get("route_id") if isinstance(metadata, dict) else None
    payload = {
        "schema": descriptor.get("schema"),
        "target": descriptor.get("target"),
        "route_id": route_id,
        "kernel": normalized_kernel_source,
        "root": descriptor.get("root"),
        "configs": dict(sorted((descriptor.get("configs") or {}).items())),
        "workgroup_count": descriptor.get("workgroup_count"),
        "scalars": sorted(descriptor.get("scalars") or [], key=lambda scalar: scalar.get("position", 0)),
        "bindings": [
            _canonical_binding(binding, descriptor=descriptor, descriptor_dir=descriptor_dir)
            for binding in sorted(descriptor["bindings"], key=lambda item: item.get("position", 0))
        ],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _int_config_with_suffix(descriptor: dict[str, Any], suffix: str) -> int | None:
    configs = descriptor.get("configs") or {}
    if not isinstance(configs, dict):
        return None
    matches = [value for key, value in configs.items() if isinstance(key, str) and key.endswith(suffix)]
    if len(matches) != 1:
        return None
    try:
        return int(matches[0])
    except (TypeError, ValueError):
        return None


def _int_shape_value(descriptor: dict[str, Any], key: str) -> int | None:
    metadata = descriptor.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    shape = metadata.get("shape")
    if not isinstance(shape, dict):
        return None
    value = shape.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _estimate_mul_mat_flops(descriptor: dict[str, Any]) -> dict[str, Any]:
    rows = _int_config_with_suffix(descriptor, ".rows") or _int_shape_value(descriptor, "d1")
    cols = _int_config_with_suffix(descriptor, ".cols") or _int_shape_value(descriptor, "d0")
    k = _int_config_with_suffix(descriptor, ".k") or _int_shape_value(descriptor, "src0_d0")
    dst_ne2 = _int_config_with_suffix(descriptor, ".dst_ne2") or _int_shape_value(descriptor, "d2") or 1
    dst_ne3 = _int_config_with_suffix(descriptor, ".dst_ne3") or _int_shape_value(descriptor, "d3") or 1
    values = {
        "rows": rows,
        "cols": cols,
        "k": k,
        "dst_ne2": dst_ne2,
        "dst_ne3": dst_ne3,
    }
    if rows is None or cols is None or k is None:
        return {
            "schema": FLOP_ESTIMATE_SCHEMA,
            "status": "insufficient_metadata",
            "op": "MUL_MAT",
            "inputs": values,
        }
    flops = 2 * rows * cols * k * dst_ne2 * dst_ne3
    return {
        "schema": FLOP_ESTIMATE_SCHEMA,
        "status": "estimated",
        "op": "MUL_MAT",
        "estimated_flops": flops,
        "formula": "2 * rows * cols * k * dst_ne2 * dst_ne3",
        "inputs": values,
        "assumptions": [
            "multiply and add are counted as separate FLOPs",
            "estimate is for one logical benchmark operation",
        ],
    }


def estimate_case_flops(case: DescriptorCase) -> dict[str, Any]:
    if case.op == "MUL_MAT":
        return _estimate_mul_mat_flops(case.descriptor)
    return {
        "schema": FLOP_ESTIMATE_SCHEMA,
        "status": "unsupported_op",
        "op": case.op,
        "estimated_flops": None,
    }


def _shape_bucket_from_descriptor(descriptor: dict[str, Any], *, estimated_flops: int | None = None) -> dict[str, Any]:
    metadata = descriptor.get("metadata", {})
    shape = metadata.get("shape") if isinstance(metadata, dict) else None
    shape = shape if isinstance(shape, dict) else {}
    configs = descriptor.get("configs") if isinstance(descriptor.get("configs"), dict) else {}
    rows = _int_config_with_suffix(descriptor, ".rows") or _int_shape_value(descriptor, "d1")
    cols = _int_config_with_suffix(descriptor, ".cols") or _int_shape_value(descriptor, "d0")
    k = _int_config_with_suffix(descriptor, ".k") or _int_shape_value(descriptor, "src0_d0")
    d2 = _int_config_with_suffix(descriptor, ".dst_ne2") or _int_shape_value(descriptor, "d2") or 1
    d3 = _int_config_with_suffix(descriptor, ".dst_ne3") or _int_shape_value(descriptor, "d3") or 1
    batch_product = d2 * d3 if isinstance(d2, int) and isinstance(d3, int) else None
    has_permutation = any("perm" in str(key) for key in shape)
    stride_values = [value for key, value in configs.items() if isinstance(key, str) and "stride" in key]
    has_stride_metadata = bool(stride_values)
    layout_kind = "permuted" if has_permutation else "strided" if has_stride_metadata else "contiguous_or_unknown"
    dtype_family = "unknown"
    bindings = descriptor.get("bindings")
    if isinstance(bindings, list):
        dtypes = [str(binding.get("dtype")) for binding in bindings if isinstance(binding, dict) and binding.get("kind") != "output"]
        if dtypes:
            dtype_family = "_".join(dtypes)
    return {
        "k": k,
        "rows": rows,
        "cols": cols,
        "dst_ne2": d2,
        "dst_ne3": d3,
        "batch_product": batch_product,
        "estimated_flops": estimated_flops,
        "flop_bucket": _flop_bucket(estimated_flops),
        "layout_kind": layout_kind,
        "dtype_family": dtype_family,
    }


def _flop_bucket(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value < 1_000:
        return "<1k"
    if value < 10_000:
        return "1k-10k"
    if value < 100_000:
        return "10k-100k"
    if value < 1_000_000:
        return "100k-1m"
    return ">=1m"


def shape_bucket_for_case(case: DescriptorCase, *, flop_estimate: dict[str, Any] | None = None) -> dict[str, Any]:
    estimate = flop_estimate or estimate_case_flops(case)
    estimated_flops = estimate.get("estimated_flops")
    return _shape_bucket_from_descriptor(
        case.descriptor,
        estimated_flops=estimated_flops if isinstance(estimated_flops, int) else None,
    )


def implementation_id_for(
    *,
    op: str,
    route_id: str,
    normalized_kernel_source: str,
    root: str,
    source_content_hash: str | None,
) -> str:
    payload = {
        "op": op,
        "route_id": route_id,
        "kernel": normalized_kernel_source,
        "root": root,
        "source_content_hash": source_content_hash,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _iter_run_manifest_paths(prepare_root: Path, op_filter: str | None = None) -> Iterable[Path]:
    if not prepare_root.is_dir():
        raise RuntimeError(f"prepare root does not exist: {prepare_root}")
    if op_filter:
        candidate = prepare_root / op_filter / "loom-execution-runs.json"
        if candidate.is_file():
            yield candidate
        return
    yield from sorted(prepare_root.glob("*/loom-execution-runs.json"))


def _case_from_prepared_entry(
    *,
    op: str,
    run_manifest_path: Path,
    entry: dict[str, Any],
    repo_root: Path,
    asset_root: Path | None,
) -> DescriptorCase | None:
    descriptor_path = _resolve_descriptor_path(entry, manifest_dir=run_manifest_path.parent)
    if descriptor_path is None or not descriptor_path.is_file():
        return None
    descriptor = load_descriptor(descriptor_path)
    normalized = _normalize_kernel_source(descriptor["kernel"], repo_root=repo_root, asset_root=asset_root)
    kernel_path = _resolve_kernel_path(descriptor["kernel"], repo_root=repo_root)
    source_hash = _sha1_file(kernel_path)
    route_id = _route_id_from_descriptor(descriptor, entry)
    implementation_id = implementation_id_for(
        op=op,
        route_id=route_id,
        normalized_kernel_source=normalized,
        root=str(descriptor["root"]),
        source_content_hash=source_hash,
    )
    execution_digest = descriptor_execution_digest(
        descriptor,
        descriptor_path=descriptor_path,
        normalized_kernel_source=normalized,
    )
    return DescriptorCase(
        op=op,
        run_manifest_path=run_manifest_path,
        descriptor_path=descriptor_path,
        descriptor=descriptor,
        prepared_entry=entry,
        normalized_kernel_source=normalized,
        source_content_hash=source_hash,
        implementation_id=implementation_id,
        execution_digest=execution_digest,
    )


def discover_cases(
    *,
    prepare_root: Path,
    repo_root: Path,
    asset_root: Path | None = None,
    op: str | None = None,
) -> list[DescriptorCase]:
    cases: list[DescriptorCase] = []
    for manifest_path in _iter_run_manifest_paths(prepare_root, op_filter=op):
        manifest = _load_json(manifest_path)
        if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
            raise RuntimeError(f"unsupported run manifest schema in {manifest_path}")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise RuntimeError(f"run manifest entries must be a list: {manifest_path}")
        manifest_op = manifest_path.parent.name
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RuntimeError(f"{manifest_path}: entries[{index}] must be an object")
            if entry.get("status") not in {"prepared", "run_passed"}:
                continue
            case = _case_from_prepared_entry(
                op=manifest_op,
                run_manifest_path=manifest_path,
                entry=entry,
                repo_root=repo_root,
                asset_root=asset_root,
            )
            if case is not None:
                cases.append(case)
    return cases


def _matches(case: DescriptorCase, args: argparse.Namespace) -> bool:
    if args.op and case.op != args.op:
        return False
    if args.route_id and case.route_id != args.route_id:
        return False
    if args.implementation_id and case.implementation_id != args.implementation_id:
        return False
    if args.kernel_source and case.normalized_kernel_source != args.kernel_source and case.kernel_source != args.kernel_source:
        return False
    if args.root and case.root != args.root:
        return False
    if args.case_id and case.case_id != args.case_id:
        return False
    return True


def bucket_cases(cases: Sequence[DescriptorCase], *, dedupe: bool = True) -> list[BenchmarkBucket]:
    buckets: dict[str, BenchmarkBucket] = {}
    seen: dict[str, set[str]] = {}
    for case in sorted(cases, key=lambda item: (item.implementation_id, item.execution_digest, item.case_id)):
        bucket = buckets.get(case.implementation_id)
        if bucket is None:
            bucket = BenchmarkBucket(
                implementation_id=case.implementation_id,
                op=case.op,
                route_id=case.route_id,
                root=case.root,
                kernel_source=case.kernel_source,
                normalized_kernel_source=case.normalized_kernel_source,
                source_content_hash=case.source_content_hash,
            )
            buckets[case.implementation_id] = bucket
            seen[case.implementation_id] = set()
        if dedupe and case.execution_digest in seen[case.implementation_id]:
            continue
        seen[case.implementation_id].add(case.execution_digest)
        bucket.cases.append(case)
    return sorted(buckets.values(), key=lambda item: (item.op, item.route_id, item.root, item.implementation_id))


def _selected_buckets(args: argparse.Namespace) -> list[BenchmarkBucket]:
    cases = discover_cases(
        prepare_root=args.prepare_root.resolve(),
        repo_root=args.repo_root.resolve(),
        asset_root=args.asset_root.resolve() if args.asset_root else None,
        op=args.op,
    )
    cases = [case for case in cases if _matches(case, args)]
    return bucket_cases(cases, dedupe=not args.no_dedupe)
