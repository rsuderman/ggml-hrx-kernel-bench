from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .fixtures import require_numpy
from .routing.api import Candidate


QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210
Q8_0_BLOCK_BYTES = 34
F32_BYTES = 4
I32_BYTES = 4
Q8_1_BLOCK_BYTES = 36


@dataclass(frozen=True)
class OracleResult:
    status: str
    oracle: str | None
    fixture_dir: Path | None
    metadata_path: Path | None
    expected_path: Path | None
    tolerance: dict[str, float] | None
    message: str | None = None

    def to_ledger(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "oracle": self.oracle,
            "fixture_dir": str(self.fixture_dir) if self.fixture_dir else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "expected_path": str(self.expected_path) if self.expected_path else None,
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass(frozen=True)
class OracleSpec:
    family_ids: tuple[str, ...]
    generate: Callable[[Any, Candidate, Path, int], OracleResult]
    write_workbench: Callable[[Candidate, Path, Path, Path], tuple[str | None, dict[str, Any]]]


@dataclass(frozen=True)
class LogicalOracleSpec:
    family_ids: tuple[str, ...]
    oracle: str
    tolerance: dict[str, float]
    build: Callable[[Any, Candidate, int], dict[str, Any]]
    exact_kernel_abi: bool = False


def q4_k_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q4_K_BLOCK_BYTES


def q5_k_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q5_K_BLOCK_BYTES


def q6_k_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q6_K_BLOCK_BYTES


def q8_1_bytes(ncols: int, nrows: int) -> int:
    return nrows * ((ncols + 31) // 32) * Q8_1_BLOCK_BYTES


def q8_0_bytes(k: int, rows: int) -> int:
    if k % 32 != 0:
        raise ValueError(f"k must be a multiple of 32: {k}")
    return rows * (k // 32) * Q8_0_BLOCK_BYTES


def f32_pattern(np: Any, shape: tuple[int, ...], *, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    pattern = np.arange(values.size, dtype=np.float32).reshape(shape)
    values += (((pattern * 17 + seed * 29) % 257) - 128).astype(np.float32) / 251.0
    return (values * np.float32(scale)).astype(np.float32)


def positive_f32_pattern(np: Any, shape: tuple[int, ...], *, seed: int, scale: float = 0.25):
    raw = f32_pattern(np, shape, seed=seed, scale=scale)
    # DIV fixtures need a stable non-zero divisor surface. Generate that
    # directly instead of shifting an arbitrary random pattern by a constant.
    return np.exp(raw).astype(np.float32)


def f16_pattern(np: Any, shape: tuple[int, ...], *, seed: int, scale: float = 1.0):
    return f32_pattern(np, shape, seed=seed, scale=scale).astype(np.float16)


def normalized_f32_rows(np: Any, shape: tuple[int, int], *, seed: int, l2_norm: float = 1.0):
    values = f32_pattern(np, shape, seed=seed)
    # Keep every RHS row at a fixed energy. This makes the dot-product error
    # budget depend on the accumulator behavior instead of accidental input
    # magnitude or cancellation from an unconstrained random vector.
    values = values - np.mean(values, axis=1, keepdims=True, dtype=np.float32)
    norms = np.linalg.norm(values.astype(np.float32), axis=1, keepdims=True).astype(np.float32)
    safe_norms = np.where(norms > np.float32(0.0), norms, np.float32(1.0))
    return (values / safe_norms * np.float32(l2_norm)).astype(np.float32)


def pack_q4_k_scales(np: Any, scales: Any, minimums: Any):
    packed = np.zeros(scales.shape[:-1] + (12,), dtype=np.uint8)
    scales_u32 = scales.astype(np.uint32)
    minimums_u32 = minimums.astype(np.uint32)
    packed[..., 0:4] = ((scales_u32[..., 0:4] & 0x3F) | ((scales_u32[..., 4:8] >> 4) << 6)).astype(np.uint8)
    packed[..., 4:8] = ((minimums_u32[..., 0:4] & 0x3F) | ((minimums_u32[..., 4:8] >> 4) << 6)).astype(np.uint8)
    packed[..., 8:12] = ((scales_u32[..., 4:8] & 0x0F) | ((minimums_u32[..., 4:8] & 0x0F) << 4)).astype(np.uint8)
    return packed


def pack_q5_k_high_bits(np: Any, quants: Any):
    high = ((quants.astype(np.uint8) & np.uint8(0x10)) >> np.uint8(4)).astype(np.uint8)
    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).reshape((1,) * (high.ndim - 2) + (8, 1))
    return np.sum(high * weights, axis=-2, dtype=np.uint8).astype(np.uint8)


def _permuted_rows(np: Any, rng: Any, base: Any, shape: tuple[int, ...]) -> Any:
    order = np.argsort(rng.random(shape + (base.size,)), axis=-1)
    return base[order].astype(base.dtype)


def q4_k_pattern(np: Any, k: int, rows: int, *, seed: int, target_rms: float = 0.5):
    blocks = rows * (k // QK_K)
    data = np.zeros((blocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    q_base = np.tile(np.arange(16, dtype=np.uint8), 2)
    # Use valid Q4_K scale/min metadata with balanced nibble coverage. The
    # minimum is chosen near scale * mean(q), so each 32-value quant group is
    # roughly centered after dequantization instead of producing huge
    # one-sided sums that swamp f16acc checks.
    scales = rng.integers(2, 7, size=(blocks, 8), dtype=np.uint8)
    minimums = np.floor(scales.astype(np.float32) * np.float32(7.5) + np.float32(0.5)).astype(np.uint8)
    q_values = _permuted_rows(np, rng, q_base, (blocks, 8))
    logical = scales.astype(np.float32)[..., None] * q_values.astype(np.float32) - minimums.astype(np.float32)[..., None]
    # Normalize per block after constructing the exact packed q/scales/mins.
    # The Loom assertion is still a plain close check, so the fixture has to
    # keep outputs in a range where the existing absolute tolerance is a
    # useful f16acc guard instead of a high-dynamic-range stress test.
    rms = np.sqrt(np.mean(logical.reshape(blocks, QK_K) ** np.float32(2.0), axis=1, dtype=np.float32)).astype(np.float32)
    d_value = np.where(rms > np.float32(0.0), np.float32(target_rms) / rms, np.float32(1.0)).astype(np.float16)
    qs = np.zeros((blocks, 128), dtype=np.uint8)
    qs[:, 0:32] = q_values[:, 0] | (q_values[:, 1] << np.uint8(4))
    qs[:, 32:64] = q_values[:, 2] | (q_values[:, 3] << np.uint8(4))
    qs[:, 64:96] = q_values[:, 4] | (q_values[:, 5] << np.uint8(4))
    qs[:, 96:128] = q_values[:, 6] | (q_values[:, 7] << np.uint8(4))
    d_bytes = d_value.reshape(blocks, 1).view(np.uint8)
    data[:, 0:2] = d_bytes
    data[:, 2:4] = d_bytes
    data[:, 4:16] = pack_q4_k_scales(np, scales, minimums)
    data[:, 16:144] = qs
    return data.reshape(-1)


def q5_k_pattern(np: Any, k: int, rows: int, *, seed: int, target_rms: float = 0.5):
    blocks = rows * (k // QK_K)
    data = np.zeros((blocks, Q5_K_BLOCK_BYTES), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    q_base = np.arange(32, dtype=np.uint8)
    scales = rng.integers(2, 5, size=(blocks, 8), dtype=np.uint8)
    minimums = np.floor(scales.astype(np.float32) * np.float32(15.5) + np.float32(0.5)).astype(np.uint8)
    q_values = _permuted_rows(np, rng, q_base, (blocks, 8))
    logical = scales.astype(np.float32)[..., None] * q_values.astype(np.float32) - minimums.astype(np.float32)[..., None]
    rms = np.sqrt(np.mean(logical.reshape(blocks, QK_K) ** np.float32(2.0), axis=1, dtype=np.float32)).astype(np.float32)
    d_value = np.where(rms > np.float32(0.0), np.float32(target_rms) / rms, np.float32(1.0)).astype(np.float16)
    q_low = q_values & np.uint8(0x0F)
    qs = np.zeros((blocks, 128), dtype=np.uint8)
    qs[:, 0:32] = q_low[:, 0] | (q_low[:, 1] << np.uint8(4))
    qs[:, 32:64] = q_low[:, 2] | (q_low[:, 3] << np.uint8(4))
    qs[:, 64:96] = q_low[:, 4] | (q_low[:, 5] << np.uint8(4))
    qs[:, 96:128] = q_low[:, 6] | (q_low[:, 7] << np.uint8(4))
    d_bytes = d_value.reshape(blocks, 1).view(np.uint8)
    data[:, 0:2] = d_bytes
    data[:, 2:4] = d_bytes
    data[:, 4:16] = pack_q4_k_scales(np, scales, minimums)
    data[:, 16:48] = pack_q5_k_high_bits(np, q_values)
    data[:, 48:176] = qs
    return data.reshape(-1)


def q6_k_pattern(np: Any, k: int, rows: int, *, seed: int, target_rms: float = 0.5):
    blocks = rows * (k // QK_K)
    data = np.zeros((blocks, Q6_K_BLOCK_BYTES), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    ql = np.zeros((blocks, 128), dtype=np.uint8)
    qh = np.zeros((blocks, 64), dtype=np.uint8)
    scales = rng.integers(1, 5, size=(blocks, 16), dtype=np.int8)
    q_signed = rng.integers(-32, 32, size=(blocks, 2, 4, 32), dtype=np.int16)
    q_u = (q_signed + np.int16(32)).astype(np.uint8)
    positions = np.arange(32)
    scale_offsets = (positions >= 16).astype(np.intp)
    logical = np.empty((blocks, 2, 4, 32), dtype=np.float32)
    for half in range(2):
        for segment in range(4):
            ql_index = half * 64 + (segment % 2) * 32
            low = q_u[:, half, segment] & np.uint8(0x0F)
            if segment < 2:
                ql[:, ql_index : ql_index + 32] |= low
            else:
                ql[:, ql_index : ql_index + 32] |= low << np.uint8(4)
            qh[:, half * 32 : half * 32 + 32] |= ((q_u[:, half, segment] >> np.uint8(4)) & np.uint8(0x03)) << np.uint8(2 * segment)
            scale_index = half * 8 + segment * 2 + scale_offsets
            logical[:, half, segment] = scales[:, scale_index].astype(np.float32) * q_signed[:, half, segment].astype(np.float32)
    rms = np.sqrt(np.mean(logical.reshape(blocks, QK_K) ** np.float32(2.0), axis=1, dtype=np.float32)).astype(np.float32)
    d_value = np.where(rms > np.float32(0.0), np.float32(target_rms) / rms, np.float32(1.0)).astype(np.float16)
    data[:, 0:128] = ql
    data[:, 128:192] = qh
    data[:, 192:208] = scales.view(np.uint8)
    data[:, 208:210] = d_value.reshape(blocks, 1).view(np.uint8)
    return data.reshape(-1)


def dequant_q4_k(np: Any, packed: Any, k: int, rows: int):
    blocks_per_row = k // QK_K
    blocks = packed.view(np.uint8).reshape(rows, blocks_per_row, Q4_K_BLOCK_BYTES)
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row)
    dmin = blocks[..., 2:4].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row)
    packed_scales = blocks[..., 4:16].astype(np.uint32)
    scale_i = np.empty((rows, blocks_per_row, 8), dtype=np.float32)
    min_i = np.empty((rows, blocks_per_row, 8), dtype=np.float32)
    low = packed_scales[..., 0:4]
    mid = packed_scales[..., 4:8]
    high = packed_scales[..., 8:12]
    scale_i[..., 0:4] = (low & 0x3F).astype(np.float32)
    min_i[..., 0:4] = (mid & 0x3F).astype(np.float32)
    scale_i[..., 4:8] = ((high & 0x0F) | ((low >> 6) << 4)).astype(np.float32)
    min_i[..., 4:8] = ((high >> 4) | ((mid >> 6) << 4)).astype(np.float32)
    qs = blocks[..., 16:144].astype(np.uint32).reshape(rows, blocks_per_row, 4, 32)
    q = np.empty((rows, blocks_per_row, 8, 32), dtype=np.float32)
    q[..., 0::2, :] = (qs & 0x0F).astype(np.float32)
    q[..., 1::2, :] = (qs >> 4).astype(np.float32)
    values = d[..., None, None] * scale_i[..., None] * q - dmin[..., None, None] * min_i[..., None]
    return values.astype(np.float32).reshape(rows, k)


def dequant_q5_k(np: Any, packed: Any, k: int, rows: int):
    blocks_per_row = k // QK_K
    blocks = packed.view(np.uint8).reshape(rows, blocks_per_row, Q5_K_BLOCK_BYTES)
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row)
    dmin = blocks[..., 2:4].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row)
    packed_scales = blocks[..., 4:16].astype(np.uint32)
    scale_i = np.empty((rows, blocks_per_row, 8), dtype=np.float32)
    min_i = np.empty((rows, blocks_per_row, 8), dtype=np.float32)
    low = packed_scales[..., 0:4]
    mid = packed_scales[..., 4:8]
    high = packed_scales[..., 8:12]
    scale_i[..., 0:4] = (low & 0x3F).astype(np.float32)
    min_i[..., 0:4] = (mid & 0x3F).astype(np.float32)
    scale_i[..., 4:8] = ((high & 0x0F) | ((low >> 6) << 4)).astype(np.float32)
    min_i[..., 4:8] = ((high >> 4) | ((mid >> 6) << 4)).astype(np.float32)
    qs = blocks[..., 48:176].astype(np.uint32).reshape(rows, blocks_per_row, 4, 32)
    q_low = np.empty((rows, blocks_per_row, 8, 32), dtype=np.uint32)
    q_low[..., 0::2, :] = qs & 0x0F
    q_low[..., 1::2, :] = qs >> 4
    qh = blocks[..., 16:48].astype(np.uint32)
    groups = np.arange(8, dtype=np.uint32).reshape(1, 1, 8, 1)
    q_high = (((qh[..., None, :] >> groups) & 0x01) << 4).astype(np.uint32)
    q = (q_low | q_high).astype(np.float32)
    values = d[..., None, None] * scale_i[..., None] * q - dmin[..., None, None] * min_i[..., None]
    return values.astype(np.float32).reshape(rows, k)


def dequant_q6_k(np: Any, packed: Any, k: int, rows: int):
    blocks_per_row = k // QK_K
    blocks = packed.view(np.uint8).reshape(rows, blocks_per_row, Q6_K_BLOCK_BYTES)
    ql = blocks[..., 0:128].astype(np.uint32).reshape(rows, blocks_per_row, 2, 2, 32)
    qh = blocks[..., 128:192].astype(np.uint32).reshape(rows, blocks_per_row, 2, 32)
    scales = blocks[..., 192:208].copy().view(np.int8).astype(np.float32).reshape(rows, blocks_per_row, 16)
    d = blocks[..., 208:210].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row)
    positions = np.arange(32)
    scale_offsets = (positions >= 16).astype(np.intp)
    values = np.empty((rows, blocks_per_row, 2, 4, 32), dtype=np.float32)
    for half in range(2):
        for segment in range(4):
            low_source = ql[..., half, segment % 2, :]
            ql_nibble = (low_source & 0x0F) if segment < 2 else ((low_source >> 4) & 0x0F)
            high = (qh[..., half, :] >> (2 * segment)) & 0x03
            q_signed = ((high << 4) | ql_nibble).astype(np.int32) - 32
            scale_index = half * 8 + segment * 2 + scale_offsets
            values[..., half, segment, :] = (
                d[..., None] * scales[..., scale_index] * q_signed.astype(np.float32)
            ).astype(np.float32)
    return values.reshape(rows, k)


def quantize_q8_0(np: Any, values: Any) -> Any:
    rows, k = values.shape
    if k % 32 != 0:
        raise ValueError(f"k must be a multiple of 32: {k}")
    blocks_per_row = k // 32
    chunks = values.astype(np.float32).reshape(rows, blocks_per_row, 32)
    amax = np.max(np.abs(chunks), axis=2).astype(np.float32)
    d = np.where(amax != np.float32(0.0), amax / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    safe_d = np.where(d != np.float32(0.0), d, np.float32(1.0)).astype(np.float32)
    scaled = chunks / safe_d[..., None]
    qs = np.where(d[..., None] != np.float32(0.0), np.rint(scaled), np.float32(0.0))
    qs = np.clip(qs, -128, 127).astype(np.int8)
    packed = np.zeros((rows, blocks_per_row, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    packed[..., 0:2] = d.astype(np.float16).reshape(rows, blocks_per_row, 1).view(np.uint8)
    packed[..., 2:34] = qs.view(np.uint8)
    return packed.reshape(-1).view(np.int8)


def dequant_q8_0(np: Any, packed: Any, k: int, rows: int) -> Any:
    blocks_per_row = k // 32
    blocks = packed.view(np.uint8).reshape(rows, blocks_per_row, Q8_0_BLOCK_BYTES)
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, blocks_per_row)
    qs = blocks[..., 2:34].copy().view(np.int8).astype(np.float32)
    return (qs * d[..., None]).astype(np.float32).reshape(rows, k)


def candidate_seed(candidate: Candidate) -> int:
    text = candidate.id.encode("utf-8")
    value = 0
    for byte in text:
        value = ((value * 131) + byte) & 0xFFFFFFFF
    return value or 1


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_oracle(candidate: Candidate, fixture_dir: Path, *, force: bool = False) -> OracleResult:
    np = require_numpy()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    seed = candidate_seed(candidate)
    family = candidate.family
    spec = ORACLE_SPECS_BY_FAMILY.get(family)
    if spec is None:
        return OracleResult(
            "unsupported_golden",
            None,
            fixture_dir,
            None,
            None,
            None,
            f"no NumPy oracle implemented for family {family}",
        )
    try:
        return spec.generate(np, candidate, fixture_dir, seed)
    except Exception as exc:
        if force:
            raise
        return OracleResult("oracle_failed", family, fixture_dir, None, None, None, str(exc))


def _mul_mat_q4_k_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    if "split_k_reduce2" in candidate.root_symbol:
        spec = LogicalOracleSpec(
            ("mul_mat_q4_k_f32",),
            "split_k_reduce2_f32_numpy_logical",
            {"atol": 1e-5, "rtol": 1e-5},
            _split_k_reduce2_arrays,
        )
        return _logical_oracle(spec, np, candidate, fixture_dir, seed)
    k = int(candidate.shape.get("k", 256))
    rows = int(candidate.shape.get("rows", 1))
    cols = int(candidate.shape.get("cols", 1))
    src0 = q4_k_pattern(np, k, rows, seed=seed)
    src1 = normalized_f32_rows(np, (cols, k), seed=seed + 1)
    weights = dequant_q4_k(np, src0, k, rows)
    expected = np.matmul(weights.astype(np.float32), src1.T.astype(np.float32)).T.reshape(cols * rows)
    dst_init = f32_pattern(np, (cols * rows,), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src0.npy", src0.view(np.int8), allow_pickle=False)
    np.save(fixture_dir / "src1.npy", src1.reshape(cols * k), allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.astype(np.float32), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.astype(np.float32), allow_pickle=False)
    meta = _metadata(candidate, seed, "mul_mat_q4_k_f32_normalized_numpy_dequant_matmul", {"atol": 0.08, "rtol": 0.02})
    meta["arrays"] = {
        "src0": str(fixture_dir / "src0.npy"),
        "src1": str(fixture_dir / "src1.npy"),
        "dst_init": str(fixture_dir / "dst_init.npy"),
        "expected": str(fixture_dir / "expected.npy"),
    }
    meta["fixture_policy"] = {
        "src0": "valid_q4_k_balanced_nibbles_centered_groups_block_rms_0.5",
        "src1": "mean_centered_rows_l2_norm_1.0",
    }
    meta["bytes"] = {
        "src0": q4_k_bytes(k, rows),
        "src1": k * cols * F32_BYTES,
        "dst": rows * cols * F32_BYTES,
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _mul_mat_id_q4_k_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    k = int(candidate.shape.get("k", candidate.shape.get("src0_d0", 256)))
    rows = int(candidate.shape.get("rows", candidate.shape.get("d0", 1)))
    nexperts = int(candidate.shape.get("nexperts", candidate.shape.get("src0_d2", 1)))
    nselected = int(candidate.shape.get("nselected", candidate.shape.get("d1", 1)))
    ntokens = int(candidate.shape.get("ntokens", candidate.shape.get("d2", 1)))
    src1_selected_stride = int(candidate.shape.get("src1_selected_stride", candidate.shape.get("src1_d1_stride", k)))
    src1_token_stride = int(candidate.shape.get("src1_token_stride", candidate.shape.get("src1_d2_stride", k * nselected)))
    idx_token_stride = int(candidate.shape.get("idx_token_stride", candidate.shape.get("src2_d1_stride", nselected)))
    dst_token_stride = int(candidate.shape.get("dst_token_stride", candidate.shape.get("dst_d2_stride", rows * nselected)))
    src0 = q4_k_pattern(np, k, nexperts * rows, seed=seed)
    weights = dequant_q4_k(np, src0, k, nexperts * rows).reshape(nexperts, rows, k)
    src1_elems = ntokens * src1_token_stride
    idx_elems = ntokens * idx_token_stride
    dst_elems = ntokens * dst_token_stride
    src1 = np.zeros((src1_elems,), dtype=np.float32)
    rhs_rows = normalized_f32_rows(np, (ntokens * nselected, k), seed=seed + 1)
    idx = np.zeros((idx_elems,), dtype=np.int32)
    dst_init = f32_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25)
    expected = dst_init.copy()
    for token in range(ntokens):
        for selected in range(nselected):
            expert = (token + selected) % nexperts
            idx[token * idx_token_stride + selected] = expert
            rhs = rhs_rows[token * nselected + selected]
            src1_base = token * src1_token_stride + selected * src1_selected_stride
            src1[src1_base : src1_base + k] = rhs
            dot = np.matmul(weights[expert].astype(np.float32), rhs.astype(np.float32))
            dst_base = token * dst_token_stride + selected * rows
            expected[dst_base : dst_base + rows] = dot.astype(np.float32)
    np.save(fixture_dir / "src0.npy", src0.view(np.int8), allow_pickle=False)
    np.save(fixture_dir / "src1.npy", src1, allow_pickle=False)
    np.save(fixture_dir / "idx.npy", idx, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.astype(np.float32), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.astype(np.float32), allow_pickle=False)
    meta = _metadata(candidate, seed, "mul_mat_id_q4_k_f32_indexed_expert_numpy_dequant_matmul", {"atol": 0.08, "rtol": 0.02})
    meta["fixture_policy"] = {
        "src0": "valid_q4_k_balanced_nibbles_centered_groups_block_rms_0.5_expert_planes",
        "src1": "mean_centered_selected_rows_l2_norm_1.0",
        "idx": "round_robin_expert_indices",
    }
    meta["bytes"] = {
        "src0": q4_k_bytes(k, nexperts * rows),
        "src1": src1_elems * F32_BYTES,
        "idx": idx_elems * I32_BYTES,
        "dst": dst_elems * F32_BYTES,
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _mul_mat_id_q5_k_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    k = int(candidate.shape.get("k", candidate.shape.get("src0_d0", 256)))
    rows = int(candidate.shape.get("rows", candidate.shape.get("d0", 1)))
    nexperts = int(candidate.shape.get("nexperts", candidate.shape.get("src0_d2", 1)))
    nselected = int(candidate.shape.get("nselected", candidate.shape.get("d1", 1)))
    ntokens = int(candidate.shape.get("ntokens", candidate.shape.get("d2", 1)))
    src1_selected_stride = int(candidate.shape.get("src1_selected_stride", candidate.shape.get("src1_d1_stride", k)))
    src1_token_stride = int(candidate.shape.get("src1_token_stride", candidate.shape.get("src1_d2_stride", k * nselected)))
    idx_token_stride = int(candidate.shape.get("idx_token_stride", candidate.shape.get("src2_d1_stride", nselected)))
    dst_token_stride = int(candidate.shape.get("dst_token_stride", candidate.shape.get("dst_d2_stride", rows * nselected)))
    src0 = q5_k_pattern(np, k, nexperts * rows, seed=seed)
    weights = dequant_q5_k(np, src0, k, nexperts * rows).reshape(nexperts, rows, k)
    src1_elems = ntokens * src1_token_stride
    idx_elems = ntokens * idx_token_stride
    dst_elems = ntokens * dst_token_stride
    src1 = np.zeros((src1_elems,), dtype=np.float32)
    rhs_rows = normalized_f32_rows(np, (ntokens * nselected, k), seed=seed + 1)
    idx = np.zeros((idx_elems,), dtype=np.int32)
    dst_init = f32_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25)
    expected = dst_init.copy()
    for token in range(ntokens):
        for selected in range(nselected):
            expert = (token + selected) % nexperts
            idx[token * idx_token_stride + selected] = expert
            rhs = rhs_rows[token * nselected + selected]
            src1_base = token * src1_token_stride + selected * src1_selected_stride
            src1[src1_base : src1_base + k] = rhs
            dot = np.matmul(weights[expert].astype(np.float32), rhs.astype(np.float32))
            dst_base = token * dst_token_stride + selected * rows
            expected[dst_base : dst_base + rows] = dot.astype(np.float32)
    np.save(fixture_dir / "src0.npy", src0.view(np.int8), allow_pickle=False)
    np.save(fixture_dir / "src1.npy", src1, allow_pickle=False)
    np.save(fixture_dir / "idx.npy", idx, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.astype(np.float32), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.astype(np.float32), allow_pickle=False)
    meta = _metadata(candidate, seed, "mul_mat_id_q5_k_f32_indexed_expert_numpy_dequant_matmul", {"atol": 0.12, "rtol": 0.04})
    meta["fixture_policy"] = {
        "src0": "valid_q5_k_balanced_quants_centered_groups_block_rms_0.5_expert_planes",
        "src1": "mean_centered_selected_rows_l2_norm_1.0",
        "idx": "round_robin_expert_indices",
    }
    meta["bytes"] = {
        "src0": q5_k_bytes(k, nexperts * rows),
        "src1": src1_elems * F32_BYTES,
        "idx": idx_elems * I32_BYTES,
        "dst": dst_elems * F32_BYTES,
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _mul_mat_id_q6_k_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    k = int(candidate.shape.get("k", candidate.shape.get("src0_d0", 256)))
    rows = int(candidate.shape.get("rows", candidate.shape.get("d0", 1)))
    nexperts = int(candidate.shape.get("nexperts", candidate.shape.get("src0_d2", 1)))
    nselected = int(candidate.shape.get("nselected", candidate.shape.get("d1", 1)))
    ntokens = int(candidate.shape.get("ntokens", candidate.shape.get("d2", 1)))
    src1_selected_stride = int(candidate.shape.get("src1_selected_stride", candidate.shape.get("src1_d1_stride", k)))
    src1_token_stride = int(candidate.shape.get("src1_token_stride", candidate.shape.get("src1_d2_stride", k * nselected)))
    idx_token_stride = int(candidate.shape.get("idx_token_stride", candidate.shape.get("src2_d1_stride", nselected)))
    dst_token_stride = int(candidate.shape.get("dst_token_stride", candidate.shape.get("dst_d2_stride", rows * nselected)))
    src0 = q6_k_pattern(np, k, nexperts * rows, seed=seed)
    weights = dequant_q6_k(np, src0, k, nexperts * rows).reshape(nexperts, rows, k)
    src1_elems = ntokens * src1_token_stride
    idx_elems = ntokens * idx_token_stride
    dst_elems = ntokens * dst_token_stride
    src1 = np.zeros((src1_elems,), dtype=np.float32)
    rhs_rows = normalized_f32_rows(np, (ntokens * nselected, k), seed=seed + 1)
    idx = np.zeros((idx_elems,), dtype=np.int32)
    dst_init = f32_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25)
    expected = dst_init.copy()
    for token in range(ntokens):
        for selected in range(nselected):
            expert = (token + selected) % nexperts
            idx[token * idx_token_stride + selected] = expert
            rhs = rhs_rows[token * nselected + selected]
            src1_base = token * src1_token_stride + selected * src1_selected_stride
            src1[src1_base : src1_base + k] = rhs
            dot = np.matmul(weights[expert].astype(np.float32), rhs.astype(np.float32))
            dst_base = token * dst_token_stride + selected * rows
            expected[dst_base : dst_base + rows] = dot.astype(np.float32)
    np.save(fixture_dir / "src0.npy", src0.view(np.int8), allow_pickle=False)
    np.save(fixture_dir / "src1.npy", src1, allow_pickle=False)
    np.save(fixture_dir / "idx.npy", idx, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.astype(np.float32), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.astype(np.float32), allow_pickle=False)
    meta = _metadata(candidate, seed, "mul_mat_id_q6_k_f32_indexed_expert_numpy_dequant_matmul", {"atol": 0.12, "rtol": 0.04})
    meta["fixture_policy"] = {
        "src0": "valid_q6_k_balanced_quants_centered_groups_block_rms_0.5_expert_planes",
        "src1": "mean_centered_selected_rows_l2_norm_1.0",
        "idx": "round_robin_expert_indices",
    }
    meta["bytes"] = {
        "src0": q6_k_bytes(k, nexperts * rows),
        "src1": src1_elems * F32_BYTES,
        "idx": idx_elems * I32_BYTES,
        "dst": dst_elems * F32_BYTES,
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _rms_norm_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    ncols, nrows, _ = _dims(candidate)
    eps = np.float32(float(candidate.values.get("eps", 0.0)))
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    scale = np.reciprocal(np.sqrt(np.mean(src * src, axis=1, keepdims=True) + eps)).astype(np.float32)
    expected = (src * scale).astype(np.float32)
    dst_init = f32_pattern(np, (nrows, ncols), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src.npy", src.reshape(nrows * ncols), allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.reshape(nrows * ncols), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.reshape(nrows * ncols), allow_pickle=False)
    meta = _metadata(candidate, seed, "rms_norm_f32_numpy", {"atol": 1e-4, "rtol": 1e-4})
    meta["eps"] = float(eps)
    meta["arrays"] = {
        "src": str(fixture_dir / "src.npy"),
        "dst_init": str(fixture_dir / "dst_init.npy"),
        "expected": str(fixture_dir / "expected.npy"),
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _rms_norm_mul_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    ncols, nrows, elems = _dims(candidate)
    eps = np.float32(0.0)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    weight = f32_pattern(np, (ncols,), seed=seed + 1, scale=0.5) + np.float32(1.0)
    scale = np.reciprocal(np.sqrt(np.mean(src * src, axis=1, keepdims=True) + eps)).astype(np.float32)
    expected = (src * scale * weight.reshape(1, ncols)).astype(np.float32)
    dst_init = f32_pattern(np, (elems,), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src.npy", src.reshape(elems), allow_pickle=False)
    np.save(fixture_dir / "weight.npy", weight, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init, allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.reshape(elems), allow_pickle=False)
    meta = _metadata(candidate, seed, "rms_norm_mul_f32_numpy", {"atol": 1e-4, "rtol": 1e-4})
    meta["eps"] = float(eps)
    meta["fixture_policy"] = {
        "src": "contiguous_f32_rows",
        "weight": "broadcast_f32_row",
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _add_rms_norm_mul_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    ncols, nrows, elems = _dims(candidate)
    eps = np.float32(0.0)
    src0 = f32_pattern(np, (nrows, ncols), seed=seed)
    src1 = f32_pattern(np, (nrows, ncols), seed=seed + 1, scale=0.25)
    weight = f32_pattern(np, (ncols,), seed=seed + 2, scale=0.5) + np.float32(1.0)
    added = (src0 + src1).astype(np.float32)
    scale = np.reciprocal(np.sqrt(np.mean(added * added, axis=1, keepdims=True) + eps)).astype(np.float32)
    expected = (added * scale * weight.reshape(1, ncols)).astype(np.float32)
    add_dst_init = f32_pattern(np, (elems,), seed=seed + 3, scale=0.25)
    dst_init = f32_pattern(np, (elems,), seed=seed + 4, scale=0.25)
    np.save(fixture_dir / "src0.npy", src0.reshape(elems), allow_pickle=False)
    np.save(fixture_dir / "src1.npy", src1.reshape(elems), allow_pickle=False)
    np.save(fixture_dir / "add_dst_init.npy", add_dst_init, allow_pickle=False)
    np.save(fixture_dir / "weight.npy", weight, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init, allow_pickle=False)
    np.save(fixture_dir / "added.npy", added.reshape(elems), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.reshape(elems), allow_pickle=False)
    meta = _metadata(candidate, seed, "add_rms_norm_mul_f32_numpy", {"atol": 1e-4, "rtol": 1e-4})
    meta["eps"] = float(eps)
    meta["fixture_policy"] = {
        "src0": "contiguous_f32_rows",
        "src1": "contiguous_f32_rows",
        "add_dst": "scratch_buffer_checked_after_add",
        "weight": "broadcast_f32_row",
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _pack_q8_1_x4_rows(np: Any, values: Any) -> Any:
    nrows, ncols = values.shape
    block_count = (ncols + 31) // 32
    outer_count = (block_count + 3) // 4
    expected = np.zeros((nrows, outer_count, 144), dtype=np.uint8)
    padded = np.zeros((nrows, block_count * 32), dtype=np.float32)
    padded[:, :ncols] = values.astype(np.float32)
    block_values = padded.reshape(nrows, block_count, 32)
    absmax = np.max(np.abs(block_values), axis=2).astype(np.float32)
    d = np.where(absmax != np.float32(0.0), absmax / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    safe_d = np.where(d != np.float32(0.0), d, np.float32(1.0)).astype(np.float32)
    qs = np.where(d[..., None] != np.float32(0.0), np.rint(block_values / safe_d[..., None]), np.float32(0.0))
    qs = np.clip(qs, -128, 127).astype(np.int8)
    s = (np.sum(qs.astype(np.float32), axis=2, dtype=np.float32) * d).astype(np.float32)
    d_bytes = d.astype(np.float16).view(np.uint8).reshape(nrows, block_count, 2)
    s_bytes = s.astype(np.float16).view(np.uint8).reshape(nrows, block_count, 2)
    block_indices = np.arange(block_count)
    outers = block_indices // 4
    inners = block_indices % 4
    expected[:, outers, inners * 4] = d_bytes[..., 0]
    expected[:, outers, inners * 4 + 1] = d_bytes[..., 1]
    expected[:, outers, inners * 4 + 2] = s_bytes[..., 0]
    expected[:, outers, inners * 4 + 3] = s_bytes[..., 1]
    for inner in range(4):
        mask = inners == inner
        expected[:, outers[mask], 16 + inner * 32 : 16 + inner * 32 + 32] = qs[:, mask].view(np.uint8)
    return expected.reshape(nrows * outer_count * 144).view(np.int8)


def _rms_norm_mul_quantize_q8_1_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    ncols, nrows, elems = _dims(candidate)
    if ncols != 3072 or nrows != 1:
        raise ValueError("rms_norm_mul_quantize_q8_1_f32 oracle currently requires ncols=3072 and nrows=1")
    eps = np.float32(0.0)
    signs = np.where((np.arange(elems, dtype=np.int32) % 2) == 0, np.float32(1.0), np.float32(-1.0))
    src = signs.reshape(nrows, ncols).astype(np.float32)
    weight = np.ones((ncols,), dtype=np.float32)
    scale = np.reciprocal(np.sqrt(np.mean(src * src, axis=1, keepdims=True) + eps)).astype(np.float32)
    quantized_values = (src * scale * weight.reshape(1, ncols)).astype(np.float32)
    expected = _pack_q8_1_x4_rows(np, quantized_values)
    dst_init = np.zeros(expected.shape, dtype=np.int8)
    np.save(fixture_dir / "src.npy", src.reshape(elems), allow_pickle=False)
    np.save(fixture_dir / "weight.npy", weight, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init, allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected, allow_pickle=False)
    meta = _metadata(candidate, seed, "rms_norm_mul_quantize_q8_1_f32_numpy", {"atol": 0.0, "rtol": 0.0})
    meta["eps"] = float(eps)
    meta["fixture_policy"] = {
        "src": "alternating_unit_values_for_exact_rms",
        "weight": "unit_broadcast_f32_row",
        "dst": "q8_1_x4_packed_i8_bytes",
    }
    meta["bytes"] = {"expected": int(expected.size)}
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _copy_f32_f16(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    src_dtype, dst_dtype = _copy_family_dtypes(candidate.family)
    common_dims = _ranked_shape(candidate)
    if common_dims is None:
        n = int(candidate.values.get("shape.copy.n") or _element_count(candidate))
        src = _copy_pattern(np, src_dtype, (n,), seed=seed)
        expected = _copy_cast(np, src, dst_dtype)
        dst_init = _copy_zeros(np, dst_dtype, (n,))
    else:
        src0_dims = _copy_tensor_dims(candidate, "src0", common_dims)
        dst_dims = _copy_tensor_dims(candidate, "dst", common_dims)
        src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
        dst_strides = _copy_tensor_strides(candidate, "dst", dst_dims)
        src0_buffer_len = max(
            _buffer_length(src0_dims, src0_strides),
            _buffer_length(common_dims, src0_strides),
        )
        dst_buffer_len = _buffer_length(dst_dims, dst_strides)
        src = _copy_pattern(np, src_dtype, (src0_buffer_len,), seed=seed)
        dst_init = _copy_zeros(np, dst_dtype, (dst_buffer_len,))
        src_index_dims = common_dims if str(candidate.route_id).endswith("_storage_4d") else src0_dims
        src_indices = _pointwise_logical_indices(np, common_dims, src_index_dims, src0_strides)
        dst_indices = _pointwise_logical_indices(np, common_dims, dst_dims, dst_strides)
        expected = dst_init.copy()
        expected[dst_indices] = _copy_cast(np, src[src_indices], dst_dtype)
    np.save(fixture_dir / "src0.npy", _copy_storage(np, src, src_dtype), allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", _copy_storage(np, dst_init, dst_dtype), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", _copy_storage(np, expected, dst_dtype), allow_pickle=False)
    meta = _metadata(candidate, seed, f"copy_{src_dtype}_{dst_dtype}_numpy_cast", {"atol": 0.0, "rtol": 0.0})
    meta["arrays"] = {
        "src0": str(fixture_dir / "src0.npy"),
        "dst_init": str(fixture_dir / "dst_init.npy"),
        "expected": str(fixture_dir / "expected.npy"),
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _cont_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    element_count = _element_count(candidate)
    src = f32_pattern(np, (element_count,), seed=seed)
    dst_init = f32_pattern(np, (element_count,), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src0.npy", src, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init, allow_pickle=False)
    np.save(fixture_dir / "expected.npy", src.copy(), allow_pickle=False)
    meta = _metadata(candidate, seed, "cont_f32_numpy_copy", {"atol": 0.0, "rtol": 0.0})
    meta["arrays"] = {
        "src0": str(fixture_dir / "src0.npy"),
        "dst_init": str(fixture_dir / "dst_init.npy"),
        "expected": str(fixture_dir / "expected.npy"),
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _ranked_shape(candidate: Candidate) -> tuple[int, ...] | None:
    ranked: list[tuple[int, int]] = []
    for name, value in candidate.shape.items():
        key = str(name)
        if not key.startswith("d") or not key[1:].isdigit():
            continue
        ranked.append((int(key[1:]), int(value)))
    if not ranked:
        return None
    ranked.sort()
    expected = list(range(len(ranked)))
    indices = [index for index, _ in ranked]
    if indices != expected:
        return None
    return tuple(value for _, value in ranked)


def _product(values: tuple[int, ...]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _contiguous_strides(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for size in dimensions:
        strides.append(stride)
        stride *= int(size)
    return tuple(strides)


def _element_count(candidate: Candidate) -> int:
    ranked = _ranked_shape(candidate)
    if ranked is not None:
        return _product(ranked)
    return int(candidate.shape.get("ncols", candidate.shape.get("cols", 1))) * int(
        candidate.shape.get("nrows", candidate.shape.get("rows", 1))
    )


def _dims(candidate: Candidate) -> tuple[int, int, int]:
    ranked = _ranked_shape(candidate)
    if ranked is not None:
        ncols = int(ranked[0])
        nrows = _product(ranked[1:]) if len(ranked) > 1 else 1
        return ncols, nrows, _product(ranked)
    ncols = int(candidate.shape.get("ncols", candidate.shape.get("cols", candidate.shape.get("k", 1))))
    nrows = int(candidate.shape.get("nrows", candidate.shape.get("rows", 1)))
    return ncols, nrows, ncols * nrows


def _pointwise_common_dims(candidate: Candidate) -> tuple[int, ...] | None:
    return _ranked_shape(candidate)


def _copy_tensor_dims(
    candidate: Candidate,
    tensor_name: str,
    common_dims: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        int(candidate.shape.get(f"{tensor_name}_d{index}", common_dims[index]))
        for index in range(len(common_dims))
    )


def _copy_tensor_strides(
    candidate: Candidate,
    tensor_name: str,
    tensor_dims: tuple[int, ...],
) -> tuple[int, ...]:
    defaults = _contiguous_strides(tensor_dims)
    return tuple(
        int(candidate.shape.get(f"{tensor_name}_d{index}_stride", defaults[index]))
        for index in range(len(tensor_dims))
    )


def _set_rows_tensor_roles(candidate: Candidate) -> tuple[str, str]:
    route = candidate.route if isinstance(candidate.route, dict) else {}
    tensors = route.get("tensors")
    if isinstance(tensors, dict) and "src2" in tensors:
        return "src1", "src2"
    return "src0", "src1"


def _pointwise_tensor_dims(
    candidate: Candidate,
    tensor_name: str,
    common_dims: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        int(candidate.shape.get(f"{tensor_name}_d{index}", common_dims[index]))
        for index in range(len(common_dims))
    )


def _pointwise_tensor_strides(
    candidate: Candidate,
    tensor_name: str,
    tensor_dims: tuple[int, ...],
) -> tuple[int, ...]:
    defaults = _contiguous_strides(tensor_dims)
    return tuple(
        int(candidate.shape.get(f"{tensor_name}_d{index}_stride", defaults[index]))
        for index in range(len(tensor_dims))
    )


def _buffer_length(dims: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return 1 + sum((int(dim) - 1) * int(stride) for dim, stride in zip(dims, strides, strict=True))


def _strided_logical_view(np: Any, buffer: Any, dims: tuple[int, ...], strides: tuple[int, ...]) -> Any:
    return np.lib.stride_tricks.as_strided(
        buffer,
        shape=dims,
        strides=tuple(int(stride) * buffer.dtype.itemsize for stride in strides),
    )


def _pointwise_logical_indices(
    np: Any,
    logical_dims: tuple[int, ...],
    tensor_dims: tuple[int, ...],
    tensor_strides: tuple[int, ...],
) -> Any:
    indices = np.indices(logical_dims, dtype=np.int64)
    linear = np.zeros(logical_dims, dtype=np.int64)
    for axis, (dim, stride) in enumerate(zip(tensor_dims, tensor_strides, strict=True)):
        linear += (indices[axis] % np.int64(dim)) * np.int64(stride)
    return linear


def _pointwise_buffers_and_views(
    np: Any,
    candidate: Candidate,
    *,
    src0_seed: int,
    src1_seed: int,
    dst_seed: int,
    src1_positive: bool = False,
    buffer_pattern: Callable[..., Any] = f32_pattern,
    src1_pattern: Callable[..., Any] | None = None,
    dst_pattern: Callable[..., Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    dst_pattern = buffer_pattern if dst_pattern is None else dst_pattern
    if src1_pattern is None:
        src1_pattern = positive_f32_pattern if src1_positive else buffer_pattern
    common_dims = _pointwise_common_dims(candidate)
    if common_dims is None:
        ncols, nrows, elems = _dims(candidate)
        src0_buffer = buffer_pattern(np, (elems,), seed=src0_seed)
        src1_buffer = src1_pattern(np, (elems,), seed=src1_seed)
        dst_init = dst_pattern(np, (elems,), seed=dst_seed, scale=0.25)
        src0_view = src0_buffer.reshape(nrows, ncols)
        src1_view = _pointwise_src1_view(np, candidate, src1_buffer)
        dst_indices = np.arange(elems, dtype=np.int64).reshape(nrows, ncols)
        return src0_buffer, src0_view, src1_buffer, src1_view, (dst_init, dst_indices)

    src0_dims = _pointwise_tensor_dims(candidate, "src0", common_dims)
    src1_dims = _pointwise_tensor_dims(candidate, "src1", common_dims)
    dst_dims = _pointwise_tensor_dims(candidate, "dst", common_dims)
    src0_strides = _pointwise_tensor_strides(candidate, "src0", src0_dims)
    src1_strides = _pointwise_tensor_strides(candidate, "src1", src1_dims)
    dst_strides = _pointwise_tensor_strides(candidate, "dst", dst_dims)
    src0_buffer = buffer_pattern(np, (_buffer_length(src0_dims, src0_strides),), seed=src0_seed)
    src1_buffer = src1_pattern(np, (_buffer_length(src1_dims, src1_strides),), seed=src1_seed)
    dst_init = dst_pattern(np, (_buffer_length(dst_dims, dst_strides),), seed=dst_seed, scale=0.25)
    src0_indices = _pointwise_logical_indices(np, common_dims, src0_dims, src0_strides)
    src1_indices = _pointwise_logical_indices(np, common_dims, src1_dims, src1_strides)
    dst_indices = _pointwise_logical_indices(np, common_dims, dst_dims, dst_strides)
    src0_view = src0_buffer[src0_indices]
    src1_view = src1_buffer[src1_indices]
    return src0_buffer, src0_view, src1_buffer, src1_view, (dst_init, dst_indices)


def _pointwise_buffer_lengths(candidate: Candidate) -> tuple[int, int, int]:
    common_dims = _pointwise_common_dims(candidate)
    if common_dims is None:
        _, _, elems = _dims(candidate)
        return elems, elems, elems
    src0_dims = _pointwise_tensor_dims(candidate, "src0", common_dims)
    src1_dims = _pointwise_tensor_dims(candidate, "src1", common_dims)
    dst_dims = _pointwise_tensor_dims(candidate, "dst", common_dims)
    src0_strides = _pointwise_tensor_strides(candidate, "src0", src0_dims)
    src1_strides = _pointwise_tensor_strides(candidate, "src1", src1_dims)
    dst_strides = _pointwise_tensor_strides(candidate, "dst", dst_dims)
    return (
        _buffer_length(src0_dims, src0_strides),
        _buffer_length(src1_dims, src1_strides),
        _buffer_length(dst_dims, dst_strides),
    )


def _pointwise_src1_view(np: Any, candidate: Candidate, src1_buffer: Any) -> Any:
    ncols, nrows, elems = _dims(candidate)
    src1_row_stride = int(
        candidate.values.get(
            "shape.pointwise.src1_d1_stride",
            candidate.values.get(
                "shape.pointwise.src1_row_stride",
                candidate.shape.get("src1_d1_stride", candidate.shape.get("src1_row_stride", ncols)),
            ),
        )
    )
    src1_ncols = int(
        candidate.values.get(
            "shape.pointwise.src1_d0",
            candidate.values.get(
                "shape.pointwise.src1_ncols",
                candidate.shape.get("src1_d0", candidate.shape.get("src1_ncols", ncols)),
            ),
        )
    )
    row_ids = np.arange(nrows, dtype=np.int64).reshape(nrows, 1)
    col_ids = np.arange(ncols, dtype=np.int64).reshape(1, ncols)
    src1_indices = row_ids * np.int64(src1_row_stride) + (col_ids % np.int64(src1_ncols))
    return src1_buffer[src1_indices]


def _pointwise_src1_values(np: Any, candidate: Candidate, seed: int) -> tuple[Any, Any]:
    _, _, elems = _dims(candidate)
    src1_buffer = f32_pattern(np, (elems,), seed=seed)
    return src1_buffer, _pointwise_src1_view(np, candidate, src1_buffer)


def _matmul_dims(candidate: Candidate) -> tuple[int, int, int]:
    k = int(
        candidate.shape.get(
            "k",
            candidate.shape.get("src0_d0", candidate.shape.get("ncols", candidate.shape.get("cols", 256))),
        )
    )
    rows = int(candidate.shape.get("rows", candidate.shape.get("nrows", candidate.shape.get("d0", 1))))
    cols = int(candidate.shape.get("cols", candidate.shape.get("ncols", candidate.shape.get("d1", 1))))
    return k, rows, cols


F16_MUL_MAT_FAMILIES = {
    "mul_mat_f16_f16_generic",
    "mul_mat_f16_f32_generic",
}
FULL_SRC0_MUL_MAT_SOURCE_IDS = {
    "mul_mat_f16_f16_contiguous",
}


def _batched_mul_mat_f16_f32_dims(candidate: Candidate) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
] | None:
    src0_dims = _captured_tensor_dims(candidate, "src0")
    dst_dims = _captured_tensor_dims(candidate, "dst")
    src1_dims = _captured_tensor_dims(candidate, "src1")
    if src0_dims is None or dst_dims is None or src1_dims is None:
        return None
    if dst_dims[2:] == (1, 1):
        return None
    src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
    src1_strides = _copy_tensor_strides(candidate, "src1", src1_dims)
    dst_strides = _copy_tensor_strides(candidate, "dst", dst_dims)
    return src0_dims, src1_dims, dst_dims, src0_strides, src1_strides, dst_strides


def _captured_tensor_dims(candidate: Candidate, tensor_name: str) -> tuple[int, int, int, int] | None:
    keys = tuple(f"{tensor_name}_d{index}" for index in range(4))
    if all(key in candidate.shape for key in keys):
        return tuple(int(candidate.shape[key]) for key in keys)
    anchor_keys = tuple(f"d{index}" for index in range(4))
    if all(key in candidate.shape for key in anchor_keys):
        anchor_dims = tuple(int(candidate.shape[key]) for key in anchor_keys)
        if tensor_name == "dst":
            return anchor_dims
        dims: list[int] = []
        for index, anchor_dim in enumerate(anchor_dims):
            value_key = f"tensor.{tensor_name}.dimensions.d{index}.size"
            shape_key = f"{tensor_name}_d{index}"
            if value_key in candidate.values:
                dims.append(int(candidate.values[value_key]))
                continue
            if shape_key in candidate.shape:
                dims.append(int(candidate.shape[shape_key]))
                continue
            if (
                tensor_name == "src0"
                and candidate.family in F16_MUL_MAT_FAMILIES
                and candidate.source_id not in FULL_SRC0_MUL_MAT_SOURCE_IDS
                and index >= 2
            ):
                dims.append(1)
                continue
            dims.append(anchor_dim)
        return tuple(dims)
    return None


def _write_arrays(np: Any, fixture_dir: Path, arrays: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = fixture_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = str(path)
    return paths


def _logical_oracle(spec: LogicalOracleSpec, np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    data = spec.build(np, candidate, seed)
    expected = data["arrays"].get("expected")
    if expected is None:
        raise ValueError(f"logical oracle {spec.oracle} did not produce an expected array")
    array_paths = _write_arrays(np, fixture_dir, data["arrays"])
    meta = _metadata(candidate, seed, spec.oracle, spec.tolerance)
    meta["exact_kernel_abi"] = spec.exact_kernel_abi
    meta["oracle_scope"] = "kernel_abi" if spec.exact_kernel_abi else "logical_numpy"
    meta["arrays"] = array_paths
    meta.update(data.get("metadata") or {})
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", spec.oracle, fixture_dir, meta_path, fixture_dir / "expected.npy", spec.tolerance)


def _logical_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[None, dict[str, Any]]:
    return None, {
        "status": "unsupported_workbench",
        "message": f"{candidate.family} has a NumPy logical oracle but no generated check.case ABI yet",
        "fixture_dir": str(fixture_dir),
    }


def _case_names(candidate: Candidate) -> tuple[str, str]:
    return f"@case_{candidate.id}", f"@bench_{candidate.id}"


def _emit_case(linked_source: Path, workbench_path: Path, case_name: str, bench_name: str, body: str) -> tuple[str, dict[str, Any]]:
    _source_plus_case(
        linked_source,
        workbench_path,
        f"""
check.case public {case_name} {{
{body.rstrip()}
  check.return
}}
""",
    )
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _copy_family_dtypes(family: str) -> tuple[str, str]:
    prefix = "copy_"
    if not family.startswith(prefix):
        raise ValueError(f"unsupported COPY family {family}")
    parts = family[len(prefix) :].split("_")
    if len(parts) != 2:
        raise ValueError(f"unsupported COPY family {family}")
    src_dtype, dst_dtype = parts
    supported = {"bf16", "f16", "f32", "i32"}
    if src_dtype not in supported or dst_dtype not in supported:
        raise ValueError(f"unsupported COPY family {family}")
    return src_dtype, dst_dtype


def _read_f32(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xf32>"""


def _read_f16(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xf16>"""


def _read_i32(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xi32>"""


def _read_i64(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xi64>"""


def _read_i16(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xi16>"""


def _f16_bits(np: Any, array: Any) -> Any:
    return np.ascontiguousarray(array.reshape(-1)).view(np.int16)


def _bf16_bits(np: Any, array: Any) -> Any:
    shape = array.shape
    values = np.ascontiguousarray(array.reshape(-1)).astype(np.float32, copy=False)
    bits = values.view(np.uint32)
    rounded = bits + (np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
    return ((rounded >> np.uint32(16)).astype(np.uint16)).view(np.int16).reshape(shape)


def _bf16_values(np: Any, array: Any) -> Any:
    shape = array.shape
    bits = _bf16_bits(np, array).reshape(-1).view(np.uint16).astype(np.uint32)
    return (bits << np.uint32(16)).view(np.float32).reshape(shape)


def _copy_pattern(np: Any, dtype: str, shape: tuple[int, ...], *, seed: int) -> Any:
    if dtype == "bf16":
        return _bf16_values(np, f32_pattern(np, shape, seed=seed))
    if dtype == "f16":
        return f16_pattern(np, shape, seed=seed)
    if dtype == "f32":
        return f32_pattern(np, shape, seed=seed)
    if dtype == "i32":
        seed_offset = np.int32(seed % 65521)
        return ((np.arange(_product(shape), dtype=np.int32).reshape(shape) * np.int32(17) + seed_offset) % np.int32(65521)).astype(np.int32)
    raise ValueError(f"unsupported COPY dtype {dtype}")


def _copy_cast(np: Any, values: Any, dtype: str) -> Any:
    if dtype == "bf16":
        return _bf16_values(np, values)
    if dtype == "f16":
        return values.astype(np.float16)
    if dtype == "f32":
        return values.astype(np.float32)
    if dtype == "i32":
        return values.astype(np.int32)
    raise ValueError(f"unsupported COPY dtype {dtype}")


def _copy_zeros(np: Any, dtype: str, shape: tuple[int, ...]) -> Any:
    if dtype == "i32":
        return np.zeros(shape, dtype=np.int32)
    if dtype == "f16":
        return np.zeros(shape, dtype=np.float16)
    return np.zeros(shape, dtype=np.float32)


def _copy_storage(np: Any, values: Any, dtype: str) -> Any:
    if dtype == "bf16":
        return _bf16_bits(np, values)
    if dtype == "f16":
        return _f16_bits(np, values)
    if dtype == "f32":
        return values.astype(np.float32)
    if dtype == "i32":
        return values.astype(np.int32)
    raise ValueError(f"unsupported COPY dtype {dtype}")


def _copy_tensor_type(dtype: str, elems: int) -> str:
    if dtype in {"bf16", "f16"}:
        return f"tensor<{elems}xi16>"
    if dtype == "i32":
        return f"tensor<{elems}xi32>"
    return f"tensor<{elems}xf32>"


def _read_copy_tensor(
    workbench_path: Path,
    fixture_dir: Path,
    name: str,
    dtype: str,
    elems: int,
) -> str:
    if dtype == "i32":
        return _read_i32(workbench_path, fixture_dir, name, elems)
    return _read_i16(workbench_path, fixture_dir, name, elems) if dtype in {"bf16", "f16"} else _read_f32(workbench_path, fixture_dir, name, elems)


def _binary_arrays(op: Callable[[Any, Any], Any]) -> Callable[[Any, Candidate, int], dict[str, Any]]:
    def build(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
        src0_buffer, src0, src1_buffer, src1, (dst_init, dst_indices) = _pointwise_buffers_and_views(
            np,
            candidate,
            src0_seed=seed,
            src1_seed=seed + 1,
            dst_seed=seed + 2,
        )
        expected = dst_init.copy()
        expected[dst_indices] = op(src0, src1).astype(np.float32)
        return {
            "arrays": {
                "src0": src0_buffer.reshape(-1),
                "src1": src1_buffer.reshape(-1),
                "dst_init": dst_init.reshape(-1),
                "expected": expected.reshape(-1),
            }
        }

    return build


def _div_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    src0_buffer, src0, src1_buffer, src1, (dst_init, dst_indices) = _pointwise_buffers_and_views(
        np,
        candidate,
        src0_seed=seed,
        src1_seed=seed + 1,
        dst_seed=seed + 2,
        src1_positive=True,
    )
    expected = dst_init.copy()
    expected[dst_indices] = (src0 / src1).astype(np.float32)
    return {
        "arrays": {
            "src0": src0_buffer.reshape(-1),
            "src1": src1_buffer.reshape(-1),
            "dst_init": dst_init.reshape(-1),
            "expected": expected.reshape(-1),
        }
    }


def _positive_f16_pattern(np: Any, shape: tuple[int, ...], *, seed: int, scale: float = 0.25):
    return positive_f32_pattern(np, shape, seed=seed, scale=scale).astype(np.float16)


def _pointwise_exact_arrays(
    op: Callable[[Any, Any], Any],
    *,
    src1_positive: bool = False,
) -> Callable[[Any, Candidate, int], dict[str, Any]]:
    def build(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
        src1_pattern = _positive_f16_pattern if src1_positive else f16_pattern
        src0_buffer, src0, src1_buffer, src1, (dst_init, dst_indices) = _pointwise_buffers_and_views(
            np,
            candidate,
            src0_seed=seed,
            src1_seed=seed + 1,
            dst_seed=seed + 2,
            buffer_pattern=f16_pattern,
            src1_pattern=src1_pattern,
        )
        expected = dst_init.copy()
        expected[dst_indices] = op(src0.astype(np.float32), src1.astype(np.float32)).astype(np.float16)
        return {
            "arrays": {
                "src0": _f16_bits(np, src0_buffer),
                "src1": _f16_bits(np, src1_buffer),
                "dst_init": _f16_bits(np, dst_init),
                "expected": _f16_bits(np, expected),
            }
        }

    return build


def _add_f16_exact_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    return _pointwise_exact_arrays(lambda lhs, rhs: lhs + rhs)(np, candidate, seed)


def _mul_f16_exact_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    return _pointwise_exact_arrays(lambda lhs, rhs: lhs * rhs)(np, candidate, seed)


def _div_f16_exact_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    return _pointwise_exact_arrays(lambda lhs, rhs: lhs / rhs, src1_positive=True)(np, candidate, seed)


def _sub_f16_exact_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    return _pointwise_exact_arrays(lambda lhs, rhs: lhs - rhs)(np, candidate, seed)


def _scale_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    scale = np.float32(0.625)
    bias = np.float32(-0.125)
    src0 = f32_pattern(np, (nrows, ncols), seed=seed)
    return {
        "arrays": {
            "src0": src0.reshape(elems),
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": (src0 * scale + bias).astype(np.float32).reshape(elems),
        },
        "metadata": {"scale": float(scale), "bias": float(bias)},
    }


def _clamp_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src0 = f32_pattern(np, (nrows, ncols), seed=seed)
    lo = np.float32(-0.45)
    hi = np.float32(0.55)
    return {
        "arrays": {
            "src0": src0.reshape(elems),
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": np.clip(src0, lo, hi).astype(np.float32).reshape(elems),
        },
        "metadata": {"min": float(lo), "max": float(hi)},
    }


def _clamp_f16_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src0 = f16_pattern(np, (nrows, ncols), seed=seed)
    lo = np.float32(-0.45)
    hi = np.float32(0.55)
    expected = np.clip(src0.astype(np.float32), lo, hi).astype(np.float16)
    return {
        "arrays": {
            "src0": _f16_bits(np, src0.reshape(elems)),
            "dst_init": _f16_bits(np, f16_pattern(np, (elems,), seed=seed + 2)),
            "expected": _f16_bits(np, expected.reshape(elems)),
        },
        "metadata": {"min": float(lo), "max": float(hi)},
    }


def _unary_source_values(np: Any, candidate: Candidate, seed: int) -> Any:
    ncols, nrows, _ = _dims(candidate)
    shape = (nrows, ncols)
    if candidate.family in {
        "elu_f32",
        "elu_f16",
        "exp_f32",
        "exp_f16",
        "expm1_f32",
        "expm1_f16",
        "gelu_f32",
        "gelu_f16",
        "gelu_erf_f32",
        "gelu_erf_f16",
        "gelu_quick_f32",
        "gelu_quick_f16",
        "sigmoid_f32",
        "sigmoid_f16",
        "silu_f32",
        "silu_f16",
        "softplus_f32",
        "softplus_f16",
        "tanh_f32",
        "tanh_f16",
    }:
        return f32_pattern(np, shape, seed=seed, scale=0.25)
    if candidate.family in {"log_f32", "log_f16", "sqrt_f32", "sqrt_f16"}:
        return positive_f32_pattern(np, shape, seed=seed, scale=0.25)
    return f32_pattern(np, shape, seed=seed)


def _unary_apply(np: Any, family: str, values: Any) -> Any:
    if family in {"abs_f32", "abs_f16"}:
        return np.abs(values)
    if family in {"ceil_f32", "ceil_f16"}:
        return np.ceil(values)
    if family in {"cos_f32", "cos_f16"}:
        return np.cos(values)
    if family in {"elu_f32", "elu_f16"}:
        return np.where(values > np.float32(0.0), values, np.exp(values) - np.float32(1.0))
    if family in {"exp_f32", "exp_f16"}:
        return np.exp(values)
    if family in {"expm1_f32", "expm1_f16"}:
        return np.exp(values) - np.float32(1.0)
    if family in {"floor_f32", "floor_f16"}:
        return np.floor(values)
    if family in {"gelu_f32", "gelu_f16"}:
        coef = np.float32(0.044715)
        sqrt_two_over_pi = np.float32(0.7978845608028654)
        return np.float32(0.5) * values * (
            np.float32(1.0)
            + np.tanh(sqrt_two_over_pi * values * (np.float32(1.0) + coef * values * values))
        )
    if family in {"gelu_erf_f32", "gelu_erf_f16"}:
        sqrt_two_inv = np.float32(0.7071067811865476)
        erf_values = np.vectorize(math.erf, otypes=[np.float32])(values * sqrt_two_inv)
        return np.float32(0.5) * values * (np.float32(1.0) + erf_values)
    if family in {"gelu_quick_f32", "gelu_quick_f16"}:
        return values / (np.float32(1.0) + np.exp(np.float32(-1.702) * values))
    if family in {"hardsigmoid_f32", "hardsigmoid_f16"}:
        return np.minimum(
            np.float32(1.0),
            np.maximum(np.float32(0.0), (values + np.float32(3.0)) / np.float32(6.0)),
        )
    if family in {"hardswish_f32", "hardswish_f16"}:
        gate = np.minimum(
            np.float32(1.0),
            np.maximum(np.float32(0.0), (values + np.float32(3.0)) / np.float32(6.0)),
        )
        return values * gate
    if family in {"leaky_relu_f32", "leaky_relu_f16"}:
        return np.where(values >= np.float32(0.0), values, values * np.float32(0.1))
    if family in {"log_f32", "log_f16"}:
        return np.log(values)
    if family in {"neg_f32", "neg_f16"}:
        return np.negative(values)
    if family in {"relu_f32", "relu_f16"}:
        return np.maximum(values, np.float32(0.0))
    if family in {"round_f32", "round_f16"}:
        return np.where(
            values >= np.float32(0.0),
            np.floor(values + np.float32(0.5)),
            np.ceil(values - np.float32(0.5)),
        )
    if family in {"sgn_f32", "sgn_f16"}:
        return np.sign(values)
    if family in {"sigmoid_f32", "sigmoid_f16"}:
        return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))
    if family in {"silu_f32", "silu_f16"}:
        return values / (np.float32(1.0) + np.exp(-values))
    if family in {"sin_f32", "sin_f16"}:
        return np.sin(values)
    if family == "softcap_f32":
        softcap = np.float32(50.0)
        return softcap * np.tanh(values / softcap)
    if family in {"softplus_f32", "softplus_f16"}:
        return np.log1p(np.exp(values))
    if family in {"sqr_f32", "sqr_f16"}:
        return np.square(values)
    if family in {"sqrt_f32", "sqrt_f16"}:
        return np.sqrt(values)
    if family in {"step_f32", "step_f16"}:
        return np.where(values > np.float32(0.0), np.float32(1.0), np.float32(0.0))
    if family in {"tanh_f32", "tanh_f16"}:
        return np.tanh(values)
    if family in {"trunc_f32", "trunc_f16"}:
        return np.trunc(values)
    if family == "xielu_f32":
        alpha_n = np.float32(4.0)
        alpha_p = np.float32(20.0)
        beta = np.float32(0.5)
        eps = np.float32(0.0000001)
        return np.where(
            values > np.float32(0.0),
            alpha_p * values * values + beta * values,
            (np.exp(np.minimum(values, eps)) - np.float32(1.0) - values) * alpha_n
            + beta * values,
        )
    raise ValueError(f"unsupported unary family {family}")


def _unary_source_buffer(np: Any, candidate: Candidate, length: int, seed: int) -> Any:
    shape = (length,)
    if candidate.family in {
        "elu_f32",
        "elu_f16",
        "exp_f32",
        "exp_f16",
        "expm1_f32",
        "expm1_f16",
        "gelu_f32",
        "gelu_f16",
        "gelu_erf_f32",
        "gelu_erf_f16",
        "gelu_quick_f32",
        "gelu_quick_f16",
        "sigmoid_f32",
        "sigmoid_f16",
        "silu_f32",
        "silu_f16",
        "softplus_f32",
        "softplus_f16",
        "tanh_f32",
        "tanh_f16",
    }:
        return f32_pattern(np, shape, seed=seed, scale=0.25)
    if candidate.family in {"log_f32", "log_f16", "sqrt_f32", "sqrt_f16"}:
        return positive_f32_pattern(np, shape, seed=seed, scale=0.25)
    return f32_pattern(np, shape, seed=seed)


def _strided_unary_arrays(
    np: Any,
    candidate: Candidate,
    seed: int,
    common_dims: tuple[int, ...],
    src0_dims: tuple[int, ...],
    src0_strides: tuple[int, ...],
) -> dict[str, Any]:
    # src0 is a non-contiguous view of a padded buffer (ggml v=1). Build the minimal enclosing
    # buffer, gather the logical view through it, and lay `expected` out as a contiguous dst.
    dst_elems = _product(common_dims)
    src0_buffer_len = _buffer_length(src0_dims, src0_strides)
    src0_indices = _pointwise_logical_indices(np, common_dims, src0_dims, src0_strides)
    dst_indices = _pointwise_logical_indices(
        np, common_dims, common_dims, _contiguous_strides(common_dims)
    )
    src_buffer = _unary_source_buffer(np, candidate, src0_buffer_len, seed)
    if candidate.family.endswith("_f16"):
        src0_buffer = src_buffer.astype(np.float16)
        applied = _unary_apply(
            np, candidate.family, src0_buffer[src0_indices].astype(np.float32)
        ).astype(np.float16)
        expected = np.zeros((dst_elems,), dtype=np.float16)
        expected[dst_indices] = applied
        return {
            "arrays": {
                "src0": _f16_bits(np, src0_buffer),
                "dst_init": np.zeros((dst_elems,), dtype=np.int16),
                "expected": _f16_bits(np, expected),
            },
            "metadata": {"dst_type": "i16"},
        }
    src0_buffer = src_buffer.astype(np.float32)
    applied = _unary_apply(np, candidate.family, src0_buffer[src0_indices]).astype(np.float32)
    dst_init = f32_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25)
    expected = dst_init.copy()
    expected[dst_indices] = applied
    return {
        "arrays": {
            "src0": src0_buffer,
            "dst_init": dst_init,
            "expected": expected,
        },
    }


def _unary_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    common_dims = _pointwise_common_dims(candidate)
    # V2 encodes shape as ranked dims "d0","d1",... (plus per-tensor deltas like
    # "src0_d1_stride" when a view is non-contiguous). Legacy V1 families
    # instead use "ncols"/"nrows"/"k", for which _ranked_shape returns None.
    # Strided views are only supported on V2 path; the stride check below (not
    # this guard) is what selects the strided branch.
    if common_dims is not None:
        src0_dims = _copy_tensor_dims(candidate, "src0", common_dims)
        src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
        if tuple(src0_strides) != _contiguous_strides(src0_dims):
            return _strided_unary_arrays(np, candidate, seed, common_dims, src0_dims, src0_strides)
    _, _, elems = _dims(candidate)
    src = _unary_source_values(np, candidate, seed)
    if candidate.family.endswith("_f16"):
        src0 = src.astype(np.float16)
        expected = _unary_apply(np, candidate.family, src0.astype(np.float32)).astype(np.float16)
        return {
            "arrays": {
                "src0": src0.reshape(elems).view(np.int16),
                "dst_init": np.zeros((elems,), dtype=np.int16),
                "expected": expected.reshape(elems).view(np.int16),
            },
            "metadata": {"dst_type": "i16"},
        }
    src0 = src.astype(np.float32)
    return {
        "arrays": {
            "src0": src0.reshape(elems),
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": _unary_apply(np, candidate.family, src0).astype(np.float32).reshape(elems),
        },
    }


def _rms_norm_mul_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    weight = f32_pattern(np, (ncols,), seed=seed + 1, scale=0.5) + np.float32(1.0)
    scale = np.reciprocal(np.sqrt(np.mean(src * src, axis=1, keepdims=True))).astype(np.float32)
    expected = (src * scale * weight.reshape(1, ncols)).astype(np.float32)
    return {
        "arrays": {
            "src": src.reshape(elems),
            "weight": weight,
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(elems),
        }
    }


def _add_rms_norm_mul_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    base = f32_pattern(np, (nrows, ncols), seed=seed)
    residual = f32_pattern(np, (nrows, ncols), seed=seed + 1, scale=0.25)
    weight = f32_pattern(np, (ncols,), seed=seed + 2, scale=0.5) + np.float32(1.0)
    added = (base + residual).astype(np.float32)
    scale = np.reciprocal(np.sqrt(np.mean(added * added, axis=1, keepdims=True))).astype(np.float32)
    expected = (added * scale * weight.reshape(1, ncols)).astype(np.float32)
    return {
        "arrays": {
            "src0": base.reshape(elems),
            "src1": residual.reshape(elems),
            "weight": weight,
            "dst_init": f32_pattern(np, (elems,), seed=seed + 3, scale=0.25),
            "expected": expected.reshape(elems),
        }
    }


def _swiglu_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    f16_family = candidate.family == "swiglu_f16"
    x = f32_pattern(np, (nrows, ncols), seed=seed)
    gate = f32_pattern(np, (nrows, ncols), seed=seed + 1)
    if f16_family:
        x = x.astype(np.float16)
        gate = gate.astype(np.float16)
    x_compute = x.astype(np.float32)
    gate_compute = gate.astype(np.float32)
    if "geglu" in candidate.root_symbol:
        inner = x_compute * (np.float32(1.0) + np.float32(0.044715) * x_compute * x_compute)
        gelu = x_compute / (np.float32(1.0) + np.exp(-np.float32(1.5957691216057308) * inner, dtype=np.float32))
        activated = gelu.astype(np.float32)
    else:
        activated = (x_compute / (np.float32(1.0) + np.exp(-x_compute, dtype=np.float32))).astype(np.float32)
    expected_f32 = (activated * gate_compute).astype(np.float32)
    src0_dims = _swiglu_tensor_dims(candidate, "src0", ncols=ncols)
    src0_strides = _swiglu_tensor_strides(candidate, "src0", src0_dims)
    store_dtype = np.int16 if f16_family else np.float32
    expected = _f16_bits(np, expected_f32.astype(np.float16)) if f16_family else expected_f32.reshape(elems)
    arrays: dict[str, Any] = {
        "dst_init": np.zeros((elems,), dtype=np.int16) if f16_family else f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
        "expected": expected,
    }
    if candidate.root_symbol.endswith("_split"):
        src1_dims = _swiglu_tensor_dims(candidate, "src1", ncols=ncols)
        src1_strides = _swiglu_tensor_strides(candidate, "src1", src1_dims)
        arrays["src0"] = _swiglu_store_rows(np, x, src0_dims, src0_strides, dtype=store_dtype)
        arrays["src1"] = _swiglu_store_rows(np, gate, src1_dims, src1_strides, dtype=store_dtype)
    else:
        swapped = _swiglu_swapped(candidate)
        first, second = (gate, x) if swapped else (x, gate)
        if str(candidate.route_id) in {"swiglu_f32_packed_contiguous_4d", "swiglu_f16_packed_contiguous_4d"}:
            packed = np.concatenate([first, second], axis=1)
            arrays["src0"] = _f16_bits(np, packed.astype(np.float16)) if f16_family else packed.reshape(elems * 2)
        else:
            arrays["src0"] = _swiglu_store_packed_rows(np, first, second, src0_dims, src0_strides, ncols, dtype=store_dtype)
    return {
        "arrays": arrays,
        "metadata": {"activation": "gelu" if "geglu" in candidate.root_symbol else "silu", "dst_type": "i16" if f16_family else "f32"},
    }


def _swiglu_swapped(candidate: Candidate) -> bool:
    for key in ("swiglu.swapped", "attribute.swapped"):
        value = candidate.values.get(key, candidate.shape.get(key))
        if value is not None:
            return int(value) != 0
    value = candidate.config.get("@shape.swiglu.swapped")
    return value is not None and int(value) != 0


def _swiglu_tensor_dims(candidate: Candidate, tensor_name: str, *, ncols: int) -> tuple[int, int, int, int]:
    default_d0 = int(candidate.shape.get("d0", ncols))
    if tensor_name == "src0" and not str(candidate.root_symbol).endswith("_split"):
        default_d0 = max(default_d0 * 2, int(candidate.shape.get("src0_d0", default_d0)))
    return tuple(
        int(candidate.shape.get(f"{tensor_name}_d{index}", candidate.shape.get(f"d{index}", default_d0 if index == 0 else 1)))
        for index in range(4)
    )


def _swiglu_tensor_strides(candidate: Candidate, tensor_name: str, dims: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    defaults = _contiguous_strides(dims)
    return tuple(
        int(candidate.shape.get(f"{tensor_name}_d{index}_stride", defaults[index]))
        for index in range(4)
    )


def _strided_storage_elements(dims: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return sum((int(dim) - 1) * int(stride) for dim, stride in zip(dims, strides, strict=True)) + 1


def _swiglu_store_rows(
    np: Any,
    rows: Any,
    dims: tuple[int, int, int, int],
    strides: tuple[int, int, int, int],
    *,
    dtype: Any = None,
) -> Any:
    storage_dtype = dtype if dtype is not None else np.float32
    storage = np.zeros((_strided_storage_elements(dims, strides),), dtype=storage_dtype)
    stored_rows = _f16_bits(np, rows.astype(np.float16)).reshape(rows.shape) if storage_dtype == np.int16 else rows
    row_index = 0
    for i3 in range(dims[3]):
        for i2 in range(dims[2]):
            for i1 in range(dims[1]):
                base = i1 * strides[1] + i2 * strides[2] + i3 * strides[3]
                storage[base : base + rows.shape[1]] = stored_rows[row_index]
                row_index += 1
    return storage


def _swiglu_store_packed_rows(
    np: Any,
    first: Any,
    second: Any,
    dims: tuple[int, int, int, int],
    strides: tuple[int, int, int, int],
    ncols: int,
    *,
    dtype: Any = None,
) -> Any:
    storage_dtype = dtype if dtype is not None else np.float32
    storage = np.zeros((_strided_storage_elements(dims, strides),), dtype=storage_dtype)
    first_rows = _f16_bits(np, first.astype(np.float16)).reshape(first.shape) if storage_dtype == np.int16 else first
    second_rows = _f16_bits(np, second.astype(np.float16)).reshape(second.shape) if storage_dtype == np.int16 else second
    row_index = 0
    for i3 in range(dims[3]):
        for i2 in range(dims[2]):
            for i1 in range(dims[1]):
                base = i1 * strides[1] + i2 * strides[2] + i3 * strides[3]
                storage[base : base + ncols] = first_rows[row_index]
                storage[base + ncols : base + ncols * 2] = second_rows[row_index]
                row_index += 1
    return storage


def _sum_rows_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, _ = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    expected = np.sum(src, axis=1).astype(np.float32)
    return {"arrays": {"src0": src.reshape(nrows * ncols), "expected": expected}}


def _candidate_f32_value(candidate: Candidate, name: str, default: float) -> float:
    value = candidate.values.get(name)
    if value is None:
        return float(default)
    return float(value)


def _candidate_f32_literal(candidate: Candidate, name: str, default: float) -> str:
    return format(_candidate_f32_value(candidate, name, default), ".17g")


def _softmax_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    scale = np.float32(_candidate_f32_value(candidate, "scale", 0.75))
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    arrays: dict[str, Any] = {
        "src0": src.reshape(elems),
        "dst_init": f32_pattern(np, (elems,), seed=seed + 1, scale=0.25),
    }
    logits = src * scale
    if "mask" in candidate.root_symbol:
        mask = f32_pattern(np, (nrows, ncols), seed=seed + 3, scale=0.125)
        arrays["mask"] = mask.reshape(elems)
        logits = logits + mask
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    expected = (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)
    arrays["expected"] = expected.reshape(elems)
    return {"arrays": arrays, "metadata": {"scale": float(scale), "has_mask": "mask" in candidate.root_symbol}}


def _argsort_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    expected = np.argsort(-src, axis=1).astype(np.int32)
    return {
        "arrays": {
            "src0": src.reshape(elems),
            "dst_init": np.full((elems,), -1, dtype=np.int32),
            "expected": expected.reshape(elems),
        }
    }


def _get_rows_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    if candidate.family == "get_rows_moe_weights_f32":
        return _get_rows_moe_weights_arrays(np, candidate, seed)
    ncols, nrows, elems = _dims(candidate)
    src0_nrows = int(candidate.values.get("shape.get_rows.src0_nrows") or candidate.values.get("get_rows.src0_nrows") or nrows)
    src_f32 = f32_pattern(np, (src0_nrows, ncols), seed=seed)
    indices = ((np.arange(nrows, dtype=np.int64) * 3 + seed) % src0_nrows).astype(np.int32)
    src = src_f32.reshape(src0_nrows * ncols)
    metadata: dict[str, Any] = {"src0_nrows": src0_nrows}
    if candidate.family == "get_rows_q4_k_f32":
        src = q4_k_pattern(np, ncols, src0_nrows, seed=seed).view(np.int8)
        src_f32 = dequant_q4_k(np, src.view(np.uint8), ncols, src0_nrows)
        metadata["q4_k_block_bytes"] = Q4_K_BLOCK_BYTES
        metadata["bytes"] = {"src0": q4_k_bytes(ncols, src0_nrows), "dst": elems * F32_BYTES}
    elif candidate.family == "get_rows_q5_k_f32":
        src = q5_k_pattern(np, ncols, src0_nrows, seed=seed).view(np.int8)
        src_f32 = dequant_q5_k(np, src.view(np.uint8), ncols, src0_nrows)
        metadata["q5_k_block_bytes"] = Q5_K_BLOCK_BYTES
        metadata["bytes"] = {"src0": q5_k_bytes(ncols, src0_nrows), "dst": elems * F32_BYTES}
    elif candidate.family == "get_rows_q6_k_f32":
        src = q6_k_pattern(np, ncols, src0_nrows, seed=seed).view(np.int8)
        src_f32 = dequant_q6_k(np, src.view(np.uint8), ncols, src0_nrows)
        metadata["q6_k_block_bytes"] = Q6_K_BLOCK_BYTES
        metadata["bytes"] = {"src0": q6_k_bytes(ncols, src0_nrows), "dst": elems * F32_BYTES}
    elif candidate.family == "get_rows_q8_0_f32":
        src = quantize_q8_0(np, src_f32)
        src_f32 = dequant_q8_0(np, src, ncols, src0_nrows)
        metadata["q8_0_block_bytes"] = Q8_0_BLOCK_BYTES
        metadata["bytes"] = {"src0": q8_0_bytes(ncols, src0_nrows), "dst": elems * F32_BYTES}
    expected = src_f32[indices].astype(np.float32)
    return {
        "arrays": {
            "src0": src,
            "indices": indices,
            "src1": indices,
            "dst_init": f32_pattern(np, (elems,), seed=seed + 1, scale=0.25),
            "expected": expected.reshape(elems),
        },
        "metadata": metadata,
    }


def _get_rows_moe_weights_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    nselected = int(candidate.shape.get("d0", candidate.values.get("shape.get_rows_moe.nselected", 8)))
    ntokens = int(candidate.shape.get("d1", candidate.values.get("shape.get_rows_moe.ntokens", 1)))
    nexperts = int(candidate.shape.get("src0_d0", candidate.values.get("shape.get_rows_moe.nexperts", 128)))
    src0_token_stride = int(
        candidate.shape.get("src0_d1_stride", candidate.values.get("shape.get_rows_moe.src0_token_stride", nexperts))
    )
    idx_token_stride = int(
        candidate.shape.get("src1_d1_stride", candidate.values.get("shape.get_rows_moe.idx_token_stride", nselected))
    )
    dst_token_stride = int(
        candidate.shape.get("dst_d1_stride", candidate.values.get("shape.get_rows_moe.dst_token_stride", nselected))
    )
    src0_elems = src0_token_stride * ntokens
    idx_elems = idx_token_stride * ntokens
    dst_elems = dst_token_stride * ntokens
    src0 = f32_pattern(np, (src0_elems,), seed=seed)
    indices = np.full((idx_elems,), -1, dtype=np.int32)
    for token in range(ntokens):
        for slot in range(nselected):
            indices[token * idx_token_stride + slot] = np.int32((token * 17 + slot * 7 + seed) % nexperts)
    dst_init = f32_pattern(np, (dst_elems,), seed=seed + 1, scale=0.25)
    expected = dst_init.copy()
    for token in range(ntokens):
        for slot in range(nselected):
            expert = int(indices[token * idx_token_stride + slot])
            if 0 <= expert < nexperts:
                expected[token * dst_token_stride + slot] = src0[token * src0_token_stride + expert]
    return {
        "arrays": {
            "src0": src0,
            "indices": indices,
            "dst_init": dst_init,
            "expected": expected,
        },
        "metadata": {
            "nexperts": nexperts,
            "nselected": nselected,
            "ntokens": ntokens,
            "src0_token_stride": src0_token_stride,
            "idx_token_stride": idx_token_stride,
            "dst_token_stride": dst_token_stride,
            "bytes": {
                "src0": src0_elems * F32_BYTES,
                "src1": idx_elems * 4,
                "dst": dst_elems * F32_BYTES,
            },
        },
    }


def _set_rows_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    common_dims = _ranked_shape(candidate)
    if common_dims is not None:
        update_tensor, index_tensor = _set_rows_tensor_roles(candidate)
        src0_dims = _copy_tensor_dims(candidate, update_tensor, common_dims)
        src1_dims = _copy_tensor_dims(candidate, index_tensor, common_dims)
        dst_dims = _copy_tensor_dims(candidate, "dst", common_dims)
        src0_strides = _copy_tensor_strides(candidate, update_tensor, src0_dims)
        src1_strides = _copy_tensor_strides(candidate, index_tensor, src1_dims)
        dst_strides = _copy_tensor_strides(candidate, "dst", dst_dims)
        src0_buffer_len = _buffer_length(src0_dims, src0_strides)
        src1_buffer_len = _buffer_length(src1_dims, src1_strides)
        dst_buffer_len = _buffer_length(dst_dims, dst_strides)
        src = f32_pattern(np, (src0_buffer_len,), seed=seed)
        dst = f32_pattern(np, (dst_buffer_len,), seed=seed + 1, scale=0.25)
        indices = np.zeros((src1_buffer_len * 2,), dtype=np.int32)
        for i12 in range(src1_dims[2]):
            for i11 in range(src1_dims[1]):
                for i in range(src1_dims[0]):
                    logical = (
                        i * src1_strides[0]
                        + i11 * src1_strides[1]
                        + i12 * src1_strides[2]
                    )
                    indices[logical * 2] = np.int32((i + i11 + i12) % dst_dims[1])
        expected = dst.copy()
        for i3 in range(src0_dims[3]):
            for i2 in range(src0_dims[2]):
                i11 = i2 % src1_dims[1]
                i12 = i3 % src1_dims[2]
                for i in range(src0_dims[1]):
                    idx_element = i * src1_strides[0] + i11 * src1_strides[1] + i12 * src1_strides[2]
                    row = int(indices[idx_element * 2])
                    for i0 in range(src0_dims[0]):
                        src_index = i0 + i * src0_strides[1] + i2 * src0_strides[2] + i3 * src0_strides[3]
                        dst_index = i0 + row * dst_strides[1] + i2 * dst_strides[2] + i3 * dst_strides[3]
                        expected[dst_index] = src[src_index]
        if "f16" in candidate.root_symbol:
            dst_bits = dst.astype(np.float16).view(np.uint16).reshape(dst_buffer_len).view(np.int16)
            expected_bits = expected.astype(np.float16).view(np.uint16).reshape(dst_buffer_len).view(np.int16)
        else:
            dst_bits = dst.reshape(dst_buffer_len)
            expected_bits = expected.reshape(dst_buffer_len)
        return {
            "arrays": {
                "src0": src.reshape(src0_buffer_len),
                "indices": indices,
                "dst_init": dst_bits,
                "expected": expected_bits,
            },
            "metadata": {
                "dst_type": "i16" if "f16" in candidate.root_symbol else "f32",
                "idx_i32_count": int(indices.size),
            },
        }
    ncols, nrows, elems = _dims(candidate)
    dst = f32_pattern(np, (nrows, ncols), seed=seed, scale=0.25)
    src = f32_pattern(np, (nrows, ncols), seed=seed + 1)
    indices = np.zeros((nrows * 2,), dtype=np.int32)
    indices[0::2] = np.arange(nrows, dtype=np.int32)
    expected = dst.copy()
    expected[np.arange(nrows, dtype=np.int32)] = src
    if "f16" in candidate.root_symbol:
        dst_bits = dst.astype(np.float16).view(np.uint16).reshape(elems).view(np.int16)
        expected_bits = expected.astype(np.float16).view(np.uint16).reshape(elems).view(np.int16)
    else:
        dst_bits = dst.reshape(elems)
        expected_bits = expected.reshape(elems)
    return {
        "arrays": {
            "src0": src.reshape(elems),
            "indices": indices,
            "dst_init": dst_bits,
            "expected": expected_bits,
        },
        "metadata": {"dst_type": "i16" if "f16" in candidate.root_symbol else "f32", "idx_i32_count": int(indices.size)},
    }


def _matmul_f32_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    kernel_abi = _matmul_f32_kernel_abi_arrays(np, candidate, seed)
    if kernel_abi is not None:
        return kernel_abi
    k, rows, cols = _matmul_dims(candidate)
    lhs = f32_pattern(np, (rows, k), seed=seed)
    rhs = f32_pattern(np, (cols, k), seed=seed + 1)
    expected = np.matmul(lhs.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0": lhs.reshape(rows * k),
            "src0_logical_f32": lhs.reshape(rows * k),
            "src1": rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {"logical_packed_weight_fixture": False},
    }


def _matmul_f32_kernel_abi_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any] | None:
    if candidate.source_id != "mul_mat_f32_f32_generic":
        return None
    src0_dims = _captured_tensor_dims(candidate, "src0")
    src1_dims = _captured_tensor_dims(candidate, "src1")
    dst_dims = _captured_tensor_dims(candidate, "dst")
    if src0_dims is None or src1_dims is None or dst_dims is None:
        return None
    src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
    src1_strides = _copy_tensor_strides(candidate, "src1", src1_dims)
    dst_strides = _copy_tensor_strides(candidate, "dst", dst_dims)
    k, rows = src0_dims[:2]
    cols = dst_dims[1]
    src0_len = _buffer_length(src0_dims, src0_strides)
    src1_len = _buffer_length(src1_dims, src1_strides)
    dst_len = _buffer_length(dst_dims, dst_strides)
    src0 = np.zeros((src0_len,), dtype=np.float32)
    src1 = np.zeros((src1_len,), dtype=np.float32)
    dst_init = f32_pattern(np, (dst_len,), seed=seed + 2, scale=0.25)
    expected = dst_init.copy()
    lhs_by_slice = {}
    for i03 in range(src0_dims[3]):
        for i02 in range(src0_dims[2]):
            lhs = f32_pattern(np, (rows, k), seed=seed + i02 + src0_dims[2] * i03)
            lhs_by_slice[(i02, i03)] = lhs
            for row in range(rows):
                src0_row_base = row * src0_strides[1] + i02 * src0_strides[2] + i03 * src0_strides[3]
                src0[src0_row_base : src0_row_base + k] = lhs[row]
    src0_scale2 = dst_dims[2] // src0_dims[2]
    src0_scale3 = dst_dims[3] // src0_dims[3]
    for i3 in range(dst_dims[3]):
        for i2 in range(dst_dims[2]):
            src0_i2 = i2 // src0_scale2
            src0_i3 = i3 // src0_scale3
            lhs = lhs_by_slice[(src0_i2, src0_i3)]
            rhs = f32_pattern(np, (cols, k), seed=seed + 1 + i2 + dst_dims[2] * i3)
            dot = np.matmul(lhs.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
            for col in range(cols):
                src1_col_base = col * src1_strides[1] + i2 * src1_strides[2] + i3 * src1_strides[3]
                src1[src1_col_base : src1_col_base + k] = rhs[col]
                dst_col_base = col * dst_strides[1] + i2 * dst_strides[2] + i3 * dst_strides[3]
                expected[dst_col_base : dst_col_base + rows] = dot[col]
    return {
        "arrays": {
            "src0": src0,
            "src0_logical_f32": src0,
            "src1": src1,
            "dst_init": dst_init,
            "expected": expected,
        },
        "metadata": {"logical_packed_weight_fixture": False, "kernel_abi_fixture": True},
    }


def _matmul_f16_rhs_arrays(np: Any, candidate: Candidate, seed: int, *, src1_dtype: str) -> dict[str, Any]:
    src1_is_f16 = src1_dtype == "f16"
    src1_pattern = f16_pattern if src1_is_f16 else f32_pattern
    src1_storage_dtype = np.float16 if src1_is_f16 else np.float32

    batched = _batched_mul_mat_f16_f32_dims(candidate)
    if batched is not None:
        (src0_dims, src1_dims, dst_dims, src0_strides, src1_strides, dst_strides) = batched
        k, rows = src0_dims[:2]
        cols = dst_dims[1]
        src0_len = _buffer_length(src0_dims, src0_strides)
        src1_len = _buffer_length(src1_dims, src1_strides)
        dst_len = _buffer_length(dst_dims, dst_strides)
        src0 = np.zeros((src0_len,), dtype=np.float16)
        src1 = np.zeros((src1_len,), dtype=src1_storage_dtype)
        dst_init = f32_pattern(np, (dst_len,), seed=seed + 2, scale=0.25)
        expected = dst_init.copy()
        lhs_by_slice = {}
        for i03 in range(src0_dims[3]):
            for i02 in range(src0_dims[2]):
                lhs = f16_pattern(np, (rows, k), seed=seed + i02 + src0_dims[2] * i03)
                lhs_by_slice[(i02, i03)] = lhs
                for row in range(rows):
                    src0_row_base = row * src0_strides[1] + i02 * src0_strides[2] + i03 * src0_strides[3]
                    src0[src0_row_base : src0_row_base + k] = lhs[row]
        src0_scale2 = dst_dims[2] // src0_dims[2]
        src0_scale3 = dst_dims[3] // src0_dims[3]
        for i3 in range(dst_dims[3]):
            for i2 in range(dst_dims[2]):
                src0_i2 = i2 // src0_scale2
                src0_i3 = i3 // src0_scale3
                lhs = lhs_by_slice[(src0_i2, src0_i3)]
                rhs = src1_pattern(np, (cols, k), seed=seed + 1 + i2 + dst_dims[2] * i3)
                dot = np.matmul(lhs.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
                for col in range(cols):
                    src1_col_base = col * src1_strides[1] + i2 * src1_strides[2] + i3 * src1_strides[3]
                    src1[src1_col_base : src1_col_base + k] = rhs[col]
                    dst_col_base = col * dst_strides[1] + i2 * dst_strides[2] + i3 * dst_strides[3]
                    expected[dst_col_base : dst_col_base + rows] = dot[col]
        return {
            "arrays": {
                "src0": _f16_bits(np, src0),
                "src1": _f16_bits(np, src1) if src1_is_f16 else src1,
                "dst_init": dst_init,
                "expected": expected,
            },
            "metadata": {"logical_packed_weight_fixture": False},
        }
    k, rows, cols = _matmul_dims(candidate)
    lhs = f16_pattern(np, (rows, k), seed=seed)
    rhs = src1_pattern(np, (cols, k), seed=seed + 1)
    expected = np.matmul(lhs.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0": _f16_bits(np, lhs),
            "src1": _f16_bits(np, rhs.reshape(cols * k)) if src1_is_f16 else rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {"logical_packed_weight_fixture": False},
    }


def _matmul_f16_f32_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    return _matmul_f16_rhs_arrays(np, candidate, seed, src1_dtype="f32")


def _matmul_f16_f16_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    return _matmul_f16_rhs_arrays(np, candidate, seed, src1_dtype="f16")


def _write_mul_mat_f32_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    kernel_abi = candidate.source_id == "mul_mat_f32_f32_generic"
    src0_name = "src0" if kernel_abi else "src0_logical_f32"
    src0_dims = _captured_tensor_dims(candidate, "src0") if kernel_abi else None
    src1_dims = _captured_tensor_dims(candidate, "src1") if kernel_abi else None
    dst_dims = _captured_tensor_dims(candidate, "dst") if kernel_abi else None
    if kernel_abi and src0_dims is not None and src1_dims is not None and dst_dims is not None:
        src0_elems = _buffer_length(src0_dims, _copy_tensor_strides(candidate, "src0", src0_dims))
        src1_elems = _buffer_length(src1_dims, _copy_tensor_strides(candidate, "src1", src1_dims))
        dst_elems = _buffer_length(dst_dims, _copy_tensor_strides(candidate, "dst", dst_dims))
    else:
        k, rows, cols = _matmul_dims(candidate)
        src0_elems = rows * k
        src1_elems = cols * k
        dst_elems = rows * cols
    case_name, bench_name = _case_names(candidate)
    lines = [
        _read_f32(workbench_path, fixture_dir, src0_name, src0_elems).replace(f"%{src0_name}", "%src0"),
        _read_f32(workbench_path, fixture_dir, "src1", src1_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
        f"  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xf32>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)",
        f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{dst_elems}xf32>",
    ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_mul_mat_f16_rhs_batched_workbench(
    candidate: Candidate,
    linked_source: Path,
    workbench_path: Path,
    fixture_dir: Path,
    *,
    src1_dtype: str,
) -> tuple[str, dict[str, Any]]:
    batched = _batched_mul_mat_f16_f32_dims(candidate)
    if batched is not None:
        (src0_dims, src1_dims, dst_dims, src0_strides, src1_strides, dst_strides) = batched
        src0_elems = _buffer_length(src0_dims, src0_strides)
        src1_elems = _buffer_length(src1_dims, src1_strides)
        dst_elems = _buffer_length(dst_dims, dst_strides)
    else:
        k, rows, cols = _matmul_dims(candidate)
        src0_elems = rows * k
        src1_elems = cols * k
        dst_elems = rows * cols
    case_name, bench_name = _case_names(candidate)
    src1_reader = _read_f16 if src1_dtype == "f16" else _read_f32
    lines = [
        _read_f16(workbench_path, fixture_dir, "src0", src0_elems),
        src1_reader(workbench_path, fixture_dir, "src1", src1_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
        f"  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xf16>, tensor<{src1_elems}x{src1_dtype}>, tensor<{dst_elems}xf32>)",
        f"  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>",
    ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_mul_mat_f16_f32_generic_workbench(
    candidate: Candidate,
    linked_source: Path,
    workbench_path: Path,
    fixture_dir: Path,
) -> tuple[str, dict[str, Any]]:
    return _write_mul_mat_f16_rhs_batched_workbench(
        candidate,
        linked_source,
        workbench_path,
        fixture_dir,
        src1_dtype="f32",
    )


def _write_mul_mat_f16_f16_workbench(
    candidate: Candidate,
    linked_source: Path,
    workbench_path: Path,
    fixture_dir: Path,
) -> tuple[str, dict[str, Any]]:
    return _write_mul_mat_f16_rhs_batched_workbench(
        candidate,
        linked_source,
        workbench_path,
        fixture_dir,
        src1_dtype="f16",
    )


def _mul_mat_q8_0_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    if "q8_1" in candidate.root_symbol:
        data = _matmul_f32_arrays(np, candidate, seed)
        data["metadata"]["message"] = "q8_0 matmul with q8_1 RHS requires q8_1 RHS ABI packer"
        data["metadata"]["logical_packed_weight_fixture"] = True
        return data
    k, rows, cols = _matmul_dims(candidate)
    lhs_f32 = f32_pattern(np, (rows, k), seed=seed)
    src0 = quantize_q8_0(np, lhs_f32)
    rhs = f32_pattern(np, (cols, k), seed=seed + 1)
    weights = dequant_q8_0(np, src0, k, rows)
    expected = np.matmul(weights.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0": src0,
            "src1": rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {
            "q8_0_block_bytes": Q8_0_BLOCK_BYTES,
            "bytes": {"src0": q8_0_bytes(k, rows), "src1": k * cols * F32_BYTES, "dst": rows * cols * F32_BYTES},
        },
    }


def _mul_mat_q5_k_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    k, rows, cols = _matmul_dims(candidate)
    src0 = q5_k_pattern(np, k, rows, seed=seed)
    rhs = normalized_f32_rows(np, (cols, k), seed=seed + 1)
    weights = dequant_q5_k(np, src0, k, rows)
    expected = np.matmul(weights.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0": src0.view(np.int8),
            "src1": rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {
            "bytes": {"src0": q5_k_bytes(k, rows), "src1": k * cols * F32_BYTES, "dst": rows * cols * F32_BYTES},
        },
    }


def _mul_mat_q6_k_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    k, rows, cols = _matmul_dims(candidate)
    src0 = q6_k_pattern(np, k, rows, seed=seed)
    rhs = normalized_f32_rows(np, (cols, k), seed=seed + 1)
    weights = dequant_q6_k(np, src0.view(np.int8), k, rows)
    expected = np.matmul(weights.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0": src0.view(np.int8),
            "src1": rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {
            "bytes": {"src0": q6_k_bytes(k, rows), "src1": k * cols * F32_BYTES, "dst": rows * cols * F32_BYTES},
        },
    }


def _quantized_matmul_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    data = _matmul_f32_arrays(np, candidate, seed)
    data["metadata"]["logical_packed_weight_fixture"] = True
    data["metadata"]["message"] = "logical f32 oracle for packed quantized family; ABI-specific packed fixtures are pending"
    return data


def _quantize_q8_1_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    block_count = (ncols + 31) // 32
    if "x4" in candidate.root_symbol:
        outer_count = (block_count + 3) // 4
        return {
            "arrays": {
                "src0": src.reshape(elems),
                "expected": _pack_q8_1_x4_rows(np, src),
            },
            "metadata": {"q8_1_x4_block_bytes": 144, "block_count": block_count, "outer_count": outer_count},
        }
    padded = np.zeros((nrows, block_count * 32), dtype=np.float32)
    padded[:, :ncols] = src
    values = padded.reshape(nrows, block_count, 32)
    absmax = np.max(np.abs(values), axis=2).astype(np.float32)
    d = np.where(absmax != np.float32(0.0), absmax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    scaled = values / d[..., None]
    qs = np.where(scaled < np.float32(0.0), np.ceil(scaled - np.float32(0.5)), np.floor(scaled + np.float32(0.5)))
    qs = np.clip(qs, -128, 127).astype(np.int8)
    s = np.sum(qs.astype(np.float32) * d[..., None], axis=2, dtype=np.float32).astype(np.float32)
    expected = np.zeros((nrows, block_count, Q8_1_BLOCK_BYTES), dtype=np.uint8)
    expected[..., 0:2] = d.astype(np.float16).reshape(nrows, block_count, 1).view(np.uint8)
    expected[..., 2:4] = s.astype(np.float16).reshape(nrows, block_count, 1).view(np.uint8)
    expected[..., 4:36] = qs.view(np.uint8)
    return {
        "arrays": {
            "src0": src.reshape(elems),
            "expected": expected.reshape(nrows * block_count * Q8_1_BLOCK_BYTES).view(np.int8),
        },
        "metadata": {"q8_1_block_bytes": Q8_1_BLOCK_BYTES, "block_count": block_count},
    }


def _rms_norm_mul_quantize_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    base = _rms_norm_mul_arrays(np, candidate, seed)
    normalized = base["arrays"]["expected"].reshape(-1)
    ncols, nrows, _ = _dims(candidate)
    pseudo = Candidate(
        id=candidate.id,
        family=candidate.family,
        op=candidate.op,
        source_id=candidate.source_id,
        source_path=candidate.source_path,
        root_symbol=candidate.root_symbol,
        export_name=candidate.export_name,
        route_id=candidate.route_id,
        route=candidate.route,
        shape=candidate.shape,
        values=candidate.values,
        config=candidate.config,
        dispatch=candidate.dispatch,
        supports=candidate.supports,
        coverage=candidate.coverage,
        status=candidate.status,
        message=candidate.message,
    )
    quant = _quantize_q8_1_arrays(np, pseudo, seed + 7)
    quant["arrays"]["src0"] = normalized.astype(np.float32)
    quant["metadata"]["source_oracle"] = "rms_norm_mul_f32_numpy"
    quant["metadata"]["bytes"] = {"expected": q8_1_bytes(ncols, nrows)}
    return quant


def _rope_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    if candidate.family == "rope_set_rows_f32":
        src0_dims = _captured_tensor_dims(candidate, "src0")
        dst_dims = _captured_tensor_dims(candidate, "dst")
        idx_dims = _captured_tensor_dims(candidate, "src1")
        if src0_dims is None or dst_dims is None or idx_dims is None:
            raise ValueError("rope_set_rows_f32 oracle requires encoded src0, src1, and dst tensor dimensions")
        ncols = int(candidate.values.get("shape.rope.ncols") or src0_dims[0])
        n_dims = int(candidate.values.get("shape.rope.n_dims") or candidate.values.get("attribute.n_dims") or ncols)
        nheads = int(candidate.values.get("shape.rope.nheads") or src0_dims[1])
        ntokens = int(candidate.values.get("shape.rope.ntokens") or src0_dims[2])
        src_elems = src0_dims[0] * src0_dims[1] * src0_dims[2] * src0_dims[3]
        dst_elems = dst_dims[0] * dst_dims[1] * dst_dims[2] * dst_dims[3]
        idx_elems = idx_dims[0] * idx_dims[1] * idx_dims[2] * idx_dims[3]
        pos_token_stride = int(candidate.values.get("shape.rope.pos_token_stride") or 1)
        pos_elems = max(pos_token_stride * ntokens, 1)
        src = f32_pattern(np, (src_elems,), seed=seed)
        positions = np.zeros((pos_elems,), dtype=np.int32)
        for token in range(ntokens):
            positions[token * pos_token_stride] = token + 1
        half_dims = max(n_dims // 2, 1)
        freq = (np.arange(half_dims, dtype=np.float32) * np.float32(0.125) + np.float32(1.0)).astype(np.float32)
        indices = (np.arange(idx_elems, dtype=np.int64) + np.int64(1)) % np.int64(dst_dims[1])
        theta_scale = np.float32(_candidate_f32_value(candidate, "theta_scale", 0.75))
        freq_scale = np.float32(_candidate_f32_value(candidate, "freq_scale", 1.1))
        attn_factor = np.float32(_candidate_f32_value(candidate, "attn_factor", 0.9))
        src_view = src.reshape(src0_dims[3], src0_dims[2], src0_dims[1], src0_dims[0])
        dst_init = f16_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25).reshape(dst_dims[3], dst_dims[2], dst_dims[1], dst_dims[0])
        expected = dst_init.copy()
        half_cols = ncols // 2
        pairs = np.arange(half_cols, dtype=np.int32)
        idx0 = pairs * 2
        idx1 = idx0 + 1
        active = pairs < (n_dims // 2)
        if np.any(active):
            active_pairs = pairs[active]
            pos = positions[np.arange(ntokens, dtype=np.int64) * pos_token_stride].astype(np.float32)
            theta = (
                pos[:, None]
                * np.power(theta_scale, active_pairs.astype(np.float32)).astype(np.float32)[None, :]
            ).astype(np.float32)
            theta = (theta / freq[active_pairs][None, :]).astype(np.float32)
            theta = (theta * freq_scale).astype(np.float32)
            c = (np.cos(theta).astype(np.float32) * attn_factor).astype(np.float32)
            s = (np.sin(theta).astype(np.float32) * attn_factor).astype(np.float32)
        for token in range(ntokens):
            dst_row = int(indices[token])
            token_src = src_view[0, token]
            token_dst = expected[0, 0, dst_row].reshape(nheads, ncols)
            x0 = token_src[:, idx0].copy()
            x1 = token_src[:, idx1].copy()
            out0 = x0.copy()
            out1 = x1.copy()
            if np.any(active):
                out0[:, active] = (
                    x0[:, active] * c[token][None, :] - x1[:, active] * s[token][None, :]
                ).astype(np.float32)
                out1[:, active] = (
                    x0[:, active] * s[token][None, :] + x1[:, active] * c[token][None, :]
                ).astype(np.float32)
            token_dst[:, idx0] = out0.astype(np.float16)
            token_dst[:, idx1] = out1.astype(np.float16)
        return {
            "arrays": {
                "src0": src.astype(np.float32),
                "positions": positions,
                "freq": freq.astype(np.float32),
                "indices": indices.astype(np.int64),
                "dst_init": _f16_bits(np, dst_init),
                "expected": _f16_bits(np, expected),
                "dst_f32_init": np.zeros((dst_elems,), dtype=np.float32),
                "expected_f32": expected.astype(np.float32).reshape(-1),
            },
            "metadata": {
                "theta_scale": float(theta_scale),
                "freq_scale": float(freq_scale),
                "attn_factor": float(attn_factor),
                "src_elems": src_elems,
                "dst_elems": dst_elems,
                "idx_elems": idx_elems,
                "pos_elems": pos_elems,
                "freq_elems": int(freq.size),
            },
        }

    ncols = int(candidate.values.get("shape.rope.ncols") or candidate.shape.get("ncols", 1))
    n_dims = int(candidate.values.get("shape.rope.n_dims") or candidate.values.get("attribute.n_dims") or candidate.shape.get("n_dims", ncols))
    nheads = int(candidate.values.get("shape.rope.nheads") or candidate.shape.get("rows", 1))
    ntokens = int(candidate.values.get("shape.rope.ntokens") or candidate.shape.get("cols", 1))
    src_head_stride = int(candidate.values.get("shape.rope.src0_head_stride") or ncols)
    src_token_stride = int(candidate.values.get("shape.rope.src0_token_stride") or (ncols * nheads))
    dst_head_stride = int(candidate.values.get("shape.rope.dst_head_stride") or ncols)
    dst_token_stride = int(candidate.values.get("shape.rope.dst_token_stride") or (ncols * nheads))
    pos_token_stride = int(candidate.values.get("shape.rope.pos_token_stride") or 1)
    src_elems = src_token_stride * ntokens
    dst_elems = dst_token_stride * ntokens
    pos_elems = pos_token_stride * ntokens
    f16_output = candidate.family in {"rope_f16", "rope_neox_f16"}
    src = (
        f16_pattern(np, (src_elems,), seed=seed)
        if f16_output
        else f32_pattern(np, (src_elems,), seed=seed)
    )
    expected = (
        f16_pattern(np, (dst_elems,), seed=seed + 1, scale=0.25)
        if f16_output
        else f32_pattern(np, (dst_elems,), seed=seed + 1, scale=0.25)
    )
    positions = np.zeros((pos_elems,), dtype=np.int32)
    for token in range(ntokens):
        positions[token * pos_token_stride] = token + 1
    half_cols = ncols // 2
    half_dims = n_dims // 2
    theta_scale = np.float32(_candidate_f32_value(candidate, "theta_scale", 0.75))
    freq_scale = np.float32(_candidate_f32_value(candidate, "freq_scale", 1.1))
    attn_factor = np.float32(_candidate_f32_value(candidate, "attn_factor", 0.9))
    output_scale = np.float32(_candidate_f32_value(candidate, "output_scale", 0.5))
    has_freq = "freq" in candidate.root_symbol
    freq = (np.arange(max(half_dims, 1), dtype=np.float32) * np.float32(0.125) + np.float32(1.0)).astype(np.float32)
    neox = "neox" in candidate.root_symbol
    scale_output = "scale" in candidate.root_symbol
    if half_cols:
        itemsize = np.dtype(src.dtype).itemsize
        src_view = np.lib.stride_tricks.as_strided(
            src,
            shape=(ntokens, nheads, ncols),
            strides=(src_token_stride * itemsize, src_head_stride * itemsize, itemsize),
            writeable=False,
        )
        dst_view = np.lib.stride_tricks.as_strided(
            expected,
            shape=(ntokens, nheads, ncols),
            strides=(dst_token_stride * itemsize, dst_head_stride * itemsize, itemsize),
            writeable=True,
        )
        pairs = np.arange(half_cols, dtype=np.int32)
        active = pairs < half_dims
        if neox:
            idx0 = pairs.copy()
            idx1 = pairs + half_dims
            if np.any(~active):
                tail_pairs = pairs[~active] - half_dims
                idx0[~active] = n_dims + tail_pairs * 2
                idx1[~active] = idx0[~active] + 1
        else:
            idx0 = pairs * 2
            idx1 = idx0 + 1

        x0 = src_view[:, :, idx0]
        x1 = src_view[:, :, idx1]
        out0 = x0.copy()
        out1 = x1.copy()
        if np.any(active):
            active_pairs = pairs[active]
            pos = positions[np.arange(ntokens, dtype=np.int64) * pos_token_stride].astype(np.float32)
            theta = (
                pos[:, None]
                * np.power(theta_scale, active_pairs.astype(np.float32)).astype(np.float32)[None, :]
            ).astype(np.float32)
            if has_freq:
                theta = (theta / freq[active_pairs][None, :]).astype(np.float32)
            theta = (theta * freq_scale).astype(np.float32)
            c = (np.cos(theta).astype(np.float32) * attn_factor).astype(np.float32)
            s = (np.sin(theta).astype(np.float32) * attn_factor).astype(np.float32)
            xa0 = x0[:, :, active]
            xa1 = x1[:, :, active]
            rot0 = (xa0 * c[:, None, :] - xa1 * s[:, None, :]).astype(np.float32)
            rot1 = (xa0 * s[:, None, :] + xa1 * c[:, None, :]).astype(np.float32)
            if scale_output:
                rot0 = (rot0 * output_scale).astype(np.float32)
                rot1 = (rot1 * output_scale).astype(np.float32)
            out0[:, :, active] = rot0
            out1[:, :, active] = rot1
        if scale_output and np.any(~active):
            out0[:, :, ~active] = (out0[:, :, ~active] * output_scale).astype(np.float32)
            out1[:, :, ~active] = (out1[:, :, ~active] * output_scale).astype(np.float32)
        dst_view[:, :, idx0] = out0.astype(np.float32)
        dst_view[:, :, idx1] = out1.astype(np.float32)
    arrays: dict[str, Any] = {
        "src0": _f16_bits(np, src) if f16_output else src.astype(np.float32),
        "positions": positions,
        "dst_init": (
            _f16_bits(np, f16_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25))
            if f16_output
            else f32_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25)
        ),
        "expected": _f16_bits(np, expected) if f16_output else expected.astype(np.float32),
    }
    if has_freq:
        arrays["freq"] = freq.astype(np.float32)
    return {
        "arrays": arrays,
        "metadata": {
            "theta_scale": float(theta_scale),
            "freq_scale": float(freq_scale),
            "attn_factor": float(attn_factor),
            "output_scale": float(output_scale),
            "has_freq": has_freq,
            "neox": neox,
            "src_elems": src_elems,
            "dst_elems": dst_elems,
            "pos_elems": pos_elems,
            "freq_elems": int(freq.size) if has_freq else 0,
        },
    }


def _softmax_kqv_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    scale = np.float32(_candidate_f32_value(candidate, "scale", 0.75))
    k, rows, cols = _matmul_dims(candidate)
    nheads_kv = int(candidate.shape.get("nheads_kv", 8))
    if nheads_kv <= 0:
        raise ValueError(f"softmax_kqv requires nheads_kv > 0, got {nheads_kv}")
    if cols % nheads_kv != 0:
        raise ValueError(
            f"softmax_kqv requires cols divisible by nheads_kv, got cols={cols} nheads_kv={nheads_kv}"
        )
    heads_per_kv = cols // nheads_kv
    kq = f32_pattern(np, (cols, k), seed=seed)
    mask = f32_pattern(np, (k,), seed=seed + 1, scale=0.125)
    shifted = (kq * scale) + mask[None, :]
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    weights = np.exp(shifted, dtype=np.float32)
    weights = (weights / np.sum(weights, axis=1, keepdims=True)).astype(np.float32)
    values = f16_pattern(np, (nheads_kv, rows, k), seed=seed + 2)
    expected = np.empty((cols, rows), dtype=np.float32)
    for head in range(cols):
        kv_head = head // heads_per_kv
        expected[head] = np.matmul(values[kv_head].astype(np.float32), weights[head]).astype(np.float32)
    return {
        "arrays": {
            "kq": kq.reshape(cols * k),
            "mask": mask,
            "v": values.reshape(nheads_kv * rows * k),
            "dst_init": f32_pattern(np, (cols * rows,), seed=seed + 3, scale=0.25),
            "expected": expected.reshape(cols * rows),
        },
        "metadata": {
            "scale": float(scale),
            "nheads_kv": nheads_kv,
            "heads_per_kv": heads_per_kv,
        },
    }


def _flash_attn_ext_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    src0_dims = _captured_tensor_dims(candidate, "src0")
    src1_dims = _captured_tensor_dims(candidate, "src1")
    src2_dims = _captured_tensor_dims(candidate, "src2")
    src3_dims = _captured_tensor_dims(candidate, "src3")
    dst_dims = _captured_tensor_dims(candidate, "dst")
    if src0_dims is None or src1_dims is None or src2_dims is None or src3_dims is None or dst_dims is None:
        raise ValueError("flash_attn_ext requires captured rank-4 src0/src1/src2/src3/dst dimensions")
    qk_dim = int(candidate.config.get("@shape.flash_attn_ext.qk_dim", src0_dims[0]))
    value_dim = int(candidate.config.get("@shape.flash_attn_ext.value_dim", dst_dims[0]))
    tokens = int(candidate.config.get("@shape.flash_attn_ext.tokens", dst_dims[2]))
    heads = int(candidate.config.get("@shape.flash_attn_ext.heads", dst_dims[1]))
    batch = int(candidate.config.get("@shape.flash_attn_ext.batch", dst_dims[3]))
    kv_len = src1_dims[1]
    kv_heads = src1_dims[2]
    if qk_dim <= 0 or value_dim <= 0 or heads <= 0 or tokens <= 0 or batch <= 0 or kv_len <= 0 or kv_heads <= 0:
        raise ValueError(f"flash_attn_ext requires positive dimensions, got dst={dst_dims} src1={src1_dims}")
    if heads % kv_heads != 0:
        raise ValueError(f"flash_attn_ext requires heads divisible by kv_heads, got heads={heads} kv_heads={kv_heads}")
    if src0_dims != (qk_dim, tokens, heads, batch):
        raise ValueError(
            f"flash_attn_ext src0 dimensions {src0_dims} do not match qk/tokens/heads/batch "
            f"{(qk_dim, tokens, heads, batch)}"
        )
    if src1_dims != (qk_dim, kv_len, kv_heads, batch):
        raise ValueError(
            f"flash_attn_ext key dimensions {src1_dims} do not match qk/kv_len/kv_heads/batch "
            f"{(qk_dim, kv_len, kv_heads, batch)}"
        )
    if src2_dims != (value_dim, kv_len, kv_heads, batch):
        raise ValueError(f"flash_attn_ext requires matching key/value dimensions, got src1={src1_dims} src2={src2_dims}")
    if src3_dims != (kv_len, tokens, 1, batch):
        raise ValueError(f"flash_attn_ext mask dimensions {src3_dims} do not match kv_len/tokens")
    if dst_dims == (value_dim, tokens, heads, batch):
        dst_layout = "dim_token_head_batch"
    elif dst_dims == (value_dim, heads, tokens, batch):
        dst_layout = "dim_head_token_batch"
    else:
        raise ValueError(
            f"flash_attn_ext dst dimensions {dst_dims} do not match supported layouts "
            f"{(value_dim, tokens, heads, batch)} or {(value_dim, heads, tokens, batch)}"
        )

    src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
    src1_strides = _copy_tensor_strides(candidate, "src1", src1_dims)
    src2_strides = _copy_tensor_strides(candidate, "src2", src2_dims)
    src3_strides = _copy_tensor_strides(candidate, "src3", src3_dims)
    dst_strides = _copy_tensor_strides(candidate, "dst", dst_dims)
    src0_buffer = f32_pattern(np, (_buffer_length(src0_dims, src0_strides),), seed=seed, scale=0.125)
    src1_buffer = f16_pattern(np, (_buffer_length(src1_dims, src1_strides),), seed=seed + 1, scale=0.125)
    src2_buffer = f16_pattern(np, (_buffer_length(src2_dims, src2_strides),), seed=seed + 2, scale=0.5)
    has_synthetic_mask = "src3" in dict(candidate.route.get("synthetic_tensors", {}))
    if has_synthetic_mask:
        src3_buffer = np.zeros((_buffer_length(src3_dims, src3_strides),), dtype=np.float16)
    else:
        src3_buffer = f16_pattern(np, (_buffer_length(src3_dims, src3_strides),), seed=seed + 3, scale=0.05)
    dst_init = f32_pattern(np, (_buffer_length(dst_dims, dst_strides),), seed=seed + 4, scale=0.25)
    expected = dst_init.copy()

    q = _strided_logical_view(np, src0_buffer, src0_dims, src0_strides)
    k = _strided_logical_view(np, src1_buffer, src1_dims, src1_strides).astype(np.float32)
    v = _strided_logical_view(np, src2_buffer, src2_dims, src2_strides).astype(np.float32)
    mask = _strided_logical_view(np, src3_buffer, src3_dims, src3_strides).astype(np.float32)
    out = _strided_logical_view(np, expected, dst_dims, dst_strides)
    scale = np.float32(_candidate_f32_value(candidate, "scale", 1.0 / math.sqrt(float(qk_dim))))
    heads_per_kv = heads // kv_heads
    for batch_id in range(batch):
        for token in range(tokens):
            mask_for_token = mask[:, token, 0, batch_id]
            for head in range(heads):
                kv_head = head // heads_per_kv
                scores = np.matmul(q[:, token, head, batch_id].astype(np.float32), k[:, :, kv_head, batch_id])
                scores = (scores * scale + mask_for_token).astype(np.float32)
                shifted = scores - np.max(scores)
                weights = np.exp(shifted, dtype=np.float32)
                weights = (weights / np.sum(weights, dtype=np.float32)).astype(np.float32)
                result = np.matmul(v[:, :, kv_head, batch_id], weights).astype(np.float32)
                if dst_layout == "dim_token_head_batch":
                    out[:, token, head, batch_id] = result
                else:
                    out[:, head, token, batch_id] = result

    return {
        "arrays": {
            "src0": src0_buffer.reshape(-1).astype(np.float32),
            "src1": _f16_bits(np, src1_buffer),
            "src2": _f16_bits(np, src2_buffer),
            "src3": _f16_bits(np, src3_buffer),
            "dst_init": dst_init.reshape(-1).astype(np.float32),
            "expected": expected.reshape(-1).astype(np.float32),
        },
        "metadata": {
            "qk_dim": qk_dim,
            "value_dim": value_dim,
            "heads": heads,
            "tokens": tokens,
            "batch": batch,
            "kv_len": kv_len,
            "kv_heads": kv_heads,
            "heads_per_kv": heads_per_kv,
            "dst_layout": dst_layout,
            "scale": float(scale),
            "has_mask": not has_synthetic_mask,
        },
    }


def _split_k_reduce2_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    rows = int(candidate.shape.get("rows", candidate.shape.get("nrows", 1)))
    plane0 = f32_pattern(np, (rows,), seed=seed)
    plane1 = f32_pattern(np, (rows,), seed=seed + 1)
    src = np.concatenate([plane0, plane1]).astype(np.float32)
    return {
        "arrays": {
            "src0": src,
            "dst_init": f32_pattern(np, (rows,), seed=seed + 2, scale=0.25),
            "expected": (plane0 + plane1).astype(np.float32),
        },
        "metadata": {"layout": "two_plane_f32_reduce2", "rows": rows},
    }


def _logical_generate(spec: LogicalOracleSpec) -> Callable[[Any, Candidate, Path, int], OracleResult]:
    return lambda np, candidate, fixture_dir, seed: _logical_oracle(spec, np, candidate, fixture_dir, seed)


LOGICAL_ORACLE_SPECS: tuple[LogicalOracleSpec, ...] = (
    LogicalOracleSpec(("add_f32",), "add_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _binary_arrays(lambda lhs, rhs: lhs + rhs)),
    LogicalOracleSpec(("mul_f32",), "mul_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _binary_arrays(lambda lhs, rhs: lhs * rhs)),
    LogicalOracleSpec(("div_f32",), "div_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _div_arrays),
    LogicalOracleSpec(("sub_f32",), "sub_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _binary_arrays(lambda lhs, rhs: lhs - rhs)),
    LogicalOracleSpec(("scale_f32",), "scale_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _scale_arrays),
    LogicalOracleSpec(("clamp_f32",), "clamp_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _clamp_arrays),
    LogicalOracleSpec(("argsort_f32_i32",), "argsort_f32_i32_numpy_desc", {"atol": 0.0, "rtol": 0.0}, _argsort_arrays),
    LogicalOracleSpec(("get_rows_f32", "get_rows_q4_k_f32", "get_rows_q5_k_f32", "get_rows_q6_k_f32", "get_rows_q8_0_f32"), "get_rows_numpy_logical", {"atol": 1e-5, "rtol": 1e-5}, _get_rows_arrays),
    LogicalOracleSpec(("get_rows_moe_weights_f32",), "get_rows_moe_weights_numpy_logical", {"atol": 1e-5, "rtol": 1e-5}, _get_rows_arrays),
    LogicalOracleSpec(("mul_mat_f16_f32_generic",), "mul_mat_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _matmul_f32_arrays),
    LogicalOracleSpec(("mul_mat_q5_k_f32", "mul_mat_q6_k_f32", "mul_mat_q8_0_f32", "mul_mat_q4_k_swiglu_f32"), "quantized_mul_mat_numpy_logical", {"atol": 0.12, "rtol": 0.04}, _quantized_matmul_arrays),
    LogicalOracleSpec(("quantize_q8_1_f32",), "quantize_q8_1_numpy", {"atol": 0.0, "rtol": 0.0}, _quantize_q8_1_arrays),
    LogicalOracleSpec(("rope_f32", "rope_neox_f32", "rope_f16", "rope_neox_f16", "rope_scale_f32", "rope_set_rows_f32"), "rope_numpy_structural_placeholder", {"atol": 1e-5, "rtol": 1e-5}, _rope_arrays),
    LogicalOracleSpec(("set_rows_f32", "cont_set_rows_f32"), "set_rows_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _set_rows_arrays),
    LogicalOracleSpec(("soft_max_f32",), "soft_max_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _softmax_arrays),
    LogicalOracleSpec(("softmax_kqv_f32_f16",), "softmax_kqv_f32_f16_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _softmax_kqv_arrays),
    LogicalOracleSpec(("flash_attn_ext_f32_f16",), "flash_attn_ext_f32_f16_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _flash_attn_ext_arrays),
    LogicalOracleSpec(("sum_rows_f32",), "sum_rows_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _sum_rows_arrays),
    LogicalOracleSpec(("swiglu_f32",), "swiglu_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _swiglu_arrays),
    LogicalOracleSpec(("swiglu_f16",), "swiglu_f16_numpy", {"atol": 1e-3, "rtol": 1e-3}, _swiglu_arrays),
)


def _metadata(candidate: Candidate, seed: int, oracle: str, tolerance: dict[str, float]) -> dict[str, Any]:
    return {
        "schema": "ggml_hrx_kernel_bench.oracle.v1",
        "candidate_id": candidate.id,
        "family": candidate.family,
        "op": candidate.op,
        "route_id": candidate.route_id,
        "root_symbol": candidate.root_symbol,
        "shape": candidate.shape,
        "values": candidate.values,
        "seed": seed,
        "oracle": oracle,
        "tolerance": tolerance,
    }


def write_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str | None, dict[str, Any]]:
    spec = ORACLE_SPECS_BY_FAMILY.get(candidate.family)
    if spec is None:
        return None, {"status": "unsupported_golden", "message": f"no generated check.case for family {candidate.family}"}
    return spec.write_workbench(candidate, linked_source, workbench_path, fixture_dir)


def _source_plus_case(linked_source: Path, workbench_path: Path, suffix: str) -> None:
    text = linked_source.read_text(encoding="utf-8")
    workbench_path.write_text(text.rstrip() + "\n\n" + suffix.lstrip(), encoding="utf-8")


def _rel_fixture(workbench_path: Path, fixture_dir: Path, name: str) -> str:
    return str((fixture_dir / name).relative_to(workbench_path.parent))


def _write_mul_mat_q4_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    if "split_k_reduce2" in candidate.root_symbol:
        return _logical_workbench(candidate, linked_source, workbench_path, fixture_dir)
    k = int(candidate.shape.get("k", 256))
    rows = int(candidate.shape.get("rows", 1))
    cols = int(candidate.shape.get("cols", 1))
    src0_elems = q4_k_bytes(k, rows)
    src1_elems = k * cols
    dst_elems = rows * cols
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{src0_elems}xi8>
  %src1 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src1.npy")}") : tensor<{src1_elems}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{dst_elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{dst_elems}xf32>
  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_mul_mat_id_q4_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    k = int(candidate.shape.get("k", candidate.shape.get("src0_d0", 256)))
    rows = int(candidate.shape.get("rows", candidate.shape.get("d0", 1)))
    nexperts = int(candidate.shape.get("nexperts", candidate.shape.get("src0_d2", 1)))
    nselected = int(candidate.shape.get("nselected", candidate.shape.get("d1", 1)))
    ntokens = int(candidate.shape.get("ntokens", candidate.shape.get("d2", 1)))
    src1_token_stride = int(candidate.shape.get("src1_token_stride", candidate.shape.get("src1_d2_stride", k * nselected)))
    idx_token_stride = int(candidate.shape.get("idx_token_stride", candidate.shape.get("src2_d1_stride", nselected)))
    dst_token_stride = int(candidate.shape.get("dst_token_stride", candidate.shape.get("dst_d2_stride", rows * nselected)))
    src0_elems = q4_k_bytes(k, nexperts * rows)
    src1_elems = ntokens * src1_token_stride
    idx_elems = ntokens * idx_token_stride
    dst_elems = ntokens * dst_token_stride
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{src0_elems}xi8>
  %src1 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src1.npy")}") : tensor<{src1_elems}xf32>
  %idx = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "idx.npy")}") : tensor<{idx_elems}xi32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{dst_elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{dst_elems}xf32>
  func.call {candidate.root_symbol}(%src0, %src1, %idx, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{idx_elems}xi32>, tensor<{dst_elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_mul_mat_id_q5_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    k = int(candidate.shape.get("k", candidate.shape.get("src0_d0", 256)))
    rows = int(candidate.shape.get("rows", candidate.shape.get("d0", 1)))
    nexperts = int(candidate.shape.get("nexperts", candidate.shape.get("src0_d2", 1)))
    nselected = int(candidate.shape.get("nselected", candidate.shape.get("d1", 1)))
    ntokens = int(candidate.shape.get("ntokens", candidate.shape.get("d2", 1)))
    src1_token_stride = int(candidate.shape.get("src1_token_stride", candidate.shape.get("src1_d2_stride", k * nselected)))
    idx_token_stride = int(candidate.shape.get("idx_token_stride", candidate.shape.get("src2_d1_stride", nselected)))
    dst_token_stride = int(candidate.shape.get("dst_token_stride", candidate.shape.get("dst_d2_stride", rows * nselected)))
    src0_elems = q5_k_bytes(k, nexperts * rows)
    src1_elems = ntokens * src1_token_stride
    idx_elems = ntokens * idx_token_stride
    dst_elems = ntokens * dst_token_stride
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{src0_elems}xi8>
  %src1 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src1.npy")}") : tensor<{src1_elems}xf32>
  %idx = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "idx.npy")}") : tensor<{idx_elems}xi32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{dst_elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{dst_elems}xf32>
  func.call {candidate.root_symbol}(%src0, %src1, %idx, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{idx_elems}xi32>, tensor<{dst_elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.12) rtol(0.04) nan(same) : tensor<{dst_elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_mul_mat_id_q6_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    k = int(candidate.shape.get("k", candidate.shape.get("src0_d0", 256)))
    rows = int(candidate.shape.get("rows", candidate.shape.get("d0", 1)))
    nexperts = int(candidate.shape.get("nexperts", candidate.shape.get("src0_d2", 1)))
    nselected = int(candidate.shape.get("nselected", candidate.shape.get("d1", 1)))
    ntokens = int(candidate.shape.get("ntokens", candidate.shape.get("d2", 1)))
    src1_token_stride = int(candidate.shape.get("src1_token_stride", candidate.shape.get("src1_d2_stride", k * nselected)))
    idx_token_stride = int(candidate.shape.get("idx_token_stride", candidate.shape.get("src2_d1_stride", nselected)))
    dst_token_stride = int(candidate.shape.get("dst_token_stride", candidate.shape.get("dst_d2_stride", rows * nselected)))
    src0_elems = q6_k_bytes(k, nexperts * rows)
    src1_elems = ntokens * src1_token_stride
    idx_elems = ntokens * idx_token_stride
    dst_elems = ntokens * dst_token_stride
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{src0_elems}xi8>
  %src1 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src1.npy")}") : tensor<{src1_elems}xf32>
  %idx = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "idx.npy")}") : tensor<{idx_elems}xi32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{dst_elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{dst_elems}xf32>
  func.call {candidate.root_symbol}(%src0, %src1, %idx, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{idx_elems}xi32>, tensor<{dst_elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.12) rtol(0.04) nan(same) : tensor<{dst_elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_rms_norm_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, _ = _dims(candidate)
    elems = ncols * nrows
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %eps = check.literal value(0.0) : f32
  %src = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src.npy")}") : tensor<{elems}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{elems}xf32>
  func.call {candidate.root_symbol}(%eps, %src, %dst) : (f32, tensor<{elems}xf32>, tensor<{elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_rms_norm_mul_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, _, elems = _dims(candidate)
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %eps = check.literal value(0.0) : f32
  %src = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src.npy")}") : tensor<{elems}xf32>
  %weight = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "weight.npy")}") : tensor<{ncols}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{elems}xf32>
  func.call {candidate.root_symbol}(%eps, %src, %weight, %dst) : (f32, tensor<{elems}xf32>, tensor<{ncols}xf32>, tensor<{elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_add_rms_norm_mul_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, _, elems = _dims(candidate)
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %eps = check.literal value(0.0) : f32
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{elems}xf32>
  %src1 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src1.npy")}") : tensor<{elems}xf32>
  %add_dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "add_dst_init.npy")}") : tensor<{elems}xf32>
  %weight = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "weight.npy")}") : tensor<{ncols}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{elems}xf32>
  %added = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "added.npy")}") : tensor<{elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{elems}xf32>
  func.call {candidate.root_symbol}(%eps, %src0, %src1, %add_dst, %weight, %dst) : (f32, tensor<{elems}xf32>, tensor<{elems}xf32>, tensor<{elems}xf32>, tensor<{ncols}xf32>, tensor<{elems}xf32>)
  check.expect.close actual(%add_dst) expected(%added) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>
  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_copy_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    src_dtype, dst_dtype = _copy_family_dtypes(candidate.family)
    common_dims = _ranked_shape(candidate)
    if common_dims is None:
        src0_elems = int(candidate.values.get("shape.copy.n") or _element_count(candidate))
        dst_elems = src0_elems
    else:
        src0_dims = _copy_tensor_dims(candidate, "src0", common_dims)
        dst_dims = _copy_tensor_dims(candidate, "dst", common_dims)
        src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
        dst_strides = _copy_tensor_strides(candidate, "dst", dst_dims)
        src0_elems = _buffer_length(src0_dims, src0_strides)
        dst_elems = _buffer_length(dst_dims, dst_strides)
    case_name, bench_name = _case_names(candidate)
    src_tensor_type = _copy_tensor_type(src_dtype, src0_elems)
    dst_tensor_type = _copy_tensor_type(dst_dtype, dst_elems)
    read_src = _read_copy_tensor(workbench_path, fixture_dir, "src0", src_dtype, src0_elems)
    read_dst = _read_copy_tensor(workbench_path, fixture_dir, "dst_init", dst_dtype, dst_elems)
    read_expected = _read_copy_tensor(workbench_path, fixture_dir, "expected", dst_dtype, dst_elems)
    read_dst = read_dst.replace("%dst_init", "%dst")
    if dst_dtype in {"bf16", "f16", "i32"}:
        check = f"  check.expect.equal actual(%dst) expected(%expected) : {dst_tensor_type}"
    else:
        check = f"  check.expect.close actual(%dst) expected(%expected) atol(0.0) rtol(0.0) nan(same) : {dst_tensor_type}"
    return _emit_case(
        linked_source,
        workbench_path,
        case_name,
        bench_name,
        "\n".join(
            [
                read_src,
                read_dst,
                read_expected,
                f"  func.call {candidate.root_symbol}(%src0, %dst) : ({src_tensor_type}, {dst_tensor_type})",
                check,
            ]
        ),
    )


def _write_cont_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    elems = _element_count(candidate)
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{elems}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{elems}xf32>
  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{elems}xf32>, tensor<{elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0) rtol(0.0) nan(same) : tensor<{elems}xf32>
  check.return
}}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_unary_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    _, _, elems = _dims(candidate)
    src0_elems, dst_elems = elems, elems
    common_dims = _pointwise_common_dims(candidate)
    if common_dims is not None:
        src0_dims = _copy_tensor_dims(candidate, "src0", common_dims)
        src0_strides = _copy_tensor_strides(candidate, "src0", src0_dims)
        if tuple(src0_strides) != _contiguous_strides(src0_dims):
            # Non-contiguous src0 (ggml v=1): read the padded src0 buffer, write a contiguous dst.
            src0_elems = _buffer_length(src0_dims, src0_strides)
            dst_elems = _product(common_dims)
    case_name, bench_name = _case_names(candidate)
    if candidate.family.endswith("_f16"):
        body = "\n".join(
            [
                _read_i16(workbench_path, fixture_dir, "src0", src0_elems),
                _read_i16(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
                _read_i16(workbench_path, fixture_dir, "expected", dst_elems),
                f"  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{src0_elems}xi16>, tensor<{dst_elems}xi16>)",
                f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{dst_elems}xi16>",
            ]
        )
        return _emit_case(linked_source, workbench_path, case_name, bench_name, body)

    atol, rtol = {
        "exp_f32": ("1e-5", "1e-5"),
        "sqrt_f32": ("1e-6", "1e-6"),
    }.get(candidate.family, ("0.0", "0.0"))
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", src0_elems),
            _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
            f"  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{src0_elems}xf32>, tensor<{dst_elems}xf32>)",
            f"  check.expect.close actual(%dst) expected(%expected) atol({atol}) rtol({rtol}) nan(same) : tensor<{dst_elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_pointwise_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    src0_elems, src1_elems, dst_elems = _pointwise_buffer_lengths(candidate)
    case_name, bench_name = _case_names(candidate)
    if candidate.family == "clamp_f16":
        lines = [
            "  %min = check.literal value(-0.45) : f32",
            "  %max = check.literal value(0.55) : f32",
            _read_i16(workbench_path, fixture_dir, "src0", src0_elems),
            _read_i16(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
            _read_i16(workbench_path, fixture_dir, "expected", dst_elems),
            f"  func.call {candidate.root_symbol}(%min, %max, %src0, %dst) : (f32, f32, tensor<{src0_elems}xi16>, tensor<{dst_elems}xi16>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{dst_elems}xi16>",
        ]
        return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))
    if candidate.family in {"add_f16", "mul_f16", "div_f16", "sub_f16"}:
        lines = [
            _read_i16(workbench_path, fixture_dir, "src0", src0_elems),
            _read_i16(workbench_path, fixture_dir, "src1", src1_elems),
            _read_i16(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
            _read_i16(workbench_path, fixture_dir, "expected", dst_elems),
            f"  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xi16>, tensor<{src1_elems}xi16>, tensor<{dst_elems}xi16>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{dst_elems}xi16>",
        ]
        return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))

    lines = [
        _read_f32(workbench_path, fixture_dir, "src0", src0_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
    ]
    if candidate.family in {"add_f32", "mul_f32", "div_f32", "sub_f32"}:
        lines.insert(1, _read_f32(workbench_path, fixture_dir, "src1", src1_elems))
        call_args = "%src0, %src1, %dst"
        call_types = f"tensor<{src0_elems}xf32>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>"
    elif candidate.family == "scale_f32":
        lines.insert(0, "  %scale = check.literal value(0.625) : f32")
        lines.insert(1, "  %bias = check.literal value(-0.125) : f32")
        call_args = "%scale, %bias, %src0, %dst"
        call_types = f"f32, f32, tensor<{src0_elems}xf32>, tensor<{dst_elems}xf32>"
    elif candidate.family == "clamp_f32":
        lines.insert(0, "  %min = check.literal value(-0.45) : f32")
        lines.insert(1, "  %max = check.literal value(0.55) : f32")
        call_args = "%min, %max, %src0, %dst"
        call_types = f"f32, f32, tensor<{src0_elems}xf32>, tensor<{dst_elems}xf32>"
    else:
        return _logical_workbench(candidate, linked_source, workbench_path, fixture_dir)
    lines.extend(
        [
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{dst_elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_swiglu_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    f16_family = candidate.family == "swiglu_f16"
    scalar_type = "i16" if f16_family else "f32"
    reader = _read_i16 if f16_family else _read_f32
    if str(candidate.route_id) in {"swiglu_f32_packed_contiguous_4d", "swiglu_f16_packed_contiguous_4d"}:
        src0_elems = elems * 2
    else:
        src0_dims = _swiglu_tensor_dims(candidate, "src0", ncols=ncols)
        src0_strides = _swiglu_tensor_strides(candidate, "src0", src0_dims)
        src0_elems = _strided_storage_elements(src0_dims, src0_strides)
    case_name, bench_name = _case_names(candidate)
    lines = [
        reader(workbench_path, fixture_dir, "src0", src0_elems),
    ]
    if candidate.root_symbol.endswith("_split"):
        src1_dims = _swiglu_tensor_dims(candidate, "src1", ncols=ncols)
        src1_strides = _swiglu_tensor_strides(candidate, "src1", src1_dims)
        src1_elems = _strided_storage_elements(src1_dims, src1_strides)
        lines.append(reader(workbench_path, fixture_dir, "src1", src1_elems))
        call_args = "%src0, %src1, %dst"
        call_types = f"tensor<{src0_elems}x{scalar_type}>, tensor<{src1_elems}x{scalar_type}>, tensor<{elems}x{scalar_type}>"
    else:
        call_args = "%src0, %dst"
        call_types = f"tensor<{src0_elems}x{scalar_type}>, tensor<{elems}x{scalar_type}>"
    lines.extend(
        [
            reader(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            reader(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
        ]
    )
    if f16_family:
        lines.append(f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{elems}x{scalar_type}>")
    else:
        lines.append(f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}x{scalar_type}>")
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_sum_rows_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", elems),
            f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'expected.npy')}\") : tensor<{nrows}xf32>",
            _read_f32(workbench_path, fixture_dir, "expected", nrows),
            f"  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{elems}xf32>, tensor<{nrows}xf32>)",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{nrows}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_argsort_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", elems),
            _read_i32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_i32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{elems}xf32>, tensor<{elems}xi32>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{elems}xi32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_get_rows_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    if candidate.family == "get_rows_moe_weights_f32":
        return _write_get_rows_moe_weights_workbench(candidate, linked_source, workbench_path, fixture_dir)
    if candidate.family not in {"get_rows_f32", "get_rows_q4_k_f32", "get_rows_q5_k_f32", "get_rows_q6_k_f32", "get_rows_q8_0_f32"}:
        return _logical_workbench(candidate, linked_source, workbench_path, fixture_dir)
    ncols, nrows, elems = _dims(candidate)
    src0_nrows = int(candidate.values.get("shape.get_rows.src0_nrows") or candidate.values.get("get_rows.src0_nrows") or nrows)
    if candidate.family == "get_rows_q4_k_f32":
        src0_elems = q4_k_bytes(ncols, src0_nrows)
        src0_read = f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'src0.npy')}\") : tensor<{src0_elems}xi8>"
        src0_type = f"tensor<{src0_elems}xi8>"
    elif candidate.family == "get_rows_q5_k_f32":
        src0_elems = q5_k_bytes(ncols, src0_nrows)
        src0_read = f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'src0.npy')}\") : tensor<{src0_elems}xi8>"
        src0_type = f"tensor<{src0_elems}xi8>"
    elif candidate.family == "get_rows_q6_k_f32":
        src0_elems = q6_k_bytes(ncols, src0_nrows)
        src0_read = f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'src0.npy')}\") : tensor<{src0_elems}xi8>"
        src0_type = f"tensor<{src0_elems}xi8>"
    elif candidate.family == "get_rows_q8_0_f32":
        src0_elems = q8_0_bytes(ncols, src0_nrows)
        src0_read = f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'src0.npy')}\") : tensor<{src0_elems}xi8>"
        src0_type = f"tensor<{src0_elems}xi8>"
    else:
        src0_elems = src0_nrows * ncols
        src0_read = _read_f32(workbench_path, fixture_dir, "src0", src0_elems)
        src0_type = f"tensor<{src0_elems}xf32>"
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            src0_read,
            _read_i32(workbench_path, fixture_dir, "indices", nrows).replace("%indices", "%idx"),
            _read_f32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}(%src0, %idx, %dst) : ({src0_type}, tensor<{nrows}xi32>, tensor<{elems}xf32>)",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_get_rows_moe_weights_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    nselected = int(candidate.shape.get("d0", 8))
    ntokens = int(candidate.shape.get("d1", 1))
    nexperts = int(candidate.shape.get("src0_d0", 128))
    src0_token_stride = int(candidate.shape.get("src0_d1_stride", nexperts))
    idx_token_stride = int(candidate.shape.get("src1_d1_stride", nselected))
    dst_token_stride = int(candidate.shape.get("dst_d1_stride", nselected))
    src0_elems = src0_token_stride * ntokens
    idx_elems = idx_token_stride * ntokens
    dst_elems = dst_token_stride * ntokens
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", src0_elems),
            _read_i32(workbench_path, fixture_dir, "indices", idx_elems).replace("%indices", "%idx"),
            _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
            f"  func.call {candidate.root_symbol}(%src0, %idx, %dst) : (tensor<{src0_elems}xf32>, tensor<{idx_elems}xi32>, tensor<{dst_elems}xf32>)",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{dst_elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_softmax_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    lines = [
        f"  %scale = check.literal value({_candidate_f32_literal(candidate, 'scale', 0.75)}) : f32",
        _read_f32(workbench_path, fixture_dir, "src0", elems),
    ]
    if "mask" in candidate.root_symbol:
        lines.append(_read_f32(workbench_path, fixture_dir, "mask", elems))
        call_args = "%scale, %src0, %mask, %dst"
        call_types = f"f32, tensor<{elems}xf32>, tensor<{elems}xf32>, tensor<{elems}xf32>"
    else:
        call_args = "%scale, %src0, %dst"
        call_types = f"f32, tensor<{elems}xf32>, tensor<{elems}xf32>"
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_set_rows_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    common_dims = _ranked_shape(candidate)
    if common_dims is not None:
        update_tensor, index_tensor = _set_rows_tensor_roles(candidate)
        src0_dims = _copy_tensor_dims(candidate, update_tensor, common_dims)
        src1_dims = _copy_tensor_dims(candidate, index_tensor, common_dims)
        dst_dims = _copy_tensor_dims(candidate, "dst", common_dims)
        elems = _buffer_length(dst_dims, _copy_tensor_strides(candidate, "dst", dst_dims))
        src0_elems = _buffer_length(src0_dims, _copy_tensor_strides(candidate, update_tensor, src0_dims))
        idx_elems = _buffer_length(src1_dims, _copy_tensor_strides(candidate, index_tensor, src1_dims)) * 2
    else:
        _, nrows, elems = _dims(candidate)
        src0_elems = elems
        idx_elems = nrows * 2
    f16_dst = "f16" in candidate.root_symbol
    dst_type = "xi16" if f16_dst else "xf32"
    reader = _read_i16 if f16_dst else _read_f32
    expect = "equal" if f16_dst else "close"
    case_name, bench_name = _case_names(candidate)
    lines = [
        _read_f32(workbench_path, fixture_dir, "src0", src0_elems),
        _read_i32(workbench_path, fixture_dir, "indices", idx_elems).replace("%indices", "%idx"),
        reader(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
        reader(workbench_path, fixture_dir, "expected", elems),
        f"  func.call {candidate.root_symbol}(%src0, %idx, %dst) : (tensor<{src0_elems}xf32>, tensor<{idx_elems}xi32>, tensor<{elems}{dst_type}>)",
    ]
    if expect == "equal":
        lines.append(f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{elems}{dst_type}>")
    else:
        lines.append(f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{elems}{dst_type}>")
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_quantize_q8_1_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    blocks = (ncols + 31) // 32
    x4 = "x4" in candidate.root_symbol
    expected_elems = nrows * ((blocks + 3) // 4) * 144 if x4 else q8_1_bytes(ncols, nrows)
    case_name, bench_name = _case_names(candidate)
    lines = [_read_f32(workbench_path, fixture_dir, "src0", elems).replace("%src0", "%src")]
    if x4 and "vk_clone" in candidate.root_symbol:
        num_blocks = ((ncols + 127) // 128) * 4
        lines = [
            f"  %ne = check.literal value({elems}) : index",
            f"  %num_blocks = check.literal value({num_blocks}) : index",
            *lines,
            f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'expected.npy')}\") : tensor<{expected_elems}xi8>",
            f"  %expected = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'expected.npy')}\") : tensor<{expected_elems}xi8>",
            f"  func.call {candidate.root_symbol}(%ne, %num_blocks, %src, %dst) : (index, index, tensor<{elems}xf32>, tensor<{expected_elems}xi8>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{expected_elems}xi8>",
        ]
    else:
        lines = [
            f"  %ne00 = check.literal value({ncols}) : index",
            f"  %s01 = check.literal value({ncols}) : index",
            f"  %s02 = check.literal value({ncols * nrows}) : index",
            f"  %s03 = check.literal value({ncols * nrows}) : index",
            f"  %ne0 = check.literal value({ncols}) : index",
            f"  %ne1 = check.literal value({nrows}) : index",
            "  %ne2 = check.literal value(1) : index",
            *lines,
            f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'expected.npy')}\") : tensor<{expected_elems}xi8>",
            f"  %expected = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'expected.npy')}\") : tensor<{expected_elems}xi8>",
            f"  func.call {candidate.root_symbol}(%ne00, %s01, %s02, %s03, %ne0, %ne1, %ne2, %src, %dst) : (index, index, index, index, index, index, index, tensor<{elems}xf32>, tensor<{expected_elems}xi8>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{expected_elems}xi8>",
        ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_mul_mat_q8_0_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str | None, dict[str, Any]]:
    if "q8_1" in candidate.root_symbol:
        return None, {"status": "unsupported_workbench", "message": "q8_0 matmul q8_1 RHS ABI packer is not implemented yet"}
    k, rows, cols = _matmul_dims(candidate)
    src0_elems = q8_0_bytes(k, rows)
    src1_elems = k * cols
    dst_elems = rows * cols
    case_name, bench_name = _case_names(candidate)
    lines = [
        f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'src0.npy')}\") : tensor<{src0_elems}xi8>",
        _read_f32(workbench_path, fixture_dir, "src1", src1_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
    ]
    if candidate.root_symbol == "@hrx2_mul_mat_q8_0_f32":
        lines = [
            f"  %k = check.literal value({k}) : index",
            f"  %rows = check.literal value({rows}) : index",
            f"  %cols = check.literal value({cols}) : index",
            *lines,
            f"  func.call {candidate.root_symbol}(%k, %rows, %cols, %src0, %src1, %dst) : (index, index, index, tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)",
        ]
    else:
        lines.append(
            f"  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)"
        )
    lines.append(f"  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>")
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_rms_norm_mul_quantize_q8_1_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    blocks = (ncols + 31) // 32
    expected_elems = nrows * ((blocks + 3) // 4) * 144
    case_name, bench_name = _case_names(candidate)
    lines = [
        "  %eps = check.literal value(0.0) : f32",
        _read_f32(workbench_path, fixture_dir, "src", elems),
        _read_f32(workbench_path, fixture_dir, "weight", ncols),
        f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'dst_init.npy')}\") : tensor<{expected_elems}xi8>",
        f"  %expected = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'expected.npy')}\") : tensor<{expected_elems}xi8>",
        f"  func.call {candidate.root_symbol}(%eps, %src, %weight, %dst) : (f32, tensor<{elems}xf32>, tensor<{ncols}xf32>, tensor<{expected_elems}xi8>)",
        f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{expected_elems}xi8>",
    ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_quantized_mul_mat_static_workbench(
    candidate: Candidate,
    linked_source: Path,
    workbench_path: Path,
    fixture_dir: Path,
) -> tuple[str, dict[str, Any]]:
    k, rows, cols = _matmul_dims(candidate)
    if candidate.family == "mul_mat_q5_k_f32":
        src0_elems = q5_k_bytes(k, rows)
        atol = 0.12
        rtol = 0.04
    elif candidate.family == "mul_mat_q6_k_f32":
        src0_elems = q6_k_bytes(k, rows)
        atol = 0.12
        rtol = 0.04
    else:
        raise ValueError(f"unsupported packed quantized workbench family: {candidate.family}")
    src1_elems = k * cols
    dst_elems = rows * cols
    case_name, bench_name = _case_names(candidate)
    lines = [
        f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, 'src0.npy')}\") : tensor<{src0_elems}xi8>",
        _read_f32(workbench_path, fixture_dir, "src1", src1_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
        f"  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)",
        f"  check.expect.close actual(%dst) expected(%expected) atol({atol}) rtol({rtol}) nan(same) : tensor<{dst_elems}xf32>",
    ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_softmax_kqv_workbench(
    candidate: Candidate,
    linked_source: Path,
    workbench_path: Path,
    fixture_dir: Path,
) -> tuple[str, dict[str, Any]]:
    k, rows, cols = _matmul_dims(candidate)
    nheads_kv = int(candidate.shape.get("nheads_kv", 8))
    src0_elems = cols * k
    mask_elems = k
    src1_elems = nheads_kv * rows * k
    dst_elems = rows * cols
    case_name, bench_name = _case_names(candidate)
    lines = [
        f"  %scale = check.literal value({_candidate_f32_literal(candidate, 'scale', 0.75)}) : f32",
        _read_f32(workbench_path, fixture_dir, "kq", src0_elems),
        _read_f32(workbench_path, fixture_dir, "mask", mask_elems),
        _read_f16(workbench_path, fixture_dir, "v", src1_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
        (
            f"  func.call {candidate.root_symbol}(%scale, %kq, %mask, %v, %dst) : "
            f"(f32, tensor<{src0_elems}xf32>, tensor<{mask_elems}xf32>, tensor<{src1_elems}xf16>, tensor<{dst_elems}xf32>)"
        ),
        f"  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>",
    ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_rope_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str | None, dict[str, Any]]:
    if candidate.family == "rope_set_rows_f32":
        src0_dims = _captured_tensor_dims(candidate, "src0")
        dst_dims = _captured_tensor_dims(candidate, "dst")
        idx_dims = _captured_tensor_dims(candidate, "src1")
        if src0_dims is None or dst_dims is None or idx_dims is None:
            return None, {"status": "unsupported_workbench", "message": "ROPE set_rows requires encoded src0, src1, and dst tensor dimensions"}
        src0_elems = src0_dims[0] * src0_dims[1] * src0_dims[2] * src0_dims[3]
        dst_elems = dst_dims[0] * dst_dims[1] * dst_dims[2] * dst_dims[3]
        idx_elems = idx_dims[0] * idx_dims[1] * idx_dims[2] * idx_dims[3]
        ntokens = int(candidate.values.get("shape.rope.ntokens") or src0_dims[2])
        pos_elems = int(candidate.values.get("shape.rope.pos_token_stride") or 1) * ntokens
        freq_elems = max(int(candidate.values.get("shape.rope.n_dims") or candidate.values.get("attribute.n_dims") or src0_dims[0]) // 2, 1)
        theta_scale = _candidate_f32_literal(candidate, "theta_scale", 0.75)
        freq_scale = _candidate_f32_literal(candidate, "freq_scale", 1.1)
        attn_factor = _candidate_f32_literal(candidate, "attn_factor", 0.9)
        case_name, bench_name = _case_names(candidate)
        unpack_symbol = f"@hrx2_test_unpack_f16_to_f32_{candidate.id}"
        suffix = f"""
kernel.def export("hrx2_test_unpack_f16_to_f32_{candidate.id}") {unpack_symbol}() {{
  %unit = index.constant 1 : index
  %minus_one = index.constant -1 : index
  %count = index.constant {dst_elems} : index
  %workgroup_size = index.constant 256 : index
  %rounding = index.add %workgroup_size, %minus_one : index
  %rounded = index.add %count, %rounding : index
  %workgroups = index.div %rounded, %workgroup_size : index
  kernel.launch.config workgroups(%workgroups, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%src: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %count = index.constant {dst_elems} : index
  %workgroup_size = index.constant 256 : index
  %workgroup = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %linear_mul = index.mul %workgroup, %workgroup_size : index
  %linear = index.add %linear_mul, %lane : index
  %in_bounds = index.cmp ult, %linear, %count : index

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<{dst_elems}xf16, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{dst_elems}xf32, #dense>

  scf.if %in_bounds {{
    %half = view.load %src_view[%linear] : view<{dst_elems}xf16, #dense> -> f16
    %value = scalar.extf %half : f16 to f32
    view.store %value, %dst_view[%linear] : f32, view<{dst_elems}xf32, #dense>
  }}
  kernel.return
}}

check.case public {case_name} {{
  %theta_scale = check.literal value({theta_scale}) : f32
  %freq_scale = check.literal value({freq_scale}) : f32
  %attn_factor = check.literal value({attn_factor}) : f32
{_read_f32(workbench_path, fixture_dir, "src0", src0_elems)}
{_read_i32(workbench_path, fixture_dir, "positions", pos_elems).replace("%positions", "%pos")}
{_read_f32(workbench_path, fixture_dir, "freq", freq_elems)}
{_read_i64(workbench_path, fixture_dir, "indices", idx_elems).replace("%indices", "%idx")}
{_read_i16(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst_bits")}
{_read_f32(workbench_path, fixture_dir, "dst_f32_init", dst_elems).replace("%dst_f32_init", "%dst")}
{_read_f32(workbench_path, fixture_dir, "expected_f32", dst_elems).replace("%expected_f32", "%expected")}
  func.call {candidate.root_symbol}(%theta_scale, %freq_scale, %attn_factor, %src0, %pos, %freq, %idx, %dst_bits) : (f32, f32, f32, tensor<{src0_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{freq_elems}xf32>, tensor<{idx_elems}xi64>, tensor<{dst_elems}xi16>)
  func.call {unpack_symbol}(%dst_bits, %dst) : (tensor<{dst_elems}xi16>, tensor<{dst_elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0005) rtol(0.0005) nan(same) : tensor<{dst_elems}xf32>
  check.return
}}
"""
        _source_plus_case(linked_source, workbench_path, suffix)
        return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}
    ncols = int(candidate.values.get("shape.rope.ncols") or candidate.shape.get("ncols", 1))
    n_dims = int(candidate.values.get("shape.rope.n_dims") or candidate.values.get("attribute.n_dims") or candidate.shape.get("n_dims", ncols))
    nheads = int(candidate.values.get("shape.rope.nheads") or candidate.shape.get("rows", 1))
    ntokens = int(candidate.values.get("shape.rope.ntokens") or candidate.shape.get("cols", 1))
    src_token_stride = int(candidate.values.get("shape.rope.src0_token_stride") or (ncols * nheads))
    dst_token_stride = int(candidate.values.get("shape.rope.dst_token_stride") or (ncols * nheads))
    pos_token_stride = int(candidate.values.get("shape.rope.pos_token_stride") or 1)
    src_elems = src_token_stride * ntokens
    dst_elems = dst_token_stride * ntokens
    pos_elems = pos_token_stride * ntokens
    freq_elems = max(n_dims // 2, 1)
    has_freq = "freq" in candidate.root_symbol
    scale_output = "scale" in candidate.root_symbol
    case_name, bench_name = _case_names(candidate)
    lines = [
        f"  %theta_scale = check.literal value({_candidate_f32_literal(candidate, 'theta_scale', 0.75)}) : f32",
        f"  %freq_scale = check.literal value({_candidate_f32_literal(candidate, 'freq_scale', 1.1)}) : f32",
        f"  %attn_factor = check.literal value({_candidate_f32_literal(candidate, 'attn_factor', 0.9)}) : f32",
    ]
    if scale_output:
        lines.append(f"  %output_scale = check.literal value({_candidate_f32_literal(candidate, 'output_scale', 0.5)}) : f32")
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "src0", src_elems),
            _read_i32(workbench_path, fixture_dir, "positions", pos_elems).replace("%positions", "%pos"),
        ]
    )
    if has_freq:
        lines.append(_read_f32(workbench_path, fixture_dir, "freq", freq_elems))
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
        ]
    )
    if scale_output:
        call_args = "%theta_scale, %freq_scale, %attn_factor, %output_scale, %src0, %pos, %freq, %dst"
        call_types = f"f32, f32, f32, f32, tensor<{src_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{freq_elems}xf32>, tensor<{dst_elems}xf32>"
    elif has_freq:
        call_args = "%theta_scale, %freq_scale, %attn_factor, %src0, %pos, %freq, %dst"
        call_types = f"f32, f32, f32, tensor<{src_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{freq_elems}xf32>, tensor<{dst_elems}xf32>"
    else:
        call_args = "%theta_scale, %freq_scale, %attn_factor, %src0, %pos, %dst"
        call_types = f"f32, f32, f32, tensor<{src_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{dst_elems}xf32>"
    lines.extend(
        [
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0005) rtol(0.0005) nan(same) : tensor<{dst_elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


ORACLE_SPECS: tuple[OracleSpec, ...] = (
    OracleSpec(
        family_ids=("mul_mat_q4_k_f32",),
        generate=_mul_mat_q4_k_f32,
        write_workbench=_write_mul_mat_q4_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_id_q4_k_f32",),
        generate=_mul_mat_id_q4_k_f32,
        write_workbench=_write_mul_mat_id_q4_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_id_q5_k_f32",),
        generate=_mul_mat_id_q5_k_f32,
        write_workbench=_write_mul_mat_id_q5_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_id_q6_k_f32",),
        generate=_mul_mat_id_q6_k_f32,
        write_workbench=_write_mul_mat_id_q6_workbench,
    ),
    OracleSpec(
        family_ids=("rms_norm_f32",),
        generate=_rms_norm_f32,
        write_workbench=_write_rms_norm_workbench,
    ),
    OracleSpec(
        family_ids=("rms_norm_mul_f32",),
        generate=_rms_norm_mul_f32,
        write_workbench=_write_rms_norm_mul_workbench,
    ),
    OracleSpec(
        family_ids=("add_rms_norm_mul_f32",),
        generate=_add_rms_norm_mul_f32,
        write_workbench=_write_add_rms_norm_mul_workbench,
    ),
    OracleSpec(
        family_ids=(
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
        ),
        generate=_copy_f32_f16,
        write_workbench=_write_copy_workbench,
    ),
    OracleSpec(
        family_ids=("cont_f32",),
        generate=_cont_f32,
        write_workbench=_write_cont_workbench,
    ),
    *(
        OracleSpec(
            family_ids=spec.family_ids,
            generate=_logical_generate(spec),
            write_workbench=_logical_workbench,
        )
        for spec in LOGICAL_ORACLE_SPECS
    ),
    OracleSpec(
        family_ids=("add_f32", "mul_f32", "div_f32", "sub_f32", "add_f16", "mul_f16", "div_f16", "sub_f16"),
        generate=_logical_generate(
            LogicalOracleSpec(
                ("add_f32", "mul_f32", "div_f32", "sub_f32", "add_f16", "mul_f16", "div_f16", "sub_f16"),
                "pointwise_binary_numpy",
                {"atol": 1e-5, "rtol": 1e-5},
                lambda np, candidate, seed: {
                    "add_f32": _binary_arrays(lambda lhs, rhs: lhs + rhs),
                    "mul_f32": _binary_arrays(lambda lhs, rhs: lhs * rhs),
                    "div_f32": _div_arrays,
                    "sub_f32": _binary_arrays(lambda lhs, rhs: lhs - rhs),
                    "add_f16": _add_f16_exact_arrays,
                    "mul_f16": _mul_f16_exact_arrays,
                    "div_f16": _div_f16_exact_arrays,
                    "sub_f16": _sub_f16_exact_arrays,
                }[candidate.family](np, candidate, seed),
                exact_kernel_abi=True,
            )
        ),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("scale_f32",),
        generate=_logical_generate(LogicalOracleSpec(("scale_f32",), "scale_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _scale_arrays, exact_kernel_abi=True)),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("clamp_f32",),
        generate=_logical_generate(LogicalOracleSpec(("clamp_f32",), "clamp_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _clamp_arrays, exact_kernel_abi=True)),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("clamp_f16",),
        generate=_logical_generate(LogicalOracleSpec(("clamp_f16",), "clamp_f16_numpy", {"atol": 0.0, "rtol": 0.0}, _clamp_f16_arrays, exact_kernel_abi=True)),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("swiglu_f32",),
        generate=_logical_generate(LogicalOracleSpec(("swiglu_f32",), "swiglu_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _swiglu_arrays, exact_kernel_abi=True)),
        write_workbench=_write_swiglu_workbench,
    ),
    OracleSpec(
        family_ids=("swiglu_f16",),
        generate=_logical_generate(LogicalOracleSpec(("swiglu_f16",), "swiglu_f16_numpy", {"atol": 1e-3, "rtol": 1e-3}, _swiglu_arrays, exact_kernel_abi=True)),
        write_workbench=_write_swiglu_workbench,
    ),
    OracleSpec(
        family_ids=("sum_rows_f32",),
        generate=_logical_generate(LogicalOracleSpec(("sum_rows_f32",), "sum_rows_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _sum_rows_arrays, exact_kernel_abi=True)),
        write_workbench=_write_sum_rows_workbench,
    ),
    OracleSpec(
        family_ids=("argsort_f32_i32",),
        generate=_logical_generate(LogicalOracleSpec(("argsort_f32_i32",), "argsort_f32_i32_numpy_desc", {"atol": 0.0, "rtol": 0.0}, _argsort_arrays, exact_kernel_abi=True)),
        write_workbench=_write_argsort_workbench,
    ),
    OracleSpec(
        family_ids=("get_rows_f32", "get_rows_moe_weights_f32", "get_rows_q4_k_f32", "get_rows_q5_k_f32", "get_rows_q6_k_f32", "get_rows_q8_0_f32"),
        generate=_logical_generate(LogicalOracleSpec(("get_rows_f32", "get_rows_moe_weights_f32", "get_rows_q4_k_f32", "get_rows_q5_k_f32", "get_rows_q6_k_f32", "get_rows_q8_0_f32"), "get_rows_numpy", {"atol": 1e-5, "rtol": 1e-5}, _get_rows_arrays, exact_kernel_abi=True)),
        write_workbench=_write_get_rows_workbench,
    ),
    OracleSpec(
        family_ids=("soft_max_f32",),
        generate=_logical_generate(LogicalOracleSpec(("soft_max_f32",), "soft_max_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _softmax_arrays, exact_kernel_abi=True)),
        write_workbench=_write_softmax_workbench,
    ),
    OracleSpec(
        family_ids=("set_rows_f32", "cont_set_rows_f32"),
        generate=_logical_generate(LogicalOracleSpec(("set_rows_f32", "cont_set_rows_f32"), "set_rows_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _set_rows_arrays, exact_kernel_abi=True)),
        write_workbench=_write_set_rows_workbench,
    ),
    OracleSpec(
        family_ids=("quantize_q8_1_f32",),
        generate=_logical_generate(LogicalOracleSpec(("quantize_q8_1_f32",), "quantize_q8_1_numpy", {"atol": 0.0, "rtol": 0.0}, _quantize_q8_1_arrays, exact_kernel_abi=True)),
        write_workbench=_write_quantize_q8_1_workbench,
    ),
    OracleSpec(
        family_ids=("rms_norm_mul_quantize_q8_1_f32",),
        generate=_rms_norm_mul_quantize_q8_1_f32,
        write_workbench=_write_rms_norm_mul_quantize_q8_1_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_q8_0_f32",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_q8_0_f32",), "mul_mat_q8_0_f32_numpy_dequant_matmul", {"atol": 0.08, "rtol": 0.02}, _mul_mat_q8_0_arrays, exact_kernel_abi=True)),
        write_workbench=_write_mul_mat_q8_0_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_q5_k_f32",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_q5_k_f32",), "mul_mat_q5_k_f32_numpy_dequant_matmul", {"atol": 0.12, "rtol": 0.04}, _mul_mat_q5_k_arrays, exact_kernel_abi=True)),
        write_workbench=_write_quantized_mul_mat_static_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_q6_k_f32",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_q6_k_f32",), "mul_mat_q6_k_f32_numpy_dequant_matmul", {"atol": 0.12, "rtol": 0.04}, _mul_mat_q6_k_arrays, exact_kernel_abi=True)),
        write_workbench=_write_quantized_mul_mat_static_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_f32_f32",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_f32_f32",), "mul_mat_f32_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _matmul_f32_arrays, exact_kernel_abi=True)),
        write_workbench=_write_mul_mat_f32_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_f16_f32_generic",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_f16_f32_generic",), "mul_mat_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _matmul_f16_f32_arrays, exact_kernel_abi=True)),
        write_workbench=_write_mul_mat_f16_f32_generic_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_f16_f16_generic",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_f16_f16_generic",), "mul_mat_f16_f16_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _matmul_f16_f16_arrays, exact_kernel_abi=True)),
        write_workbench=_write_mul_mat_f16_f16_workbench,
    ),
    OracleSpec(
        family_ids=("softmax_kqv_f32_f16",),
        generate=_logical_generate(
            LogicalOracleSpec(
                ("softmax_kqv_f32_f16",),
                "softmax_kqv_f32_f16_numpy_logical",
                {"atol": 0.08, "rtol": 0.02},
                _softmax_kqv_arrays,
                exact_kernel_abi=True,
            )
        ),
        write_workbench=_write_softmax_kqv_workbench,
    ),
    OracleSpec(
        family_ids=("flash_attn_ext_f32_f16",),
        generate=_logical_generate(
            LogicalOracleSpec(
                ("flash_attn_ext_f32_f16",),
                "flash_attn_ext_f32_f16_numpy_logical",
                {"atol": 0.08, "rtol": 0.02},
                _flash_attn_ext_arrays,
                exact_kernel_abi=True,
            )
        ),
        write_workbench=_logical_workbench,
    ),
    OracleSpec(
        family_ids=("rope_f32", "rope_neox_f32", "rope_scale_f32"),
        generate=_logical_generate(LogicalOracleSpec(("rope_f32", "rope_neox_f32", "rope_scale_f32"), "rope_f32_numpy", {"atol": 5e-4, "rtol": 5e-4}, _rope_arrays, exact_kernel_abi=True)),
        write_workbench=_write_rope_workbench,
    ),
    OracleSpec(
        family_ids=("rope_f16", "rope_neox_f16"),
        generate=_logical_generate(LogicalOracleSpec(("rope_f16", "rope_neox_f16"), "rope_f16_numpy", {"atol": 2e-3, "rtol": 2e-3}, _rope_arrays, exact_kernel_abi=True)),
        write_workbench=_write_rope_workbench,
    ),
    OracleSpec(
        family_ids=("rope_set_rows_f32",),
        generate=_logical_generate(LogicalOracleSpec(("rope_set_rows_f32",), "rope_set_rows_f32_numpy", {"atol": 0.0, "rtol": 0.0}, _rope_arrays, exact_kernel_abi=True)),
        write_workbench=_write_rope_workbench,
    ),
    OracleSpec(
        family_ids=(
            "abs_f32",
            "abs_f16",
            "ceil_f32",
            "ceil_f16",
            "cos_f32",
            "cos_f16",
            "elu_f32",
            "elu_f16",
            "exp_f32",
            "exp_f16",
            "expm1_f32",
            "expm1_f16",
            "floor_f32",
            "floor_f16",
            "gelu_f32",
            "gelu_f16",
            "gelu_erf_f32",
            "gelu_erf_f16",
            "gelu_quick_f32",
            "gelu_quick_f16",
            "hardsigmoid_f32",
            "hardsigmoid_f16",
            "hardswish_f32",
            "hardswish_f16",
            "leaky_relu_f32",
            "leaky_relu_f16",
            "log_f32",
            "log_f16",
            "neg_f32",
            "neg_f16",
            "relu_f32",
            "relu_f16",
            "round_f32",
            "round_f16",
            "sgn_f32",
            "sgn_f16",
            "sigmoid_f32",
            "sigmoid_f16",
            "silu_f32",
            "silu_f16",
            "sin_f32",
            "sin_f16",
            "softcap_f32",
            "softplus_f32",
            "softplus_f16",
            "sqr_f32",
            "sqr_f16",
            "sqrt_f32",
            "sqrt_f16",
            "step_f32",
            "step_f16",
            "tanh_f32",
            "tanh_f16",
            "trunc_f32",
            "trunc_f16",
            "xielu_f32",
        ),
        generate=_logical_generate(
            LogicalOracleSpec(
                (
                    "abs_f32",
                    "abs_f16",
                    "ceil_f32",
                    "ceil_f16",
                    "cos_f32",
                    "cos_f16",
                    "elu_f32",
                    "elu_f16",
                    "exp_f32",
                    "exp_f16",
                    "expm1_f32",
                    "expm1_f16",
                    "floor_f32",
                    "floor_f16",
                    "gelu_f32",
                    "gelu_f16",
                    "gelu_erf_f32",
                    "gelu_erf_f16",
                    "gelu_quick_f32",
                    "gelu_quick_f16",
                    "hardsigmoid_f32",
                    "hardsigmoid_f16",
                    "hardswish_f32",
                    "hardswish_f16",
                    "leaky_relu_f32",
                    "leaky_relu_f16",
                    "log_f32",
                    "log_f16",
                    "neg_f32",
                    "neg_f16",
                    "relu_f32",
                    "relu_f16",
                    "round_f32",
                    "round_f16",
                    "sgn_f32",
                    "sgn_f16",
                    "sigmoid_f32",
                    "sigmoid_f16",
                    "silu_f32",
                    "silu_f16",
                    "sin_f32",
                    "sin_f16",
                    "softcap_f32",
                    "softplus_f32",
                    "softplus_f16",
                    "sqr_f32",
                    "sqr_f16",
                    "sqrt_f32",
                    "sqrt_f16",
                    "step_f32",
                    "step_f16",
                    "tanh_f32",
                    "tanh_f16",
                    "trunc_f32",
                    "trunc_f16",
                    "xielu_f32",
                ),
                "unary_numpy",
                {"atol": 1e-5, "rtol": 1e-5},
                _unary_arrays,
                exact_kernel_abi=True,
            )
        ),
        write_workbench=_write_unary_workbench,
    ),
)


ORACLE_SPECS_BY_FAMILY: dict[str, OracleSpec] = {
    family_id: spec
    for spec in ORACLE_SPECS
    for family_id in spec.family_ids
}
