from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .import_models import ImportedCase


CasePredicate = Callable[[ImportedCase], bool]
LoweringFn = Callable[[ImportedCase], tuple[list[str], list[int]]]


@dataclass(frozen=True)
class ImportMappingRule:
    op: str
    dtype_filters: Mapping[str, str]
    kernel_family: str
    predicate: CasePredicate
    lowering: LoweringFn
    notes: str | None = None


@dataclass(frozen=True)
class MappingCandidate:
    rule: ImportMappingRule
    score: int = 0


def _dtype_matches(dtype: Mapping[str, Any], filters: Mapping[str, str]) -> bool:
    for key, expected in filters.items():
        actual = dtype.get(key)
        if str(actual).lower() != expected.lower():
            return False
    return True


def compatible_rules_for_op(case: ImportedCase, rules: list[ImportMappingRule]) -> list[ImportMappingRule]:
    op = case.op.upper()
    return [rule for rule in rules if rule.op.upper() == op]


def compatible_rules_for_op_dtype(case: ImportedCase, rules: list[ImportMappingRule]) -> list[ImportMappingRule]:
    return [rule for rule in compatible_rules_for_op(case, rules) if _dtype_matches(case.dtype, rule.dtype_filters)]


def match_rules(case: ImportedCase, rules: list[ImportMappingRule]) -> list[MappingCandidate]:
    candidates: list[MappingCandidate] = []
    for rule in compatible_rules_for_op_dtype(case, rules):
        if not rule.predicate(case):
            continue
        candidates.append(MappingCandidate(rule=rule))
    return candidates


def _int_list(case: ImportedCase, key: str) -> list[int]:
    raw = case.normalized_params.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"expected a non-empty list for {key}")
    values: list[int] = []
    for index, value in enumerate(raw):
        if not isinstance(value, int):
            raise ValueError(f"{key}[{index}] must be an integer")
        values.append(value)
    return values


def _all_zero(values: list[int]) -> bool:
    return all(value == 0 for value in values)


def copy_f32_f16_contiguous(case: ImportedCase) -> bool:
    params = case.normalized_params
    if int(params.get("_src_transpose", 0)) != 0:
        return False
    try:
        permute_src = _int_list(case, "permute_src")
        permute_dst = _int_list(case, "permute_dst")
        ne = _int_list(case, "ne")
    except ValueError:
        return False
    return len(ne) == 4 and _all_zero(permute_src) and _all_zero(permute_dst)


def lower_copy_f32_f16_contiguous(case: ImportedCase) -> tuple[list[str], list[int]]:
    ne = _int_list(case, "ne")
    return ["nrows", "ncols"], [1, math.prod(ne)]


IMPORT_MAPPING_RULES: list[ImportMappingRule] = [
    ImportMappingRule(
        op="CPY",
        dtype_filters={"type_src": "f32", "type_dst": "f16"},
        kernel_family="copy_f32_f16",
        predicate=copy_f32_f16_contiguous,
        lowering=lower_copy_f32_f16_contiguous,
        notes="maps contiguous F32->F16 copies into the generic flattened copy kernel",
    )
]
