from __future__ import annotations


VALIDATED_IMPORT_OPS = frozenset({"ADD", "CPY", "MUL"})


def validated_import_ops() -> frozenset[str]:
    return VALIDATED_IMPORT_OPS
