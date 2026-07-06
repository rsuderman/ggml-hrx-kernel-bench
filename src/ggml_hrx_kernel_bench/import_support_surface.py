from __future__ import annotations


VALIDATED_IMPORT_OPS = frozenset({"ABS", "ADD", "CPY", "DIV", "MUL", "SUB"})


def validated_import_ops() -> frozenset[str]:
    return VALIDATED_IMPORT_OPS
