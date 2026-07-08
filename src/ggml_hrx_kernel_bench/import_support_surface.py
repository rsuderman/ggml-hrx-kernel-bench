from __future__ import annotations


VALIDATED_IMPORT_OPS = frozenset(
    {
        "ABS",
        "ADD",
        "CLAMP",
        "CONT",
        "CPY",
        "DIV",
        "MUL",
        "SCALE",
        "SET_ROWS",
        "SUB",
    }
)


def validated_import_ops() -> frozenset[str]:
    return VALIDATED_IMPORT_OPS
