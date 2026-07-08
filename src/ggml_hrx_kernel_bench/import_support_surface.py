from __future__ import annotations


VALIDATED_IMPORT_OPS = frozenset(
    {
        "ABS",
        "ADD",
        "CLAMP",
        "CONT",
        "CPY",
        "DIV",
        "EXP",
        "MUL",
        "NEG",
        "RELU",
        "SCALE",
        "SET_ROWS",
        "SQR",
        "SQRT",
        "SUB",
        "SUM_ROWS",
    }
)


def validated_import_ops() -> frozenset[str]:
    return VALIDATED_IMPORT_OPS
