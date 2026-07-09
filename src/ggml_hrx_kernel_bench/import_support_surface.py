from __future__ import annotations


VALIDATED_IMPORT_OPS = frozenset(
    {
        "ABS",
        "ADD",
        "ARGSORT",
        "CLAMP",
        "CONT",
        "CPY",
        "DIV",
        "EXP",
        "FLASH_ATTN_EXT",
        "GET_ROWS",
        "MUL",
        "MUL_MAT",
        "NEG",
        "ROPE",
        "ROPE_SCALE",
        "ROPE_SET_ROWS",
        "RMS_NORM",
        "RELU",
        "SCALE",
        "SET_ROWS",
        "SOFT_MAX",
        "SQR",
        "SQRT",
        "SUB",
        "SWIGLU",
        "SUM_ROWS",
    }
)


def validated_import_ops() -> frozenset[str]:
    return VALIDATED_IMPORT_OPS
