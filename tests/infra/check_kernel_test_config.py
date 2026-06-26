from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {"id", "op", "cases"}
OPTIONAL_TOP_LEVEL_KEYS = {"$schema", "source", "root_symbol"}
ALLOWED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS | OPTIONAL_TOP_LEVEL_KEYS
REQUIRED_CASE_KEYS = {"id", "inputs", "outputs"}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_shape(shape: object, *, context: str) -> None:
    _expect(isinstance(shape, dict), f"{context} must be an object")
    _expect(bool(shape), f"{context} must not be empty")
    for axis, value in shape.items():
        _expect(isinstance(axis, str) and axis, f"{context} has an invalid axis name")
        _expect(isinstance(value, int) and value > 0, f"{context}.{axis} must be a positive integer")


def _validate_tensor_shapes(value: object, *, context: str) -> None:
    _expect(isinstance(value, dict), f"{context} must be an object")
    _expect(bool(value), f"{context} must not be empty")
    for tensor_name, shape in value.items():
        _expect(isinstance(tensor_name, str) and tensor_name, f"{context} has an invalid tensor name")
        _validate_shape(shape, context=f"{context}.{tensor_name}")


def validate_config(data: object) -> None:
    _expect(isinstance(data, dict), 'config must be a JSON object')
    extra_keys = set(data) - ALLOWED_TOP_LEVEL_KEYS
    _expect(not extra_keys, f"unexpected top-level keys: {sorted(extra_keys)}")
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(data)
    _expect(not missing_keys, f"missing top-level keys: {sorted(missing_keys)}")
    _expect(isinstance(data['id'], str) and data['id'], 'id must be a non-empty string')
    _expect(isinstance(data['op'], str) and data['op'], 'op must be a non-empty string')
    _expect(isinstance(data['cases'], list) and data['cases'], 'cases must be a non-empty array')
    if '$schema' in data:
        _expect(isinstance(data['$schema'], str) and data['$schema'], '$schema must be a non-empty string')
    if 'source' in data:
        _expect(isinstance(data['source'], str) and data['source'], 'source must be a non-empty string')
    if 'root_symbol' in data:
        _expect(isinstance(data['root_symbol'], str) and data['root_symbol'], 'root_symbol must be a non-empty string')

    for index, case in enumerate(data['cases']):
        _expect(isinstance(case, dict), f"cases[{index}] must be an object")
        extra_case_keys = set(case) - REQUIRED_CASE_KEYS
        _expect(not extra_case_keys, f"cases[{index}] has unexpected keys: {sorted(extra_case_keys)}")
        missing_case_keys = REQUIRED_CASE_KEYS - set(case)
        _expect(not missing_case_keys, f"cases[{index}] is missing keys: {sorted(missing_case_keys)}")
        _expect(isinstance(case['id'], str) and case['id'], f"cases[{index}].id must be a non-empty string")
        _validate_tensor_shapes(case['inputs'], context=f"cases[{index}].inputs")
        _validate_tensor_shapes(case['outputs'], context=f"cases[{index}].outputs")


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate a kernel test config JSON file.')
    parser.add_argument('config_path')
    args = parser.parse_args()

    path = Path(args.config_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    validate_config(data)
    print(f"validated {path} with {len(data['cases'])} case(s)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
