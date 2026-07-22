from __future__ import annotations

import json
import importlib.util
from io import StringIO
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = PROJECT_ROOT / "tools" / "v2-route-selector" / "python_v2_route_selector.py"
ROUTING_DIR = PROJECT_ROOT / "catalog" / "v2"


class _FailingFlushOutput:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        raise BrokenPipeError("output pipe is closed")


def _load_selector_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("python_v2_route_selector", SELECTOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _query(*, operation: str = "CLAMP") -> dict[str, object]:
    tensor = {
        "dtype": "F32",
        "dimensions": [5, 7],
        "strides": [1, 5],
    }
    return {
        "op": operation,
        "tensors": {"src0": tensor, "dst": tensor},
        "attributes": {},
    }


def _run_selector(
    *,
    input_path: str,
    routing_dir: Path = ROUTING_DIR,
    standard_input: str | None = None,
    batch: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SELECTOR),
        "--input",
        input_path,
        "--routing-dir",
        str(routing_dir),
    ]
    if batch:
        command.append("--batch")
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        input=standard_input,
        check=False,
        capture_output=True,
        text=True,
    )


def test_selects_route_from_standard_input() -> None:
    result = _run_selector(input_path="-", standard_input=json.dumps(_query()))

    assert result.returncode == 0, result.stderr
    assert result.stdout == "clamp_f32_contiguous_4d\n"
    assert result.stderr == ""


def test_selects_route_from_file(tmp_path: Path) -> None:
    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps(_query()), encoding="utf-8")

    result = _run_selector(input_path=str(query_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == "clamp_f32_contiguous_4d\n"
    assert result.stderr == ""


def test_returns_one_when_no_route_matches() -> None:
    result = _run_selector(
        input_path="-",
        standard_input=json.dumps(_query(operation="DOES_NOT_EXIST")),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "NO_MATCH" in result.stderr
    assert "DOES_NOT_EXIST" in result.stderr


def test_returns_two_for_malformed_json() -> None:
    result = _run_selector(input_path="-", standard_input="not-json\n")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr


def test_returns_two_for_invalid_route_query() -> None:
    result = _run_selector(input_path="-", standard_input=json.dumps({"op": "CLAMP"}))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "missing required field 'tensors'" in result.stderr


def test_returns_two_for_unreadable_input_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    result = _run_selector(input_path=str(missing_path))

    assert result.returncode == 2
    assert result.stdout == ""
    assert str(missing_path) in result.stderr


def test_returns_two_for_invalid_routing_catalog(tmp_path: Path) -> None:
    missing_routing_dir = tmp_path / "missing-catalog"

    result = _run_selector(
        input_path="-",
        routing_dir=missing_routing_dir,
        standard_input=json.dumps(_query()),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "router.json" in result.stderr


def test_returns_two_for_usage_error() -> None:
    result = subprocess.run(
        [sys.executable, str(SELECTOR), "--input", "-"],
        cwd=PROJECT_ROOT,
        input=json.dumps(_query()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "--routing-dir" in result.stderr


def test_batch_selects_from_standard_input_and_preserves_physical_line_numbers() -> None:
    match = json.dumps(_query())
    no_match = json.dumps(_query(operation="DOES_NOT_EXIST"))

    result = _run_selector(
        input_path="-",
        standard_input=f"\r\n{match}\r\n \t\r\n{no_match}\r\n",
        batch=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        '{"line":2,"status":"MATCH","route_id":"clamp_f32_contiguous_4d"}',
        (
            '{"line":4,"status":"NO_MATCH",'
            '"diagnostic":"no route matched operation \'DOES_NOT_EXIST\'"}'
        ),
    ]
    assert result.stderr == ""


def test_batch_reads_file_and_processes_final_line_without_newline(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(json.dumps(_query()), encoding="utf-8")

    result = _run_selector(input_path=str(query_path), batch=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        '{"line":1,"status":"MATCH","route_id":"clamp_f32_contiguous_4d"}\n'
    )
    assert result.stderr == ""


def test_batch_continues_after_malformed_and_invalid_rows() -> None:
    rows = [
        "not-json",
        json.dumps({"op": "CLAMP"}),
        json.dumps(_query(operation="DOES_NOT_EXIST")),
        json.dumps(_query()),
    ]

    result = _run_selector(
        input_path="-",
        standard_input="\n".join(rows),
        batch=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == (
        '{"line":1,"status":"ERROR","diagnostic":"malformed JSON"}'
    )
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"line": 1, "status": "ERROR", "diagnostic": "malformed JSON"},
        {
            "line": 2,
            "status": "ERROR",
            "diagnostic": "route query is missing required field 'tensors'",
        },
        {
            "line": 3,
            "status": "NO_MATCH",
            "diagnostic": "no route matched operation 'DOES_NOT_EXIST'",
        },
        {"line": 4, "status": "MATCH", "route_id": "clamp_f32_contiguous_4d"},
    ]
    assert result.stderr == ""


def test_batch_accepts_empty_input() -> None:
    result = _run_selector(input_path="-", standard_input="", batch=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_batch_does_not_treat_unicode_whitespace_as_a_blank_jsonl_row() -> None:
    result = _run_selector(
        input_path="-",
        standard_input="\N{NO-BREAK SPACE}\n" + json.dumps(_query()),
        batch=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        '{"line":1,"status":"ERROR","diagnostic":"malformed JSON"}',
        '{"line":2,"status":"MATCH","route_id":"clamp_f32_contiguous_4d"}',
    ]
    assert result.stderr == ""


def test_batch_returns_two_for_unreadable_input_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"

    result = _run_selector(input_path=str(missing_path), batch=True)

    assert result.returncode == 2
    assert result.stdout == ""
    assert str(missing_path) in result.stderr


def test_batch_returns_two_for_non_utf8_input_file(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    query_path.write_bytes(b"\xff\n")

    result = _run_selector(input_path=str(query_path), batch=True)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "decode" in result.stderr


def test_batch_returns_two_for_invalid_routing_catalog(tmp_path: Path) -> None:
    result = _run_selector(
        input_path="-",
        routing_dir=tmp_path / "missing-catalog",
        standard_input=json.dumps(_query()),
        batch=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "router.json" in result.stderr


def test_batch_returns_two_for_malformed_routing_catalog(tmp_path: Path) -> None:
    routing_dir = tmp_path / "routing"
    routing_dir.mkdir()
    (routing_dir / "router.json").write_text("not-json", encoding="utf-8")

    result = _run_selector(
        input_path="-",
        routing_dir=routing_dir,
        standard_input=json.dumps(_query()),
        batch=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr


def test_batch_returns_two_when_output_flush_fails() -> None:
    selector_module = _load_selector_module()
    standard_output = _FailingFlushOutput()
    standard_error = StringIO()
    selector_module.sys = SimpleNamespace(
        stdin=StringIO(json.dumps(_query())),
        stdout=standard_output,
        stderr=standard_error,
    )

    return_code = selector_module.main(
        ["--input", "-", "--routing-dir", str(ROUTING_DIR), "--batch"]
    )

    assert return_code == 2
    assert standard_output.writes == [
        '{"line":1,"status":"MATCH","route_id":"clamp_f32_contiguous_4d"}\n'
    ]
    assert standard_error.getvalue() == (
        "python_v2_route_selector.py: error: output pipe is closed\n"
    )
    assert "Traceback" not in standard_error.getvalue()
    assert selector_module.sys.stdout is None
