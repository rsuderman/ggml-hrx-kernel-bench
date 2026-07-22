from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARITY_TOOL_DIR = PROJECT_ROOT / "tests" / "infra"
sys.path.insert(0, str(PARITY_TOOL_DIR))

import check_route_selector_parity as route_selector_parity  # noqa: E402
from check_route_selector_parity import (  # noqa: E402
    compare_route_selectors,
    format_route_selector_parity_report,
)


CHECKER = PROJECT_ROOT / "tests" / "infra" / "check_route_selector_parity.py"
ENFORCE_ROUTER_PARITY_ENV = "ENFORCE_ROUTER_PARITY"


def _jsonl(*records: dict[str, object]) -> str:
    return "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)


def _match(line_number: int, route_id: str = "route.shared") -> dict[str, object]:
    return {"line": line_number, "status": "MATCH", "route_id": route_id}


def _failure(line_number: int, status: str, diagnostic: str) -> dict[str, object]:
    return {"line": line_number, "status": status, "diagnostic": diagnostic}


def _completed(command: list[str], stdout: str, stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_compare_invokes_each_selector_once_with_complete_unchanged_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_text = (
        "\r\n"
        '{"case":"same"}\r\n'
        " \t\r\n"
        '{"case":"different"}\r\n'
        '{"case":"both-no-match"}\n'
        '{"case":"native-unsupported"}\n'
        '{"case":"native-missing"}\n'
        '{"case":"both-error"}'
    )
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_bytes(source_text.encode("utf-8"))
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any):
        calls.append((command, kwargs))
        if command[0] == sys.executable:
            stdout = _jsonl(
                _match(2),
                _match(4, "route.python"),
                _failure(5, "NO_MATCH", "no Python route"),
                _match(6),
                _match(7),
                _failure(8, "ERROR", "bad Python query"),
            )
        else:
            stdout = _jsonl(
                _match(2),
                _match(4, "route.native"),
                _failure(5, "NO_MATCH", "no native route"),
                _failure(6, "UNSUPPORTED", "native cannot evaluate op"),
                _failure(8, "ERROR", "bad native query"),
            )
        return _completed(command, stdout)

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
        timeout_seconds=0.25,
    )

    assert len(calls) == 2
    assert [call[1]["input"] for call in calls] == [source_text, source_text]
    assert calls[0][0] == [
        sys.executable,
        str(tmp_path / "python-selector.py"),
        "--input",
        "-",
        "--routing-dir",
        str(tmp_path / "routing"),
        "--batch",
    ]
    assert calls[1][0] == [
        str(tmp_path / "native-selector"),
        "--input",
        "-",
        "--batch",
    ]
    for _, kwargs in calls:
        assert kwargs == {
            "input": source_text,
            "text": True,
            "capture_output": True,
            "check": False,
            "timeout": 0.25,
        }

    assert not report.has_global_errors
    assert report.checked_count == 6
    assert report.skipped_blank_count == 2
    assert [failure.line_number for failure in report.discrepancies] == [4, 5, 6, 7, 8]
    assert report.discrepancies[3].reason == "native selector omitted the result row"
    assert report.discrepancies[4].python_result is not None
    assert report.discrepancies[4].python_result.status == "ERROR"

    diagnostic = format_route_selector_parity_report(report)
    assert f"{query_path}:4" in diagnostic
    assert f"{query_path}:8" in diagnostic
    assert "Python selector result: MATCH route_id='route.python'" in diagnostic
    assert "native selector result: UNSUPPORTED" in diagnostic
    assert "native selector result: <missing>" in diagnostic


def test_compare_reports_missing_rows_against_source_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text('{"case":1}\n\n{"case":2}\n', encoding="utf-8")
    call_count = 0

    def fake_run(command: list[str], **_: Any):
        nonlocal call_count
        call_count += 1
        stdout = _jsonl(_match(1)) if command[0] == sys.executable else _jsonl(_match(1), _match(3))
        return _completed(command, stdout)

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
    )

    assert call_count == 2
    assert not report.has_global_errors
    assert len(report.discrepancies) == 1
    assert report.discrepancies[0].line_number == 3
    assert report.discrepancies[0].reason == "Python selector omitted the result row"


@pytest.mark.parametrize(
    ("bad_stdout", "expected_error"),
    [
        ("not JSON\n", "invalid JSON"),
        ("[]\n", "record must be a JSON object"),
        ('{"line":1,"status":"MAYBE","diagnostic":"x"}\n', "field 'status'"),
        ('{"line":1,"status":"MATCH"}\n', "MATCH record must contain exactly"),
        (
            '{"line":1,"status":"MATCH","route_id":"route","diagnostic":"x"}\n',
            "MATCH record must contain exactly",
        ),
        (
            '{"line":1,"status":"NO_MATCH","route_id":"route"}\n',
            "NO_MATCH record must contain exactly",
        ),
        (
            '{"line":true,"status":"MATCH","route_id":"route"}\n',
            "field 'line' must be a positive integer",
        ),
        (
            '{"line":1,"line":1,"status":"MATCH","route_id":"route"}\n',
            "duplicate field 'line'",
        ),
        ("\n" + _jsonl(_match(1)), "blank record"),
        (_jsonl(_match(1), _match(1)), "duplicate result for source line 1"),
        (_jsonl(_match(2)), "unexpected result for source line 2"),
    ],
)
def test_compare_rejects_malformed_selector_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_stdout: str,
    expected_error: str,
) -> None:
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text("{}\n", encoding="utf-8")

    def fake_run(command: list[str], **_: Any):
        stdout = bad_stdout if command[0] == sys.executable else _jsonl(_match(1))
        return _completed(command, stdout)

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
    )

    assert report.has_global_errors
    assert any(expected_error in error for error in report.python_process.protocol_errors)
    assert not report.native_process.global_errors


def test_compare_rejects_out_of_order_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text("{}\n{}\n{}\n", encoding="utf-8")

    def fake_run(command: list[str], **_: Any):
        if command[0] == sys.executable:
            return _completed(command, _jsonl(_match(1), _match(3), _match(2)))
        return _completed(command, _jsonl(_match(1), _match(2), _match(3)))

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
    )

    assert report.python_process.protocol_errors == (
        "output line 3: result for source line 2 is out of order",
    )
    assert [failure.line_number for failure in report.discrepancies] == [2]


@pytest.mark.parametrize(
    ("python_result", "native_result", "expected_python_error", "expected_native_error"),
    [
        (
            subprocess.TimeoutExpired(
                ["python-selector"],
                0.25,
                output="partial output",
                stderr="partial error",
            ),
            FileNotFoundError("native selector is missing"),
            "timed out after 0.25 seconds",
            "could not launch selector: native selector is missing",
        ),
        (
            subprocess.CompletedProcess([], 2, "partial output", "fatal setup\n"),
            subprocess.CompletedProcess([], 0, _jsonl(_match(1)), "warning\n"),
            "exited with return code 2",
            "printed unexpected stderr",
        ),
    ],
)
def test_compare_accumulates_global_process_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_result: subprocess.CompletedProcess[str] | BaseException,
    native_result: subprocess.CompletedProcess[str] | BaseException,
    expected_python_error: str,
    expected_native_error: str,
) -> None:
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text("{}\n", encoding="utf-8")
    calls = 0

    def fake_run(command: list[str], **_: Any):
        nonlocal calls
        calls += 1
        outcome = python_result if command[0] == sys.executable else native_result
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(
            command,
            outcome.returncode,
            outcome.stdout,
            outcome.stderr,
        )

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
        timeout_seconds=0.25,
    )

    assert calls == 2
    assert report.has_global_errors
    assert expected_python_error in report.python_process.global_errors
    assert expected_native_error in report.native_process.global_errors
    assert report.discrepancies == ()
    diagnostic = format_route_selector_parity_report(report)
    assert expected_python_error in diagnostic
    assert expected_native_error in diagnostic
    assert "partial output" in diagnostic


def test_compare_empty_input_still_invokes_both_selectors_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_path = tmp_path / "empty.jsonl"
    query_path.write_text("", encoding="utf-8")
    calls: list[str] = []

    def fake_run(command: list[str], **kwargs: Any):
        calls.append(kwargs["input"])
        return _completed(command, "")

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
    )

    assert calls == ["", ""]
    assert report.checked_count == 0
    assert report.passed


def test_compare_treats_unicode_whitespace_as_a_nonblank_source_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text("\N{NO-BREAK SPACE}\n", encoding="utf-8")

    def fake_run(command: list[str], **_: Any):
        return _completed(command, _jsonl(_failure(1, "ERROR", "malformed JSON")))

    monkeypatch.setattr(route_selector_parity.subprocess, "run", fake_run)
    report = compare_route_selectors(
        route_queries_path=query_path,
        routing_dir=tmp_path / "routing",
        python_selector_path=tmp_path / "python-selector.py",
        native_selector_path=tmp_path / "native-selector",
    )

    assert report.checked_count == 1
    assert report.skipped_blank_count == 0
    assert [failure.line_number for failure in report.discrepancies] == [1]


_FAKE_SELECTOR_SOURCE = r"""#!/usr/bin/env python3
import base64
import json
import os
import sys
from pathlib import Path

payload = sys.stdin.buffer.read()
selector_name = Path(sys.argv[0]).name
selector_kind = "python" if selector_name.startswith("python-") else "native"
log_path = Path(os.environ["ROUTE_SELECTOR_PARITY_TEST_LOG_DIR"]) / f"{selector_kind}.log"
with log_path.open("a", encoding="ascii") as log_file:
    log_file.write(base64.b64encode(payload).decode("ascii") + "\n")

for line_number, raw_line in enumerate(payload.splitlines(), start=1):
    if not raw_line.strip():
        continue
    case = json.loads(raw_line)["case"]
    if case == "missing" and selector_kind == "python":
        continue
    if case == "malformed-output" and selector_kind == "python":
        print("not JSON")
        continue
    if case == "different":
        record = {
            "line": line_number,
            "status": "MATCH",
            "route_id": "route.python" if selector_kind == "python" else "route.native",
        }
    elif case == "no-match" and selector_kind == "native":
        record = {
            "line": line_number,
            "status": "NO_MATCH",
            "diagnostic": "no native route",
        }
    else:
        record = {"line": line_number, "status": "MATCH", "route_id": "route.shared"}
    print(json.dumps(record, separators=(",", ":")))
"""


def _write_fake_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setenv("ROUTE_SELECTOR_PARITY_TEST_LOG_DIR", str(log_dir))

    python_selector = tmp_path / "python-selector.py"
    native_selector = tmp_path / "native-selector"
    python_selector.write_text(textwrap.dedent(_FAKE_SELECTOR_SOURCE), encoding="utf-8")
    native_selector.write_text(textwrap.dedent(_FAKE_SELECTOR_SOURCE), encoding="utf-8")
    native_selector.chmod(0o755)
    return python_selector, native_selector, log_dir


def _recorded_inputs(log_path: Path) -> list[bytes]:
    return [base64.b64decode(line) for line in log_path.read_text(encoding="ascii").splitlines()]


def _run_checker(
    *,
    query_path: Path,
    routing_dir: Path,
    python_selector: Path,
    native_selector: Path,
    enforce_router_parity: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop(ENFORCE_ROUTER_PARITY_ENV, None)
    if enforce_router_parity is not None:
        environment[ENFORCE_ROUTER_PARITY_ENV] = enforce_router_parity
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--route-queries",
            str(query_path),
            "--routing-dir",
            str(routing_dir),
            "--python-selector",
            str(python_selector),
            "--native-selector",
            str(native_selector),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def test_checker_cli_passes_and_sends_complete_input_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_selector, native_selector, log_dir = _write_fake_selectors(tmp_path, monkeypatch)
    source = b'\r\n{"case":"same"}\r\n \t\n{"case":"same"}'
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_bytes(source)
    routing_dir = tmp_path / "routing"
    routing_dir.mkdir()

    result = _run_checker(
        query_path=query_path,
        routing_dir=routing_dir,
        python_selector=python_selector,
        native_selector=native_selector,
    )

    assert result.returncode == 0
    assert result.stdout == "Route selector parity passed for 2 RouteQuery row(s).\n"
    assert result.stderr == ""
    assert _recorded_inputs(log_dir / "python.log") == [source]
    assert _recorded_inputs(log_dir / "native.log") == [source]


@pytest.mark.parametrize(
    ("enforce_router_parity", "expected_returncode"),
    [(None, 77), ("", 1)],
)
def test_checker_cli_prints_all_line_discrepancies_and_honors_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enforce_router_parity: str | None,
    expected_returncode: int,
) -> None:
    python_selector, native_selector, _ = _write_fake_selectors(tmp_path, monkeypatch)
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text(
        '{"case":"different"}\n\n{"case":"no-match"}\n{"case":"missing"}\n',
        encoding="utf-8",
    )
    routing_dir = tmp_path / "routing"
    routing_dir.mkdir()

    result = _run_checker(
        query_path=query_path,
        routing_dir=routing_dir,
        python_selector=python_selector,
        native_selector=native_selector,
        enforce_router_parity=enforce_router_parity,
    )

    assert result.returncode == expected_returncode
    assert result.stdout == ""
    assert "failed for 3 of 3 RouteQuery row(s)" in result.stderr
    assert f"{query_path}:1" in result.stderr
    assert f"{query_path}:3" in result.stderr
    assert f"{query_path}:4" in result.stderr
    assert "selected different routes" in result.stderr
    assert "native selector returned NO_MATCH: no native route" in result.stderr
    assert "Python selector result: <missing>" in result.stderr
    if enforce_router_parity is None:
        assert f"set {ENFORCE_ROUTER_PARITY_ENV}" in result.stderr
    else:
        assert f"set {ENFORCE_ROUTER_PARITY_ENV}" not in result.stderr


def test_checker_cli_returns_two_for_malformed_selector_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_selector, native_selector, _ = _write_fake_selectors(tmp_path, monkeypatch)
    query_path = tmp_path / "route-queries.jsonl"
    query_path.write_text('{"case":"malformed-output"}\n', encoding="utf-8")
    routing_dir = tmp_path / "routing"
    routing_dir.mkdir()

    result = _run_checker(
        query_path=query_path,
        routing_dir=routing_dir,
        python_selector=python_selector,
        native_selector=native_selector,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "selector process or protocol failed" in result.stderr
    assert "Python selector:" in result.stderr
    assert "protocol error: output line 1: invalid JSON" in result.stderr


def test_checker_cli_returns_two_when_jsonl_cannot_be_read(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"
    result = _run_checker(
        query_path=missing_path,
        routing_dir=tmp_path / "routing",
        python_selector=tmp_path / "python-selector.py",
        native_selector=tmp_path / "native-selector",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert f"cannot read RouteQuery JSONL {missing_path}" in result.stderr
