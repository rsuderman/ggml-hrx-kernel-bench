"""Compare native and Python route selectors over RouteQuery JSONL input."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SELECTOR_TIMEOUT_SECONDS = 10.0

_DIAGNOSTIC_STATUSES = frozenset({"NO_MATCH", "UNSUPPORTED", "ERROR"})
_SUPPORTED_STATUSES = frozenset({"MATCH", *_DIAGNOSTIC_STATUSES})
_BLANK_LINE_CHARACTERS = " \t\r\n\v\f"

ENFORCE_ROUTER_PARITY_ENV = "ENFORCE_ROUTER_PARITY"
PARITY_MISMATCH_SKIP_RETURN_CODE = 77


@dataclass(frozen=True)
class SelectorRowResult:
    line_number: int
    status: str
    route_id: str | None = None
    diagnostic: str | None = None


@dataclass(frozen=True)
class SelectorProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    process_error: str | None = None
    rows: tuple[SelectorRowResult, ...] = ()
    protocol_errors: tuple[str, ...] = ()

    @property
    def global_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.process_error is not None:
            errors.append(self.process_error)
        elif self.returncode != 0:
            errors.append(f"exited with return code {self.returncode}")
        if self.stderr:
            errors.append("printed unexpected stderr")
        errors.extend(self.protocol_errors)
        return tuple(errors)


@dataclass(frozen=True)
class RouteSelectorDiscrepancy:
    line_number: int
    raw_json: str
    reason: str
    python_result: SelectorRowResult | None
    native_result: SelectorRowResult | None


@dataclass(frozen=True)
class RouteSelectorParityReport:
    route_queries_path: Path
    checked_count: int
    skipped_blank_count: int
    discrepancies: tuple[RouteSelectorDiscrepancy, ...]
    python_process: SelectorProcessResult
    native_process: SelectorProcessResult

    @property
    def has_global_errors(self) -> bool:
        return bool(self.python_process.global_errors or self.native_process.global_errors)

    @property
    def passed(self) -> bool:
        return not self.has_global_errors and not self.discrepancies


class _DuplicateJsonKey(ValueError):
    pass


def _exception_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_selector(
    command: Sequence[str],
    *,
    source_text: str,
    timeout_seconds: float,
) -> SelectorProcessResult:
    try:
        result = subprocess.run(
            list(command),
            input=source_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return SelectorProcessResult(
            returncode=None,
            stdout=_exception_output(exc.stdout),
            stderr=_exception_output(exc.stderr),
            process_error=f"timed out after {timeout_seconds:g} seconds",
        )
    except (OSError, UnicodeError) as exc:
        return SelectorProcessResult(
            returncode=None,
            stdout="",
            stderr="",
            process_error=f"could not launch selector: {exc}",
        )
    return SelectorProcessResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate field {key!r}")
        result[key] = value
    return result


def _parse_row(output_line: str) -> SelectorRowResult:
    try:
        payload = json.loads(output_line, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, _DuplicateJsonKey) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("record must be a JSON object")

    line_number = payload.get("line")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1:
        raise ValueError("field 'line' must be a positive integer")

    status = payload.get("status")
    if not isinstance(status, str) or status not in _SUPPORTED_STATUSES:
        raise ValueError(
            "field 'status' must be MATCH, NO_MATCH, UNSUPPORTED, or ERROR"
        )

    if status == "MATCH":
        expected_fields = {"line", "status", "route_id"}
        if set(payload) != expected_fields:
            raise ValueError(
                "MATCH record must contain exactly 'line', 'status', and 'route_id'"
            )
        route_id = payload["route_id"]
        if (
            not isinstance(route_id, str)
            or not route_id
            or route_id != route_id.strip()
        ):
            raise ValueError("field 'route_id' must be a nonempty, trimmed string")
        return SelectorRowResult(
            line_number=line_number,
            status=status,
            route_id=route_id,
        )

    expected_fields = {"line", "status", "diagnostic"}
    if set(payload) != expected_fields:
        raise ValueError(
            f"{status} record must contain exactly 'line', 'status', and 'diagnostic'"
        )
    diagnostic = payload["diagnostic"]
    if not isinstance(diagnostic, str) or not diagnostic:
        raise ValueError("field 'diagnostic' must be a nonempty string")
    return SelectorRowResult(
        line_number=line_number,
        status=status,
        diagnostic=diagnostic,
    )


def _validate_selector_output(
    result: SelectorProcessResult,
    *,
    expected_line_numbers: tuple[int, ...],
) -> SelectorProcessResult:
    # Partial output from a process that did not complete cannot be treated as a
    # result stream. Preserve it for diagnostics, but do not manufacture missing
    # row discrepancies for every query.
    if result.process_error is not None or result.returncode != 0 or result.stderr:
        return result

    expected_positions = {
        line_number: position
        for position, line_number in enumerate(expected_line_numbers)
    }
    seen_line_numbers: set[int] = set()
    last_position = -1
    rows: list[SelectorRowResult] = []
    protocol_errors: list[str] = []

    for output_line_number, output_line in enumerate(result.stdout.splitlines(), start=1):
        if not output_line:
            protocol_errors.append(f"output line {output_line_number}: blank record")
            continue
        try:
            row = _parse_row(output_line)
        except ValueError as exc:
            protocol_errors.append(f"output line {output_line_number}: {exc}")
            continue

        if row.line_number in seen_line_numbers:
            protocol_errors.append(
                f"output line {output_line_number}: duplicate result for source line "
                f"{row.line_number}"
            )
            continue
        seen_line_numbers.add(row.line_number)

        position = expected_positions.get(row.line_number)
        if position is None:
            protocol_errors.append(
                f"output line {output_line_number}: unexpected result for source line "
                f"{row.line_number}"
            )
            continue
        if position <= last_position:
            protocol_errors.append(
                f"output line {output_line_number}: result for source line "
                f"{row.line_number} is out of order"
            )
            continue
        last_position = position
        rows.append(row)

    return SelectorProcessResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        process_error=result.process_error,
        rows=tuple(rows),
        protocol_errors=tuple(protocol_errors),
    )


def _describe_row(name: str, row: SelectorRowResult | None) -> str:
    if row is None:
        return f"{name} selector omitted the result row"
    if row.status == "MATCH":
        return f"{name} selector returned MATCH route_id={row.route_id!r}"
    return f"{name} selector returned {row.status}: {row.diagnostic}"


def _discrepancy_reason(
    python_result: SelectorRowResult | None,
    native_result: SelectorRowResult | None,
) -> str | None:
    if python_result is None or native_result is None:
        return "; ".join(
            description
            for description, result in (
                (_describe_row("Python", python_result), python_result),
                (_describe_row("native", native_result), native_result),
            )
            if result is None
        )
    if python_result.status == "MATCH" and native_result.status == "MATCH":
        if python_result.route_id == native_result.route_id:
            return None
        return (
            f"selected different routes: Python={python_result.route_id!r}, "
            f"native={native_result.route_id!r}"
        )
    return "; ".join(
        (
            _describe_row("Python", python_result),
            _describe_row("native", native_result),
        )
    )


def _read_route_queries(path: Path) -> tuple[str, tuple[tuple[int, str], ...], int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as query_file:
            source_text = query_file.read()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read RouteQuery JSONL {path}: {exc}") from exc

    nonblank_lines: list[tuple[int, str]] = []
    skipped_blank_count = 0
    for line_number, raw_json in enumerate(io.StringIO(source_text, newline=""), start=1):
        if raw_json.strip(_BLANK_LINE_CHARACTERS):
            nonblank_lines.append((line_number, raw_json))
        else:
            skipped_blank_count += 1
    return source_text, tuple(nonblank_lines), skipped_blank_count


def compare_route_selectors(
    *,
    route_queries_path: Path,
    routing_dir: Path,
    python_selector_path: Path,
    native_selector_path: Path,
    timeout_seconds: float = DEFAULT_SELECTOR_TIMEOUT_SECONDS,
) -> RouteSelectorParityReport:
    source_text, source_rows, skipped_blank_count = _read_route_queries(route_queries_path)
    expected_line_numbers = tuple(line_number for line_number, _ in source_rows)

    python_command = (
        sys.executable,
        str(python_selector_path),
        "--input",
        "-",
        "--routing-dir",
        str(routing_dir),
        "--batch",
    )
    native_command = (str(native_selector_path), "--input", "-", "--batch")

    python_process = _validate_selector_output(
        _run_selector(
            python_command,
            source_text=source_text,
            timeout_seconds=timeout_seconds,
        ),
        expected_line_numbers=expected_line_numbers,
    )
    native_process = _validate_selector_output(
        _run_selector(
            native_command,
            source_text=source_text,
            timeout_seconds=timeout_seconds,
        ),
        expected_line_numbers=expected_line_numbers,
    )

    python_rows = {row.line_number: row for row in python_process.rows}
    native_rows = {row.line_number: row for row in native_process.rows}
    python_stream_available = not (
        python_process.process_error
        or python_process.returncode != 0
        or python_process.stderr
    )
    native_stream_available = not (
        native_process.process_error
        or native_process.returncode != 0
        or native_process.stderr
    )

    discrepancies: list[RouteSelectorDiscrepancy] = []
    if python_stream_available and native_stream_available:
        for line_number, raw_json in source_rows:
            python_result = python_rows.get(line_number)
            native_result = native_rows.get(line_number)
            reason = _discrepancy_reason(python_result, native_result)
            if reason is not None:
                discrepancies.append(
                    RouteSelectorDiscrepancy(
                        line_number=line_number,
                        raw_json=raw_json,
                        reason=reason,
                        python_result=python_result,
                        native_result=native_result,
                    )
                )

    return RouteSelectorParityReport(
        route_queries_path=route_queries_path,
        checked_count=len(source_rows),
        skipped_blank_count=skipped_blank_count,
        discrepancies=tuple(discrepancies),
        python_process=python_process,
        native_process=native_process,
    )


def _format_process_result(name: str, result: SelectorProcessResult) -> list[str]:
    return_code = "<not available>" if result.returncode is None else str(result.returncode)
    lines = [
        f"{name} selector:",
        f"  return code: {return_code}",
        f"  stdout: {result.stdout!r}",
        f"  stderr: {result.stderr!r}",
    ]
    if result.process_error is not None:
        lines.append(f"  process error: {result.process_error}")
    elif result.returncode != 0:
        lines.append(f"  process error: exited with return code {result.returncode}")
    if result.stderr:
        lines.append("  process error: printed unexpected stderr")
    for error in result.protocol_errors:
        lines.append(f"  protocol error: {error}")
    return lines


def _format_row_result(name: str, result: SelectorRowResult | None) -> str:
    if result is None:
        return f"  {name} result: <missing>"
    if result.status == "MATCH":
        detail = f"route_id={result.route_id!r}"
    else:
        detail = f"diagnostic={result.diagnostic!r}"
    return f"  {name} result: {result.status} {detail}"


def format_route_selector_parity_report(report: RouteSelectorParityReport) -> str:
    lines: list[str] = []
    if report.has_global_errors:
        lines.append("Route selector parity failed because a selector process or protocol failed.")
        if report.python_process.global_errors:
            lines.extend(("", *_format_process_result("Python", report.python_process)))
        if report.native_process.global_errors:
            lines.extend(("", *_format_process_result("native", report.native_process)))

    if report.discrepancies:
        if lines:
            lines.append("")
        lines.append(
            f"Route selector parity failed for {len(report.discrepancies)} "
            f"of {report.checked_count} RouteQuery row(s)."
        )
        for discrepancy in report.discrepancies:
            display_json = discrepancy.raw_json.rstrip("\r\n")
            lines.extend(
                [
                    "",
                    f"{report.route_queries_path}:{discrepancy.line_number}",
                    f"  reason: {discrepancy.reason}",
                    f"  raw JSON: {display_json}",
                    _format_row_result("Python selector", discrepancy.python_result),
                    _format_row_result("native selector", discrepancy.native_result),
                ]
            )
    if not lines:
        return "Route selector parity passed."
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Python and native route selection for RouteQuery JSONL rows."
    )
    parser.add_argument("--route-queries", type=Path, required=True)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--python-selector", type=Path, required=True)
    parser.add_argument("--native-selector", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = compare_route_selectors(
            route_queries_path=args.route_queries,
            routing_dir=args.routing_dir,
            python_selector_path=args.python_selector,
            native_selector_path=args.native_selector,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if report.has_global_errors:
        print(format_route_selector_parity_report(report), file=sys.stderr)
        return 2

    if report.discrepancies:
        print(format_route_selector_parity_report(report), file=sys.stderr)
        if ENFORCE_ROUTER_PARITY_ENV in os.environ:
            return 1
        print(
            f"note: set {ENFORCE_ROUTER_PARITY_ENV} to make route mismatches fail",
            file=sys.stderr,
        )
        return PARITY_MISMATCH_SKIP_RETURN_CODE

    print(f"Route selector parity passed for {report.checked_count} RouteQuery row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
