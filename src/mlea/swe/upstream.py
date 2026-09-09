"""Adapter to the real SWE-bench grader.

Optional, like the MLE-bench adapter. When ``swebench`` is importable, the same
log this package grades is also graded by upstream's ``get_eval_report`` -- the
real log parser, the real FAIL_TO_PASS/PASS_TO_PASS logic, the real resolution
rule -- and the two verdicts are compared.

Grading needs no Docker. Upstream's containers exist to *produce* a test log
safely; deciding what a log means is log parsing plus set comparison, and runs
anywhere.
"""

from __future__ import annotations

from pathlib import Path


class UpstreamUnavailable(RuntimeError):
    pass


INSTALL_HINT = "upstream swebench is not importable. Install it with `pip install swebench`."


def available() -> bool:
    try:
        import swebench.harness.grading  # noqa: F401
        import swebench.types  # noqa: F401
    except Exception:
        return False
    return True


def _require() -> None:
    if not available():
        raise UpstreamUnavailable(INSTALL_HINT)


def parsers() -> list[str]:
    _require()
    from swebench.harness.grading import PARSER_REGISTRY

    return sorted(PARSER_REGISTRY)


def grade_with_upstream(
    log_path: str | Path, instance_id: str,
    fail_to_pass: list[str], pass_to_pass: list[str],
    *, log_parser: str = "parse_log_pytest",
) -> dict:
    """Grade a log with the real SWE-bench grader.

    Returns upstream's own report dict for the instance -- ``resolved`` plus the
    per-set success/failure breakdown -- so a comparison is against the real
    structure and not a summary of it.
    """
    _require()
    from swebench.harness.grading import get_eval_report
    from swebench.types import TestSpec

    spec = TestSpec(
        instance_id=instance_id,
        image="none",
        eval_script_list=[],
        repo="synthetic/repo",
        version="1.0",
        FAIL_TO_PASS=list(fail_to_pass),
        PASS_TO_PASS=list(pass_to_pass),
        log_parser=log_parser,
        eval_type="pass_and_fail",
    )
    prediction = {
        "instance_id": instance_id,
        "model_patch": "x",
        "model_name_or_path": "mlea",
    }
    report = get_eval_report(spec, prediction, str(log_path), True)
    return report[instance_id]


__all__ = [
    "INSTALL_HINT",
    "UpstreamUnavailable",
    "available",
    "grade_with_upstream",
    "parsers",
]
