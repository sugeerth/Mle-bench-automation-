"""SWE-bench: instance generation, resolution, and conformance with upstream."""

import os
import shutil

import pytest

from mlea.swe import (
    EXPECTED,
    STRATEGIES,
    SweSpec,
    grade_log,
    make_instance,
    make_suite,
    patch_for,
    run_agent,
    upstream,
)
from mlea.swe.bench import BUGS, BUGS_BY_NAME, InstanceInvalid
from mlea.swe.grade import APPLY_PATCH_FAIL, parse_pytest_statuses, run_tests
from mlea.swe.harness import apply_patch

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or os.name != "posix",
    reason="needs git on POSIX",
)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    return make_suite(tmp_path_factory.mktemp("swe"))


# --- the mechanical oracle ---


def test_every_instance_verifies_against_its_gold_patch(suite):
    """An instance is well-formed iff its gold patch flips FAIL_TO_PASS and
    leaves PASS_TO_PASS alone. make_instance refuses to emit one that does not,
    which is the correctness oracle a Kaggle-derived benchmark cannot have."""
    assert len(suite) == len(BUGS)
    for inst in suite:
        assert inst.fail_to_pass, inst.instance_id
        assert inst.pass_to_pass, inst.instance_id
        assert set(inst.fail_to_pass).isdisjoint(inst.pass_to_pass)
        assert inst.gold_patch.strip().startswith("diff --git")


def test_a_gold_patch_that_fixes_nothing_is_refused(tmp_path):
    from mlea.swe import bench

    dud = bench.Bug(
        name="dud", module="m.py", broken="def f():\n    return 1\n",
        fixed="def f():\n    return 1\n",  # identical: fixes nothing
        tests="from m import f\n\n\ndef test_f():\n    assert f() == 1\n",
        problem_statement="nothing is wrong",
    )
    bench.BUGS_BY_NAME["dud"] = dud
    try:
        with pytest.raises(InstanceInvalid, match="fixes nothing"):
            make_instance(SweSpec("synth__dud-1", "dud"), tmp_path)
    finally:
        del bench.BUGS_BY_NAME["dud"]


def test_a_gold_patch_that_breaks_a_passing_test_is_refused(tmp_path):
    from mlea.swe import bench

    saboteur = bench.Bug(
        name="sab", module="m.py",
        broken="def f():\n    return 1\n\n\ndef g():\n    return 2\n",
        fixed="def f():\n    return 9\n\n\ndef g():\n    return 0\n",
        tests=("from m import f, g\n\n\n"
               "def test_g():\n    assert g() == 2\n\n\n"
               "def test_f():\n    assert f() == 9\n"),
        problem_statement="f should return 9",
    )
    bench.BUGS_BY_NAME["sab"] = saboteur
    try:
        with pytest.raises(InstanceInvalid, match="breaks"):
            make_instance(SweSpec("synth__sab-1", "sab"), tmp_path)
    finally:
        del bench.BUGS_BY_NAME["sab"]


def test_unknown_bug_is_rejected():
    with pytest.raises(ValueError, match="unknown bug"):
        SweSpec("x", "not-a-bug")


# --- the resolution rule ---


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_reference_agent_hits_its_branch(suite, tmp_path, strategy):
    """Every branch of the rule is exercised by a real run, not a fixture."""
    inst = suite[0]
    log = run_agent(inst, patch_for(inst, strategy), workdir=tmp_path)
    report = grade_log(log, inst.fail_to_pass, inst.pass_to_pass, inst.instance_id)
    assert report.summary().startswith(EXPECTED[strategy].split(" ")[0])


def test_a_regressive_patch_is_not_a_resolution(suite, tmp_path):
    """The clause a naive harness omits: fixing the bug while breaking something
    else scores as resolved if only FAIL_TO_PASS is checked."""
    inst = suite[0]
    log = run_agent(inst, patch_for(inst, "regressive"), workdir=tmp_path)
    report = grade_log(log, inst.fail_to_pass, inst.pass_to_pass, inst.instance_id)
    assert not report.resolved
    assert report.regressed
    assert not report.f2p_failure, "the reported bug WAS fixed"
    assert report.p2p_failure, "...and something else broke"


def test_an_unapplied_patch_reports_no_test_results(suite, tmp_path):
    """Found by conformance: marking every test failed asserts something the log
    does not support, and would blame an agent whose patch never ran."""
    inst = suite[0]
    log = run_agent(inst, patch_for(inst, "broken"), workdir=tmp_path)
    report = grade_log(log, inst.fail_to_pass, inst.pass_to_pass, inst.instance_id)
    assert not report.patch_applied and not report.resolved
    assert report.f2p_failure == [] and report.p2p_failure == []
    assert report.f2p_success == [] and report.p2p_success == []


def test_a_missing_test_does_not_count_as_a_pass():
    """Scoring a suite that never ran as a resolution is the classic failure."""
    log = (">>>>> Start Test Output\nPASSED a.py::test_a\n"
           ">>>>> Test Exit Code: 0\n>>>>> End Test Output\n")
    report = grade_log(log, ["a.py::test_a"], ["a.py::test_absent"])
    assert not report.resolved
    assert report.p2p_failure == ["a.py::test_absent"]


def test_a_log_without_markers_is_not_a_result():
    report = grade_log("pytest exploded", ["t::a"], ["t::b"])
    assert not report.patch_applied and not report.resolved


def test_apply_patch_reports_why_it_failed(suite, tmp_path):
    work = tmp_path / "copy"
    shutil.copytree(suite[0].repo_dir, work)
    ok, message = apply_patch(work, "not a diff at all")
    assert not ok and message


def test_an_empty_patch_counts_as_applied(suite, tmp_path):
    work = tmp_path / "empty"
    shutil.copytree(suite[0].repo_dir, work)
    ok, _ = apply_patch(work, "")
    assert ok, "doing nothing is not a mechanical failure"


def test_runs_happen_on_a_copy(suite, tmp_path):
    """One agent's patch must not leak into the next agent's run."""
    inst = suite[0]
    before = (inst.repo_dir / inst.module).read_text()
    run_agent(inst, patch_for(inst, "gold"), workdir=tmp_path)
    assert (inst.repo_dir / inst.module).read_text() == before


def test_parser_needs_the_summary_lines(suite):
    """-rA is what makes pytest emit per-test PASSED/FAILED lines."""
    statuses = parse_pytest_statuses(run_tests(suite[0].repo_dir, suite[0].test_file))
    assert len(statuses) >= 3
    assert set(statuses.values()) <= {"PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"}


# --- conformance with the real grader ---


upstream_only = pytest.mark.skipif(
    not upstream.available(), reason="upstream swebench not importable"
)


@upstream_only
def test_upstream_exposes_its_parsers():
    assert len(upstream.parsers()) > 40
    assert "parse_log_pytest" in upstream.parsers()


@upstream_only
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_we_agree_with_the_real_grader(suite, tmp_path, strategy):
    """The check that makes this a SWE-bench harness: same log, both graders,
    full FAIL_TO_PASS and PASS_TO_PASS breakdown compared."""
    for inst in suite:
        log = run_agent(inst, patch_for(inst, strategy), workdir=tmp_path)
        log_path = tmp_path / f"{inst.instance_id}-{strategy}.log"
        log_path.write_text(log)

        ours = grade_log(log, inst.fail_to_pass, inst.pass_to_pass, inst.instance_id)
        theirs = upstream.grade_with_upstream(
            log_path, inst.instance_id, inst.fail_to_pass, inst.pass_to_pass)
        status = theirs.get("tests_status", {})

        assert ours.resolved == theirs["resolved"], inst.instance_id
        for key, ok, bad in (
            ("FAIL_TO_PASS", ours.f2p_success, ours.f2p_failure),
            ("PASS_TO_PASS", ours.p2p_success, ours.p2p_failure),
        ):
            block = status.get(key, {})
            assert set(ok) == set(block.get("success", [])), f"{inst.instance_id} {key}"
            assert set(bad) == set(block.get("failure", [])), f"{inst.instance_id} {key}"


@upstream_only
def test_gold_resolves_and_noop_does_not_for_upstream_too(suite, tmp_path):
    inst = suite[0]
    for strategy, expected in (("gold", True), ("noop", False)):
        log = run_agent(inst, patch_for(inst, strategy), workdir=tmp_path)
        p = tmp_path / f"{strategy}.log"
        p.write_text(log)
        theirs = upstream.grade_with_upstream(
            p, inst.instance_id, inst.fail_to_pass, inst.pass_to_pass)
        assert theirs["resolved"] is expected
