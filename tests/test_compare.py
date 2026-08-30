import pytest

from mlea.compare import compare
from mlea.records import Fingerprint, IncomparableError, RunRecord, RunSet


def rs(label, split, medals, seeds=3, config="default"):
    """medals: {competition: n_medals_out_of_seeds}"""
    runs = [
        RunRecord(c, s, medal=(s < k)) for c, k in medals.items() for s in range(seeds)
    ]
    return RunSet(label, Fingerprint(split_id=split, container_config=config), runs)


def test_identical_arms_show_no_difference():
    m = {f"c{i}": 2 for i in range(22)}
    r = compare(rs("a", "low", m), rs("b", "low", m))
    assert r.difference == pytest.approx(0.0)
    assert not r.significant


def test_large_uniform_improvement_is_detected():
    lo = {f"c{i}": 0 for i in range(22)}
    hi = {f"c{i}": 3 for i in range(22)}
    r = compare(rs("weak", "low", lo), rs("strong", "low", hi))
    assert r.difference == pytest.approx(1.0)
    assert r.significant


def test_two_competition_swing_on_lite_is_not_significant():
    """The headline case from PLAN.md 6: 2 of 22 is ~9 points and is noise."""
    base = {f"c{i}": 3 if i < 11 else 0 for i in range(22)}
    better = dict(base)
    better["c11"] = 3
    better["c12"] = 3
    r = compare(rs("a", "low", base), rs("b", "low", better))
    assert r.difference == pytest.approx(2 / 22)
    assert not r.significant, "a 2-of-22 swing must not read as a real improvement"


def test_split_mismatch_raises_by_default():
    with pytest.raises(IncomparableError):
        compare(rs("a", "split75", {"c1": 3}), rs("b", "low", {"c1": 3}))


def test_matched_pairs_permit_split_mismatch():
    pre = rs("pre", "pre-cutoff", {f"p{i}": 3 for i in range(8)})
    post = rs("post", "post-cutoff", {f"q{i}": 0 for i in range(8)})
    pairs = [(f"p{i}", f"q{i}") for i in range(8)]
    r = compare(pre, post, pairs=pairs)
    assert r.matched
    assert r.difference == pytest.approx(-1.0)
    assert r.significant


def test_matched_design_warns_about_residual_mismatch():
    pre = rs("pre", "pre", {f"p{i}": 2 for i in range(8)})
    post = rs("post", "post", {f"q{i}": 1 for i in range(8)})
    r = compare(pre, post, pairs=[(f"p{i}", f"q{i}") for i in range(8)])
    assert any("difficulty mismatch" in w for w in r.warnings)


def test_matched_pairs_missing_runs_raise():
    pre = rs("pre", "pre", {"p0": 3})
    post = rs("post", "post", {"q0": 3})
    with pytest.raises(ValueError, match="no gradeable runs"):
        compare(pre, post, pairs=[("p0", "q0"), ("p1", "q1")])


def test_underpowered_design_is_flagged():
    m = {f"c{i}": 0 for i in range(4)}
    better = {f"c{i}": 3 for i in range(4)}
    r = compare(rs("a", "low", m), rs("b", "low", better))
    assert r.underpowered_by_construction
    assert not r.significant
    assert "cannot produce" in r.summary()


def test_non_overlapping_competitions_are_dropped_with_a_warning():
    a = rs("a", "low", {f"c{i}": 3 for i in range(10)})
    b = rs("b", "low", {f"c{i}": 3 for i in range(5)})
    r = compare(a, b)
    assert r.n_units == 5
    assert any("dropped" in w for w in r.warnings)


def test_low_seed_count_warns():
    m = {f"c{i}": 1 for i in range(22)}
    r = compare(rs("a", "low", m, seeds=1), rs("b", "low", m, seeds=1))
    assert any("seed(s) per competition" in w for w in r.warnings)


def test_infra_failures_are_reported_and_excluded():
    runs_a = [RunRecord("c1", s, medal=True) for s in range(3)]
    runs_b = [RunRecord("c1", 0, medal=True), RunRecord("c1", 1, False, True)]
    a = RunSet("a", Fingerprint("low"), runs_a)
    b = RunSet("b", Fingerprint("low"), runs_b)
    r = compare(a, b)
    assert r.rate_b == pytest.approx(1.0)
    assert any("infra failure" in w for w in r.warnings)


def test_comparison_is_deterministic_given_a_seed():
    m1 = {f"c{i}": i % 4 for i in range(22)}
    m2 = {f"c{i}": (i + 1) % 4 for i in range(22)}
    a, b = rs("a", "low", m1), rs("b", "low", m2)
    assert compare(a, b, seed=1).p_value == compare(a, b, seed=1).p_value


def test_summary_renders():
    m = {f"c{i}": 2 for i in range(22)}
    text = compare(rs("a", "low", m), rs("b", "low", m)).summary()
    assert "difference" in text and "p (paired perm)" in text
