import json

import pytest

from mlea.records import (
    Fingerprint,
    IncomparableError,
    RunRecord,
    RunSet,
    assert_comparable,
    common_competitions,
)


def make_set(label, split, medals, seeds=1, config="default"):
    runs = [
        RunRecord(competition_id=c, seed=s, medal=m)
        for c, m in medals.items()
        for s in range(seeds)
    ]
    return RunSet(label, Fingerprint(split_id=split, container_config=config), runs)


def test_refuses_to_compare_different_splits():
    """The 65.3%-vs-68.2% failure mode: full-set and lite numbers are not comparable."""
    a = make_set("mlevolve", "split75", {"c1": True})
    b = make_set("rd-agent", "low", {"c1": True})
    with pytest.raises(IncomparableError, match="split_id"):
        assert_comparable(a, b)


def test_refuses_to_compare_different_container_configs():
    a = make_set("a", "low", {"c1": True}, config="reference")
    b = make_set("b", "low", {"c1": True}, config="cheap-8gb")
    with pytest.raises(IncomparableError, match="container_config"):
        assert_comparable(a, b)


def test_allows_mismatch_only_when_explicitly_requested():
    a = make_set("pre", "pre-cutoff", {"c1": True})
    b = make_set("post", "post-cutoff", {"c2": True})
    assert_comparable(a, b, allow_fingerprint_mismatch=True)


def test_reruns_are_averaged_not_replaced():
    """Re-running until it medals must not work."""
    runs = [
        RunRecord("c1", seed=0, medal=False),
        RunRecord("c1", seed=0, medal=False),
        RunRecord("c1", seed=0, medal=True),  # the lucky re-run
    ]
    rs = RunSet("a", Fingerprint("low"), runs)
    assert rs.medal_rate_by_competition()["c1"] == pytest.approx(1 / 3)


def test_infra_failures_excluded_from_capability():
    runs = [
        RunRecord("c1", 0, medal=True),
        RunRecord("c1", 1, medal=False, infra_failure=True),
    ]
    rs = RunSet("a", Fingerprint("low"), runs)
    assert rs.medal_rate_by_competition()["c1"] == 1.0
    assert rs.n_infra_failures == 1


def test_infra_failure_cannot_have_medal():
    with pytest.raises(ValueError, match="infra failure"):
        RunRecord("c1", 0, medal=True, infra_failure=True)


def test_headline_rate_weights_competitions_equally():
    """Extra seeds on one competition must not give it extra weight."""
    runs = [RunRecord("c1", s, medal=True) for s in range(10)]
    runs += [RunRecord("c2", 0, medal=False)]
    rs = RunSet("a", Fingerprint("low"), runs)
    assert rs.any_medal_rate() == pytest.approx(0.5)


def test_common_competitions_requires_overlap():
    a = make_set("a", "low", {"c1": True})
    b = make_set("b", "low", {"c2": True})
    with pytest.raises(IncomparableError, match="share no competitions"):
        common_competitions(a, b)


def test_from_json_requires_split_id(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"fingerprint": {}, "runs": []}))
    with pytest.raises(IncomparableError, match="split_id is required"):
        RunSet.from_json(p)


def test_from_json_roundtrip(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "label": "aide",
                "fingerprint": {"split_id": "low", "harness_version": "v1"},
                "runs": [
                    {"competition_id": "c1", "seed": 0, "any_medal": True},
                    {"competition_id": "c1", "seed": 1, "any_medal": False},
                ],
            }
        )
    )
    rs = RunSet.from_json(p)
    assert rs.label == "aide"
    assert rs.fingerprint.split_id == "low"
    assert rs.medal_rate_by_competition()["c1"] == pytest.approx(0.5)
