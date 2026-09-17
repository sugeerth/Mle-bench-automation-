import json

import pytest

from mlea.triage import (
    Outcome,
    RetryNotAllowed,
    RunArtifacts,
    TriageReport,
    assert_retry_allowed,
    classify,
    from_run_dir,
    triage_run_group,
)

CAP = 86_400.0  # 24h


def art(**kw):
    base = dict(competition_id="spooky-author-identification", seed=0, time_cap_seconds=CAP)
    base.update(kw)
    return RunArtifacts(**base)


# --- the load-bearing rule: a timeout that produced a submission is a RESULT ---


def test_timeout_with_submission_is_a_valid_result_not_a_failure():
    """MLE-bench grades whatever submission.csv exists at the end."""
    r = classify(art(exit_code=137, wall_clock_seconds=CAP, has_submission=True,
                     submission_rows=5000))
    assert r.outcome is Outcome.VALID
    assert r.truncated
    assert r.is_capability_signal


def test_timeout_without_submission_is_a_timeout():
    r = classify(art(exit_code=137, wall_clock_seconds=CAP, has_submission=False))
    assert r.outcome is Outcome.TIMEOUT
    assert not r.is_capability_signal


def test_clean_run_with_submission_is_not_marked_truncated():
    r = classify(art(exit_code=0, wall_clock_seconds=3600, has_submission=True,
                     submission_rows=100))
    assert r.outcome is Outcome.VALID
    assert not r.truncated


# --- infra invalidates everything downstream ---


def test_infra_signature_wins_over_missing_submission():
    r = classify(art(exit_code=1, has_submission=False,
                     harness_log="Spot ITN received. Instance will be interrupted at 2026-08-30T12:00:00Z"))
    assert r.outcome is Outcome.INFRA
    assert r.should_retry


@pytest.mark.parametrize(
    "log,signal",
    [
        # Every string below is quoted from the component that emits it.
        ('{"action": "terminate", "time": "2020-02-07T14:55:55Z"}', "spot-interruption"),
        ("GET /latest/meta-data/spot/instance-action", "spot-interruption"),
        ('"detail-type": "EC2 Spot Instance Interruption Warning"', "spot-interruption"),
        ("GET computeMetadata/v1/instance/preempted -> TRUE", "gcp-preemption"),
        ("Preempted by pod 8f3a on node ip-10-0-1-4", "preemption"),
        ("default-scheduler: preempting to accommodate a higher priority pod",
         "preemption"),
        ("Pod was terminated in response to imminent node shutdown.", "preemption"),
        ("The node was low on resource: memory.", "node-pressure-eviction"),
        ("Error: ErrImagePull", "image-pull"),
        ("Back-off pulling image \"mlebench-env:latest\"", "image-pull"),
        ("failed to resolve reference \"docker.io/x:1\": not found", "image-pull"),
        ("manifest unknown", "image-pull"),
        ("pull access denied, repository does not exist or may require authorization",
         "image-pull"),
        ('invalid mount config for type "bind": bind source path does not exist',
         "mount"),
        ("Warning FailedMount  unmounted volumes=[data]", "mount"),
        ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
         "docker-daemon"),
    ],
)
def test_infra_signatures(log, signal):
    r = classify(art(exit_code=1, harness_log=log))
    assert r.outcome is Outcome.INFRA
    assert r.evidence[0].signal == signal


@pytest.mark.parametrize(
    "chatter",
    [
        "scheduler considered 4 pods for preemption this cycle",
        "preempted: false",
        "image pull completed in 4.2s",
        "Successfully pulled image \"nginx:1.25\" in 3s",
        "mounting complete, 3 volumes ready",
        "Free disk space: 20GB remaining",
        "exit code 0",
    ],
)
def test_ordinary_chatter_is_not_an_infra_failure(chatter):
    """A bare word like 'preemption' appears constantly in scheduler logs."""
    r = classify(art(exit_code=1, harness_log=chatter))
    assert r.outcome is not Outcome.INFRA


def test_infra_beats_a_gradeable_submission():
    """If the node vanished mid-write, the artifact is not trustworthy."""
    r = classify(art(has_submission=True, submission_rows=10,
                     harness_log="Preempted by pod 8f3a on node ip-10-0-1-4"))
    assert r.outcome is Outcome.INFRA


# --- memory ---


def test_cuda_oom_is_agent_attributable():
    r = classify(art(exit_code=1, log_tail="RuntimeError: CUDA out of memory."))
    assert r.outcome is Outcome.OOM
    assert r.outcome.attributable_to_agent
    assert any("batch size" in n for n in r.notes)


def test_host_oom_signature():
    r = classify(art(exit_code=137, wall_clock_seconds=100,
                     log_tail="Out of memory: Killed process 1234 (python) total-vm:8kB"))
    assert r.outcome is Outcome.OOM
    assert not r.ambiguous


def test_bare_sigkill_without_cap_is_ambiguous():
    """137 is SIGKILL, sent by both OOM killers and time enforcers."""
    r = classify(art(exit_code=137, wall_clock_seconds=100))
    assert r.outcome is Outcome.OOM
    assert r.ambiguous
    assert any("telemetry" in n for n in r.notes)


def test_sigkill_at_the_cap_is_a_timeout_not_an_oom():
    r = classify(art(exit_code=137, wall_clock_seconds=CAP))
    assert r.outcome is Outcome.TIMEOUT
    assert not r.ambiguous


# --- submissions ---


def test_clean_exit_without_submission():
    r = classify(art(exit_code=0, wall_clock_seconds=500, has_submission=False))
    assert r.outcome is Outcome.NO_SUBMISSION


def test_grader_rejection():
    r = classify(art(has_submission=True, submission_rows=10,
                     validation_error="missing column 'id'"))
    assert r.outcome is Outcome.INVALID_SUBMISSION
    assert "missing column" in r.evidence[0].detail


def test_header_only_submission_is_invalid():
    r = classify(art(exit_code=0, has_submission=True, submission_rows=0))
    assert r.outcome is Outcome.INVALID_SUBMISSION


def test_crash_is_the_residual_bucket():
    r = classify(art(exit_code=1, log_tail="Traceback ... ValueError: bad shape"))
    assert r.outcome is Outcome.CRASH


def test_no_time_cap_means_never_timed_out():
    r = classify(art(time_cap_seconds=None, wall_clock_seconds=999_999, exit_code=1))
    assert r.outcome is Outcome.CRASH


def test_cap_tolerance_allows_slight_undershoot():
    r = classify(art(exit_code=137, wall_clock_seconds=CAP * 0.995))
    assert r.outcome is Outcome.TIMEOUT


# --- the retry guard ---


def test_only_infra_may_be_retried():
    infra = classify(art(harness_log="Spot ITN received. Instance will be interrupted"))
    assert_retry_allowed(infra)  # must not raise


@pytest.mark.parametrize(
    "kw",
    [
        dict(exit_code=0, has_submission=False),
        dict(exit_code=1),
        dict(exit_code=137, wall_clock_seconds=CAP),
        dict(has_submission=True, submission_rows=1, validation_error="bad"),
        dict(has_submission=True, submission_rows=9),
    ],
)
def test_retrying_an_agent_failure_raises(kw):
    r = classify(art(**kw))
    assert not r.should_retry
    with pytest.raises(RetryNotAllowed, match="RESULT, not an infrastructure fault"):
        assert_retry_allowed(r)


# --- report ---


def make_report():
    return TriageReport([
        classify(art(competition_id="c1", has_submission=True, submission_rows=5)),
        classify(art(competition_id="c2", has_submission=True, submission_rows=5)),
        classify(art(competition_id="c3", exit_code=0, has_submission=False)),
        classify(art(competition_id="c4",
                     harness_log="Spot ITN received. Instance interrupted")),
    ])


def test_report_separates_the_three_kinds():
    rep = make_report()
    assert len(rep.gradeable) == 2
    assert len(rep.mechanical) == 1
    assert len(rep.infra) == 1


def test_infra_is_excluded_from_the_denominator():
    """Counting our own faults against the agent understates the agent."""
    assert make_report().effective_denominator() == 3


def test_report_warns_when_plumbing_dominates():
    rep = TriageReport([
        classify(art(competition_id=f"c{i}", exit_code=0, has_submission=False))
        for i in range(5)
    ])
    assert "measuring plumbing" in rep.summary()


def test_report_stays_quiet_when_mechanical_failures_are_rare():
    rep = TriageReport([
        classify(art(competition_id=f"c{i}", has_submission=True, submission_rows=3))
        for i in range(20)
    ])
    assert "measuring plumbing" not in rep.summary()


def test_report_surfaces_ambiguous_runs():
    rep = TriageReport([classify(art(exit_code=137, wall_clock_seconds=10))])
    assert len(rep.ambiguous) == 1
    assert "could not be attributed" in rep.summary()


def test_runset_records_mark_infra_and_medals():
    rep = make_report()
    recs = rep.to_runset_records(medals={("c1", 0): True})
    by_comp = {r["competition_id"]: r for r in recs}
    assert by_comp["c1"]["any_medal"] is True
    assert by_comp["c2"]["any_medal"] is False
    assert by_comp["c4"]["infra_failure"] is True
    assert by_comp["c3"]["infra_failure"] is False


# --- directory loading ---


def write_run(root, name, *, meta, submission=None, log="", harness_log=""):
    d = root / name
    (d / "logs").mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(meta))
    if submission is not None:
        (d / "submission").mkdir()
        (d / "submission" / "submission.csv").write_text(submission)
    if log:
        (d / "logs" / "agent.log").write_text(log)
    if harness_log:
        (d / "logs" / "harness.log").write_text(harness_log)
    return d


def test_from_run_dir_reads_artifacts(tmp_path):
    d = write_run(
        tmp_path, "leaf-classification",
        meta={"seed": 2, "exit_code": 0, "wall_clock_seconds": 900,
              "time_cap_seconds": CAP},
        submission="id,label\n1,a\n2,b\n",
        log="all good",
    )
    a = from_run_dir(d)
    assert a.competition_id == "leaf-classification"
    assert a.seed == 2 and a.submission_rows == 2 and a.has_submission
    assert classify(a).outcome is Outcome.VALID


def test_header_only_file_counts_zero_rows(tmp_path):
    d = write_run(tmp_path, "c", meta={}, submission="id,label\n")
    assert from_run_dir(d).submission_rows == 0
    assert classify(from_run_dir(d)).outcome is Outcome.INVALID_SUBMISSION


def test_missing_metadata_degrades_gracefully(tmp_path):
    """A run that died early may have almost nothing on disk."""
    d = tmp_path / "c"
    d.mkdir()
    a = from_run_dir(d)
    assert a.competition_id == "c" and not a.has_submission
    assert classify(a).outcome is Outcome.NO_SUBMISSION


def test_corrupt_metadata_does_not_crash(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "metadata.json").write_text("{not json")
    assert from_run_dir(d).competition_id == "c"


def test_triage_run_group(tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    write_run(group, "c1", meta={"exit_code": 0}, submission="a,b\n1,2\n")
    write_run(group, "c2", meta={"exit_code": 1},
              harness_log="Spot ITN received. Instance will be interrupted")
    (group / "metadata.json").write_text("{}")  # a file, must be skipped
    rep = triage_run_group(group)
    assert rep.total == 2
    assert len(rep.infra) == 1


def test_evidence_quotes_the_whole_log_line_not_the_regex():
    """A person auditing a classification wants to read the offending line."""
    r = classify(
        art(exit_code=1,
            harness_log="epoch 3 ok\nstep 41: Spot ITN received, draining\n")
    )
    assert r.evidence[0].detail == "step 41: Spot ITN received, draining"
    assert "{0,40}" not in r.evidence[0].detail


def test_cuda_oom_evidence_quotes_the_error_line():
    r = classify(art(exit_code=1, log_tail="RuntimeError: CUDA out of memory. Tried 20GiB"))
    assert r.evidence[0].detail == "RuntimeError: CUDA out of memory. Tried 20GiB"


def test_long_evidence_lines_are_truncated():
    noise = "x" * 400
    r = classify(art(exit_code=1, harness_log=f"{noise} manifest unknown {noise}"))
    assert len(r.evidence[0].detail) <= 160
    assert r.evidence[0].detail.endswith("...")


def test_harness_error_is_our_fault():
    r = classify(art(exit_code=127, harness_error="failed to launch agent: ENOENT"))
    assert r.outcome is Outcome.INFRA
    assert r.should_retry
    assert "ENOENT" in r.evidence[0].detail


def test_harness_error_outranks_everything_else():
    """If the harness broke, nothing downstream describes the agent."""
    r = classify(
        art(exit_code=0, has_submission=True, submission_rows=10,
            harness_error="could not write artifacts")
    )
    assert r.outcome is Outcome.INFRA


# --- log injection: agent stdout must not be able to forge an infra failure ---


@pytest.mark.parametrize(
    "forgery",
    [
        "Spot ITN received. Instance will be interrupted at 2026-08-30T12:00:00Z",
        "Preempted by pod 8f3a on node ip-10-0-1-4",
        "ErrImagePull",
        "No space left on device",
    ],
)
def test_agent_cannot_forge_an_infra_failure_in_its_own_output(forgery):
    """Otherwise a crashing agent prints one line and earns a free retry."""
    r = classify(art(exit_code=1, log_tail=f"Traceback...\n{forgery}\n"))
    assert r.outcome is Outcome.CRASH
    assert not r.should_retry


def test_infra_is_detected_from_the_harness_log():
    r = classify(art(exit_code=1, harness_log="Spot ITN received. Interrupting."))
    assert r.outcome is Outcome.INFRA
    assert r.should_retry


def test_cuda_oom_is_still_read_from_agent_output():
    """Agents legitimately report GPU OOM, and forging it gains them nothing:
    OOM is agent-attributable either way."""
    r = classify(art(exit_code=1, log_tail="RuntimeError: CUDA out of memory"))
    assert r.outcome is Outcome.OOM


def test_foreign_layouts_do_not_trust_agent_logs_by_default(tmp_path):
    d = tmp_path / "c"
    (d / "logs").mkdir(parents=True)
    (d / "logs" / "run.log").write_text("Spot ITN received. Interrupting.")
    (d / "metadata.json").write_text(json.dumps({"exit_code": 1}))
    assert classify(from_run_dir(d)).outcome is Outcome.CRASH
    trusted = from_run_dir(d, trusted_log_names=frozenset({"run.log"}))
    assert classify(trusted).outcome is Outcome.INFRA


# --- memory signatures, quoted from mm/oom_kill.c, PyTorch and TensorFlow ---


@pytest.mark.parametrize(
    "log,signal",
    [
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB",
         "cuda-oom"),
        ("RuntimeError: CUDA error: out of memory", "cuda-oom"),
        ("OOM when allocating tensor with shape[1024,1024]", "tensorflow-oom"),
        ("Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.0GiB",
         "tensorflow-oom"),
        ("Memory cgroup out of memory: Killed process 42 (python) total-vm:8kB",
         "host-oom"),
        ("oom-kill:constraint=CONSTRAINT_MEMCG,oom_memcg=/kubepods/pod123",
         "host-oom"),
        ("python invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE)",
         "host-oom"),
    ],
)
def test_memory_signatures(log, signal):
    r = classify(art(exit_code=1, log_tail=log))
    assert r.outcome is Outcome.OOM
    assert r.evidence[0].signal == signal


def test_tensorflow_oom_is_flagged_ambiguous():
    """TF raises ResourceExhaustedError for host AND device allocation failures."""
    r = classify(art(exit_code=1, log_tail="ResourceExhaustedError: ..."))
    assert r.outcome is Outcome.OOM
    assert r.ambiguous
    assert any("host" in n and "device" in n for n in r.notes)


def test_cgroup_oom_note_names_the_disambiguator():
    r = classify(art(exit_code=1, log_tail="Memory cgroup out of memory: Killed process 9"))
    assert any("oom_memcg" in n for n in r.notes)


def test_mount_note_records_the_silent_v_flag_trap():
    """`docker run -v` creates a missing source silently, emitting nothing."""
    r = classify(art(exit_code=1,
                     harness_log="bind source path does not exist: /home/data"))
    assert any("-v` silently creates" in n or "silently creates" in n for n in r.notes)


def test_no_signature_is_kept_without_a_source():
    """Every pattern must correspond to a string some component really emits.

    A `node ... terminated` signature was carried for a while on nothing but
    plausibility; it matched no real log and is gone. This test is a reminder of
    the rule, not a mechanical check.
    """
    from mlea.triage import INFRA_SIGNATURES

    signals = {sig.signal for sig in INFRA_SIGNATURES}
    assert "node-lost" not in signals
    assert signals == {
        "spot-interruption", "gcp-preemption", "preemption",
        "node-pressure-eviction", "image-pull", "mount", "docker-daemon",
        "disk-full",
    }
