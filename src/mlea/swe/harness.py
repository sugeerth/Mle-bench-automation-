"""Apply an agent's patch and run the suite against it."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .bench import SweInstance
from .grade import APPLY_PATCH_FAIL, END_TEST_OUTPUT, START_TEST_OUTPUT, run_tests


def apply_patch(repo: str | Path, patch: str) -> tuple[bool, str]:
    """Apply a unified diff. Returns ``(applied, message)``.

    An empty patch counts as applied -- an agent that produced nothing has done
    nothing wrong mechanically, it has simply not fixed the bug, and conflating
    the two would hide the difference between a broken agent and a stuck one.
    """
    repo = Path(repo)
    if not patch.strip():
        return True, "empty patch"
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as fh:
        fh.write(patch if patch.endswith("\n") else patch + "\n")
        patch_path = fh.name
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "apply", "--verbose", patch_path],
            capture_output=True, text=True,
        )
    finally:
        Path(patch_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[:400]
    return True, "applied"


def run_agent(
    instance: SweInstance, patch: str, *, workdir: str | Path | None = None,
    timeout: float = 120.0,
) -> str:
    """Apply a patch to a fresh copy of the repo and return the test log.

    The repo is copied first. Running against the instance directory itself
    would let one agent's patch leak into the next agent's run -- the same class
    of mistake as reusing a competition directory across sweeps.
    """
    scratch = Path(workdir or tempfile.mkdtemp(prefix="mlea-swe-"))
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch / instance.instance_id
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(instance.repo_dir, work)

    applied, message = apply_patch(work, patch)
    if not applied:
        return (
            f"{APPLY_PATCH_FAIL}\n{message}\n"
            f"{START_TEST_OUTPUT}\n{END_TEST_OUTPUT}\n"
        )
    return run_tests(work, instance.test_file, timeout=timeout)


__all__ = ["apply_patch", "run_agent"]
