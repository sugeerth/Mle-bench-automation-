"""Generate SWE-bench-shaped instances with a verified mechanical oracle.

Each instance is a small real git repository containing a real bug, a test suite
that fails on it, and a gold patch that fixes it. The FAIL_TO_PASS and
PASS_TO_PASS sets are not declared -- they are **measured**, by running the
suite before and after applying the gold patch.

That measurement is also the validity check. An instance is well-formed iff
every FAIL_TO_PASS test fails before and passes after, and every PASS_TO_PASS
test passes in both. :func:`make_instance` refuses to emit an instance that does
not satisfy it, which is the correctness oracle a Kaggle-derived benchmark
cannot have.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class InstanceInvalid(RuntimeError):
    """A generated instance failed its own oracle. Always a generator bug."""


@dataclass(frozen=True)
class Bug:
    """A defect template: the broken source, the fix, and the tests."""

    name: str
    module: str
    broken: str
    fixed: str
    tests: str
    #: A short issue description, as an agent would receive it.
    problem_statement: str
    #: Fixes the reported bug *and* breaks something else. The case a harness
    #: that only checks FAIL_TO_PASS scores as a resolution.
    regressive: str = ""


BUGS: tuple[Bug, ...] = (
    Bug(
        name="zero-division",
        module="calc.py",
        broken=(
            "def add(a, b):\n"
            "    return a + b\n\n\n"
            "def divide(a, b):\n"
            "    return a / b\n"
        ),
        fixed=(
            "def add(a, b):\n"
            "    return a + b\n\n\n"
            "def divide(a, b):\n"
            "    if b == 0:\n"
            '        raise ValueError("division by zero")\n'
            "    return a / b\n"
        ),
        tests=(
            "import pytest\n\n"
            "from calc import add, divide\n\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n\n\n"
            "def test_divide():\n"
            "    assert divide(6, 3) == 2\n\n\n"
            "def test_divide_by_zero_raises():\n"
            "    with pytest.raises(ValueError):\n"
            "        divide(1, 0)\n"
        ),
        problem_statement=(
            "divide() raises ZeroDivisionError when the denominator is 0.\n\n"
            "It should raise ValueError('division by zero') instead, so callers "
            "can handle it alongside the library's other input errors."
        ),
        regressive=(
            "def add(a, b):\n"
            "    return a - b\n\n\n"
            "def divide(a, b):\n"
            "    if b == 0:\n"
            '        raise ValueError("division by zero")\n'
            "    return a / b\n"
        ),
    ),
    Bug(
        name="off-by-one",
        module="paging.py",
        broken=(
            "def page_count(total, per_page):\n"
            "    return total // per_page\n\n\n"
            "def page_slice(items, page, per_page):\n"
            "    start = page * per_page\n"
            "    return items[start : start + per_page]\n"
        ),
        fixed=(
            "def page_count(total, per_page):\n"
            "    return -(-total // per_page)\n\n\n"
            "def page_slice(items, page, per_page):\n"
            "    start = page * per_page\n"
            "    return items[start : start + per_page]\n"
        ),
        tests=(
            "from paging import page_count, page_slice\n\n\n"
            "def test_exact_pages():\n"
            "    assert page_count(20, 10) == 2\n\n\n"
            "def test_slice():\n"
            "    assert page_slice(list(range(10)), 1, 3) == [3, 4, 5]\n\n\n"
            "def test_partial_page_counts():\n"
            "    assert page_count(21, 10) == 3\n"
        ),
        problem_statement=(
            "page_count() drops the final partial page.\n\n"
            "page_count(21, 10) returns 2, so the last item is unreachable. "
            "It should return 3."
        ),
        regressive=(
            "def page_count(total, per_page):\n"
            "    return -(-total // per_page)\n\n\n"
            "def page_slice(items, page, per_page):\n"
            "    start = page * per_page\n"
            "    return items[start : start + per_page + 1]\n"
        ),
    ),
    Bug(
        name="mutable-default",
        module="registry.py",
        broken=(
            "def collect(item, bucket=[]):\n"
            "    bucket.append(item)\n"
            "    return bucket\n\n\n"
            "def size(bucket):\n"
            "    return len(bucket)\n"
        ),
        fixed=(
            "def collect(item, bucket=None):\n"
            "    if bucket is None:\n"
            "        bucket = []\n"
            "    bucket.append(item)\n"
            "    return bucket\n\n\n"
            "def size(bucket):\n"
            "    return len(bucket)\n"
        ),
        tests=(
            "from registry import collect, size\n\n\n"
            "def test_collect_appends():\n"
            "    assert collect(1, []) == [1]\n\n\n"
            "def test_size():\n"
            "    assert size([1, 2]) == 2\n\n\n"
            "def test_default_is_not_shared():\n"
            "    collect('a')\n"
            "    assert collect('b') == ['b']\n"
        ),
        problem_statement=(
            "collect() shares state between calls.\n\n"
            "The default argument is a mutable list evaluated once at definition "
            "time, so items accumulate across unrelated calls."
        ),
        regressive=(
            "def collect(item, bucket=None):\n"
            "    if bucket is None:\n"
            "        bucket = []\n"
            "    bucket.append(item)\n"
            "    return bucket\n\n\n"
            "def size(bucket):\n"
            "    return len(bucket) + 1\n"
        ),
    ),
)

BUGS_BY_NAME = {b.name: b for b in BUGS}


@dataclass(frozen=True)
class SweSpec:
    instance_id: str
    bug: str = "zero-division"

    def __post_init__(self) -> None:
        if self.bug not in BUGS_BY_NAME:
            raise ValueError(
                f"unknown bug {self.bug!r}; available: {sorted(BUGS_BY_NAME)}"
            )


@dataclass
class SweInstance:
    """A generated instance, with its measured test sets."""

    instance_id: str
    repo_dir: Path
    module: str
    test_file: str
    problem_statement: str
    gold_patch: str
    bug: str = ""
    regressive_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo_dir": str(self.repo_dir),
            "module": self.module,
            "test_file": self.test_file,
            "problem_statement": self.problem_statement,
            "gold_patch": self.gold_patch,
            "bug": self.bug,
            "regressive_patch": self.regressive_patch,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
        }

    @staticmethod
    def load(path: str | Path) -> "SweInstance":
        blob = json.loads(Path(path).read_text())
        return SweInstance(
            instance_id=blob["instance_id"],
            repo_dir=Path(blob["repo_dir"]),
            module=blob["module"],
            test_file=blob["test_file"],
            problem_statement=blob["problem_statement"],
            gold_patch=blob["gold_patch"],
            bug=blob.get("bug", ""),
            regressive_patch=blob.get("regressive_patch", ""),
            fail_to_pass=blob["FAIL_TO_PASS"],
            pass_to_pass=blob["PASS_TO_PASS"],
        )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


def make_instance(spec: SweSpec, root: str | Path) -> SweInstance:
    """Create one instance and verify it against its own oracle.

    Raises :class:`InstanceInvalid` if the gold patch does not do exactly what
    the instance claims. That check is what a Kaggle-derived benchmark has no
    equivalent of, and it is the reason a SWE-bench-shaped generator can be
    trusted in a way an MLE-bench-shaped one cannot.
    """
    from .grade import parse_pytest_statuses, run_tests

    bug = BUGS_BY_NAME[spec.bug]
    repo = Path(root) / spec.instance_id
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)

    test_file = f"test_{bug.module}"
    (repo / bug.module).write_text(bug.broken)
    (repo / test_file).write_text(bug.tests)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "bench@example.invalid")
    _git(repo, "config", "user.name", "bench")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")

    before = parse_pytest_statuses(run_tests(repo, test_file))

    # The gold patch, as a real unified diff against the committed state.
    (repo / bug.module).write_text(bug.fixed)
    gold_patch = _git(repo, "diff").stdout
    after = parse_pytest_statuses(run_tests(repo, test_file))
    _git(repo, "checkout", "--", bug.module)

    regressive_patch = ""
    if bug.regressive:
        (repo / bug.module).write_text(bug.regressive)
        regressive_patch = _git(repo, "diff").stdout
        _git(repo, "checkout", "--", bug.module)

    fail_to_pass = sorted(
        t for t, s in after.items()
        if s == "PASSED" and before.get(t) in ("FAILED", "ERROR")
    )
    pass_to_pass = sorted(
        t for t, s in after.items()
        if s == "PASSED" and before.get(t) == "PASSED"
    )

    if not fail_to_pass:
        raise InstanceInvalid(
            f"{spec.instance_id}: the gold patch fixes nothing -- no test goes "
            f"from failing to passing"
        )
    regressed = sorted(
        t for t, s in before.items()
        if s == "PASSED" and after.get(t) != "PASSED"
    )
    if regressed:
        raise InstanceInvalid(
            f"{spec.instance_id}: the gold patch breaks {regressed}"
        )
    if not gold_patch.strip():
        raise InstanceInvalid(f"{spec.instance_id}: the gold patch is empty")

    instance = SweInstance(
        instance_id=spec.instance_id,
        repo_dir=repo,
        module=bug.module,
        test_file=test_file,
        problem_statement=bug.problem_statement,
        gold_patch=gold_patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        bug=bug.name,
        regressive_patch=regressive_patch,
    )
    (repo / "instance.json").write_text(json.dumps(instance.to_dict(), indent=2))
    return instance


SUITE: tuple[SweSpec, ...] = tuple(
    SweSpec(f"synth__{b.name}-1", b.name) for b in BUGS
)


def make_suite(root: str | Path, specs: tuple[SweSpec, ...] = SUITE) -> list[SweInstance]:
    return [make_instance(s, root) for s in specs]


__all__ = [
    "BUGS",
    "BUGS_BY_NAME",
    "Bug",
    "InstanceInvalid",
    "SUITE",
    "SweInstance",
    "SweSpec",
    "make_instance",
    "make_suite",
]
