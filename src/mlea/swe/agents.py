"""Reference patch producers, one per outcome a harness has to tell apart.

Not models -- fixed patches, chosen so that every branch of the resolution rule
is exercised by a real run rather than a fixture. ``regressive`` is the one that
matters: it fixes the reported bug and breaks something else, which a harness
that checks only FAIL_TO_PASS scores as a resolution.
"""

from __future__ import annotations

from .bench import SweInstance

#: Expected outcome per strategy, asserted by the conformance check.
EXPECTED = {
    "gold": "resolved",
    "noop": "unresolved",
    "regressive": "regressed",
    "broken": "patch did not apply",
    "irrelevant": "unresolved",
}
STRATEGIES = tuple(EXPECTED)

_IRRELEVANT = """\
diff --git a/README.md b/README.md
new file mode 100644
--- /dev/null
+++ b/README.md
@@ -0,0 +1 @@
+notes
"""

_BROKEN = """\
diff --git a/does_not_exist.py b/does_not_exist.py
--- a/does_not_exist.py
+++ b/does_not_exist.py
@@ -1,3 +1,3 @@
-this context line is not in any file
+neither is this one
 nor this
"""


def patch_for(instance: SweInstance, strategy: str) -> str:
    """The patch a given reference agent would submit."""
    if strategy not in EXPECTED:
        raise ValueError(f"unknown strategy {strategy!r}; one of {STRATEGIES}")
    if strategy == "gold":
        return instance.gold_patch
    if strategy == "noop":
        return ""
    if strategy == "regressive":
        if not instance.regressive_patch:
            raise ValueError(f"{instance.instance_id} has no regressive variant")
        return instance.regressive_patch
    if strategy == "irrelevant":
        return _IRRELEVANT
    return _BROKEN


__all__ = ["EXPECTED", "STRATEGIES", "patch_for"]
