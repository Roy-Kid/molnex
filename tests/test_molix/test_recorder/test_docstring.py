"""RED test for the molrec citation in the recorder module docstring.

Acceptance trace:
    ac-012 → :func:`test_module_docstring_cites_molrec`
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_module_docstring_cites_molrec() -> None:
    """ac-012: ``src/molix/recorder/__init__.py`` cites molrec spec §metrics.

    The module docstring must mention the four molrec spec section
    names we adopt and the local-disk path to the spec file, so that
    a future reader (or the spec-drift auditor) lands on the source
    of truth without needing to grep the rest of the codebase.
    """
    here = Path(__file__).resolve()
    init_path = (here.parents[3] / "src" / "molix" / "recorder" / "__init__.py").resolve()
    assert init_path.is_file(), f"Could not locate {init_path!s}"

    source = init_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    docstring = ast.get_docstring(module)
    assert docstring is not None, (
        f"{init_path!s} must declare a module-level docstring citing molrec spec §metrics."
    )

    expected_sections = ("Structure", "Metric records", "Metric types", "Key namespace")
    for section in expected_sections:
        assert section in docstring, (
            f"Module docstring must reference molrec spec section "
            f"{section!r}; current docstring is missing it."
        )

    expected_path = "/Users/roykid/work/molcrafts/molrec/docs/spec/metrics.md"
    assert expected_path in docstring, (
        f"Module docstring must cite the local-disk path to the molrec "
        f"spec ({expected_path!r}) so spec-drift audits can find the "
        f"source of truth."
    )
