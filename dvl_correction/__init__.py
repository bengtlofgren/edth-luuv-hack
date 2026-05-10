"""Import shim for running the source checkout from the monorepo root.

The installable ``dvl_correction`` package lives in ``dvl_correction/src``.
When commands are run from the monorepo root, Python sees this outer directory
first and would otherwise create an empty namespace package. Keep the source
directory first on the package path, then execute the real package initializer.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "src"

__path__ = [str(_SOURCE), str(_ROOT)]

_init_path = _SOURCE / "__init__.py"
exec(compile(_init_path.read_text(encoding="utf-8"), str(_init_path), "exec"))
