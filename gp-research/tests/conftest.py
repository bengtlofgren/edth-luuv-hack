"""Put gp-research/ on sys.path so tests can `import eval.xxx` (the eval/
package is meant to be imported the same way eval_snapir.py / eval_solaqua.py
will import it as a sibling package -- gp-research/ itself is not a valid
dotted module name because of the hyphen)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test may download data; skipped if no local "
        "cache exists and a quick download attempt fails.")

