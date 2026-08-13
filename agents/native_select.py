"""Loader for the native PUCT selection module (#45a).

Same shape as `engine/game.py`'s C++ engine import: put the CMake output
directory on the path, try the import, and degrade to pure Python if it is not
there. Kept in its own module so `agents/mcts.py` does not grow a second copy
of the path dance, and so a caller can ask whether the fast path exists without
importing the whole search.

Unlike the engine import, this one is SILENT on failure. The engine's banner is
useful because a missing engine makes everything 20-50x slower; a missing
selector only means `MCTS(native_select=True)` will refuse, and it refuses
loudly at the point of use. A print here would fire on every tool that imports
mcts, including ones that never search.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))          # .../agents/
_cpp_build = _os.path.join(_os.path.dirname(_here), "engine", "cpp",
                           "build", "Release")
if _cpp_build not in _sys.path:
    _sys.path.insert(0, _cpp_build)

try:
    from uttt_select import ChildArray, SOLVED_NONE, Probe, noop0, noop1, noop2
    HAVE_NATIVE_SELECT = True
    IMPORT_ERROR = None
except ImportError as _e:                                     # pragma: no cover
    ChildArray = None
    Probe = None
    noop0 = noop1 = noop2 = None
    # Must still be defined: the mirror encodes Python's `None` as this value
    # and the encoding is part of the format, not of the extension.
    SOLVED_NONE = 2
    HAVE_NATIVE_SELECT = False
    IMPORT_ERROR = _e


def require(what="native selection"):
    """Raise with a build hint rather than let a None sail into the search."""
    if not HAVE_NATIVE_SELECT:
        raise RuntimeError(
            f"{what} requires the uttt_select extension, which did not import "
            f"({IMPORT_ERROR}). Build it with:\n"
            f"    cmake --build engine/cpp/build --config Release")
    return ChildArray
