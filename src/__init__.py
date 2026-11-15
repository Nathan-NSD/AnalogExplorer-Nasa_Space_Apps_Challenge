"""AnalogExplorer package: data preparation, similarity engine, reports and the Streamlit application."""
from pathlib import Path


def source_version() -> tuple:
    """(name, size, mtime) of the package's source files: changes whenever new code is deployed."""
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files)


LOADED_VERSION = source_version()   # code version these modules were loaded from (read by app.py)
