"""Configure PROJ so pyogrio and pyproj share a valid proj.db."""

from __future__ import annotations

import os
from pathlib import Path


def _proj_db_path(directory: str | Path) -> Path:
    return Path(directory) / "proj.db"


def ensure_proj_database() -> str:
    """Point PROJ at a valid proj.db and reset pyproj's global context.

    Jupyter kernels often inherit a stale ``PROJ_DATA`` (for example from an old
    pip ``pyogrio`` wheel's ``proj_data`` directory). That leads to
    ``proj_create: no database context specified`` when GeoPandas assigns a CRS
    after reading Natural Earth via pyogrio.
    """
    import pyproj

    data_dir = Path(pyproj.datadir.get_data_dir())
    if not _proj_db_path(data_dir).is_file():
        msg = (
            "PROJ database (proj.db) not found. Install proj-data from conda-forge, "
            f"e.g. `conda install -c conda-forge proj-data`, then restart the kernel. "
            f"Looked in: {data_dir}"
        )
        raise RuntimeError(msg)

    data_dir_str = str(data_dir)
    for var in ("PROJ_DATA", "PROJ_LIB"):
        current = os.environ.get(var)
        if current and _proj_db_path(current).is_file():
            continue
        os.environ[var] = data_dir_str

    pyproj.datadir.set_data_dir(data_dir_str)
    return data_dir_str


ensure_proj_database()
