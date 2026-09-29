"""Intake readers for sources that xarray cannot open from a plain path or URL.

Both readers are catalog-entry ``reader:`` targets, referenced by import string so a
catalog file names them without importing this module. ``CopernicusMarineReader`` is a
copy of the reader of the same name in ocean-skill (``ocean_skill.readers``); catalog
entries written by ocean-skill name that path, which :func:`catalog.map_reader`
redirects here when ocean-skill is not installed. Both belong in a shared package.
"""

from __future__ import annotations

from importlib.util import find_spec

import xarray as xr
from intake.readers.readers import BaseReader

from roms_tools.datasets import download

_POOCH_FETCHERS = {
    "topo": download.download_topo,
    "correction": download.download_correction_data,
    "river": download.download_river_data,
    "sal": download.download_sal_data,
}


class PoochReader(BaseReader):
    """Open a file that ROMS-Tools downloads and caches through pooch.

    Parameters
    ----------
    registry : str
        Which pooch registry holds the file: ``"topo"``, ``"correction"``,
        ``"river"`` or ``"sal"``.
    filename : str
        File name within that registry.
    **kwargs
        Passed to :func:`xarray.open_dataset`.
    """

    output_instance = "xarray:Dataset"

    def _read(self, registry: str, filename: str, **kwargs):
        try:
            fetch = _POOCH_FETCHERS[registry]
        except KeyError as err:
            msg = f"Unknown pooch registry {registry!r}; expected one of {sorted(_POOCH_FETCHERS)}."
            raise ValueError(msg) from err
        return xr.open_dataset(fetch(filename), **kwargs)


class CopernicusMarineReader(BaseReader):
    """Open a Copernicus Marine ARCO store by ``dataset_id`` via ``copernicusmarine``.

    The store is not anonymously readable at chunk level, so the toolbox, which holds
    the user's login, opens it. ``copernicusmarine`` is an optional dependency imported
    only when a read happens.

    Parameters
    ----------
    dataset_id : str
        Stable Copernicus Marine dataset id.
    service : str, optional
        ``"arco-geo-series"`` (fast for maps) or ``"arco-time-series"`` (fast for
        point time series).
    dataset_version : str, optional
        Pin a dataset version; default is the current one.
    check_login : bool, optional
        Fail early with a clear message when not logged in.
    **kwargs
        Passed to ``copernicusmarine.open_dataset`` (``start_datetime``, ...).
    """

    output_instance = "xarray:Dataset"

    def _read(
        self,
        dataset_id: str,
        service: str = "arco-time-series",
        dataset_version: str | None = None,
        check_login: bool = True,
        **kwargs,
    ):
        if find_spec("copernicusmarine") is None:
            msg = (
                f"Reading the Copernicus Marine source {dataset_id!r} needs the "
                "'copernicusmarine' package (`pip install roms-tools[stream]`) and a "
                "login (`copernicusmarine login`)."
            )
            raise RuntimeError(msg)
        import copernicusmarine

        if check_login and not copernicusmarine.login(check_credentials_valid=True):
            msg = (
                f"Not authenticated with Copernicus Marine, so {dataset_id!r} cannot "
                "be read. Run `copernicusmarine login` and retry."
            )
            raise RuntimeError(msg)
        opts = dict(kwargs)
        if dataset_version is not None:
            opts["dataset_version"] = dataset_version
        return copernicusmarine.open_dataset(
            dataset_id=dataset_id, service=service, **opts
        )
