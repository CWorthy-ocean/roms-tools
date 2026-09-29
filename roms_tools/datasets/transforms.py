"""Named dataset transforms referenced from catalog entries.

A transform fixes up a source dataset that the generic ``LatLonDataset`` engine
cannot handle on its own (a coordinate stored as a data variable, a mask derived from
a field). Catalog entries name transforms by import string, for example
``roms_tools.datasets.transforms:etopo5_assign_coords``, so a source that needs one
does not need its own subclass.

A transform receives the dataset together with the current name maps and returns both,
so a transform that must change the maps (drop a variable, rename a dimension) does so
by return value rather than by mutating an object.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import xarray as xr


@dataclass
class Names:
    """The role-to-name maps a dataset object carries."""

    var_names: dict[str, str]
    dim_names: dict[str, str]
    opt_var_names: dict[str, str] = field(default_factory=dict)


# ``(ds, names) -> (ds, names)``
Transform = Callable[[xr.Dataset, Names], tuple[xr.Dataset, Names]]


def load_transform(import_string: str) -> Transform:
    """Resolve a ``module:function`` string to a transform.

    Parameters
    ----------
    import_string : str
        Import path in ``module:function`` form.

    Returns
    -------
    Transform
        The referenced function.

    Raises
    ------
    ValueError
        If the string is not in ``module:function`` form.
    """
    module_name, sep, func_name = import_string.partition(":")
    if not sep or not module_name or not func_name:
        msg = f"Transform {import_string!r} must be of the form 'module:function'."
        raise ValueError(msg)
    return getattr(importlib.import_module(module_name), func_name)


def etopo5_assign_coords(ds: xr.Dataset, names: Names) -> tuple[xr.Dataset, Names]:
    """Assign ``lon`` and ``lat`` coordinates from the ETOPO5 ``topo_lon``/``topo_lat``.

    Same operation as :meth:`ETOPO5Dataset.clean_up`.
    """
    ds = ds.assign_coords({"lon": ds["topo_lon"], "lat": ds["topo_lat"]})
    return ds, names


def glorys_masks(ds: xr.Dataset, names: Names) -> tuple[xr.Dataset, Names]:
    """Add ``mask`` (from ``zeta``) and ``mask_vel`` (from ``u``) to a GLORYS dataset.

    Same operation as :meth:`GLORYSDataset.post_process`: the masks are 1 where the
    first time step (and surface level for velocity) is valid and 0 where it is NaN.
    """
    var_names, dim_names = names.var_names, names.dim_names
    zeta = ds[var_names["zeta"]]
    u = ds[var_names["u"]]

    zeta_ref = (
        zeta.isel({dim_names["time"]: 0}) if dim_names["time"] in zeta.dims else zeta
    )
    u_ref = u.isel({dim_names["time"]: 0}) if dim_names["time"] in u.dims else u
    if dim_names["depth"] in u_ref.dims:
        u_ref = u_ref.isel({dim_names["depth"]: 0})

    ds = ds.copy()
    ds["mask"] = xr.where(zeta_ref.isnull(), 0, 1)
    ds["mask_vel"] = xr.where(u_ref.isnull(), 0, 1)
    return ds, names
