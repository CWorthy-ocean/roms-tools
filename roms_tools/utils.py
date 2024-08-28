from numbers import Integral

import numpy as np
import xarray as xr


def partition(ds: xr.Dataset, nx: int = 1, ny: int = 1) -> list[xr.Dataset]:
    """
    Split a ROMS dataset up into nx by ny spatial tiles.
    """

    if not isinstance(nx, Integral) or not isinstance(ny, Integral):
        raise ValueError("nx and ny must be integers")

    # if eta is periodic there are no ghost cells along those dimensions
    if ds.sizes["eta_rho"] == ds.sizes["eta_v"]:
        n_eta_ghost_cells = 0
    else:
        n_eta_ghost_cells = 1

    # if xi is periodic there are no ghost cells along those dimensions
    if ds.sizes["xi_rho"] == ds.sizes["xi_u"]:
        n_xi_ghost_cells = 0
    else:
        n_xi_ghost_cells = 1

    def integer_division_or_raise(a: int, b: int) -> int:
        remainder = a % b
        if remainder == 0:
            return a // b
        else:
            raise ValueError(
                f"Partitioning nx = {nx} ny = {ny} does not divide the domain into subdomains of integer size."
            )

    eta_rho_domain_size = integer_division_or_raise(
        ds.sizes["eta_rho"] - 2 * n_eta_ghost_cells, nx
    )
    eta_v_domain_size = integer_division_or_raise(
        ds.sizes["eta_v"] - 1 * n_eta_ghost_cells, nx
    )

    xi_rho_domain_size = integer_division_or_raise(
        ds.sizes["xi_rho"] - 2 * n_xi_ghost_cells, ny
    )
    xi_u_domain_size = integer_division_or_raise(
        ds.sizes["xi_u"] - 1 * n_xi_ghost_cells, ny
    )

    eta_coarse_domain_size = integer_division_or_raise(
        ds.sizes["eta_coarse"] - 2 * n_eta_ghost_cells, nx
    )
    xi_coarse_domain_size = integer_division_or_raise(
        ds.sizes["xi_coarse"] - 2 * n_xi_ghost_cells, ny
    )

    # TODO refactor to use two functions for odd- and even-length dimensions
    partitioned_sizes = {
        "eta_rho": [eta_rho_domain_size + n_eta_ghost_cells]
        + [eta_rho_domain_size] * (nx - 2)
        + [eta_rho_domain_size + n_eta_ghost_cells],
        "eta_v": [eta_v_domain_size] * (nx - 1)
        + [eta_v_domain_size + n_eta_ghost_cells],
        "xi_rho": [xi_rho_domain_size + n_xi_ghost_cells]
        + [xi_rho_domain_size] * (nx - 2)
        + [xi_rho_domain_size + n_xi_ghost_cells],
        "xi_u": [xi_u_domain_size] * (nx - 1) + [xi_u_domain_size + n_xi_ghost_cells],
        "eta_coarse": [eta_coarse_domain_size + n_eta_ghost_cells]
        + [eta_coarse_domain_size] * (nx - 2)
        + [eta_coarse_domain_size + n_eta_ghost_cells],
        "xi_coarse": [xi_coarse_domain_size + n_xi_ghost_cells]
        + [xi_coarse_domain_size] * (nx - 2)
        + [xi_coarse_domain_size + n_xi_ghost_cells],
    }

    def cumsum(pmf):
        """Implementation of cumsum which ensures the result starts with zero"""
        cdf = np.empty(len(pmf) + 1, dtype=int)
        cdf[0] = 0
        np.cumsum(pmf, out=cdf[1:])
        return cdf

    partitioned_datasets = []
    for i in range(nx):
        for j in range(ny):
            # file_number = j + (i * nx)

            eta_rho_partition_indices = cumsum(partitioned_sizes["eta_rho"])
            eta_v_partition_indices = cumsum(partitioned_sizes["eta_v"])
            xi_rho_partition_indices = cumsum(partitioned_sizes["xi_rho"])
            xi_u_partition_indices = cumsum(partitioned_sizes["xi_u"])
            eta_coarse_partition_indices = cumsum(partitioned_sizes["eta_coarse"])
            xi_coarse_partition_indices = cumsum(partitioned_sizes["xi_coarse"])

            indexer = dict(
                eta_rho=slice(
                    int(eta_rho_partition_indices[i]),
                    int(eta_rho_partition_indices[i + 1]),
                ),
                eta_v=slice(
                    int(eta_v_partition_indices[i]), int(eta_v_partition_indices[i + 1])
                ),
                xi_rho=slice(
                    int(xi_rho_partition_indices[j]),
                    int(xi_rho_partition_indices[j + 1]),
                ),
                xi_u=slice(
                    int(xi_u_partition_indices[j]), int(xi_u_partition_indices[j + 1])
                ),
                eta_coarse=slice(
                    int(eta_coarse_partition_indices[i]),
                    int(eta_coarse_partition_indices[i + 1]),
                ),
                xi_coarse=slice(
                    int(xi_coarse_partition_indices[j]),
                    int(xi_coarse_partition_indices[j + 1]),
                ),
            )

            partitioned_ds = ds.isel(**indexer)

            partitioned_datasets.append(partitioned_ds)

    # TODO return the file number index too?
    return partitioned_datasets
