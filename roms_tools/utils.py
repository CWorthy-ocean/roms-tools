from numbers import Integral

import numpy as np
import xarray as xr


def partition(
    ds: xr.Dataset, nx: int = 1, ny: int = 1
) -> tuple[list[int], list[xr.Dataset]]:
    """
    Split a ROMS dataset up into nx by ny spatial tiles.
    """

    # TODO also check they are positive integers
    if not isinstance(nx, Integral) or not isinstance(ny, Integral):
        raise ValueError("nx and ny must be integers")

    # 'eta_rho' and 'xi_rho' are always expected to be present
    partitionable_dims_maybe_present = ["eta_v", "xi_u", "eta_coarse", "xi_coarse"]
    dims_to_partition = ["eta_rho", "xi_rho"] + [
        d for d in partitionable_dims_maybe_present if d in ds.dims
    ]

    # if eta is periodic there are no ghost cells along those dimensions
    if "eta_v" in ds.sizes and ds.sizes["eta_rho"] == ds.sizes["eta_v"]:
        # TODO how are we supposed to know if eta is periodic if eta_v doesn't appear? partit.F doesn't say...
        n_eta_ghost_cells = 0
    else:
        n_eta_ghost_cells = 1

    # if xi is periodic there are no ghost cells along those dimensions
    if "xi_u" in ds.sizes and ds.sizes["xi_rho"] == ds.sizes["xi_u"]:
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
    xi_rho_domain_size = integer_division_or_raise(
        ds.sizes["xi_rho"] - 2 * n_xi_ghost_cells, ny
    )

    if "eta_v" in dims_to_partition:
        eta_v_domain_size = integer_division_or_raise(
            ds.sizes["eta_v"] - 1 * n_eta_ghost_cells, nx
        )
    if "xi_u" in dims_to_partition:
        xi_u_domain_size = integer_division_or_raise(
            ds.sizes["xi_u"] - 1 * n_xi_ghost_cells, ny
        )

    if "eta_coarse" in dims_to_partition:
        eta_coarse_domain_size = integer_division_or_raise(
            ds.sizes["eta_coarse"] - 2 * n_eta_ghost_cells, nx
        )
    if "xi_coarse" in dims_to_partition:
        xi_coarse_domain_size = integer_division_or_raise(
            ds.sizes["xi_coarse"] - 2 * n_xi_ghost_cells, ny
        )

    # unpartitioned dimensions should have sizes unchanged
    partitioned_sizes = {
        dim: [size] for dim, size in ds.sizes.items() if dim in dims_to_partition
    }

    # TODO refactor to use two functions for odd- and even-length dimensions
    if "eta_v" in dims_to_partition:
        partitioned_sizes["eta_v"] = [eta_v_domain_size] * (nx - 1) + [
            eta_v_domain_size + n_eta_ghost_cells
        ]
    if "xi_u" in dims_to_partition:
        partitioned_sizes["xi_u"] = [xi_u_domain_size] * (ny - 1) + [
            xi_u_domain_size + n_xi_ghost_cells
        ]

    if nx > 1:
        partitioned_sizes["eta_rho"] = (
            [eta_rho_domain_size + n_eta_ghost_cells]
            + [eta_rho_domain_size] * (nx - 2)
            + [eta_rho_domain_size + n_eta_ghost_cells]
        )

        if "eta_coarse" in dims_to_partition:
            partitioned_sizes["eta_coarse"] = (
                [eta_coarse_domain_size + n_eta_ghost_cells]
                + [eta_coarse_domain_size] * (nx - 2)
                + [eta_coarse_domain_size + n_eta_ghost_cells]
            )

    if ny > 1:
        partitioned_sizes["xi_rho"] = (
            [xi_rho_domain_size + n_xi_ghost_cells]
            + [xi_rho_domain_size] * (ny - 2)
            + [xi_rho_domain_size + n_xi_ghost_cells]
        )

        if "xi_coarse" in dims_to_partition:
            partitioned_sizes["xi_coarse"] = (
                [xi_coarse_domain_size + n_xi_ghost_cells]
                + [xi_coarse_domain_size] * (ny - 2)
                + [xi_coarse_domain_size + n_xi_ghost_cells]
            )

    def cumsum(pmf):
        """Implementation of cumsum which ensures the result starts with zero"""
        cdf = np.empty(len(pmf) + 1, dtype=int)
        cdf[0] = 0
        np.cumsum(pmf, out=cdf[1:])
        return cdf

    file_numbers = []
    partitioned_datasets = []
    for j in range(ny):
        for i in range(nx):
            file_number = i + (j * ny)
            file_numbers.append(file_number)

            eta_rho_partition_indices = cumsum(partitioned_sizes["eta_rho"])
            xi_rho_partition_indices = cumsum(partitioned_sizes["xi_rho"])

            indexers = {
                "eta_rho": slice(
                    int(eta_rho_partition_indices[i]),
                    int(eta_rho_partition_indices[i + 1]),
                ),
                "xi_rho": slice(
                    int(xi_rho_partition_indices[j]),
                    int(xi_rho_partition_indices[j + 1]),
                ),
            }

            if "eta_v" in dims_to_partition:
                eta_v_partition_indices = cumsum(partitioned_sizes["eta_v"])
                indexers["eta_v"] = slice(
                    int(eta_v_partition_indices[i]),
                    int(eta_v_partition_indices[i + 1]),
                )
            if "xi_u" in dims_to_partition:
                xi_u_partition_indices = cumsum(partitioned_sizes["xi_u"])
                indexers["xi_u"] = slice(
                    int(xi_u_partition_indices[j]), int(xi_u_partition_indices[j + 1])
                )

            if "eta_coarse" in dims_to_partition:
                eta_coarse_partition_indices = cumsum(partitioned_sizes["eta_coarse"])
                indexers["eta_coarse"] = slice(
                    int(eta_coarse_partition_indices[i]),
                    int(eta_coarse_partition_indices[i + 1]),
                )

            if "xi_coarse" in dims_to_partition:
                xi_coarse_partition_indices = cumsum(partitioned_sizes["xi_coarse"])
                indexers["xi_coarse"] = slice(
                    int(xi_coarse_partition_indices[j]),
                    int(xi_coarse_partition_indices[j + 1]),
                )

            partitioned_ds = ds.isel(**indexers)

            partitioned_datasets.append(partitioned_ds)

    return file_numbers, partitioned_datasets
