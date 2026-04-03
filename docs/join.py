# %% [markdown]
# # Joining partitioned ROMS files
# In addition to requiring partitioned input files, ROMS produces partitioned output files, which need to be joined to be analyzed.
# We can join these files using the `join_netcdf` function.
#
# ## Writing some partitioned example files
# As in the page in [partitioning](https://roms-tools.readthedocs.io/en/latest/partition.html), we will create and partition some example files to demonstrate the use of the joining tool:

# %%
from roms_tools import Grid

# %%
grid = Grid(
    nx=300,
    ny=150,
    size_x=23000,
    size_y=12000,
    center_lon=-161.0,
    center_lat=14.4,
    rot=-3.0,
)

# %%
grid.plot()

# %%
filepath_grid="my_roms_grid.nc"
grid.save(filepath_grid)

# %%
from roms_tools import partition_netcdf
partition_netcdf(filepath_grid, np_xi=5, np_eta=3, output_dir="to_join/")

# %% [markdown]
# ## Joining the example files
# Each of these files has a subset of the original grid in it. To reassemble them into a coherent whole, we use the `join_netcdf` tool. The tool takes a wildcard pattern or list of files, and infers the original layout. We can also provide an optional `output_path`, or `ROMS-Tools` will take the common root filename by default:

# %%
from roms_tools import join_netcdf

# %%
%%time
joined_grid_path = join_netcdf("to_join/my_roms_grid.??.nc")

# %% [markdown]
# Comparing with the original grid, we can see the partitions have joined correctly:

# %%
joined_grid = Grid.from_file("to_join/my_roms_grid.nc")

# %%
joined_grid.plot()
