# Test and visualize close_narrow_channels functionality
# Paste this into a Jupyter notebook cell

import numpy as np
import matplotlib.pyplot as plt
from roms_tools import Grid
from roms_tools.setup.mask import _close_narrow_channels, add_velocity_masks

# ============================================================================
# Test 1: Closing narrow 1-pixel water channels
# Note: In ROMS masks, 1 = OCEAN (water) and 0 = LAND
# ============================================================================
print("=" * 80)
print("Test 1: Closing narrow 1-pixel water channels")
print("=" * 80)

# Create a small grid with close_narrow_channels=False to avoid closing during init
grid1 = Grid(
    nx=15,
    ny=15,
    size_x=100,
    size_y=100,
    center_lon=-20,
    center_lat=64,
    rot=0,
    N=3,
    close_narrow_channels=False,
)

# Get the actual shape of mask_rho (grid adds boundary cells, so it's nx+2, ny+2)
mask_shape = grid1.ds["mask_rho"].shape

# Create a mask with narrow 1-pixel water channels
# Start with all land (0)
mask = np.zeros(mask_shape, dtype=np.int32)

# Create a narrow 1-pixel water channel in north-south direction
mask[8, 8] = 1  # water cell (ocean)
mask[7, 8] = 0  # land above
mask[9, 8] = 0  # land below

# Create a narrow 1-pixel water channel in east-west direction
mask[11, 11] = 1  # water cell (ocean)
mask[11, 10] = 0  # land left
mask[11, 12] = 0  # land right

# Set the mask in the grid
grid1.ds["mask_rho"].values[:] = mask

# Save original mask for comparison (as DataArray for plotting)
mask_before_1 = grid1.ds.mask_rho.copy()

print(f"\nBefore closing:")
print(f"  NS water channel at [8, 8]: {mask_before_1.values[8, 8]} (should be 1 = ocean)")
print(f"  EW water channel at [11, 11]: {mask_before_1.values[11, 11]} (should be 1 = ocean)")

# Close narrow channels directly (as update_mask would do)
grid1.ds = _close_narrow_channels(
    grid1.ds,
    mask_var="mask_rho",
    max_iterations=10,
    connectivity=4,
    min_region_fraction=0.1,
    inplace=True,
    verbose=True,
)
# Update velocity masks after modifying mask_rho
grid1.ds = add_velocity_masks(grid1.ds)

# Get the mask after closing (as DataArray for plotting)
mask_after_1 = grid1.ds.mask_rho.copy()

print(f"\nAfter closing:")
print(f"  NS water channel at [8, 8]: {mask_after_1.values[8, 8]} (should be 0 = land, channel closed)")
print(f"  EW water channel at [11, 11]: {mask_after_1.values[11, 11]} (should be 0 = land, channel closed)")

# ============================================================================
# Test 2: Filling small lakes (removing small isolated water regions)
# Note: In ROMS masks, 1 = OCEAN (water) and 0 = LAND
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Filling small lakes - removing small isolated water regions")
print("=" * 80)

# Create a small grid with close_narrow_channels=False to avoid closing during init
grid2 = Grid(
    nx=20,
    ny=20,
    size_x=100,
    size_y=100,
    center_lon=-20,
    center_lat=64,
    rot=0,
    N=3,
    close_narrow_channels=False,
)

# Get the actual shape of mask_rho
mask_shape = grid2.ds["mask_rho"].shape

# Create a mask with a large ocean region and a small isolated lake
mask = np.zeros(mask_shape, dtype=np.int32)  # Start with all land (0)

# Create a large ocean region (1) - this should be preserved as the largest region
mask[6:16, 6:16] = 1  # Large ocean region

# Create a small isolated lake (water region, 1) - this should be filled (converted to land)
mask[3:5, 3:5] = 1  # Small 2x2 isolated lake

# Set the mask in the grid
grid2.ds["mask_rho"].values[:] = mask

# Save original mask (as DataArray for plotting)
mask_before_2 = grid2.ds.mask_rho.copy()

print(f"\nBefore closing:")
print(f"  Small isolated lake [3:5, 3:5]: {mask_before_2.values[3:5, 3:5].sum()} cells (should be 4 = water)")
print(f"  Large ocean region [6:16, 6:16]: {mask_before_2.values[6:16, 6:16].sum()} cells (should be 100 = water)")

# Close narrow channels directly (as update_mask would do)
grid2.ds = _close_narrow_channels(
    grid2.ds,
    mask_var="mask_rho",
    max_iterations=10,
    connectivity=4,
    min_region_fraction=0.1,
    inplace=True,
    verbose=True,
)
# Update velocity masks after modifying mask_rho
grid2.ds = add_velocity_masks(grid2.ds)

# Get the mask after closing (as DataArray for plotting)
mask_after_2 = grid2.ds.mask_rho.copy()

print(f"\nAfter closing:")
print(f"  Small isolated lake [3:5, 3:5]: {mask_after_2.values[3:5, 3:5].sum()} cells (should be 0 = land, lake filled)")
print(f"  Large ocean region [6:16, 6:16]: {mask_after_2.values[6:16, 6:16].sum()} cells (should be 100 = water, preserved)")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 12))

# Test 1: Narrow passages
# Calculate difference for visualization
mask_diff_1 = mask_before_1.values - mask_after_1.values

# Get coordinates for plotting
lon1 = grid1.ds.lon_rho.values
lat1 = grid1.ds.lat_rho.values

# Plot Test 1: Before
im1 = axes[0, 0].pcolormesh(lon1, lat1, mask_before_1.values, cmap='YlOrRd', shading='nearest')
axes[0, 0].set_title("Test 1: Before (narrow water channels)")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[0, 0], label='Mask (0=land, 1=ocean)')

# Plot Test 1: After
im2 = axes[0, 1].pcolormesh(lon1, lat1, mask_after_1.values, cmap='YlOrRd', shading='nearest')
axes[0, 1].set_title("Test 1: After (channels closed)")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")
plt.colorbar(im2, ax=axes[0, 1], label='Mask (0=land, 1=ocean)')

# Plot Test 1: Difference
im3 = axes[0, 2].pcolormesh(lon1, lat1, mask_diff_1, cmap='RdBu_r', shading='nearest', vmin=-1, vmax=1)
axes[0, 2].set_title("Test 1: Difference")
axes[0, 2].set_xlabel("Longitude")
axes[0, 2].set_ylabel("Latitude")
plt.colorbar(im3, ax=axes[0, 2], label='Change (1=water→land, -1=land→water)')

# Test 2: Hole filling
# Calculate difference for visualization
mask_diff_2 = mask_before_2.values - mask_after_2.values

# Get coordinates for plotting
lon2 = grid2.ds.lon_rho.values
lat2 = grid2.ds.lat_rho.values

# Plot Test 2: Before
im4 = axes[1, 0].pcolormesh(lon2, lat2, mask_before_2.values, cmap='YlOrRd', shading='nearest')
axes[1, 0].set_title("Test 2: Before (with small lake)")
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")
plt.colorbar(im4, ax=axes[1, 0], label='Mask (0=land, 1=ocean)')

# Plot Test 2: After
im5 = axes[1, 1].pcolormesh(lon2, lat2, mask_after_2.values, cmap='YlOrRd', shading='nearest')
axes[1, 1].set_title("Test 2: After (lake filled)")
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")
plt.colorbar(im5, ax=axes[1, 1], label='Mask (0=land, 1=ocean)')

# Plot Test 2: Difference
im6 = axes[1, 2].pcolormesh(lon2, lat2, mask_diff_2, cmap='RdBu_r', shading='nearest', vmin=-1, vmax=1)
axes[1, 2].set_title("Test 2: Difference")
axes[1, 2].set_xlabel("Longitude")
axes[1, 2].set_ylabel("Latitude")
plt.colorbar(im6, ax=axes[1, 2], label='Change (1=water→land, -1=land→water)')

plt.tight_layout()
plt.show()

print("\nVisualization complete!")
print("\nLegend:")
print("  - Red areas in difference plots: water converted to land (channels closed, lakes filled)")
print("  - Blue areas in difference plots: land converted to water")
print("  - White: no change")
print("\nNote: In ROMS masks, 1 = OCEAN (water) and 0 = LAND")
