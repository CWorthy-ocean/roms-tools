"""
Visualization script for close_narrow_channels functionality.
This script demonstrates the effect of closing narrow channels on ROMS masks.
"""

import numpy as np
import matplotlib.pyplot as plt
from roms_tools import Grid
from roms_tools.setup.mask import _close_narrow_channels, add_velocity_masks
from roms_tools.plot import plot

# ============================================================================
# Test 1: Closing narrow 1-pixel passages
# ============================================================================
print("=" * 80)
print("Test 1: Closing narrow 1-pixel passages")
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

# Create a mask with narrow 1-pixel land passages
# Start with all water (0)
mask = np.zeros(mask_shape, dtype=np.int32)

# Create a narrow 1-pixel land passage in north-south direction
# A land cell with water above and below (vertically isolated)
mask[8, 8] = 1  # land cell
mask[7, 8] = 0  # water above
mask[9, 8] = 0  # water below

# Create a narrow 1-pixel land passage in east-west direction
# A land cell with water left and right (horizontally isolated)
mask[11, 11] = 1  # land cell
mask[11, 10] = 0  # water left
mask[11, 12] = 0  # water right

# Set the mask in the grid
grid1.ds["mask_rho"].values[:] = mask

# Save original mask for comparison
mask_before_1 = grid1.ds.mask_rho.values.copy()

print(f"\nBefore closing:")
print(f"  NS passage at [8, 8]: {mask_before_1[8, 8]} (should be 1)")
print(f"  EW passage at [11, 11]: {mask_before_1[11, 11]} (should be 1)")

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

# Get the mask after closing
mask_after_1 = grid1.ds.mask_rho.values.copy()

print(f"\nAfter closing:")
print(f"  NS passage at [8, 8]: {mask_after_1[8, 8]} (should be 0)")
print(f"  EW passage at [11, 11]: {mask_after_1[11, 11]} (should be 0)")

# ============================================================================
# Test 2: Hole filling (removing small isolated regions)
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: Hole filling - removing small isolated regions")
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

# Create a mask with a large land region and a small isolated land region
mask = np.zeros(mask_shape, dtype=np.int32)  # Start with all water (0)

# Create a large land region (1) - this should be preserved as the largest region
mask[6:16, 6:16] = 1

# Create a small isolated land region (1) - this should be removed
mask[3:5, 3:5] = 1  # Small 2x2 isolated region

# Set the mask in the grid
grid2.ds["mask_rho"].values[:] = mask

# Save original mask
mask_before_2 = grid2.ds.mask_rho.values.copy()

print(f"\nBefore closing:")
print(f"  Small isolated region [3:5, 3:5]: {mask_before_2[3:5, 3:5].sum()} cells (should be 4)")
print(f"  Large land region [6:16, 6:16]: {mask_before_2[6:16, 6:16].sum()} cells (should be 100)")

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

# Get the mask after closing
mask_after_2 = grid2.ds.mask_rho.values.copy()

print(f"\nAfter closing:")
print(f"  Small isolated region [3:5, 3:5]: {mask_after_2[3:5, 3:5].sum()} cells (should be 0)")
print(f"  Large land region [6:16, 6:16]: {mask_after_2[6:16, 6:16].sum()} cells (should be 100)")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Test 1: Narrow passages
ax1 = plt.subplot(2, 3, 1)
plot(mask_before_1, grid_ds=grid1.ds, apply_mask=False, ax=ax1, title="Test 1: Before (narrow passages)")
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = plt.subplot(2, 3, 2)
plot(mask_after_1, grid_ds=grid1.ds, apply_mask=False, ax=ax2, title="Test 1: After (passages closed)")
ax2.set_xlabel("")
ax2.set_ylabel("")

ax3 = plt.subplot(2, 3, 3)
mask_diff_1 = mask_before_1 - mask_after_1
plot(mask_diff_1, grid_ds=grid1.ds, apply_mask=False, ax=ax3, title="Test 1: Difference", cmap_name="RdBu_r")
ax3.set_xlabel("")
ax3.set_ylabel("")

# Test 2: Hole filling
ax4 = plt.subplot(2, 3, 4)
plot(mask_before_2, grid_ds=grid2.ds, apply_mask=False, ax=ax4, title="Test 2: Before (with small hole)")
ax4.set_xlabel("")
ax4.set_ylabel("")

ax5 = plt.subplot(2, 3, 5)
plot(mask_after_2, grid_ds=grid2.ds, apply_mask=False, ax=ax5, title="Test 2: After (hole filled)")
ax5.set_xlabel("")
ax5.set_ylabel("")

ax6 = plt.subplot(2, 3, 6)
mask_diff_2 = mask_before_2 - mask_after_2
plot(mask_diff_2, grid_ds=grid2.ds, apply_mask=False, ax=ax6, title="Test 2: Difference", cmap_name="RdBu_r")
ax6.set_xlabel("")
ax6.set_ylabel("")

plt.tight_layout()
plt.show()

print("\nVisualization complete!")
print("\nLegend:")
print("  - Red areas in difference plots: land converted to water (passages/holes removed)")
print("  - Blue areas in difference plots: water converted to land")
print("  - White: no change")
