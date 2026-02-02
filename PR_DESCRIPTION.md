# Integrate narrow channel closing into mask generation

## Summary

This PR integrates the `_close_narrow_channels` functionality directly into the `Grid.update_mask()` method, making it a part of the mask generation workflow. The function closes narrow 1-pixel wide water channels in ROMS land-sea masks to prevent numerical issues.

## Key Changes

### Functionality Integration
- **Integrated `_close_narrow_channels` into `update_mask`**: The functionality is now accessible via the `close_narrow_channels` parameter in `Grid.update_mask()` or as a `Grid` initialization parameter
- **Made function internal**: Renamed to `_close_narrow_channels` to indicate it's an internal implementation detail


### Behavior
- **Closes narrow water channels**: Converts 1-pixel wide water channels to land to prevent numerical issues in ROMS simulations
- **Iterative algorithm**: Processes channels in both north-south and east-west directions iteratively until no more narrow channels are found

### Logging Improvements
- **Simplified logging**: When `verbose=True`, only logs "Closing narrow channels"
- **Silent by default**: When `verbose=False`, no log output is produced

### API Changes
- Added `close_narrow_channels: bool = False` parameter to `Grid` dataclass initialization
- Added `close_narrow_channels: bool | None = None` parameter to `Grid.update_mask()` method
- Updated `ChildGrid.update_mask()` to match parent class signature

### Testing
- Updated `test_close_narrow_channels` to test channel closing with:
  - A vertical ocean line
  - A small lake connected by a narrow channel
  - Verification that narrow channels are properly closed

### Documentation
- Updated `Grid` class docstring to document the new parameter
- Updated `update_mask` docstring to explain the functionality
- Updated notebook examples to show usage via `update_mask`

## Usage

### During Grid Initialization
```python
grid = Grid(
    nx=100,
    ny=100,
    size_x=500,
    size_y=500,
    center_lon=-20,
    center_lat=64,
    close_narrow_channels=True,  # Enable closing narrow channels
)
grid.update_mask()
```

### During Mask Update
```python
grid = Grid(...)
grid.update_mask(
    mask_shapefile="/path/to/shapefile.shp",
    verbose=True,
    close_narrow_channels=True,  # Override initialization setting
)
```

### In Child Grids
```python
child_grid = ChildGrid(...)
child_grid.update_mask(
    close_narrow_channels=True,  # Works the same way
)
```

## Technical Details

- **Algorithm**: Iteratively closes 1-pixel wide channels in both north-south and east-west directions
- **Iteration limit**: Uses `max_iterations=10` (configurable) to prevent infinite loops
- **Performance**: Fixed parameters (`max_iterations=10`) are used when called from `update_mask`

## Testing

- [x] Tests added for `_close_narrow_channels` functionality
- [x] Tests verify narrow channels are closed correctly
- [x] All existing tests pass
- [x] Pre-commit checks pass (ruff, mypy)

## Documentation

- [x] Function docstrings updated
- [x] Class docstrings updated
- [x] Notebook examples updated
- [x] Mask convention clearly documented throughout
