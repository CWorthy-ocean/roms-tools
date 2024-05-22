
@dataclass(frozen=True, kw_only=True)
class ERA5:

    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        #ds = fetch_tpxo()
        ds = xr.open_mfdataset('/glade/derecho/scratch/bachman/ERA5/NA/ERA5*.nc')
        ds['longitude'] = xr.where(ds.longitude <= 0, ds.longitude + 360, ds.longitude)

        object.__setattr__(self, "ds", ds)

    def 

@dataclass(frozen=True, kw_only=True)
class AtmosphericForcing:
    grid: Grid
    model_reference_date: datetime = datetime(2000, 1, 1)
    source: str = "era5"
    ds: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        if self.source == "era5":
            era5 = ERA5()

            tides = tpxo.get_corrected_tides(self.model_reference_date, self.alan_factor)
            # rename dimension and select desired number of constituents
            for k in tides.keys():
                tides[k] = tides[k].rename({"nc": "ntides"})
                tides[k] = tides[k].isel(ntides=slice(None, self.nc))

            # Interpolate onto desired grid
            # Wind
            u10 = ds_ERA["u10"].interp(longitude=grid.ds.lon_rho, latitude=grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
            v10 = ds_ERA["v10"].interp(longitude=grid.ds.lon_rho, latitude=grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
            u10_grid = u10 * np.cos(self.grid.ds.angle) + v10 * np.sin(self.grid.ds.angle)
            v10_grid = v10 * np.cos(self.grid.ds.angle) - u10 * np.sin(self.grid.ds.angle)
            # Radiation
            swr = ds_ERA["ssr"].interp(longitude=grid.ds.lon_rho, latitude=grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
            lwr = ds_ERA["strd"].interp(longitude=grid.ds.lon_rho, latitude=grid.ds.lat_rho, method='nearest').drop_vars(["longitude", "latitude"])
            # Translate to fluxes. ERA5 stores values integrated over 1 hour
            swr = swr / 3600
            lwr = lwr / 3600
            
            swr = era5.correct_shortwave_radiation(swr, grid)

	        swr = get_frc_era(data,grd,'ssr',irec,'linear');  % downward_shortwave_flux [J/m2]
	lwr = get_frc_era(data,grd,'strd',irec,'linear'); % downward_longwave_flux [J/m2]



        # save in new dataset
        ds = xr.Dataset()

        ds["uwnd"] = u10_grid
        ds["uwnd"].attrs["long_name"] = "10 meter wind in x-direction"
        ds["uwnd"].attrs["units"] = "m s**-1"

        ds["vwnd"] = v10_grid
        ds["vwnd"].attrs["long_name"] = "10 meter wind in y-direction"
        ds["vwnd"].attrs["units"] = "m s**-1"

        object.__setattr__(self, "ds", ds)
