from dataclasses import dataclass
from roms_tools.setup.grid import Grid
from roms_tools.setup.vertical_coordinate import VerticalCoordinate
from roms_tools.setup.fill import fill_and_interpolate
from roms_tools.setup.utils import extrapolate_deepest_to_bottom, interpolate_from_rho_to_u, interpolate_from_rho_to_v
import xarray as xr
import numpy as np

@dataclass(frozen=True, kw_only=True)
class ROMSToolsMixin():
    """
    Represents a mixin tool for ROMS-Tools with capabilities shared by the various
    ROMS-Tools dataclasses.

    Parameters
    ----------
    grid : Grid
        Object representing the grid information used for the model.
    vertical_coordinate : VerticalCoordinate
        Object representing the vertical coordinate system. Defaults to None.

    """
    
    grid: Grid
    vertical_coordinate: VerticalCoordinate = None

    def regrid_data(self, data, vars_2d, vars_3d):

        lon = self.grid.ds.lon_rho
        lat = self.grid.ds.lat_rho

        # operate on longitudes between -180 and 180 unless ROMS domain lies at least 5 degrees in lontitude away from Greenwich meridian
        lon = xr.where(lon > 180, lon - 360, lon)
        straddle = True
        if not self.grid.straddle and abs(lon).min() > 5:
            lon = xr.where(lon < 0, lon + 360, lon)
            straddle = False

        # Restrict data to relevant subdomain to achieve better performance and to avoid discontinuous longitudes introduced by converting
        # to a different longitude range (+- 360 degrees). Discontinues longitudes can lead to artifacts in the interpolation process that
        # would not be detected by the nan_check function.
        data.choose_subdomain(
            latitude_range=[lat.min().values, lat.max().values],
            longitude_range=[lon.min().values, lon.max().values],
            margin=2,
            straddle=straddle,
        )

        # interpolate onto desired grid
        data_vars = {}

        # 2d interpolation
        fill_dims = [data.dim_names["latitude"], data.dim_names["longitude"]]
        coords = {data.dim_names["latitude"]: lat, data.dim_names["longitude"]: lon}
        for var in vars_2d:
            mask = xr.where(data.ds[data.var_names[var]].isel(time=0).isnull(), 0, 1)

            data_vars[var] = fill_and_interpolate(
                data.ds[data.var_names[var]].astype(np.float64),
                mask,
                fill_dims=fill_dims,
                coords=coords,
                method="linear",
            )

        # 3d interpolation
        coords = {
            data.dim_names["latitude"]: lat,
            data.dim_names["longitude"]: lon,
            data.dim_names["depth"]: self.vertical_coordinate.ds["layer_depth_rho"],
        }
        # extrapolate deepest value all the way to bottom ("flooding")
        for var in vars_3d:
            data.ds[data.var_names[var]] = extrapolate_deepest_to_bottom(
                data.ds[data.var_names[var]], data.dim_names["depth"]
            )
            mask = xr.where(data.ds[data.var_names[var]].isel(time=0).isnull(), 0, 1)

            # setting fillvalue_interp to None means that we allow extrapolation in the
            # interpolation step to avoid NaNs at the surface if the lowest depth in original
            # data is greater than zero

            data_vars[var] = fill_and_interpolate(
                data.ds[data.var_names[var]].astype(np.float64),
                mask,
                fill_dims=fill_dims,
                coords=coords,
                method="linear",
                fillvalue_interp=None,
            )
            if data.dim_names["time"] != "time":
                data_vars[var] = data_vars[var].rename({data.dim_names["time"]: "time"})

        return data_vars

    def process_velocities(self, data_vars):
        
        angle = self.grid.ds.angle
        
        # rotate velocities to grid orientation
        u_rot = data_vars["u"] * np.cos(angle) + data_vars["v"] * np.sin(angle)
        v_rot = data_vars["v"] * np.cos(angle) - data_vars["u"] * np.sin(angle)

        # interpolate to u- and v-points
        u = interpolate_from_rho_to_u(u_rot)
        v = interpolate_from_rho_to_v(v_rot)

        # 3d masks for ROMS domain
        umask = self.grid.ds.mask_u.expand_dims({"s_rho": u.s_rho})
        vmask = self.grid.ds.mask_v.expand_dims({"s_rho": v.s_rho})

        data_vars["u"] = u * umask
        data_vars["v"] = v * vmask

        # Compute barotropic velocity
        # thicknesses
        dz = -self.vertical_coordinate.ds["interface_depth_rho"].diff(dim="s_w")
        dz = dz.rename({"s_w": "s_rho"})
        # thicknesses at u- and v-points
        dzu = interpolate_from_rho_to_u(dz)
        dzv = interpolate_from_rho_to_v(dz)

        data_vars["ubar"] = (
                (dzu * data_vars["u"]).sum(dim="s_rho") / dzu.sum(dim="s_rho")
        ).transpose("time", "eta_rho", "xi_u")
        data_vars["vbar"] = (
                (dzv * data_vars["v"]).sum(dim="s_rho") / dzv.sum(dim="s_rho")
                ).transpose("time", "eta_v", "xi_rho")

        return data_vars

    def get_variable_metadata(self):

        d = {
                "temp": {"long_name": "potential temperature", "units": "Celsius"},
                "salt": {"long_name": "salinity", "units": "PSU"},
                "zeta": {"long_name": "sea surface height", "units": "m"},
                "u": {"long_name": "u-flux component", "units": "m/s"},
                "v": {"long_name": "v-flux component", "units": "m/s"},
                "w": {"long_name": "w-flux component", "units": "m/s"},
                "ubar": {"long_name": "vertically integrated u-flux component", "units": "m/s"},
                "vbar": {"long_name": "vertically integrated v-flux component", "units": "m/s"},
                "PO4": {"long_name": "dissolved inorganic phosphate", "units": "mmol/m^3"},
                "NO3": {"long_name": "dissolved inorganic nitrate", "units": "mmol/m^3"},
                "SiO3": {"long_name": "dissolved inorganic silicate", "units": "mmol/m^3"},
                "NH4": {"long_name": "dissolved ammonia", "units": "mmol/m^3"},
                "Fe": {"long_name": "dissolved inorganic iron", "units": "mmol/m^3"},
                "Lig": {"long_name": "iron binding ligand", "units": "mmol/m^3"},
                "O2": {"long_name": "dissolved oxygen", "units": "mmol/m^3"},
                "DIC": {"long_name": "dissolved inorganic carbon", "units": "mmol/m^3"},
                "DIC_ALT_CO2": {"long_name": "dissolved inorganic carbon, alternative CO2", "units": "mmol/m^3"},
                "ALK": {"long_name": "alkalinity", "units": "meq/m^3"},
                "ALK_ALT_CO2": {"long_name": "alkalinity, alternative CO2", "units": "meq/m^3"},
                "DOC": {"long_name": "dissolved organic carbon", "units": "mmol/m^3"},
                "DON": {"long_name": "dissolved organic nitrogen", "units": "mmol/m^3"},
                "DOP": {"long_name": "dissolved organic phosphorus", "units": "mmol/m^3"},
                "DOCr": {"long_name": "refractory dissolved organic carbon", "units": "mmol/m^3"},
                "DONr": {"long_name": "refractory dissolved organic nitrogen", "units": "mmol/m^3"},
                "DOPr": {"long_name": "refractory dissolved organic phosphorus", "units": "mmol/m^3"},
                "zooC": {"long_name": "zooplankton carbon", "units": "mmol/m^3"},
                "spChl": {"long_name": "small phytoplankton chlorophyll", "units": "mg/m^3"},
                "spC": {"long_name": "small phytoplankton carbon", "units": "mmol/m^3"},
                "spP": {"long_name": "small phytoplankton phosphorous", "units": "mmol/m^3"},
                "spFe": {"long_name": "small phytoplankton iron", "units": "mmol/m^3"},
                "spCaCO3": {"long_name": "small phytoplankton CaCO3", "units": "mmol/m^3"},
                "diatChl": {"long_name": "diatom chloropyll", "units": "mg/m^3"},
                "diatC": {"long_name": "diatom carbon", "units": "mmol/m^3"},
                "diatP": {"long_name": "diatom phosphorus", "units": "mmol/m^3"},
                "diatFe": {"long_name": "diatom iron", "units": "mmol/m^3"},
                "diatSi": {"long_name": "diatom silicate", "units": "mmol/m^3"},
                "diazChl": {"long_name": "diazotroph chloropyll", "units": "mg/m^3"},
                "diazC": {"long_name": "diazotroph carbon", "units": "mmol/m^3"},
                "diazP": {"long_name": "diazotroph phosphorus", "units": "mmol/m^3"},
                "diazFe": {"long_name": "diazotroph iron", "units": "mmol/m^3"}
                }
        return d
    
    def get_boundary_info(self):

        # Boundary coordinates
        bdry_coords = {
                "rho": {
            "south": {"eta_rho": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_rho": 0},
        },
                "u": {
            "south": {"eta_rho": 0},
            "east": {"xi_u": -1},
            "north": {"eta_rho": -1},
            "west": {"xi_u": 0},
        },
                "v":
        {
            "south": {"eta_v": 0},
            "east": {"xi_rho": -1},
            "north": {"eta_v": -1},
            "west": {"xi_rho": 0},
        }
        }

        # How to rename the dimensions

        rename = {"rho": {
            "south": {"xi_rho": "xi_rho_south"},
            "east": {"eta_rho": "eta_rho_east"},
            "north": {"xi_rho": "xi_rho_north"},
            "west": {"eta_rho": "eta_rho_west"},
        },
                  "u": {
            "south": {"xi_u": "xi_u_south"},
            "east": {"eta_rho": "eta_u_east"},
            "north": {"xi_u": "xi_u_north"},
            "west": {"eta_rho": "eta_u_west"},
        },
                  "v": {
            "south": {"xi_rho": "xi_v_south"},
            "east": {"eta_v": "eta_v_east"},
            "north": {"xi_rho": "xi_v_north"},
            "west": {"eta_v": "eta_v_west"},
        }
                  }

        return bdry_coords, rename

