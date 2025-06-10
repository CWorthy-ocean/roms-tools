# Datasets

To use `ROMS-Tools`, the user is required to [download](#downloading-era5-and-glorys) the following datasets:


| **Dataset** | **Supported Versions** | **Required Fields** | **Field Description**                                                      | **Available at**                                                                 | **Required For**    |
|-------------|------------------------|---------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|
| SRTM15        | V2.6   | `lat`             | Latitude (degrees north)                                     | [USCD SRTM15+ Product](https://topex.ucsd.edu/WWW_html/srtm15_plus.html)                             | Grid (Topography)       |
|         |    | `lon`             | Longitude (degrees east)                                     |                              |        |
|         |    | `z`             | Topography                                     |                              |        |
|-------------|------------------------|---------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|
| TPXO        | TPXO9v5a (1/6 degree)  | `lat_z`             | Latitude of z nodes (degrees north)                                     | [OSU TPXO Tide Models](https://www.tpxo.net/global)                             | Tidal Forcing       |
|             | TPXO10v2 (1/6 degree)  | `lon_z`             | Longitude of z nodes (degrees east)                                     |                                                                                  |                     |
|             | TPXO10v2a (1/6 degree) | `lat_u`             | Latitude of u nodes (degrees north)                                     |                                                                                  |                     |
|             |                        | `lon_u`             | Longitude of u nodes (degrees east)                                     |                                                                                  |                     |
|             |                        | `lat_v`             | Latitude of v nodes (degrees north)                                     |                                                                                  |                     |
|             |                        | `lon_v`             | Longitude of v nodes (degrees east)                                     |                                                                                  |                     |
|             |                        | `mz`                | Water/land mask for z nodes                                             |                                                                                  |                     |
|             |                        | `mu`                | Water/land mask for u nodes                                             |                                                                                  |                     |
|             |                        | `mv`                | Water/land mask for v nodes                                             |                                                                                  |                     |
|             |                        | `hRe`               | Tidal elevation, real part (m)                                              |                                                                                  |                     |
|             |                        | `hIm`               | Tidal elevation, imaginary part (m)                                         |                                                                                  |                     |
|             |                        | `URe`               | Tidal transport WE, real part (m²/s)                                       |                                                                                  |                     |
|             |                        | `UIm`               | Tidal transport WE, imaginary part (m²/s)                                  |                                                                                  |                     |
|             |                        | `VRe`               | Tidal transport SN, real part (m²/s)                                       |                                                                                  |                     |
|             |                        | `VIm`               | Tidal transport SN, imaginary part (m²/s)                                  |                                                                                  |                     |
|-------------|------------------------|---------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|
| ERA5        | -                      | `time`              | Time                                                                      | [Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | Surface Forcing     |
|             |                        | `latitude`          | Latitude (degrees north)                                                  |                                                                                  |                     |
|             |                        | `longitude`         | Longitude (degrees east)                                                  |                                                                                  |                     |
|             |                        | `u10`               | 10m U wind component (m/s)                                                 |                                                                                  |                     |
|             |                        | `v10`               | 10m V wind component (m/s)                                                 |                                                                                  |                     |
|             |                        | `ssr`               | Surface net short-wave (solar) radiation (W/m²)                            |                                                                                  |                     |
|             |                        | `strd`              | Surface long-wave (thermal) radiation downwards (W/m²)                     |                                                                                  |                     |
|             |                        | `t2m`               | 2m temperature (K)                                                        |                                                                                  |                     |
|             |                        | `d2m`               | 2m dewpoint temperature (K)                                               |                                                                                  |                     |
|             |                        | `tp`                | Total precipitation (m) (converted to mm for use in model)                 |                                                                                  |                     |
|             |                        | `sst`               | Sea surface temperature (K) — used for land masking                        |                                                                                  |                     |
|-------------|------------------------|---------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|
| GLORYS      | -                      | `time`              | Time                                                                      | [Mercator Ocean](https://www.mercator-ocean.eu/en/ocean-science/glorys/) | Initial Conditions, Boundary Forcing     |
|             |                        | `latitude`          | Latitude (degrees north)                                                  |                                                                                  |                     |
|             |                        | `longitude`         | Longitude (degrees east)                                                  |                                                                                  |                     |
|             |                        | `depth`         | Depth (m)                                                  |                                                                                  |                     |
|             |                        | `zos`               | Sea surface height (m)                                                    |                                                                                  |                     |
|             |                        | `thetao`            | Temperature (degrees C)                                                   |                                                                                  |                     |
|             |                        | `so`                | Salinity (psu)                                                            |                                                                                  |                     |
|             |                        | `uo`                | Eastward velocity (m/s)                                                   |                                                                                  |                     |
|             |                        | `vo`                | Northward velocity (m/s)                                                  |                                                                                  |                     |
|-------------|------------------------|---------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|
| Dai and Trenberth (coastal station volume, monthly series); **Note: downloaded internally**      | 2019                      | `station`              | Station index                                                                      | [NCAR Research Data Archive](https://rda.ucar.edu/datasets/d551000/dataaccess/#) | River Forcing     |
|   |                       | `time`              | Time                                                                      | |      |
|  | -                      | `lat_mou`              | River mouth latitude                                                  |  |      |
|  | -                      | `lon_mou`              | River mouth longitude                                                  |  |      |
|  | -                      | `FLOW`              | Monthly mean volume at station                                                  |  |      |
|  | -                      | `ratio_m2s`              | Ratio of volume between river mouth and station                                                  |  |      |
|  | -                      | `riv_name`              | River name                                                  |  |      |
|  | -                      | `vol_stn` (optional)             | Annual volume at station                                                  |  |      |

# Downloading ERA5 and GLORYS

## Prerequisites

Install 2 packages into your conda python environment:

**Copernicusmarine**: https://pypi.org/project/copernicusmarine/
**cdsapi**: https://anaconda.org/conda-forge/cdsapi

## GLORYS

### Setup

Installation instructions and set up account credentials in the shell: [here](https://help.marine.copernicus.eu/en/articles/7970514-copernicus-marine-toolbox-installation)
But, essentially do:
- 1st need to make an account with marine.copernicus: https://marine.copernicus.eu/
- Export account credentials in `~/.bashrc`:
- Linux:

```bash
export COPERNICUSMARINE_SERVICE_USERNAME=username
export COPERNICUSMARINE_SERVICE_PASSWORD=password
```

### Download

Run the download script (via Scott B) below to extract data for *salt, temp, u, v, sea surface height*. Specify the years, month, and number of days per month. Adjust lat lon if needed. Pulls from Global Ocean Physics Reanalysis.

```python
import copernicusmarine
years = ['2012']
months = ['01','02','03','04','05']
days = [31,29,31,30,31]
for y in years:
  for count,m in enumerate(months):
    for d in range(1,days[count]+1):
      if d < 10:
        extra_str='0'
      else:
        extra_str=''
      copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["so", "thetao", "uo", "vo", "zos"],
        minimum_longitude=-75,
        maximum_longitude=-15,
        minimum_latitude=2,
        maximum_latitude=50,
        start_datetime= y + "-" + m + "-" + extra_str + str(d) + "T00:00:00",
        end_datetime= y + "-" + m + "-" + extra_str + str(d) + "T00:00:00",
        minimum_depth=0.49402499198913574,
        maximum_depth=5727.9,
)
```
## ERA5

### Setup

Installation instructions and set up account credentials in the shell: [here](https://help.marine.copernicus.eu/en/articles/7970514-copernicus-marine-toolbox-installation)

But, essentially do:

- Make account at: https://cds.climate.copernicus.eu/
- Login on web
- Setup the api: https://cds.climate.copernicus.eu/how-to-api
    - Add this to `$HOME/.cdsapirc`:

```bash
url: https://cds.climate.copernicus.eu/api
key: 77deaea8-7d41-4b8e-967f-90d6d1849c0e
```

- Make sure to have correct version: `$ pip install "cdsapi>=0.7.4"`
- Sign the license agreement: https://cds.climate.copernicus.eu/profile?tab=licences

### Download

Run the download script (via Scott B via Abigale) below to extract data for *wind, dewpoint, temp, sst, precipitation, net solar radiation, surface downward thermal radiation*. Specify the years, months.

```python
# This script downloads the ERA5 files using cdsapi
# The variables selected are those needed to create forcing files in roms-tools as of May 2025
# before using this script, you must be in an environment with cdsapi installed 

import cdsapi
import zipfile
import shutil
import os
from tempfile import mkdtemp

for yr in range(2017,2023):
    for m in range(1,13):
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "sea_surface_temperature",
                "total_precipitation",
                "surface_net_solar_radiation",
                "surface_thermal_radiation_downwards"
            ],
            "year": [str(yr)],
            "month": [str(m).zfill(2)],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "zip",
#            "area": [80, -295, -80, -40]
        }

        target = dataset+ '_' + str(yr)  + '-' + str(m).zfill(2)
        client = cdsapi.Client()
        client.retrieve(dataset, request, target + '.zip')

#Unzip files
        print('unzipping file')
        with zipfile.ZipFile('./' + target + '.zip', 'r') as zip_ref:
            directory_to_extract_to = mkdtemp()
            zip_ref.extractall(directory_to_extract_to)

            source_path = directory_to_extract_to + '/' + 'data_stream-oper_stepType-instant.nc'
            destination_path = './' + target + '_inst.nc'
            shutil.move(source_path, destination_path)

            source_path = directory_to_extract_to + '/' + 'data_stream-oper_stepType-accum.nc'
            destination_path = './' + target + '_accum.nc'
            shutil.move(source_path, destination_path)

        os.remove('./' + target + '.zip')
        print(target + ' download complete')
```
