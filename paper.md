---
title: 'ROMS-Tools: Reproducible Preprocessing and Analysis for Regional Ocean Modeling with ROMS'
tags:
  - Python
  - ocean modeling
  - ROMS
  - marine carbon dioxide removal
authors:
  - name: Nora Loose
    orcid: 0000-0002-3684-9634
    affiliation: 1
  - name: Tom Nicholas
    orcid: 0000-0002-2176-0530
    affiliation: 2
  - name: Scott Eilerman
    orcid:
    affiliation: 1
  - name: Christopher McBride
    orcid:
    affiliation: 1
  - name: Sam Maticka
    orcid:
    affiliation: 1
  - name: Dafydd Stephenson
    orcid:
    affiliation: 1
  - name: Scott Bachman
    orcid:
    affiliation: 1
  - name: Pierre Damien
    orcid:
    affiliation: 3
  - name: Ulla Heede
    orcid:
    affiliation: 1
  - name: Alicia Karspeck
    orcid:
    affiliation: 1
  - name: Matthew C. Long
    orcid:
    affiliation: 1
  - name: M. Jeroen Molemaker
    orcid:
    affiliation: 3
  - name: Abigale Wyatt
    orcid: 0000-0003-2428-4370
    affiliation: 1
affiliations:
 - name: '[C]Worthy LLC, Boulder, CO, United States'
   index: 1
 - name: Earthmover PBC
   index: 2
 - name: University of California, Los Angeles, CA, United States
   index: 3
date: 3 November 2025
bibliography: docs/references.bib

---

# Summary

The ocean regulates Earth’s climate and sustains marine ecosystems by circulating and storing heat, carbon, oxygen, and nutrients, while exchanging heat and gases with the atmosphere. Scientists study these processes using ocean models, which simulate the ocean on a grid.
**Regional ocean models** focus computational resources on a limited geographical area with fine grid spacing, and can resolve fine-scale phenomena such as mesoscale and submesoscale features, tidal dynamics, coastal currents, upwelling, and detailed biogeochemical (BGC) processes. A widely used regional ocean model is the **Regional Ocean Modeling System (ROMS)** [@shchepetkin_regional_2005]. ROMS has been coupled to the Marine Biogeochemistry Library (MARBL) [@long_simulations_2021; @ucla-roms] to link physical and BGC processes. ROMS-MARBL supports research on environmental management, fisheries, regional climate impacts, and ocean-based carbon dioxide removal (CDR) strategies.

Configuring a regional ocean model like ROMS-MARBL is technically challenging; it requires initialization and time-dependent forcing from oceanic and atmospheric data drawn from multiple external sources in diverse formats. These global datasets can reach several petabytes and must be subsetted, processed, and mapped onto the target domain’s geometry, yielding input datasets of 10–100 terabytes for large regional models. Generating these input files is time-consuming, error-prone, and hard to reproduce, creating a bottleneck for both new and experienced users. The Python package `ROMS-Tools` addresses this challenge by providing efficient, `dask`-backed [@dask], user-friendly tools that can be installed via Conda or PyPI and run interactively from Jupyter notebooks. The package supports creating regional grids, preprocessing all required model inputs, and postprocessing and analysis. Current capabilities are fully compatible with UCLA-ROMS [@ucla-roms; @ucla-roms-cworthy], with potential support for other ROMS versions, such as Rutgers ROMS [@rutgers-roms], in the future.


## Input Data and Preprocessing

`ROMS-Tools` generates the following input files for ROMS-MARBL:

1. **Model Grid**: Customizable, curvilinear, and orthogonal grid designed to maintain a nearly uniform horizontal resolution across the domain. The grid is rotatable to align with coastlines and features a terrain-following vertical coordinate.
2. **Bathymetry**: Derived from **SRTM15** [@tozer_global_2019].
3. **Land Mask**: Inferred from coastlines provided by **Natural Earth** or the Global Self-consistent, Hierarchical, High-resolution Geography (**GSHHG**) Database [@wessel_global_1996].
4. **Physical Ocean Conditions**:  Initial and open boundary conditions for sea surface height, temperature, salinity, and velocities derived from the 1/12° Global Ocean Physics Reanalysis (**GLORYS**) [@jean-michel_copernicus_2021].
5. **BGC Ocean Conditions**: Initial and open boundary conditions for dissolved inorganic carbon, alkalinity, and other biogeochemical tracers from Community Earth System Model (**CESM**) output [@yeager_2022] or hybrid observational-model sources [@garcia2019woa; @lauvset_new_2016; @huang_data-driven_2022; @yang_global_2020; @yeager_2022].
6. **Meteorological forcing**: Wind, radiation, precipitation, and air temperature/humidity processed from the global 1/4° ECMWF Reanalysis v5 (**ERA5**) [@hersbach_era5_2020] with optional corrections for radiation bias and coastal wind.
7. **BGC surface forcing**: Partial pressure of carbon dioxide, as well as iron, dust, and nitrogen deposition from **CESM** output [@yeager_2022] or hybrid observational-model sources [@landschutzer_decadal_2016; @kok_improved_2021; @hamilton_earth_2022; @yeager_2022].
8. **Tidal Forcing:** Tidal potential, elevation, and velocities derived from **TPXO** [@egbert_efficient_2002] including self-attraction and loading (SAL) corrections.
9. **River Forcing:** Freshwater runoff derived from **Dai & Trenberth** [@dai_estimates_2002] or user-provided custom files.
10. **CDR Forcing**: User-defined interventions that inject BGC tracers at point sources or as larger-scale Gaussian perturbations to simulate CDR interventions. The CDR forcing is prescribed as volume and tracer fluxes (e.g., alkalinity for ocean alkalinity enhancement, iron for iron fertilization, or other BGC constituents).  Users can control the magnitude, spatial footprint, and temporal evolution, allowing flexible representation of CDR interventions.

Some source datasets are accessed automatically by the package, including Natural Earth, Dai & Trenberth runoff, and ERA5 meteorology, while users must manually download SRTM15, GSHHG, GLORYS, the BGC datasets, and TPXO tidal files. Although these are the datasets currently supported, the package’s modular design makes it straightforward to add new source datasets in the future.

To generate the model inputs, `ROMS-Tools` automates several intermediate processing steps, including:

* **Bathymetry processing**: The bathymetry is smoothed in two stages, first across the entire model domain and then locally in areas with steep slopes, to ensure local steepness ratios do not exceed a prescribed threshold in order to reduce pressure-gradient errors. A minimum depth is enforced to prevent water levels from becoming negative during large tidal excursions.
* **Mask definition**: The land-sea mask is generated by comparing the ROMS grid’s horizontal coordinates with a coastline dataset using the `regionmask` package [@hauser_regionmaskregionmask_2024]. Enclosed basins are subsequently filled with land.
* **Land value handling**: Land values are filled via an algebraic multigrid method using `pyamg` [@pyamg2023] prior to horizontal regridding. This extends ocean values into land areas to resolve discrepancies between source data and ROMS land masks, preventing land-originating values from appearing in ocean cells.
* **Regridding**: Ocean and atmospheric fields are horizontally and vertically regridded from standard latitude-longitude-depth grids to the model’s curvilinear grid with a terrain-following vertical coordinate using `xarray` [@hoyer2017xarray] and `xgcm` [@xgcm]. Velocities are rotated to align with the curvilinear ROMS grid.
* **Longitude conventions**: `ROMS-Tools` handles differences in longitude conventions, converting between [-180°, 180°] and [0°, 360°] as needed.
* **River locations**: Rivers that fall within the model domain are automatically identified and relocated to the nearest coastal grid cell. Rivers that need to be shifted manually or span multiple cells can be configured by the user.
* **Data streaming**: ERA5 atmospheric data can be accessed directly from the cloud, removing the need for users to pre-download large datasets locally. Similar streaming capabilities may be implemented for other datasets in the future.

Users can quickly design and visualize regional grids and inspect all input fields with built-in plotting utilities. An example of surface initial conditions generated for a California Current System simulation at 5 km horizontal grid spacing is shown in \autoref{fig:example}.

![Surface initial conditions for the California Current System created with `ROMS-Tools` from GLORYS. Left: potential temperature. Right: zonal velocity. Shown for January 1, 2000.\label{fig:example}](docs/images/ics_from_glorys.png){ width=100% }

## Postprocessing and Analysis

`ROMS-Tools` supports postprocessing and analysis of ROMS-MARBL output, including regridding from the native curvilinear, terrain-following grid to a standard latitude-longitude-depth grid using `xesmf` [@xesmf], with built-in plotting for both grid types. The analysis layer also includes specialized utilities for evaluating carbon dioxide removal (CDR) interventions, such as generating carbon uptake and efficiency curves.

# Statement of Need

Setting up a regional ocean model is a major technical undertaking. Traditionally, this work has relied on a patchwork of custom scripts and lab-specific workflows, which can be time-consuming, error-prone, and difficult to reproduce. These challenges slow down science, create a steep barrier to entry for new researchers, and limit collaboration across groups.

Within the ROMS community, the preprocessing landscape has been shaped by tools like `pyroms` [@pyroms]. While providing valuable low-level utilities, `pyroms` presents challenges for new users: installation is cumbersome due to Python/Fortran dependencies, the API is inconsistent, documentation is limited, and it lacks tests, CI, and support for modern Python tools like `xarray` and `dask`. Since active development has largely ceased, its suitability for new projects, such as CDR simulations, is limited.

Tools from other modeling communities cannot simply be adopted, since each ocean model has distinct structural requirements. For example, the `regional-mom6` package [@barnes_regional-mom6_2024], developed for the Modular Ocean Model v6 (MOM6) [@adcroft_gfdl_2019], cannot be used to generate ROMS inputs, because ROMS employs a terrain-following vertical coordinate system that requires a specialized vertical regridding approach, whereas MOM6 accepts inputs on arbitrary depth levels and does not require vertical regridding at all. Several other differences further prevent cross-compatibility. Together, these limitations underscored the need for a modern, maintainable, and reproducible tool designed specifically for ROMS.\footnote{In the future, packages like ROMS-Tools and regional-mom6 could share a common backbone, with model-specific adaptations layered on top.}

`ROMS-Tools` was developed to meet this need. It draws on the legacy of the MATLAB preprocessing scripts developed at UCLA [@ucla-matlab], which encapsulate decades of expertise in configuring regional ocean model inputs. While many of the core algorithms and design principles are retained, `ROMS-Tools` provides an open-source Python implementation of these MATLAB tools using an object-oriented programming paradigm. This implementation supports a modern workflow with high-level API calls, improving reproducibility, minimizing user errors, and allowing easy extension to new features, forcing datasets, and use cases. In some cases, `ROMS-Tools` diverges from the MATLAB implementation to take advantage of new methods or better integration with the modern Python ecosystem. By streamlining input generation and analysis, `ROMS-Tools` reduces technical overhead, lowers the barrier to entry, and enables scientists to focus on research rather than data preparation.


# Software Design

`ROMS-Tools` is designed to balance **ease of use, flexibility, reproducibility, and scalability** by combining high-level user interfaces with a modular, extensible architecture.

## Lessons from MATLAB Tools

The legacy MATLAB preprocessing scripts were powerful but required users to edit source code directly to configure simulations. This workflow led to frequent errors for new users, made it difficult to track completed steps, and limited reproducibility. `ROMS-Tools` addresses these issues with **high-level API calls**, automated error-prone steps, and explicit workflow state management via YAML.

## Design Trade-Offs

A central design trade-off in `ROMS-Tools` is between **user control** and **automation**. Rather than enforcing a fixed workflow, the package exposes key choices such as physical options (e.g., corrections for radiation or wind), interpolation and fill methods, and computational backends. This approach contrasts with opinionated frameworks that fix defaults and directory structures to maximize automation. While users must make explicit decisions, some steps remain automated to prevent errors. For example, bathymetry smoothing is applied automatically with a non-tunable parameter, since overly small smoothing factors could produce rough bathymetry and crash simulations due to pressure gradient errors. This design choice directly addresses issues new users faced in the MATLAB scripts, and balances **flexibility** and **safety**, enabling transparent experimentation without exposing users to avoidable pitfalls.

Another key design consideration is balancing **modular, incremental workflow steps** with **reproducibility**. `ROMS-Tools` organizes tasks (such as creating `InitialConditions`, `BoundaryForcing`, and `SurfaceForcing`) into small, composable components that can be executed, saved, and revisited independently, rather than following a monolithic, fixed workflow. All components depend on the `Grid`, but once it is created, the remaining objects are independent. This modular approach avoids unnecessary recomputation when only some inputs change but requires careful tracking of workflow state. To ensure reproducibility, all configuration choices are stored in compact, text-based YAML files. These files are version-controllable, easy to share, and eliminate the need to transfer large model input NetCDF datasets. This design directly addresses the MATLAB scripts’ lack of explicit workflow tracking.

## Architecture

At the user-facing level, `ROMS-Tools` provides high-level objects such as `Grid`, `InitialConditions`, and `BoundaryForcing`. Each object exposes a consistent interface (`.ds`, `.plot()`, `.save()`, `.to_yaml()`), allowing users to call the same methods in sequence or inspect attributes that are always present. This design reduces cognitive overhead, makes workflows predictable, and removes the need for new users to edit raw scripts or manually track intermediate files, as was required with the MATLAB tools.

Internally, `ROMS-Tools` follows a **layered, modular architecture**. Low-level classes (`LatLonDataset`, `ROMSDataset`) handle data ingestion and preprocessing, including common operations such as subdomain selection and lateral filling. Source-specific datasets (e.g., `ERA5Dataset`, `GLORYSDataset`, `SRTMDataset`) inherit from these base classes and encode dataset-specific conventions like variable names, coordinates, and masking. Adding support for a new data source typically requires only a small subclass to define variable mappings while reusing existing logic, minimizing changes to the core code. High-level classes (`Grid`, `InitialConditions`, `BoundaryForcing`) build on these low-level datasets to produce ready-to-use modeling inputs, performing tasks such as regridding and final assembly. This layered design enhances **extensibility and maintainability**, avoiding the pitfalls of the monolithic MATLAB scripts.

## Computational and Data Model Choices

`ROMS-Tools` is built on `xarray`, which provides a clear, consistent interface for exploring and inspecting labeled, multi-dimensional geophysical datasets. Users can take advantage of `xarray`’s intuitive indexing, plotting, and metadata handling. Optional `dask` support allows workflows to scale from laptops to HPC systems, enabling parallel and out-of-core computation for very large input and output datasets. By combining modern Python tools with a user-friendly interface, `ROMS-Tools` addresses the usability challenges that hampered new users in the MATLAB-based workflow.

# Research Impact Statement

`ROMS-Tools` is used by two primary research communities. First, regional ocean modelers use it to generate reproducible input datasets for ROMS simulations; external users include researchers at **PNNL**, **WHOI**, and **UCLA**. Second, researchers in the ocean-based carbon dioxide removal (CDR) community use `ROMS-Tools` to configure reproducible ROMS–MARBL simulations of climate intervention scenarios, with adopters including **[C]Worthy**, **Carbon to Sea**, **Ebb Carbon**, and **SCCWRP**. All of these groups have contacted the developers directly or engaged in offline discussions regarding their use of the package.

Additional evidence of community uptake comes from public usage metrics. At the time of writing, the GitHub repository shows **119 unique cloners in the past 14 days**, with stars from users at institutions including the University of Waikato, NCAR, University of Maryland, National Oceanography Centre, McGill University, UC Santa Cruz, and others. Distribution statistics indicate **over 3,100 conda-forge downloads in the past six months**, including **68 downloads of the most recent release (v3.3.0)**, and **more than 48,000 total PyPI downloads** (noting that PyPI counts include automated CI usage, whereas conda downloads do not).

`ROMS-Tools` is also integrated into broader workflows, including **C-Star** [@cstar], an open-source platform to provide scientifically credible monitoring, reporting, and verification (MRV) for the emerging marine carbon market.

# AI Usage Disclosure

Generative AI tools were used to assist with writing docstrings and developing tests for the `ROMS-Tools` software, to improve the clarity and readability of the documentation, and to shorten and edit portions of the manuscript text. All AI-assisted content was reviewed and verified by the authors for technical accuracy and correctness.

# Acknowledgements

Development of `ROMS-Tools` has been supported by ARPA-E (DE-AR0001838) and philanthropic donations to [C]Worthy from the Grantham Foundation for the Environment, the Chan Zuckerberg Initiative, Founders Pledge, and the Ocean Resilience Climate Alliance.

# References
