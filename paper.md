---
title: 'ROMS-Tools: A Python Package for Preparing and Analyzing ROMS Simulations'
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

The ocean shapes Earth’s climate and sustains marine ecosystems by circulating and storing vast amounts of heat, oxygen, carbon, and nutrients, while exchanging heat and gases with the atmosphere. To understand these complex dynamics, scientists rely on ocean models, powerful computer simulations of physical circulation and biogeochemical (BGC) processes. These models represent the ocean on a grid of cells, where finer grid spacing (more, smaller cells) provides higher fidelity and greater detail at the cost of  more computing power. While global ocean models simulate the entire ocean, **regional ocean models** focus computational resources on a specific area to achieve much finer grid spacing than is computationally feasible over the global domain. This finer grid spacing enables regional ocean models to explicitly resolve fine-scale phenomena, like mesoscale (10-100 km) and submesoscale (0.1-10 km) features, tidal dynamics, coastal currents, upwelling, and detailed BGC processes. Capturing these dynamics and processes at high fidelity is essential for applications in environmental management, fisheries, for assessing regional impacts of climate change, and for evaluating ocean-based carbon dioxide removal (CDR) strategies.

A widely used regional ocean model is the **Regional Ocean Modeling System (ROMS)** [@shchepetkin_regional_2005]. To connect physical circulation with ecosystem dynamics and the ocean carbon cycle, ROMS has been coupled to a BGC model called the Marine Biogeochemistry Library (MARBL) [@long_simulations_2021; @ucla-roms]. This coupled framework allows researchers to explore a variety of scientific and practical questions.  For example, it can be used to investigate the potential of ocean-based carbon removal strategies, such as adding alkaline materials to the ocean to sequester atmospheric carbon dioxide. It can also be used to study how physical processes drive ecosystem dynamics, such as how nutrient-rich waters from upwelling fuel the phytoplankton blooms that form the base of the marine food web [@gruber_eddy-resolving_2006].

## Input Data and Preprocessing

Whether for research or industrial-focused applications, configuring a regional ocean model like ROMS-MARBL remains a major technical challenge. The model must be initialized and forced over time with relevant oceanic and atmospheric data, which often come from multiple external data providers in a variety of formats and can reach several petabytes for global, multi-purpose datasets. These data must be subsetted, processed, and mapped onto the specific model geometry of the target domain, resulting in input datasets that can still be on the order of 10-100 terabytes for larger regional models. Generating these bespoke input files is time-consuming, error-prone, and difficult to reproduce, creating a bottleneck for both new and experienced model users. The Python package `ROMS-Tools` addresses this challenge by providing a set of  efficient, user-friendly, and extensible tools to design new regional grids for  ROMS-MARBL and to process and stage all required model input files. `ROMS-Tools` supports reproducible and easy-to-interpret workflows that enable faster and more robust ROMS-MARBL setups. The package’s user interface and underlying data model are based on `xarray` [@hoyer2017xarray], allowing seamless handling of multidimensional datasets with rich metadata and optional parallelization via a `dask` [@dask] backend.

`ROMS-Tools` can automatically process commonly used datasets or incorporate custom user data and routines. Currently, it can generate the following inputs:

1. **Model Grid**: Customizable, curvilinear, and orthogonal grid designed to maintain a nearly uniform horizontal resolution across the domain. The grid is rotatable to align with coastlines and features a terrain-following vertical coordinate.
2. **Bathymetry**: Derived from **SRTM15** [@tozer_global_2019].
3. **Land Mask**: Inferred from coastlines provided by **Natural Earth** or the Global Self-consistent, Hierarchical, High-resolution Geography (**GSHHG**) Database [@wessel_global_1996].
4. **Physical Ocean Conditions**:  Initial and open boundary conditions for sea surface height, temperature, salinity, and velocities derived from the 1/12° Global Ocean Physics Reanalysis (**GLORYS**) [@jean-michel_copernicus_2021].
5. **BGC Ocean Conditions**: Initial and open boundary conditions for dissolved inorganic carbon, alkalinity, and other biogeochemical tracers from Community Earth System Model (**CESM**) output [@yeager_2022] or hybrid observational-model sources [@garcia2019woa; @lauvset_new_2016; @huang_data-driven_2022; yang_global_2020; @yeager_2022]
6. **Meteorological forcing**: Wind, radiation, precipitation, and air temperature/humidity processed from the global 1/4° ECMWF Reanalysis v5 (**ERA5**) [@hersbach_era5_2020] with optional corrections for radiation bias and coastal wind.
7. **BGC surface forcing**: Partial pressure of carbon dioxide, as well as iron, dust, and nitrogen deposition from **CESM** output [@yeager_2022] or hybrid observational-model sources [@landschutzer_decadal_2016; @kok_improved_2021; @hamilton_earth_2022; @yeager_2022].
8. **Tidal Forcing:** Tidal potential, elevation, and velocities derived from **TPXO** [@egbert_efficient_2002] including self-attraction and loading (SAL) corrections.
9. **River Forcing:** Freshwater runoff derived from **Dai & Trenberth** [@dai_estimates_2002] or user-provided custom files.
10. **CDR Forcing**: User-defined interventions that inject BGC tracers at point sources or as larger-scale Gaussian perturbations, designed to simulate CDR interventions. The CDR forcing provides an external forcing term prescribed as volume and tracer fluxes (e.g., alkalinity for ocean alkalinity enhancement, iron for iron fertilization, or other BGC constituents).  Users can specify the magnitude, spatial footprint, and time dependence of the forcing, enabling flexible representation of CDR interventions.

Some source datasets are accessed automatically by the package, including Natural Earth, Dai & Trenberth runoff, and ERA5 meteorology, while users must manually download SRTM15, GSHHG, GLORYS, the BGC datasets, and TPXO tidal files. While the source datasets listed above are the ones currently supported, the package’s modular design makes it straightforward to add new data sources or custom routines in the future.
To generate the model inputs listed above, `ROMS-Tools` automates several intermediate processing steps, including:

* **Bathymetry processing**: The bathymetry is smoothed in two stages, first across the entire model domain and then locally in areas with steep slopes, to ensure local steepness ratios do not exceed a prescribed threshold in order to reduce pressure-gradient errors. A minimum depth is enforced to prevent water levels from becoming negative during large tidal excursions.
* **Mask definition**: The land-sea mask is generated by comparing the ROMS grid’s horizontal coordinates with a coastline dataset using the `regionmask` package [@hauser_regionmaskregionmask_2024]. Enclosed basins are subsequently filled with land.
* **Land value handling**: Land values are filled via an algebraic multigrid method using `pyamg` [@pyamg2023] prior to horizontal regridding. This extends ocean values into land areas to resolve discrepancies between source data and ROMS land masks, preventing land-originating values from appearing in ocean cells.
* **Regridding**: Ocean and atmospheric fields are horizontally and vertically regridded from standard latitude-longitude-depth grids to the model’s curvilinear grid with a terrain-following vertical coordinate using `xarray` [@hoyer2017xarray]. Optional sea surface height corrections can be applied, and velocities are rotated to align with the curvilinear ROMS grid.
* **Longitude conventions**: `ROMS-Tools` handles differences in longitude conventions, converting between [-180°, 180°] and [0°, 360°] as needed.
* **River locations**: Rivers that fall within the model domain are automatically identified and relocated to the nearest coastal grid cell. Rivers that need to be shifted manually or span multiple cells can be configured by the user.
* **Data streaming**: ERA5 atmospheric data can be accessed directly from the cloud, removing the need for users to pre-download large datasets locally. Similar streaming capabilities may be implemented for other datasets in the future.

Users can quickly design and visualize regional grids and inspect all input fields with built-in plotting utilities. An example of surface initial conditions generated for a California Current System simulation at 5 km horizontal grid spacing is shown in \autoref{fig:example}.

![Surface initial conditions for the California Current System created with `ROMS-Tools` from GLORYS. Left: potential temperature. Right: zonal velocity. Shown for January 1, 2000.\label{fig:example}](docs/images/ics_from_glorys.png){ width=100% }

`ROMS-Tools` also includes features that facilitate simulation management. It supports partitioning input files to enable parallelized ROMS simulations across multiple nodes, and writes NetCDF outputs with metadata fully compatible with ROMS-MARBL. Currently, all capabilities in ROMS-Tools are fully compatible with UCLA-ROMS [@ucla-roms; @ucla-roms-cworthy], with the potential to add other ROMS versions, such as Rutgers ROMS [@rutgers-roms], in the future.

## Postprocessing and Analysis

`ROMS-Tools` also includes analysis tools for postprocessing ROMS-MARBL output. It first provides a joining tool (the counterpart to the input file partitioning utility described earlier) that merges ROMS output files produced as tiles from multi-node simulations. Beyond file management, there are `ROMS-Tools` analysis utilities for general-purpose tasks, such as loading model output directly into an `xarray` dataset with additional useful metadata, enabling seamless use of the Pangeo scientific Python ecosystem for further analysis and visualization. The analysis layer also supports regridding from the native curvilinear ROMS grid with terrain-following coordinate to a standard latitude-longitude-depth grid using `xesmf` [@xesmf], and includes built-in plotting on both the native and latitude–longitude–depth grids. Beyond these general-purpose features, the `ROMS-Tools` analysis layer offers a suite of targeted tools for evaluating CDR interventions. These include utilities for generating standard plots, such as CDR efficiency curves, and performing specialized tasks essential for CDR monitoring, reporting, and verification.

## Workflow, Reproducibility, and Performance

`ROMS-Tools` is designed to support modern, reproducible workflows. It is easily installable via Conda or PyPI and can be run interactively from Jupyter Notebooks.
To ensure reproducibility and facilitate collaboration, each workflow is defined in a simple YAML configuration file. These compact, text-based YAML files can be version-controlled and easily shared, eliminating the need to transfer large NetCDF files between researchers, as source data like GLORYS and ERA5 are accessible in the cloud.
For performance, the package is integrated with `dask` [@dask] to enable efficient, out-of-core computations on large datasets.
Finally, to ensure reliability, the software is rigorously tested with continuous integration (CI) and supported by comprehensive documentation with examples and tutorials.

# Statement of Need

Setting up a regional ocean model is a major undertaking. It requires generating a wide range of complex input files, including the model grid, initial and boundary conditions, and forcing from the atmosphere, tides, and rivers. Traditionally, this work has depended on a patchwork of custom scripts and lab-specific workflows, which can be time-consuming, error-prone, and difficult to reproduce.
These challenges slow down science, create a steep barrier to entry for new researchers, and limit collaboration across groups.

Within the ROMS community, the preprocessing landscape has been shaped by tools like `pyroms` [@pyroms]. While `pyroms` has long provided valuable low-level utilities, it also presents challenges for new users. Installation can be cumbersome due to its Python and Fortran dependencies, and its inconsistent Application Programming Interface (API) and limited documentation make it hard to learn. The package was not designed with reproducible workflows in mind, and it lacks tests, CI, and support for modern Python tools such as `xarray` and `dask`. Since development of `pyroms` has largely ceased, its suitability for new projects, such as CDR simulations, is increasingly limited.
Furthermore, tools from other modeling communities cannot simply be adopted, since each ocean model has distinct structural requirements. For example, the new `regional-mom6` package [@barnes_regional-mom6_2024], developed for the Modular Ocean Model v6 (MOM6) [@adcroft_gfdl_2019], cannot be used to generate ROMS inputs, because ROMS employs a terrain-following vertical coordinate system that requires a specialized vertical regridding approach, whereas MOM6 accepts inputs on arbitrary depth levels and does not require vertical regridding at all. Several other differences further prevent cross-compatibility. Together, these limitations underscored the need for a modern, maintainable, and reproducible tool designed specifically for ROMS.\footnote{In the future, packages like ROMS-Tools and regional-mom6 could share a common backbone, with model-specific adaptations layered on top.}

`ROMS-Tools` was developed to meet this need. It draws on the legacy of the MATLAB preprocessing scripts developed at UCLA [@ucla-matlab], which encapsulate decades of expertise in configuring regional ocean model inputs. While many of the core algorithms and design principles are retained, `ROMS-Tools` provides an open-source Python implementation of these MATLAB tools using an object-oriented programming paradigm. This implementation enables a modernized workflow driven by high-level user API calls, enhancing reproducibility, reducing the potential for user errors, and supporting extensibility for additional features, forcing datasets, and use cases. In some cases, `ROMS-Tools` diverges from the MATLAB implementation to take advantage of new methods or better integration with the modern Python ecosystem.
By streamlining input generation and analysis, `ROMS-Tools` reduces technical overhead, lowers the barrier to entry, and enables scientists to focus on research rather than data preparation. The primary users of the package include (i) ocean modelers developing new domains for any regional modeling application and (ii) researchers in the ocean-based CDR community who use `ROMS-Tools` to set up simulations that mimic climate intervention scenarios.

# Acknowledgements

Development of `ROMS-Tools` has been supported by ARPA-E (DE-AR0001838) and philanthropic donations to [C]Worthy from the Grantham Foundation for the Environment, the Chan Zuckerberg Initiative, Founders Pledge, and the Ocean Resilience Climate Alliance.

# References
