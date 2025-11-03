---
title: 'ROMS-Tools: A Python Package for Preparing and Analyzing ROMS Simulations'
tags:
  - Python
  - ocean modeling
  - ROMS
authors:
  - name: Nora Loose
    orcid: 0000-0002-3684-9634
    affiliation: 1
  - name: Tom Nicholas
    orcid:
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
  - name: Jeroen Molemaker
    orcid:
    affiliation: 3
  - name: Abigale Wyatt
    orcid:
    affiliation: 1
affiliations:
 - name: '[C]Worthy LLC, Boulder, CO, United States'
   index: 1
 - name: Earthmover
   index: 2
 - name: University of California, Los Angeles, CA, United States
   index: 3
date: 3 November 2025
bibliography: docs/references.bib

---

# Summary

The ocean shapes Earth’s climate and sustains marine ecosystems by circulating and storing vast amounts of heat, oxygen, carbon, and nutrients, while exchanging heat and gases with the atmosphere. To understand these complex dynamics and processes, scientists rely on ocean models, powerful computer simulations of physical circulation and biogeochemical (BGC) dynamics. These models represent the ocean on a grid of cells, where finer grid spacing (more, smaller cells) provides higher fidelity and greater detail but requires significantly more computing power. While global ocean models simulate the entire ocean, **regional ocean models** focus computational resources on a specific area to achieve much finer grid spacing than is computationally feasible over the global domain. This finer grid spacing enables regional ocean models to explicitly resolve fine-scale phenomena, like mesoscale (10-100 km) and submesoscale (0.1-10 km) features, tidal dynamics, coastal currents, upwelling, and detailed BGC processes. Capturing these dynamics and processes at high fidelity is essential for applications in environmental management, fisheries, for assessing regional impacts of climate change, and for evaluating ocean-based carbon dioxide removal (CDR) strategies.

A widely used regional ocean model is the **Regional Ocean Modeling System (ROMS)** [@shchepetkin_regional_2005]. To connect physical circulation with ecosystem dynamics and the ocean carbon cycle, ROMS has been coupled to a BGC model called  the Marine Biogeochemistry Library (MARBL) [@long_simulations_2021]. This coupled framework allows researchers to explore a variety of scientific and practical questions.  For example, it can be used to investigate the potential of ocean-based carbon removal strategies, such as adding alkaline materials to the ocean to sequester atmospheric carbon dioxide. It can also be used to study how physical processes drive ecosystem dynamics, such as how nutrient-rich waters from upwelling fuel the phytoplankton blooms that form the base of the marine food web [@gruber_eddy-resolving_2006].

## Input Data and Preprocessing
Whether for research or industrial-focused applications, configuring a regional ocean model like ROMS-MARBL remains a major technical challenge. Generating the required input files is time-consuming, error-prone, and difficult to reproduce, creating a bottleneck for both new and experienced model users. The Python package `ROMS-Tools` addresses this challenge by providing an efficient, user-friendly, and reproducible workflow to generate all required model input files. Its user interface and underlying data model are based on `xarray` [@hoyer2017xarray] , enabling seamless handling of multidimensional datasets with rich metadata and optional parallelization via a `dask` [@dask] backend.

`ROMS-Tools` can automatically process commonly used datasets or incorporate custom user data and routines. Currently, it can generate the following inputs:

1. **Model Grid**: Customizable, curvilinear grid, rotatable to align with coastlines, with a terrain-following vertical coordinate.
2. **Bathymetry**: Derived from **SRTM15** [@tozer_global_2019].
3. **Land Mask**: Inferred from **Natural Earth** coastlines.
4. **Physical Ocean Conditions**:  Initial and open boundary conditions for sea surface height, temperature, salinity, and velocities derived from GLORYS [@jean-michel_copernicus_2021].
5. **BGC Ocean Conditions**: Initial and open boundary conditions for dissolved inorganic carbon, alkalinity, and other biogeochemical tracers from CESM output [@yeager_seasonal--multiyear_2022] or hybrid observational-model sources.
6. **Meteorological forcing**: Wind, radiation, precipitation, and air temperature/humidity processed from ERA5 [@hersbach_era5_2020] with optional radiation bias and coastal wind corrections.
7. **BGC surface forcing**: pCO~2~, iron, dust, nitrogen deposition from CESM output [@yeager_seasonal--multiyear_2022] or hybrid observational-model sources.
8. **Tidal Forcing:** Tidal potential, elevation, and velocities derived from **TPXO** [@egbert_efficient_2002] including self-attraction and loading (SAL) corrections.
9. **River Forcing:** Freshwater runoff derived from **Dai & Trenberth** [@dai_estimates_2002] or custom files.
10. **CDR Forcing**: User-defined interventions that inject BGC tracers at point sources or as larger-scale Gaussian perturbations, suitable for the simulation of field- or large-scale CDR experiments.

While the source datasets listed above are the ones currently supported, the package’s modular design  makes it straightforward to add new data sources or custom routines in the future.
To generate the model inputs listed above, ROMS-Tools automates several intermediate processing steps, including:

**Bathymetry processing**: The bathymetry is smoothed in two stages, first across the entire domain and then along the shelf, to ensure local steepness ratios are not exceeded and to reduce pressure-gradient errors. A minimum depth is enforced to prevent water levels from becoming negative during large tidal excursions.
**Mask definition**: The land-sea mask is generated by comparing the ROMS grid’s horizontal coordinates with a coastline dataset using regionmask [@hauser_regionmaskregionmask_2024]. Enclosed basins are subsequently filled with land.
**Land value handling**: Land values are filled via an algebraic multigrid method using pyamg [@pyamg2023] prior to horizontal regridding. This extends ocean values into land areas to resolve discrepancies between source data and ROMS land masks that could otherwise produce artificial values in ocean cells.
**Regridding**: Ocean and atmospheric fields are horizontally and vertically regridded from standard lat-lon-depth grids to the model’s curvilinear grid with a terrain-following vertical coordinate using xarray [@hoyer2017xarray]. Optional sea surface height corrections can be applied, and velocities are rotated to align with the rotated ROMS grid
**Longitude conventions**: `ROMS-Tools` handles differences in longitude conventions, converting between −180°–180° and 0°–360° as needed.
**River forcing**: Relevant rivers are automatically selected and relocated to the nearest coastal cell, while multi-cell or moving rivers can be managed manually.
**Atmospheric data streaming**: ERA5 atmospheric data can be accessed directly from the cloud, removing the need for users to pre-download large datasets locally.
Users can quickly design and visualize regional grids and inspect all input fields with built-in plotting utilities. An example of surface initial conditions generated for a California Current System simulation is shown in Figure \autoref{fig:example}.
![Surface initial conditions for the California Current System created with `ROMS-Tools` from GLORYS. Left: potential temperature. Right: zonal velocity. Shown for January 1, 2000.\label{fig:example}](docs/images/ics_from_glorys.png){ width=100% }
`ROMS-Tools` also includes features that facilitate simulation and output management. It supports partitioning and recombining input and output files to enable parallelized ROMS simulations across multiple nodes, and writes NetCDF outputs with metadata fully compatible with ROMS-MARBL. Currently, UCLA-ROMS [@ucla-roms] is fully supported, with the potential to add other ROMS versions, such as Rutgers ROMS [@rutgers-roms], in the future.

## Analysis Layer
`ROMS-Tools` includes an analysis layer for postprocessing ROMS-MARBL output. It provides utilities for general-purpose tasks, such as loading model output directly into an Xarray dataset with additional metadata, enabling seamless use of the Pangeo scientific Python ecosystem for further analysis and visualization. The analysis layer also supports regridding from the native curvilinear ROMS grid with terrain-following coordinate to a standard latitude-longitude-depth grid using xesmf [@xesmf]. Beyond these general-purpose features, the analysis layer offers a suite of targeted tools for evaluating CDR interventions. These include utilities for generating standard plots, such as CDR uptake efficiency curves, and performing specialized tasks essential for CDR monitoring, reporting, and verification.

## Workflow, Reproducibility, and Performance

`ROMS-Tools` is designed to support modern, reproducible workflows. It is easily installable via Conda or PyPI and can be run interactively from Jupyter Notebooks.
To ensure reproducibility and facilitate collaboration, each workflow is defined in a simple YAML configuration file. These compact, text-based YAML files can be version-controlled and easily shared, eliminating the need to transfer large NetCDF files between researchers, as source data like GLORYS and ERA5 are accessible in the cloud.

For performance, the package is integrated with `dask` [@dask] to enable efficient, out-of-core computations on large datasets.
Finally, to ensure reliability, the software is rigorously tested with continuous integration (CI) and supported by comprehensive documentation.

# Statement of need

Setting up a regional ocean model is a major undertaking. It requires generating a wide range of complex input files, including the model grid, initial and boundary conditions, and forcing from the atmosphere, tides, and rivers. Traditionally, this work has depended on a patchwork of custom scripts and lab-specific workflows, which can be time-consuming, error-prone, and difficult to reproduce.
These challenges slow down science, create a steep barrier to entry for new researchers, and limit collaboration across groups.

Within the ROMS community, the preprocessing landscape has been shaped by tools like `pyroms` [@pyroms].
While `pyroms` has long provided valuable low-level utilities, it also presents challenges for new users. Installation can be cumbersome due to its Python and Fortran dependencies, and its inconsistent API and limited documentation make it hard to learn. The package was not designed with reproducible workflows in mind, and it lacks tests, continuous integration, and support for modern Python tools such as `xarray` and `dask`. Since development of `pyroms` has largely ceased, its suitability for new projects is increasingly limited.
Importantly, tools from other modeling communities cannot simply be adapted, since each ocean model has distinct structural requirements. For example, the new `regional-mom6` package [@barnes_regional-mom6_2024], developed for MOM6 [@adcroft_gfdl_2019], cannot be used to generate ROMS inputs, because ROMS employs a terrain-following vertical coordinate system that requires a fundamentally different regridding approach, whereas MOM6 accepts inputs on arbitrary depth levels. Several other differences further prevent cross-compatibility. Together, these limitations underscored the need for a modern, maintainable, and reproducible tool designed specifically for ROMS.\footnote{In the future, packages like `ROMS-Tools` and `regional-mom6` could share a common backbone, with model-specific adaptations layered on top.}


`ROMS-Tools` was developed to meet this need. It draws on the legacy of MATLAB preprocessing scripts developed at UCLA [@ucla-matlab], which encapsulate decades of expertise in configuring regional ocean model inputs. While many of the core algorithms and design principles are retained, `ROMS-Tools` provides an open-source Python implementation using an object-oriented programming paradigm. This implementation enables a modernized workflow driven by high-level user Application Programming Interface (API) calls, enhancing reproducibility, reducing the potential for user errors, and supporting extensibility for additional features, forcing datasets, and use cases. In some cases, ROMS Tools diverges from the MATLAB implementation to take advantage of new methods or better integration with the modern Python ecosystem.
By streamlining input generation and analysis, `ROMS-Tools` reduces technical overhead, lowers the barrier to entry, and enables scientists to focus on research rather than data preparation. The primary users of the package include ocean modelers developing new domains and researchers in the CDR community, who use it to test climate intervention scenarios.

# Acknowledgements

Acknowledgement of any financial support.


# References
