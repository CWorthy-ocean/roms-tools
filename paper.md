---
title: 'ROMS-Tools: A Python package for galactic dynamics'
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
  - name: Matt Long
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
date: 26 September 2025
bibliography: docs/references.bib

---

# Summary

The ocean shapes Earth's climate and sustains marine ecosystems by transporting and exchanging vast quantities of heat, oxygen, carbon, and nutrients with the atmosphere. To understand these complex processes, scientists rely on ocean models, powerful computer simulations of physical circulation and biogeochemical (BGC) dynamics. These models represent the ocean on a grid of cells, where higher resolution (more, smaller cells) provides greater detail but requires significantly more computing power. While global models simulate the entire ocean, **regional models** focus computational resources on a specific area to achieve much higher resolution and can therefore resolve fine-scale processes like tidal dynamics, coastal circulation, submesoscale features (0.1–10 km), and detailed biogeochemical cycles.

A widely used regional ocean model is the **Regional Ocean Modeling System (ROMS)** (`[@shchepetkin_regional_2005]`). Configured at high resolution, it can resolve the fine-scale coastal currents and upwelling that shape regions like the California Current System (e.g., `[@marchesiello_equilibrium_2003]`). To connect physical circulation with ecosystem dynamics, ROMS can be coupled to a biogeochemical model, for example the Marine Biogeochemistry Library (MARBL) (`[@long_simulations_2021]`). This coupled framework allows researchers to explore how physical processes drive ecosystem dynamics, such as how nutrient-rich waters from upwelling fuel the phytoplankton blooms that form the base of the marine food web (e.g., `[@gruber_eddy-resolving_2006]`).
High-resolution physical-BGC simulations provide the foundation for diverse applications, including environmental forecasting, sustainable fisheries management, and assessing the regional impacts of climate change.

Yet configuring a regional ocean model like ROMS-MARBL remains a major challenge. Generating the required input files is time-consuming, error-prone, and difficult to reproduce, creating a bottleneck for both new and experienced researchers. The Python package **ROMS-Tools** addresses this challenge by providing an efficient, user-friendly, and reproducible workflow to generate all required inputs, including:
- **Model Grid**: A customizable, curvilinear grid that can be rotated to efficiently align with complex coastlines and uses a terrain-following vertical coordinate to accurately represent seafloor bathymetry.
- **Bathymetry and Land Mask**: High-resolution seafloor depth from **SRTM15** `[@tozer_global_2019]` and a corresponding land-sea mask from **Natural Earth**.
- **Atmospheric Forcing:** Meteorological drivers (wind, radiation, precipitation) from ERA5 `[@hersbach_era5_2020]` and surface BGC forcing (pCO₂, dust, nitrogen deposition) from hybrid observational-model sources.
- **Tidal Forcing:** Tidal potential, elevation, and velocities derived from **TPXO** `[@egbert_efficient_2002]` including corrections for self-attraction and loading (SAL).
- **River Forcing:** Freshwater runoff from **Dai & Trenberth** `[@dai_estimates_2002]` or custom user-provided files.
- **CDR Forcing**: Flexible user-defined tracers for Carbon Dioxide Removal (CDR) or other interventions.

![Initial conditions.\label{fig:example}](docs/images/ics_from_glorys.png){ width=100% }

An example of the generated inputs is shown in Figure \autoref{fig:example}, which illustrates initial conditions for the California Current System created with ROMS-Tools.

In addition to generating input files, ROMS-Tools handles several complex processing steps. It performs horizontal and vertical regridding from standard lat-lon-depth grids to the model's curvilinear grid with terrain-following vertical coordinate (using libraries like xarray and xgcm), and fills land values in the input data efficiently via an algebraic multigrid method (using pyamg). The workflow also manages longitude conversions (between -180∘ to 180∘ and 0∘ to 360∘), applies bathymetry smoothing to reduce pressure-gradient errors, and formats all NetCDF outputs with the metadata expected by ROMS-MARBL. Currently, ROMS-Tools fully supports UCLA-ROMS; support for other versions, such as Rutgers ROMS, may be added in the future with community contributions. A notable feature is the ability to stream ERA5 data directly from the cloud, so users do not need to download the data beforehand.
ROMS-Tools offers functionality for partitioning input files, a requirement for parallelized ROMS simulations across multiple nodes. Additionally, it can recombine the partitioned model output files.
For analysis, the package includes postprocessing utilities for regridding model output from the native ROMS grid to a standard lat-lon-depth grid (using xesmf), generating plots, and performing specialized tasks useful for CDR monitoring, reporting, and verification (MRV).

ROMS-Tools is designed to support modern, reproducible workflows. It is easily installable via Conda and PyPI and can be run interactively from Jupyter Notebooks.
To ensure reproducibility and facilitate collaboration, each workflow is defined in a simple YAML configuration file. These compact, text-based YAML files can be version-controlled and easily shared, eliminating the need to transfer large NetCDF files between researchers.
For performance, the package is integrated with dask to enable efficient, out-of-core computations on large datasets.
Finally, to guarantee reliability, the software is rigorously tested with continuous integration (CI) and supported by comprehensive documentation.

# Statement of need

Setting up a regional ocean model is a major undertaking. It requires generating a wide range of complex input files, including the model grid, initial and boundary conditions, and forcing from the atmosphere, tides, and rivers. Traditionally, this work has depended on a patchwork of custom scripts and lab-specific workflows: a fragmented approach that is time-consuming, error-prone, and difficult to reproduce. These challenges slow down science, create a steep barrier to entry for new researchers, and limit collaboration across groups.

Within the ROMS community, the preprocessing landscape has been shaped by tools like pyroms. While pyroms offers a valuable collection of low-level utilities, it poses significant challenges for new users: installation is tricky because it requires both Python and Fortran dependencies, the API is inconsistent, documentation is limited, and the design does not naturally support reproducible workflows.
Importantly, tools from other modeling communities cannot simply be adapted, since each ocean model has distinct structural requirements. For example, the new regional-mom6 package `[@barnes_regional-mom6_2024]`, developed for MOM6 `[@adcroft_gfdl_2019]`, cannot be used to generate ROMS inputs, because ROMS employs a terrain-following vertical coordinate system that requires a fundamentally different regridding approach, whereas MOM6 accepts inputs on arbitrary depth levels. Several other differences further prevent cross-compatibility. Together, these limitations underscored the need for a modern, maintainable, and reproducible tool designed specifically for ROMS.

ROMS-Tools was developed to meet this need. Algorithmically, it builds on the legacy of the original UCLA MATLAB preprocessing scripts, which encapsulate decades of community experience in setting up regional ocean model inputs. Rather than being a direct line-by-line translation, ROMS-Tools reimplements and modernizes those approaches in Python, leveraging this accumulated knowledge while introducing improved workflows and enhanced reproducibility. By streamlining input generation and analysis, ROMS-Tools reduces technical overhead, lowers the barrier to entry, and enables scientists to focus on research rather than data preparation.
While designed for ocean modelers developing new domains, the package is also gaining traction in the Carbon Dioxide Removal (CDR) community, where it enables testing of different climate intervention scenarios within existing, well-validated model setups.

# Acknowledgements

Acknowledgement of any financial support.


# References
