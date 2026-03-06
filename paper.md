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

The ocean regulates Earth’s climate and sustains marine ecosystems by circulating and storing heat, carbon, oxygen, and nutrients, while exchanging heat and gases with the atmosphere. Scientists study these processes using ocean models, which simulate the ocean on a grid. **Regional ocean models** focus computational resources on a limited geographical area with fine grid spacing, and can resolve fine-scale phenomena such as mesoscale and submesoscale features, tidal dynamics, coastal currents, upwelling, and detailed biogeochemical (BGC) processes. A widely used regional ocean model is the **Regional Ocean Modeling System (ROMS)** [@shchepetkin_regional_2005]. ROMS has been coupled to the Marine Biogeochemistry Library (MARBL) [@long_simulations_2021; @ucla-roms] to link physical and BGC processes. ROMS-MARBL supports research on environmental management, fisheries, regional climate impacts, and ocean-based carbon dioxide removal (CDR) strategies.

`ROMS-Tools` is a Python package that streamlines the **preparation and analysis of ROMS-MARBL simulations** by enabling users to generate regional grids, prepare model inputs efficiently, and analyze model outputs. A detailed overview of the package's functionality is available in the `ROMS-Tools` [documentation]((https://roms-tools.readthedocs.io). By providing a modern, user-friendly interface, `ROMS-Tools` lowers technical barriers, improves reproducibility, and allows scientists to focus on research rather than data preparation. The package is installable via Conda or PyPI and can be run interactively in Jupyter notebooks.

# Statement of Need

Regional ocean models are essential tools for research in marine ecosystems, climate dynamics, and ocean-based CDR. However, configuring a regional ocean model like ROMS-MARBL is technically demanding.
Model setup requires initialization and time-dependent forcing from oceanic and atmospheric datasets, drawn from multiple external sources in diverse formats. These global source datasets can span petabytes and must be subsetted, processed, and mapped onto the target model grid, producing 10–100 terabytes of input data for large regional domains. Generating these input files is time-consuming, error-prone, and difficult to reproduce. These challenges create a bottleneck for both new and experienced users, slow down science, and limit collaboration across groups.

Existing tools within the ocean modeling ecosystem do not fully address these challenges for ROMS-MARBL or ROMS users. While legacy MATLAB-based scripts developed at UCLA [@ucla-matlab] and Python packages such as `pyroms` [@pyroms] provide critical functionality, both rely on low-level, manually coordinated steps that limit reproducibility, maintainability, and accessibility. Moreover, frameworks developed for other ocean models cannot be directly applied to ROMS due to fundamental differences in grid geometry, vertical coordinates, and model input requirements. As a result, users lack a modern, integrated framework for reproducible model setup and analysis that is specifically designed for ROMS and ROMS-MARBL.

`ROMS-Tools` was developed to fill this gap. It is an open-source Python framework designed for researchers and practitioners who run ROMS or ROMS-MARBL regional ocean simulations, including users in physical oceanography, marine biogeochemistry, and ocean-based CDR applications. Current capabilities are fully compatible with UCLA-ROMS [@ucla-roms; @ucla-roms-cworthy], with potential support for other ROMS implementations, such as Rutgers ROMS [@rutgers-roms], in the future.
The package handles large input and output datasets via parallel computation with `dask` [@dask], making workflows scalable from laptops to high-performance computing clusters. Built-in visualization tools enable quick inspection of regional grids as well as model input and output fields. For example, \autoref{fig:example} shows surface initial conditions for a California Current System simulation at 5 km horizontal resolution, generated and visualized directly using `ROMS-Tools`.
By lowering technical barriers and improving transparency and reproducibility, `ROMS-Tools` enables more efficient model development, facilitates scientific collaboration, and supports applications such as verification of marine carbon removal strategies.

![Surface initial conditions for the California Current System created and visualized with `ROMS-Tools`. Left: potential temperature. Right: grid-aligned horizontal velocity in $\xi$-direction. Shown for January 1, 2000.\label{fig:example}](docs/images/ics_from_glorys.png){ width=100% }

# State of the Field

Historically, setting up a regional ocean model required a patchwork of custom scripts and lab-specific workflows, resulting in error-prone and difficult-to-reproduce processes. Within the ROMS community, tools like `pyroms` [@pyroms] addressed some of these issues by providing low-level Python utilities for preprocessing ROMS model inputs. However, `pyroms` has several limitations: installation is cumbersome due to Python/Fortran dependencies, the API is inconsistent, and documentation and tests are missing. The package does not support modern tools such as `xarray` [@hoyer2017xarray], nor reproducible workflows. Active development has ceased, and maintenance (including compatibility with newer Python versions) is no longer provided. Together, these limitations make it very difficult to add new features, such as support for BGC and CDR applications, and improvements to user-friendliness.

Tools from other modeling communities cannot be directly applied to ROMS because each model has distinct structural requirements and input conventions. For example, the `regional-mom6` package [@barnes_regional-mom6_2024], developed for regional configurations of the Modular Ocean Model v6 (MOM6) [@adcroft_gfdl_2019], cannot generate ROMS inputs. ROMS uses a terrain-following vertical coordinate system that requires specialized vertical regridding, whereas MOM6 accepts inputs on arbitrary depth levels and does not require vertical regridding at all. While ROMS and MOM6 differ in fundamental ways, `regional-mom6` represents the closest comparable tool to `ROMS-Tools` in the wider modeling ecosystem. Notably, the main development cycles of `regional-mom6` and `ROMS-Tools` overlapped (`regional-mom6`: 2023–2024; `ROMS-Tools`: 2024–2025, based on public GitHub commits). Had the developers been aware of each other, a shared framework could potentially have been created, with model-specific adaptations layered on top. Adapting one framework to the other now would require extensive architectural changes.

Legacy MATLAB preprocessing scripts developed at UCLA [@ucla-matlab] encapsulate decades of expertise in configuring regional ocean models, but require users to edit source code directly, making workflows error-prone, difficult to reproduce, and challenging to extend to new datasets or applications. `ROMS-Tools` provides a modern, open-source Python implementation of these scripts, retaining core algorithms while offering high-level APIs, automated intermediate steps, and explicit workflow state management via YAML. This object-oriented design improves reproducibility, reduces user errors, and supports extensibility, while leveraging modern Python tools such as `xarray` and `dask`. In some cases, `ROMS-Tools` diverges from the original MATLAB implementation to incorporate improved methods or better integrate with the Python ecosystem.


# Software Design

`ROMS-Tools` emphasizes ease of use, flexibility, reproducibility, and scalability through a modular architecture and high-level user interfaces.

## Design Trade-Offs

A central design trade-off in `ROMS-Tools` is between **user control** and **automation**. Rather than enforcing a fixed workflow, the package exposes key choices such as physical options (e.g., corrections for radiation or wind), interpolation and fill methods, and computational backends. This approach contrasts with opinionated frameworks that fix defaults and directory structures to maximize automation. While users must make explicit decisions, some steps remain automated to prevent errors. For example, bathymetry smoothing is applied automatically using a fixed, non-tunable parameter, since insufficient or omitted smoothing can crash simulations due to pressure gradient errors. This design choice addresses issues encountered by new users of the original UCLA MATLAB scripts and balances flexibility with safety, enabling experimentation while avoiding common pitfalls.

Another key design consideration is balancing **modular, incremental workflow steps** with **reproducibility**. `ROMS-Tools` organizes tasks (such as creating `InitialConditions`, `BoundaryForcing`, and `SurfaceForcing`) into small, composable components that can be executed, saved, and revisited independently, rather than following a monolithic, fixed workflow. All components depend on the `Grid`, but once it is created, the remaining objects are independent. This modular approach avoids unnecessary recomputation when only some inputs change but requires careful tracking of workflow state. To ensure reproducibility, all configuration choices are stored in compact, text-based YAML files. These files are version-controllable, easy to share, and eliminate the need to transfer large model input NetCDF datasets.

## Architecture

At the user-facing level, `ROMS-Tools` provides high-level objects such as `Grid`, `InitialConditions`, and `BoundaryForcing`. Each object exposes a consistent interface (`.ds`, `.plot()`, `.save()`, `.to_yaml()`), allowing users to apply the same methods across workflow steps and inspect standardized attributes that are always present. This consistency reduces cognitive overhead and makes workflows predictable.

Internally, `ROMS-Tools` follows a layered, modular architecture. Low-level dataset classes (`LatLonDataset`, `ROMSDataset`) handle data ingestion and preprocessing tasks such as subdomain selection and lateral land filling. Source-specific datasets (e.g., `ERA5Dataset`, `GLORYSDataset`, `SRTMDataset`) inherit from these base classes and encode dataset-specific conventions like variable names, coordinates, and masking. Supporting a new data source typically requires only a small subclass defining these mappings while reusing existing preprocessing logic, minimizing changes to the core code.

High-level classes (`Grid`, `InitialConditions`, `BoundaryForcing`) build on these low-level dataset abstractions to generate ready-to-use modeling inputs through operations such as regridding and final assembly. This layered design improves **extensibility and maintainability**.

## Computational and Data Model Choices

`ROMS-Tools` is built on `xarray`, which provides a clear, consistent interface for exploring and inspecting labeled, multi-dimensional geophysical datasets. Users can take advantage of `xarray`’s intuitive indexing, plotting, and metadata handling. Optional `dask` enables parallel and out-of-core computation for very large input and output datasets.

# Research Impact Statement

`ROMS-Tools` is used by two primary research communities. First, regional ocean modelers use it to generate input datasets for ROMS simulations; external users include researchers at PNNL, WHOI, Stanford University, and UCLA. Second, researchers in the ocean-based carbon dioxide removal (CDR) community use `ROMS-Tools` to configure reproducible ROMS–MARBL simulations of climate intervention scenarios, with adopters including [C]Worthy, Carbon to Sea, Ebb Carbon, and SCCWRP.
All of these groups have contacted the developers directly or engaged with the project through GitHub or offline discussions. Several manuscripts from these communities are currently in preparation.

Beyond standalone use, `ROMS-Tools` is integrated into broader scientific workflows, including C-Star [@cstar], an open-source platform under development to provide scientifically credible monitoring, reporting, and verification (MRV) for the emerging marine carbon market.

Additional evidence of community uptake comes from public usage metrics. At the time of writing, the GitHub repository shows 119 unique cloners in the past 14 days, with stars from users at institutions including the University of Waikato, NCAR, University of Maryland, National Oceanography Centre, McGill University, UC Santa Cruz, and others. Distribution statistics indicate over 3,100 conda-forge downloads in the past six months, including 68 downloads of the most recent release (`v3.3.0`), and more than 48,000 total PyPI downloads. PyPI counts include automated continuous integration (CI) usage by `ROMS-Tools`, in addition to direct user installations. In contrast, conda-forge downloads of `v3.3.0` reflect exclusively human-initiated installs, as C-Star's CI workflows currently pin pre-`v3.3.0` releases of `ROMS-Tools`.

# AI Usage Disclosure

Generative AI tools were used to help write docstrings, develop tests, and improve the clarity and readability of both the `ROMS-Tools` documentation and manuscript text. All AI-assisted content was reviewed and verified by the authors for technical accuracy and correctness.

# Acknowledgements

Development of `ROMS-Tools` has been supported by ARPA-E (DE-AR0001838) and philanthropic donations to [C]Worthy from the Grantham Foundation for the Environment, the Chan Zuckerberg Initiative, Founders Pledge, and the Ocean Resilience Climate Alliance.

# References
