# Publications

Code and analysis materials associated with research papers by Songyan Zhu and collaborators. Each top-level folder corresponds to one paper and has its own README with details of the study, available code, inputs, and data sources.

## Papers

| Folder | Paper | Materials |
| --- | --- | --- |
| [COSORE](COSORE/) | *Global soil respiration and its share of ecosystem respiration: Spatial patterns and environmental controls* | [Data-analysis script](COSORE/COSORE_analysis.py) and [study-specific README](COSORE/README.md) |
| [Urban_FLUX](Urban_FLUX/) | *Disproportionate tree carbon uptake and sustainable urban greening beyond green space area* | [Data-analysis script](Urban_FLUX/urban_uflux_analysis.py) and [study-specific README](Urban_FLUX/README.md) |

### Urban FLUX

The Urban FLUX study examines ecosystem carbon uptake and water use across 72 cities during 2016–2025 using 30-m UFLUX estimates. It considers the contributions of tree cover to positive net ecosystem production (NEP), carbon–water relationships, land-cover transitions, and spatial differences between city centres and peripheries.

The [Urban_FLUX folder](Urban_FLUX/) contains the extracted analysis script and instructions for its inputs, dependencies and workflows. The accompanying manuscript is being prepared for submission to *Nature Cities*. The code covers the analyses documented in the study-specific README; it does not reproduce every result or figure in the manuscript.

## Using this repository

Open the folder for the paper of interest and read its README before running the code. Scripts may require separately archived data, processed model outputs, or optional Python packages. Each folder states the inputs and units used for its analyses.

The code available in a folder reflects the materials described there; a data-analysis script should not be assumed to include every data-processing step or model-training run used for the paper.

## Data and citation

Data access and citation details are provided in each paper's folder. For the COSORE study, the supporting data are available in [Zenodo record 23011179](https://zenodo.org/records/23011179). Please cite the relevant paper and original data sources when using materials from this repository. The COSORE folder's README will be updated with the journal citation and DOI when available.

For the Urban FLUX study, the manuscript identifies the annual flux grids and city-level summary tables at [Zenodo, DOI: 10.5281/zenodo.24105442](https://doi.org/10.5281/zenodo.24105442). Dataset coverage and additional input requirements are described in the [study-specific README](Urban_FLUX/README.md). This data DOI should not be treated as a code archive DOI.

Current manuscript citation:

> Zhu, S., Dong, W., Xu, J., Gharun, M., Zheng, Z., Wang, L., Chen, M. & Dash, J. *Disproportionate tree carbon uptake and sustainable urban greening beyond green space area*. Manuscript in preparation.
