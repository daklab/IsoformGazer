<table>
<tr>
<td>
  <img src="src/isoformgazer/assets/Isoform-Gazer-Logo.png" alt="Isoform Gazer Logo" width="1000">
</td>
<td>
  <h1 style="margin-bottom:0;">Isoform Gazer</h1>
  <p style="margin-top:5px;"><em>A webtool for visualizing alternative RNA isoforms across integrated long-read RNA-seq and pseudobulked single-cell short-read RNA-seq data.</em></p>
</td>
</tr>
</table>

<div align="center">

[![Live Application](https://img.shields.io/badge/Live_Application-isoformgazer.nygenome.org-9b59b6?style=for-the-badge&logo=firefox&logoColor=white)](https://isoformgazer.nygenome.org)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-8e44ad?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-9b59b6?style=for-the-badge)](LICENSE)

</div>

## Quick Start

**For most users, we recommend using the live web application at [isoformgazer.nygenome.org](https://isoformgazer.nygenome.org)**.

The web application provides full access to all features without any installation required.

For local deployment or development, please see the installation instructions below. Please note that the local version utilizes SQLite out of necessity and lacks the same optimized Redis caching and efficiency of the deployed application's PostgreSQL database. Additionally, the resultant SQLite file created based on the prerelease data is large (>10GB).

## Data Overview

Isoform Gazer integrates two complementary data modalities using a shared GTF reference to provide integration across sequencing technologies and datasets: 

### Long-Read Isoform Data (Human & Mouse)

- **Data Sources:** ENCODE4 PacBio long-read RNA-seq
- **Data Overview:** 199,406 RNA transcripts across 55 human cell types and 170,977 transcripts across 9 mouse cell types
- **Data Processing:**
  - Raw reads aligned with Minimap2 and filtered for high-quality mappings (MAPQ ≥ 60)
  - Full-length reads selected based on annotated first and last exon overlap
  - Isoforms collapsed using FLAIR logic based on shared internal junction coordinates
  - High-confidence transcripts required ≥10 supporting reads across all samples
  - Novel isoforms assigned reproducible hash-based identifiers (SHA256)

For a full description of the data processing pipeline, please refer to the associated preprint by [Schertzer, M., Park, S., et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.07.02.662769v1) and [its accompanying repository](https://github.com/daklab/Perplexity_Isoform_MasterTable) for all analysis scripts.

### Single-Cell Smart-seq2 Junction Data (Human & Mouse)

- **Data Sources:** Publicly available Smart-seq2 datasets from:
  - Tabula Sapiens V2 (human)
  - Tabula Muris Senis (mouse)
  - Allen Brain Atlas human brain single-nuclei Smart-seq2
  - Allen Brain Atlas mouse single-nuclei RNA-seq
- **Data Overview:**: 12,491,650 junctions across 50 human cell types and 14,462,791 junctions across 50+ mouse cell types
- **Data Processing:**
  - Splice junctions extracted using Regtools and ATSEmapper pipeline
  - Validated canonical splice site motifs (GT-AG, GC-AG, AT-AC)
  - Annotated relative to long-read derived GTF files for direct integration
  - Alternative Transcript Structure Events (ATSEs) identified via splice graph construction
  - Single-cell counts aggregated into pseudobulk profiles by cell type

For a full description of the data processing pipeline, please refer to the associated LeafletFA preprint by [Isaev, K. (2025)](https://www.biorxiv.org/content/10.64898/2025.12.30.697080v1.abstract) and [the accompanying ATSEmapper pipeline repository](https://github.com/daklab/ATSEmapper) for all analysis scripts.

## Data Availability

To run Isoform Gazer locally, master table data is available through the [v0.0.0 prerelease](https://github.com/daklab/IsoformGazer/releases/tag/v0.0.0).

#### Option 1: Automatic Download (Recommended)
```bash
python src/isoformgazer/download_master_table_data.py
```

#### Option 2: Manual Download

Download the data files from the prerelease and place them in the `src/isoformgazer/data/` directory.

## Installation

> **Note:** Installation is currently supported for **Linux** and **macOS**.

### Option 1: Poetry (Recommended)

**1. Ensure Python 3.12 or greater is installed**
```bash
python3 --version
```

**2. Clone the repository**
```bash
git clone https://github.com/daklab/IsoformGazer.git
cd IsoformGazer
```

**3. Install Poetry** (if not already installed)

Via pipx:
```bash
python3 -m pip install pipx
python3 -m pipx ensurepath
pipx install poetry
```

Or via curl:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**4. Install dependencies**
```bash
poetry install
```

> **Tip:** After running `git pull` to update Isoform Gazer, always rerun `poetry install` to ensure dependencies are up to date.

**5. Explore Poetry options** (optional)
```bash
poetry show --help
```

### Option 2: Conda (Coming Soon)

A bioconda recipe and environment creation instructions will be added in a future release.


## Usage

After installing dependencies, launch Isoform Gazer locally:

```bash
poetry run python src/isoformgazer/app.py
```

This will start the application on localhost. Open the displayed address (e.g., `http://127.0.0.1:8050/`) in your browser.


## Contributing

Isoform Gazer is under active development. Please feel free to submit feedback and open issues or pull requests.


## License

See [LICENSE](LICENSE) for details.


## Acknowledgments

Isoform Gazer is developed and maintained by the [Knowles Lab](https://daklab.github.io/) at the New York Genome Center: 
- **Julia T. Lewandowski**: Application backend and front-end development, database design and management, deployment
- **Megan D. Schertzer**: ENCODE4 PacBio LRS data processing and analysis
- **Keren Isaev**: Single-cell SRS data processing and analysis
- **Stella H. Park**: ENCODE4 PacBio LRS data processing and analysis
- **David A. Knowles**: Project support, supervision, and funding 

We sincerely thank and acknowledge **Steve Brock** (Principal Scientific System Administrator, New York Genome Center) for their extensive support in the successful deployment of this webtool. 

## Citations 
Reese, F. et al. (2023). The ENCODE4 long-read RNA-seq collection reveals distinct classes of transcript structure diversity. bioRxiv, doi: 10.1101/2023.05.15.540865.

Quake, S. & The Tabula Sapiens Consortium (2024). Tabula Sapiens reveals transcription factor expression, senescence effects, and sex-specific features in cell types from 28 human organs and tissues. bioRxiv, doi: 10.1101/2024.12.03.626516.

The Tabula Muris Consortium (2020). A single-cell transcriptomic atlas characterizes ageing tissues in the mouse. Nature, 583, 590–595. 

Yao, Z., Velthoven N., Nguyen, T., et al. (2021). A taxonomy of transcriptomic cell types across the isocortex and hippocampal formation. Cell, 184, 3222-3241. 

Bakken, T., Jorstad, N., Hu, Q., et al. (2021) Comparative cellular analysis of motor cortex in human, marmoset, and mouse. Nature, 598, 111-119.

Schertzer, M. D., Park, S. H., Su, J., Sheynkman, G. M., & Knowles, D. A. (2025). Perplexity as a Metric for Isoform Diversity in the Human Transcriptome. bioRxiv, doi: https://doi.org/10.1101/2025.07.02.662769  

Isaev et al. (2025). LeafletFA: A comprehensive framework for alternative splicing analysis
from single cell RNA-seq data. bioRxiv, doi: https://doi.org/10.64898/2025.12.30.697080
