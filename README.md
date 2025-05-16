# IsoformGazer
Isoform Gazer is a comprehensive dashboard application for visualizing RNA splicing events across pseudobulked single-cell junction usage and long-read isoform expression data.

## Installation 
Installation is currently supported for Linux and MacOS. 

### Option 1: Poetry (Recommended)

1. Ensure you are using Python version 3.12 or greater.

2. Clone the repository and ``cd`` in:
```
git clone https://github.com/daklab/IsoformGazer.git
cd IsoformGazer
```

3. Install Poetry if not already installed: 
```
python3 -m pip install pipx
python3 -m pipx ensurepath
pipx install poetry
```
Alternatively, you can also install Poetry using: 
```
curl -sSL https://install.python-poetry.org | python3 -
```

4. Within the ``IsoformGazer`` repository, use Poetry to install and manage all dependencies: 
```
poetry install
```

If you're interested in viewing details on the package dependencies and other options Poetry offers, within the IsoformGazer repository you can run: 
```
poetry show --help 
```

### Option 2: Conda (TO DO)
A bioconda recipe may be created at a later date. We will also add instructions for creating the necessary environment from scratch.