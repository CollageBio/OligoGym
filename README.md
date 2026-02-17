# OligoGym 🏃
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Roche/OligoGym)
## Description

OligoGym is a package that streamlines the training and evaluation of predictive models of oligonucleotide (ASOs, siRNAs) properties. The core components of OligoGym are its featurizers and models. The featurizers convert compounds represented using the HELM notation into a set of features that can be used by machine learning models. The models are implemented using PyTorch Lightning and scikit-learn, and they can be trained and evaluated on various datasets. They are implemented in a way that allows for easy integration with the featurizers, making it simple to switch between different featurizers and models.

OligoGym is designed to be easy to use and flexible, making it suitable for both researchers and practitioners in the field of oligonucleotide design and optimization. 

## Example code
```python
from oligogym.features import KMersCounts
from oligogym.models import LinearModel
from oligogym.data import DatasetDownloader

downloader = DatasetDownloader()
data = downloader.download("TLR8")
X_train, X_test, y_train, y_test = data.split(split_strategy="random")
feat = KMersCounts(k=[1, 2, 3], modification_abundance=True)
X_kmer_train = feat.fit_transform(X_train)
X_kmer_test = feat.transform(X_test)

model = LinearModel()
model.fit(X_kmer_train, y_train)
y_pred = model.predict(X_kmer_test)
```

## Featurizers
The following featurizers are currently implemented:

- KMersCounts
- OneHotEncoder
- Thermodynamics

## Models
The following models are currently implemented:

- SKLearnModel
    - NearestNeighborsModel
    - RandomForestModel
    - XGBoostModel
    - LinearModel
    - GaussianProcessModel
    - TabPFNModel
- LightningModel
    - MLP
    - CNN
    - CausalCNN
    - GRU

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

Install uv (if not already installed):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository and install dependencies:

```bash
git clone github.com/Roche/oligogym
cd oligogym
uv sync
```

This creates a virtual environment in `.venv/` and installs all dependencies (including dev dependencies) from the lockfile (`uv.lock`).

To install without dev dependencies:

```bash
uv sync --no-dev
```

To update the lockfile after changing `pyproject.toml`:

```bash
uv lock
uv sync
```

## Usage

Run commands inside the managed virtual environment:

```bash
uv run python your_script.py
```

Or activate the virtual environment directly:

```bash
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows
```

## Development
### Code Formatting

Format code using Black:

```bash
uv run black oligogym/ tests/
```

### Linting

Lint code using Flake8:

```bash
uv run flake8 oligogym/ tests/
```

### Testing

Run tests using Pytest:

```bash
uv run pytest
```

### Running Experiments

Run the ASO benchmark experiments:

```bash
uv run python experiments/benchmark_aso.py
```
