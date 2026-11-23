# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: <https://tiziano1590.github.io/comp_astro_26/>.

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_26.git
cd comp_astro_26
pip install .
```

### Development installation

```bash
pip install -e '.[dev,docs]'
```
The Random Forest detection additionally needs `scikit-learn`, `pandas`,
`matplotlib`, `joblib` and `pyyaml`.

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --transit [--output lightcurve.png]
daneel -i <input_file> --detection rf
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-d, --detection`: : Run exoplanet detection with the chosen method.
  Currently available: `rf` (Random Forest)
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for the transit plot (default: `lc.png`)

### Examples

```bash
# Run exoplanet detection
daneel -i examples/params.yaml --transit --output lightcurve.png

# Train and evaluate a Random Forest transit detector
daneel -i path/to/parameters.yaml --detecton rf

# Run both detection and atmospheric analysis
daneel -i examples/params.yaml --atmosphere
```
## Random Forest detection (`-d rf`)

The command trains a Random Forest classifier on the TESS light curves in
`tess_data.csv` and evaluates it on a held-out test set:

```bash
daneel -i path/to/parameters.yaml -d rf
```

### What it does

1. Loads the flux columns (`flux_0000` ... `flux_0999`) and the `label`
   column (0 = non-planet, 1 = planet) from the CSV file.
2. Splits the data into training and test sets (stratified, so both keep the
   same class balance).
3. Balances the **training set only** with light augmentation (noise, scaling
   and shifts), so the test set stays untouched.
4. Trains the Random Forest.
5. Chooses a decision threshold from the ROC curve (Youden's J statistic)
   in addition to the default threshold of 0.5.
6. Prints the statistics and saves the results (see below).

### Parameters

To be included in the .YAML as shown below

```yaml
detection:
  data_path: data/tess_data.csv      # path to the dataset
  n_bins: 1000                  # number of flux bins per light curve
  test_size: 0.2                # fraction of data used for testing
  random_state: 42              # seed for reproducibility
  samples_per_class: 350        # size of each class after augmentation (train set)
  use_scaler: false             # standardise features (optional for forests)
  output_dir: rf_results        # folder where results are saved
  random_forest:                # passed to the Random Forest
    n_estimators: 500
    max_depth: null             # null = no limit
```

### Output

The statistics are printed to the terminal and everything is saved inside  `rf_results/`:

| File | Content |
|---|---|
| `report_random_forest.txt` | Hyperparameters, Area Under the ROC Curve, OOB, and for both the default (0.5) and the optimal threshold: confusion matrix, precision, recall, accuracy and classification report |
| `confusion_matrix_rf.png` | Confusion matrix at the optimal threshold |
| `roc_curve_rf.png` | ROC curve |
| `pr_curve_rf.png` | Precision-recall curve |
| `probability_hist_rf.png` | Predicted planet probabilities for each true class |
| `tess_rf_model.joblib` | The trained model |
| `rf_optimal_threshold.npy` | The optimal threshold|

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.
Add detection settings go in a `detection` section of the YAML file, next to
the other sections (for example `transit`) for the -d command.

## Development

Run the test suite and style checks with:

```bash
pytest
ruff check .
```

Build the documentation with `make -C docs html`.

## License

This project is licensed under the GNU General Public License v3.0.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
