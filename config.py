from pathlib import Path


# Experiment configuration
SEED = 42
HORIZON = 10_000

# Numerical configuration
NUMERICAL_TOLERANCE = 1e-12

# Output directories
RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
FIGURE_DIR = RESULTS_DIR / "figures"
