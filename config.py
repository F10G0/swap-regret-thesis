from pathlib import Path

# Experiment configuration
SEED = 42
HORIZON = 1_000
BANDIT_REPLICATES = 20

# Numerical configuration
NUMERICAL_TOLERANCE = 1e-12
STATIONARITY_TOLERANCE = 1e-10
STATIONARY_METHOD = "solve"

# Output directories
RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
FIGURE_DIR = RESULTS_DIR / "figures"
