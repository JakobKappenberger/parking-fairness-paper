# Script for fairness paper

## Setup
*Before beginning, make sure to have 
 [NetLogo 6.2](https://ccl.northwestern.edu/netlogo/download.shtml) installed. On Windows, the installation should be detected automatically. Linux users must specify the respective path.*
```
# Build Environment
cd project_folder
conda env create -f environment.yml
# Activate Environment
conda activate thesis_gutmann
```

## Using this Repository

```1
# To run simulation with turtle documentation
# The runs from which the pricing is to derived must be moved to 'project_folder/src/runs_for_prices'
cd project_folder/src
python run_simulation.py 1
```

- **episodes** (required): Number of episodes to run the baseline for
- **--[m]odel_size**: Size of the NetLogo grid to use (either "training" or "evaluation"(default))
- **--[n]etlogo_[p]ath**: Path to NetLogo installation (for Linux users only)
- **--gui**: Boolean for NetLogo UI (default False)

All results are written to the baseline subfolder in the experiments directory with a dedicated timestamp to identify them.

