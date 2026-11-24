# Analyses

Runs of the plume model(s) organised per directory:

| Directory | What it does |
|---|---|
| `validation/` | Compares the model with LES (`les_*.py`) and (in the future) observations. Data for these validations should go in `validation/cases/` and are not part of this repo. |
| `crosscheck/` | Checks that the wur_plume model and the wildfire-meteo-dmt tool give the same plume. |

Each sub-directory runs independently of the others. They import `plume_model` and `config` from the rest of the repo.

## Adding an analysis

1. Make a sub-directory, e.g. `analyses/entrainment_comparison/`.
2. Import the model, the plotting conventions and your own modules:

   ```python
   import config
   from plume_model.wur_plume import Plume
   import environments   # analyses/entrainment_comparison/environments.py
   ```