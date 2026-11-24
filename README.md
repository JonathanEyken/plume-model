# plume-model

A (collection of) simple entraining plume model(s) of the dynamics of wildfire plumes for education and research.

## Install

```bash
python -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install -e .
```
`plume_model` and `config` contain importable modules for analysis scripts that use the various plume models.

## How to run

Create an environment, initialise a plume given an environment and a fire, and raise the plume.

```python
from plume_model import Environment
from plume_model.wur_plume import Plume

# Define on any ascending height grid z, T, Td, p, u, v in SI units.
env = Environment(z, T, Td, p, u, v)

# A fire with sensible and latent heat flux H, LE [W/m2], and a fire_area [m2] of depth fire_depth [m].
plume = Plume(env, H=50e3, LE=5e3, fire_area=1e6, fire_depth=15.0)
plume.ascend()
```

## The default model
The default plume (`wur_plume.Plume`) solves the following explicit finite difference approximations of
budgets of `thetal` (liquid-water potential temperature), `qt` (total-water specific humidity), 
`w` (vertical velocity) and plume mass (proportional to its cross-sectional `area`) over height (`z`):

- `dM/dz = M (eps - delta)`, where `M = rho * area * w` is mass flux
- `d phi/dz = -eps (phi - phi_env)` with `phi` in `thetal`, `qt`;
- `eps = fac_ent/sqrt(A_0) + a_buoyant max(B, 0)/w²` with `fac_ent = 0.1`, `a_buoyant = 0.5`;
- `delta = beta fac_ent/sqrt(A_0) - c_det/w dw/dz` (the last term only where `w` reduces with height),
  with `beta = 0.5`, `c_det = 2`;
- `d(w²)/dz = 2 (a_w B - b_w eps w²)` with `a_w = 0.5`, `b_w = 0.3`;
- a base state at `h0 = 20 m` from free-convective scaling, ventilated by the wind across
  the fire front of depth `fire_depth` (switch off with `vent_base=False`);
- `dz = 10 m`.

After `ascend()`, every state variable and several diagnostic variables are attributes of the
`plume`, as profiles on the plume's own height grid `z` (from `h0` upward in steps of `dz`):

- the plume: `w`, `area`, `mass_flux`, `thetal`, `qt`, `T`, `Tv`, `thetav`, `Td`, `buoy`, `u`,
  `v`, `x`, `y`;
- condensate: `ql` formed at each level and removed there (pseudoadiabatically), i.e. it
  is not carried up with the plume;
- the environment: `p`, `exner_env`, `T_env`, `Td_env`, `theta_env`, `thetav_env`, `qt_env`,
  `rho_env`, `u_env`, `v_env`;
- the total fractional entrainment and detrainment rates and their associated fluxes: `eps`,
  `delta`, `entrainment_flux`, `detrainment_flux`;
- the two terms of the `w²` budget: `dw2dz_buoy`, `dw2dz_drag`.

Four scalars describe where the ascent ended: `k_top` and `z_top` are the index and height of
the plume top; `stopped` is false if the plume reached `z_max` without stopping; `k_lcl` is the
index of the first saturated level, or -1 if the plume never condenses.

If the environment comes as potential temperature and total water, as LES output usually
does, make the environment with `environment_from_theta(z, theta, qt, p, u, v)` instead.

## Analyses

Scripts that run the plume models live in `analyses/` (see [`analyses/README.md`](analyses/README.md)):

You may choose to adopt the plot styles from `config`:

```python
import config
config.apply()
```

## Cross-check against [wildfire-meteo-dmt](https://github.com/wildfire-meteo/wildfire-meteo-dmt)

One deployment of the default plume model is in [wildfire-meteo-dmt](https://github.com/wildfire-meteo/wildfire-meteo-dmt),
so to make sure these are consistent, `analyses/crosscheck/crosscheck_js.py` compares plumes from this package and
that repo. It is local and optional: it needs `node` and `wildfire-meteo-dmt`, found via `$WILDFIRE_METEO_DMT` or at
`../wildfire-meteo-dmt`.

```bash
WILDFIRE_METEO_DMT=/path/to/wildfire-meteo-dmt python analyses/crosscheck/crosscheck_js.py
```