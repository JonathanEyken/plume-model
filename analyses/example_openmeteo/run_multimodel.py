"""
An example of how the openmeteo input can be used to create a plume model forecast showing multiple weather models in the same plot.
"""
import matplotlib.pyplot as plt

from plume_model.environment import environment_from_nwp
from plume_model.wur_plume import Plume

# Settings:
########################
locname = "Wageningen"
lat = 51.99
lon = 5.66

timestep = "2026-11-27T12:00"

models = ["ECMWF","ICOND2","AROME"]

areas  = [0.2, 0.5, 1, 1.5, 2, 3, 5,6,7,8,9, 10, 20, 50, 75, 100]
########################
fig,ax = plt.subplots()
ax.semilogx()

for model in models:
    env = environment_from_nwp(lat,lon,timestep,model=model)

    tops = []
    for A in areas:
        plu = Plume(env, 100e3, 10e3, A*1e4, 20)
        plu.surface()
        plu.ascend()
        tops.append(plu.z_top)

    ax.plot(areas,tops,label=model)

ax.legend()
ax.set(ylabel="plume top height (m AGL)",
       xlabel="plume area (Ha)",
       title=f"{locname}, {timestep}")
plt.show()
