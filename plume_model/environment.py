#
# Copyright 2026 Wageningen University & Research (WUR)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import time
# import pandas as pd
import openmeteo_requests

from .thermo import Rd, dewpoint, exner, esat, qsat, virtual_temp, q_from_e_p


class Environment:
    """
    The environment the plume rises through, on its own ascending height grid.
    Can be constructed using either openmeteo NWP or radiosounding profiles.
    """
    def __init__(self, z, T, Td, p, u=None, v=None, elevation=None):
        self.z = np.asarray(z, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.Td = np.asarray(Td, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.u = np.zeros_like(self.z) if u is None else np.asarray(u, dtype=float)
        self.v = np.zeros_like(self.z) if v is None else np.asarray(v, dtype=float)

        self.exner = exner(self.p)
        self.theta = self.T / self.exner
        self.qt = qsat(self.Td, self.p)
        self.thetav = virtual_temp(self.theta, self.qt)
        self.rho = self.p / (Rd * self.exner * self.thetav)
        self.z_top = float(self.z[-1])

        if elevation:
            self.z_msl = self.z + elevation


def environment_from_theta(z, theta, qt, p, u=None, v=None):
    """Build an Environment from (theta, qt), the form LES and models usually give."""
    pi = exner(np.asarray(p, dtype=float))
    T = np.asarray(theta, dtype=float) * pi
    return Environment(z, T, dewpoint(np.asarray(qt, dtype=float), p), p, u, v)


def environment_from_nwp(lat,lon,timestep,model="ICON"):
    """
    Load openmeteo profiles of the required variables to initialize an environment.
    lat = latitude  (decimal degrees, float)
    lon = longitude (decimal degrees, float)
    timestep = string in the format "yyyy-mm-ddTHH:MM", UTC

    model = string specifying the model. Options:
    "ECMWF"    --> global     , 0.25° resolution , 15 days forecast 
    "ICON"     --> global     , 11 km resolution , 7.5 days forecast
    "ICONEU"   --> EU         , 7 km resolution  , 5 day forecast
    "ICOND2"   --> central-EU , 2 km resolution  , 2 day forecast
    "AROME"    --> NW-EU      , 2.5 km resolution, 2 day forecast
    """
    if check_forecast_horizon(timestep,model):
        print("ERROR: The selected date and time are too far into the future, please select another model or a sooner timestep.")
        return 0
    
    if ishistorical(timestep):
        url        = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        url        = "https://api.open-meteo.com/v1/forecast"

    p_vars = []
    if model == "ECMWF":
        pressure_levs = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
        params = {"latitude":  lat,"longitude": lon,"hourly": p_vars,"models": "ecmwf_ifs025","start_hour": timestep,"end_hour": timestep,"elevation":"nan","wind_speed_unit":"ms",}
    elif model == "ICON":
        pressure_levs = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]
        params = {"latitude":  lat,"longitude": lon,"hourly": p_vars,"models": "dwd_icon","start_hour": timestep,"end_hour": timestep,"elevation":"nan","wind_speed_unit":"ms",}
    elif model == "ICONEU":
        pressure_levs = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]
        params = {"latitude":  lat,"longitude": lon,"hourly": p_vars,"models": "dwd_icon_eu","start_hour": timestep,"end_hour": timestep,"elevation":"nan","wind_speed_unit":"ms",}
    elif model == "ICOND2":
        pressure_levs = [1000, 975, 950, 850, 700, 600, 500, 400, 300, 250, 200]
        params = {"latitude":  lat,"longitude": lon,"hourly": p_vars,"models": "dwd_icon_d2","start_hour": timestep,"end_hour": timestep,"elevation":"nan","wind_speed_unit":"ms",}   
    elif model == "AROME": 
        pressure_levs = [1000, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 275, 250, 225, 200, 175, 150, 125, 100]
        params = {"latitude":  lat,"longitude": lon,"hourly": p_vars,"models": "meteofrance_arome_france0025","start_hour": timestep,"end_hour": timestep,"elevation":"nan","wind_speed_unit":"ms",}
    else:
        print("ERROR: Please select model = 'ECMWF', 'ICON', 'ICONEU', 'ICOND2', or 'AROME'")
        return 0

    n_levs        = len(pressure_levs)
    vars          = ['temperature', 'relative_humidity','wind_u_component','wind_v_component','geopotential_height']
    for var in vars:
        for l in pressure_levs:
            p_vars.append(f"{var}_{l}hPa")
    
    openmeteo = openmeteo_requests.Client()
    response  = openmeteo.weather_api(url, params = params)[0]
    api_outp  = response.Hourly()
    timestep  = time.strftime("%Y-%m-%dT%H:%M", time.gmtime(api_outp.Time()))
    
    print(f"Loaded {model} profiles for temperature, humidity and wind for:")
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Time: {timestep} UTC")

    data = {}     # dict to store the data
    vi = 0        # counter for indexing the variables
    for var in vars:
        data[var] = np.empty(n_levs)
        for l in range(n_levs):
            data[var][l] = api_outp.Variables(vi*n_levs + l).ValuesAsNumpy()[0]
        vi += 1

    # Compute the necessary variables:
    elev  = response.Elevation()
    z_msl = data["geopotential_height"]
    z_agl = data["geopotential_height"] - elev
    
    # Converting units:
    p     = np.array(pressure_levs)*1e2    # to hPa 
    RH    = data["relative_humidity"]/1e2  # to fraction
    T     = data["temperature"] + 273.15   # to K

    # Calculate derived variables:
    e_sat = esat(T)
    e     = RH*e_sat
    q     = q_from_e_p(e,p)
    Td    = dewpoint(q,p)

    env = Environment(z_agl,T,Td,p,
                      data["wind_u_component"], data["wind_v_component"],
                      elevation = elev)
    return env


def check_forecast_horizon(timestep,model):
    """
    Returns 1 if the selected date is (potentially) too far into the future.
    (Needed because Openmeteo does not handle this error and silently returns nan's)

    ECMWF:  max 15 days - 6 hours update cycle
    ICON:   max 7.5 days - 6 hours update cycle
    ICONEU: max 5 days - 3 hours update cycle
    ICOND2: max 2 days - 3 hours update cycle
    AROME:  max 2 days - 3 hours update cycle
    """
    if model == "ECMWF":
        dtmax = 15*24*3600 - 6*3600
    elif model == "ICON":
        dtmax = 7.5*24*3600 - 6*3600
    elif model == "ICONEU":
        dtmax = 5*24*3600 - 3*3600       
    elif model == "ICOND2":
        dtmax = 2*24*3600 - 3*3600
    elif model == "AROME":
        dtmax = 2*24*3600 - 3*3600
    else:
        print("ERROR: Please select model = 'ECMWF', 'ICON', 'ICONEU', 'ICOND2', or 'AROME'")
        return 0 

    t_current = time.gmtime()
    t_input   = time.strptime(timestep, "%Y-%m-%dT%H:%M")
    delta_t   = time.mktime(t_input) - time.mktime(t_current)
    return delta_t > dtmax


def ishistorical(timestep):
    """
    Check if the timestep is longer than 4 days in the past,
    used to select the correct historical or forecast openmeteo url.
    """
    t_current = time.gmtime()
    t_input   = time.strptime(timestep, "%Y-%m-%dT%H:%M")
    delta_t   = time.mktime(t_input) - time.mktime(t_current)
    return delta_t < -4*24*3600