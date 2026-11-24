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

"""Steady bulk plume, integrated explicitly upward in height at the environmental pressure."""

import numpy as np

from .thermo import Lv, Rd, cp, dewpoint, exner, g, qsat, sat_adjust, virtual_temp

A_W = 0.5        # Buoyancy coefficient of the w equation
B_W = 0.3        # Drag coefficient of the w equation
FAC_ENT = 0.1    # Morton entrainment fac_ent / sqrt(A_0)
A_BUOYANT = 0.5  # Buoyancy-driven entrainment a B / w**2
BETA = 0.5       # Ratio of fractional detrainment to Morton entrainment
C_DET = 2.0      # Dynamic detrainment -c_det / w dw/dz where w decreases
H0_PLUME = 20.0  # Height of the plume base, over which the surface flux sets w0
DZ_PLUME = 10.0

W_EPS = 1e-6


class Plume:
    """A Plume is a trajectory through an environment (env) above a fire with sensible and
    latent heat flux H, LE over fire_area, with a front of along-wind depth fire_depth, from
    a base at h0, on a grid with spacing dz until z_max.

    The surface flux sets plume excesses at h0 through free-convective scaling:
    w0**2 = K dtheta with K = 3 g a_w h0 / (2 thetav (1 + b_w)), from the model's w equation.
    With vent_base (the default), the heat leaves a control volume of height h0 and
    along-wind depth fire_depth D both upward and downwind at u_vent = s h0 / D, with s the environmental
    wind speed at h0:

        H = rho cp dtheta (w0 + u_vent),    so    w0**2 (w0 + u_vent) = K H / (rho cp).

    If vent_base=False, u_vent = 0 and w0**3 = K H / (rho cp).
    TODO: The base area A_0 (w0 + u_vent) / w0 includes the air vented downwind, so the plume still receives H.

    Plume equations, with M = rho_e A w and phi in (thetal, qt, u, v):

        dM/dz       = (eps - delta) M,
        dphi/dz     = -eps (phi - phi_e),
        d(w**2)/dz  = 2 (a_w B - b_w eps w**2),
        dx/dz       = u / w,    dy/dz = v / w,

    with pseudoadiabatic saturation adjustment, and entrainment and detrainment

        eps   = fac_ent / sqrt(A_0) + a_buoyant max(B, 0) / w**2,
        delta = beta fac_ent / sqrt(A_0) - c_det / w dw/dz   (dw/dz < 0 only).

    B / w**2 is the inverse length over which buoyancy accelerates the plume, which equals
    the fractional entrainment at constant area. Dynamic detrainment prevents plume area from
    going to infinity as the plume decelerates.

    The main function is ascend, which integrates upwards on a uniform grid. All profiles
    are calculated on this grid, including the environment on the plume's own grid and the
    terms of the w**2 budget.
    """

    def __init__(self, env, H, LE, fire_area, fire_depth, vent_base=True, fac_ent=FAC_ENT,
                 a_buoyant=A_BUOYANT, beta=BETA, c_det=C_DET, a_w=A_W, b_w=B_W,
                 h0=H0_PLUME, dz=DZ_PLUME, z_max=None, full_ascent=False):
        self.env = env
        self.H = H
        self.LE = LE
        self.fire_area = fire_area
        self.vent_base = vent_base
        self.fire_depth = fire_depth
        self.fac_ent = fac_ent
        self.a_buoyant = a_buoyant
        self.beta = beta
        self.c_det = c_det
        self.a_w = a_w
        self.b_w = b_w
        self.h0 = h0
        self.dz = dz
        self.z_max = env.z_top if z_max is None else z_max
        self.full_ascent = full_ascent

        self.k_top = -1
        self.k_lcl = -1
        self.z_top = np.nan
        self.stopped = False

    def surface(self):
        """(dtheta, dq, w0, area) the fire imposes on the environment at h0."""
        if self.H <= 0:
            return 0.0, 0.0, 0.0, self.fire_area

        rho = self.env.rho[0]
        thetav = self.env.thetav[0]
        K = 3 * g * self.a_w * self.h0 / (2 * thetav * (1 + self.b_w))
        R = K * self.H / (rho * cp)

        w0 = np.cbrt(R)
        u_vent = 0.0
        if self.vent_base:
            if self.fire_depth is None:
                raise ValueError("A ventilated plume base needs a fire_depth.")
            wind_speed = np.hypot(np.interp(self.h0, self.env.z, self.env.u),
                                  np.interp(self.h0, self.env.z, self.env.v))
            u_vent = wind_speed * self.h0 / self.fire_depth

            # Newton solver for w0**3 + u_vent w0**2 = R, starting from w0 when u_vent=0.
            for _ in range(50):
                dw0 = (w0 ** 3 + u_vent * w0 ** 2 - R) / (3 * w0 ** 2 + 2 * u_vent * w0)
                w0 -= dw0
                if abs(dw0) < 1e-10 * w0:
                    break
        dtheta = w0 ** 2 / K
        dq = self.LE / (rho * Lv * (w0 + u_vent))
        area = self.fire_area * (w0 + u_vent) / w0
        return dtheta, dq, w0, area

    def entrainment(self, w, w_prev, buoy):
        """Fractional entrainment and detrainment."""

        # Entrainment
        eps_morton = self.fac_ent / np.sqrt(self.fire_area)
        eps_buoyant = self.a_buoyant * max(buoy, 0.0) / max(w, W_EPS) ** 2
        eps = eps_morton + eps_buoyant

        # Detrainment
        delta_morton = self.beta * eps_morton
        delta_dyn = 0.0
        if W_EPS <= w < w_prev:
            # Backward difference of ln w, applied explicitly over the next step.
            delta_dyn -= self.c_det * np.log(w / w_prev) / self.dz
        delta = delta_morton + delta_dyn
        return eps, delta

    def ascend(self):
        """Integrate the plume from h0 up to where it stops, or to z_max."""
        env = self.env
        dz = self.dz
        n = int(np.floor((self.z_max - self.h0) / dz)) + 1

        # Allocate all profiles; the base state is the plume at h0, so the integration starts there.
        self.z = self.h0 + np.arange(n, dtype=float) * dz
        for name in ("thetal", "qt", "ql", "qi", "thetav", "T", "Tv", "Td", "area", "w",
                     "mass_flux", "entrainment_flux", "detrainment_flux", "eps", "delta",
                     "buoy", "u", "v", "x", "y", "dw2dz_buoy", "dw2dz_drag"):
            setattr(self, name, np.zeros(n))
        self.saturated = np.zeros(n, dtype=bool)
        z = self.z

        # The environment is interpolated to the plume grid, then its variables are derived.
        self.T_env = np.interp(z, env.z, env.T)
        self.Td_env = np.interp(z, env.z, env.Td)
        self.p = np.interp(z, env.z, env.p)
        self.u_env = np.interp(z, env.z, env.u)
        self.v_env = np.interp(z, env.z, env.v)
        self.exner_env = exner(self.p)
        self.theta_env = self.T_env / self.exner_env
        self.qt_env = qsat(self.Td_env, self.p)
        self.thetav_env = virtual_temp(self.theta_env, self.qt_env)
        self.rho_env = self.p / (Rd * self.exner_env * self.thetav_env)

        dtheta, dq, w0, area = self.surface()

        # If the plume's initialisation gives no updraft, stop immediately
        if w0 < W_EPS:
            self._finish(0)
            return self

        # Execute the first model level separately
        self._start(dtheta, dq, w0, area)
        self.eps[0], self.delta[0] = self.entrainment(w0, w0, self.buoy[0])
        self.entrainment_flux[0] = self.eps[0] * self.mass_flux[0]
        self.detrainment_flux[0] = self.delta[0] * self.mass_flux[0]
        self.dw2dz_buoy[0] = 2 * self.a_w * self.buoy[0]
        self.dw2dz_drag[0] = -2 * self.b_w * self.eps[0] * w0 ** 2

        # Iterate until plume top, or until w < W_EPS, or until mass flux <= 0
        i = 1
        while i < n:

            # Main dynamical vertical upwards integration
            self._step(i)

            # The rates and fluxes at level i drive the step from i to i + 1.
            self.eps[i], self.delta[i] = self.entrainment(self.w[i], self.w[i - 1], self.buoy[i])
            self.entrainment_flux[i] = self.eps[i] * self.mass_flux[i]
            self.detrainment_flux[i] = self.delta[i] * self.mass_flux[i]

            if self.w[i] < W_EPS or self.mass_flux[i] <= 0:
                if self.k_top == -1:
                    self.k_top = i - 1
                    self.stopped = True
                if not self.full_ascent:
                    break
            i += 1

        if self.k_top == -1:
            self.k_top = i - 1

        self._finish(i)
        return self

    def _start(self, dtheta, dq, w0, area):
        # The fire perturbation is a dry heat excess (ql = 0 here), so thetal == theta.
        self.thetal[0] = self.theta_env[0] + dtheta
        self.qt[0] = self.qt_env[0] + dq

        T, ql, qi = sat_adjust(self.thetal[0], self.qt[0], self.p[0])
        self.T[0] = T
        self.ql[0] = ql
        self.qi[0] = qi
        self.Tv[0] = virtual_temp(T, self.qt[0], ql, qi)
        self.Td[0] = min(dewpoint(self.qt[0], self.p[0]), T)
        self.thetav[0] = self.Tv[0] / self.exner_env[0]
        self.buoy[0] = g / self.thetav_env[0] * (self.thetav[0] - self.thetav_env[0])
        self.area[0] = area
        self.w[0] = w0
        self.mass_flux[0] = self.rho_env[0] * area * w0

        self.u[0] = self.u_env[0]
        self.v[0] = self.v_env[0]

    def _step(self, i):
        j = i - 1
        dz = self.dz
        M = self.mass_flux
        E = self.entrainment_flux

        M[i] = M[j] + (E[j] - self.detrainment_flux[j]) * dz

        # theta_env stands in for thetal_env, exact only while the environment is unsaturated.
        self.thetal[i] = self.thetal[j] - E[j] * (self.thetal[j] - self.theta_env[j]) / M[j] * dz
        self.qt[i] = self.qt[j] - E[j] * (self.qt[j] - self.qt_env[j]) / M[j] * dz

        alive = self.k_top == -1

        if alive:
            self.u[i] = self.u[j] - E[j] * (self.u[j] - self.u_env[j]) / M[j] * dz
            self.v[i] = self.v[j] - E[j] * (self.v[j] - self.v_env[j]) / M[j] * dz
            self.x[i] = self.x[j] + self.u[j] / self.w[j] * dz
            self.y[i] = self.y[j] + self.v[j] / self.w[j] * dz
        else:
            self.u[i] = self.u[j]
            self.v[i] = self.v[j]
            self.x[i] = self.x[j]
            self.y[i] = self.y[j]

        T, ql, qi = sat_adjust(self.thetal[i], self.qt[i], self.p[i])

        # Pseudoadiabatic: the condensate is removed rather than carried up with the plume.
        self.qt[i] -= ql
        self.thetal[i] = T / self.exner_env[i]

        self.T[i] = T
        self.ql[i] = ql
        self.qi[i] = qi
        self.Tv[i] = virtual_temp(T, self.qt[i], 0.0, 0.0)
        # Above the LCL the plume is saturated, so its dewpoint is capped at T.
        self.Td[i] = min(dewpoint(self.qt[i], self.p[i]), T)
        self.thetav[i] = self.Tv[i] / self.exner_env[i]
        self.saturated[i] = ql > 0 or qi > 0

        self.buoy[i] = g / self.thetav_env[i] * (self.thetav[i] - self.thetav_env[i])

        self.dw2dz_buoy[i] = 2 * self.a_w * self.buoy[i]
        self.dw2dz_drag[i] = -2 * self.b_w * self.eps[j] * self.w[j] ** 2
        w2 = self.w[j] ** 2 + (self.dw2dz_buoy[i] + self.dw2dz_drag[i]) * dz
        self.w[i] = np.sqrt(max(0.0, w2)) if alive else 0.0

        self.area[i] = M[i] / (self.rho_env[i] * self.w[i]) if self.w[i] > W_EPS else np.nan

    def _finish(self, n_keep):
        # Cut every profile down to the part that was actually integrated.
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                setattr(self, name, value[:n_keep])

        self.k_lcl = int(np.argmax(self.saturated)) if self.saturated.any() else -1
        self.z_top = float(self.z[self.k_top]) if self.k_top >= 0 else np.nan
