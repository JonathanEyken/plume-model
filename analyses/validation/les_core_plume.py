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

"""The default model against the LES plume core.

The default Plume has a ventilated base on the LES fire-front depth,
d(w**2)/dz = 2 (a_w B - b_w eps w**2), Morton and dynamic detrainment, and

    eps = fac_ent / sqrt(A_0) + a max(B, 0) / w**2.

Its targets are the LES core (les.py): theta' on the centreline of the 95th-percentile w, w_95,
the displacement and u of that centreline, and the plume top. Area has no LES core counterpart,
so it is shown against the LES mean plume.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import config
from plume_model.wur_plume import A_BUOYANT, A_W, B_W, FAC_ENT

import les
from les_dry_line_fire import at_height, fli_values, guide_line, run_model

OUTPUT = Path(__file__).parent / "output" / "les_dry_line_fire" / "core_plume"

# The buoyant term makes eps*dz ~ 0.3 at h0 for dz = 10 m.
DZ = 2.5

Z_TABLE = (20.0, 50.0, 100.0, 250.0, 500.0, 800.0)
FLI_PROFILES = 10e6


def run_core(case, dz=DZ):
    return run_model(case, dz, ventilated=True)


def core_pairs(case, plume):
    """(name, model profile, LES core profile) for the core targets."""
    return [
        ("theta'", plume.thetav - plume.thetav_env, case.theta_prime_core),
        ("w", plume.w, case.w_core),
        ("x", plume.x, case.x_core),
        ("u", plume.u, case.u_core),
    ]


def print_summary(cases, runs):
    """runs: (label, plumes) pairs."""
    low = np.array([c.U <= 3 for c in cases])
    high = np.array([c.U >= 9 for c in cases])
    print("Against the LES core, dz = %g m. Default model: fac_ent = %g, a = %g, a_w = %g, b_w = %g,"
          " ventilated base on the LES front depth" % (DZ, FAC_ENT, A_BUOYANT, A_W, B_W))
    print("\nz_top: model / LES, median (min - max), and median by wind")
    print("%-24s %20s %8s %8s %8s" % ("", "all", "U <= 3", "U >= 9", "rms [m]"))
    for label, plumes in runs:
        r = np.array([p.z_top / c.z_top for c, p in zip(cases, plumes)])
        err = np.array([p.z_top - c.z_top for c, p in zip(cases, plumes)])
        print("%-24s %6.2f (%4.2f - %4.2f) %8.2f %8.2f %8.0f"
              % (label, np.median(r), r.min(), r.max(), np.median(r[low]), np.median(r[high]),
                 np.sqrt(np.mean(err**2))))

    print("\nmodel / LES core, median over cases with both plumes at that height")
    for k, (name, _, _) in enumerate(core_pairs(cases[0], runs[0][1][0])):
        print("\n%-24s" % name + "".join("%7.0f" % z for z in Z_TABLE))
        for label, plumes in runs:
            row = []
            for z_at in Z_TABLE:
                ratios = []
                for c, p in zip(cases, plumes):
                    _, model, ref = core_pairs(c, p)[k]
                    ratios.append(at_height(p.z, model, z_at) / at_height(c.z, ref, z_at))
                row.append(np.nanmedian(ratios))
            print("%-24s" % label + "".join("%7.2f" % v for v in row))


def plot_overview(cases, plumes, path):
    """Plume top over (FLI, U), and theta', w and area at FLI_PROFILES for every U."""
    fig, axes = plt.subplot_mosaic([["top", "top", "scatter"], ["theta", "w", "area"]],
                                   figsize=(13, 9), height_ratios=[1, 1.3])

    ax = axes["top"]
    for i, fli in enumerate(fli_values(cases)):
        pairs = [(c, p) for c, p in zip(cases, plumes) if c.fli == fli]
        color = config.run_color(i)
        label = "FLI = %g MW/m" % (fli / 1e6)
        ax.errorbar([c.U for c, _ in pairs], [c.z_top / 1e3 for c, _ in pairs],
                    yerr=[c.z_top_std / 1e3 for c, _ in pairs], color=color,
                    linestyle=config.LINESTYLES["reference"], marker=config.MARKERS["case"],
                    capsize=3)
        ax.plot([c.U for c, _ in pairs], [p.z_top / 1e3 for _, p in pairs], color=color,
                linestyle=config.LINESTYLES["model"], label=label)
        axes["scatter"].errorbar([c.z_top / 1e3 for c, _ in pairs],
                                 [p.z_top / 1e3 for _, p in pairs],
                                 xerr=[c.z_top_std / 1e3 for c, _ in pairs], color=color,
                                 marker=config.MARKERS["case"], linestyle="none", capsize=3)
    ax.set_xlabel("U [m/s]")
    ax.set_ylabel("plume top z_top [km]")
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=config.COLORS["env"], linestyle=config.LINESTYLES["model"],
               label="model"),
        Line2D([], [], color=config.COLORS["env"], linestyle=config.LINESTYLES["reference"],
               marker=config.MARKERS["case"], label="LES, $\\pm$1 std over time")]

    ax = axes["scatter"]
    lims = [1.0, 4.0]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.plot(lims, lims, color=config.COLORS["env"], linestyle=config.LINESTYLES["zero"],
            linewidth=config.LINEWIDTHS["guide"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("LES z_top [km]")
    ax.set_ylabel("model z_top [km]")

    profile = [(c, p) for c, p in zip(cases, plumes) if c.fli == FLI_PROFILES]
    colors = config.sweep_colors(len(profile))
    for color, (c, p) in zip(colors, profile):
        zk, zm = c.z / 1e3, p.z / 1e3
        les_style = dict(color=color, linestyle=config.LINESTYLES["reference"],
                         linewidth=config.LINEWIDTHS["reference"])
        axes["theta"].plot(c.theta_prime_core, zk, **les_style)
        axes["theta"].plot(p.thetav - p.thetav_env, zm, color=color,
                           label="U = %g m/s" % c.U)
        axes["w"].plot(c.w_core, zk, **les_style)
        axes["w"].plot(p.w, zm, color=color)
        axes["area"].plot(c.area, zk, color=color, linestyle=config.LINESTYLES["zero"],
                          linewidth=config.LINEWIDTHS["guide"])
        axes["area"].plot(p.area, zm, color=color)

    h_abl = np.mean([c.h_abl for c, _ in profile])
    for name in ("theta", "w", "area"):
        guide_line(axes[name], h_abl / 1e3)
        axes[name].set_ylim(0, 3)
    guide_line(axes["theta"], 0.0, vertical=True)
    axes["theta"].set_xlim(-3, 30)
    axes["area"].set_xscale("log")
    axes["theta"].set_xlabel(r"$\theta'$ [K]")
    axes["w"].set_xlabel("w [m/s]")
    axes["area"].set_xlabel("area [m$^2$]")
    axes["theta"].set_ylabel("z [km]")
    env = config.COLORS["env"]
    axes["theta"].legend(handles=axes["theta"].get_legend_handles_labels()[0] + [
        Line2D([], [], color=env, linestyle=config.LINESTYLES["model"], label="model"),
        Line2D([], [], color=env, linestyle=config.LINESTYLES["reference"], label="LES core"),
        Line2D([], [], color=env, linestyle=config.LINESTYLES["zero"],
               linewidth=config.LINEWIDTHS["guide"], label="LES mean plume (area only)")],
        fontsize=7, loc="upper right")
    axes["w"].set_title("FLI = %g MW/m; dotted grey: boundary-layer top" % (FLI_PROFILES / 1e6),
                        fontsize=9)
    fig.suptitle("Default model (fac_ent = %g, a = %g, a_w = %g, b_w = %g, ventilated base) "
                 "against the LES core" % (FAC_ENT, A_BUOYANT, A_W, B_W))
    fig.savefig(path)
    plt.close(fig)


def main():
    config.apply()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [les.LESCase(name) for name in les.case_names()]
    plumes = [run_core(c) for c in cases]
    print_summary(cases, [("default model", plumes)])
    path = OUTPUT / "overview.png"
    plot_overview(cases, plumes, path)
    print("\nFigure written to %s" % path)


if __name__ == "__main__":
    main()
