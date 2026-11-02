"""Make the figures of report.md from the Newton logs of the runs in output/newton_refinement/.

    python notes/newton-mesh-refinement/make_figures.py [figure_dir]

The runs are one period (200 steps of dt = 0.0025 from t = 16.5525) on each mesh of the family
data/meshes/fsi2/refinement/mesh_sec_ref{0,1,2}, from the settled states, in the configurations
j0_pred, j0_nopred (J_0 reassembled in every iteration, with and without the prediction, tight
relative tolerance), full_pred (the full Jacobian with the prediction, tight relative tolerance) and
j0_reuse (J_0 with the prediction and reuse, gamma 0.2, default tolerances).

The time step sweep and the spectral calculation are in output/newton_dt/: runs over the first
WINDOW = 0.12 s of that period, from the same states, spec_ref<level>_dt<dt> (the full Jacobian with
the prediction, writing the spectrum of J_0^-1 N every 0.005 s), j0_ref<level>_dt<dt> (J_0 with the
prediction, reassembled in every iteration) and split_<part>_ref0_dt<dt> (the spectrum with only the
momentum or only the continuity part of the fluid ALE derivatives left out of J_0).

The eigenvectors are in output/newton_modes/: runs over the first 0.04 s, from the same states,
modes_ref<level>_dt<dt> (the full Jacobian with the prediction, writing the 4 dominant eigenpairs of
J_0^-1 N every 0.01 s), measured by analyse_modes.py into output/newton_modes/analysis/. Writes SVG
and PNG figures into figure_dir, by default plots/ next to this file.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

from xfsi_solver.scripts.plot_newton_logs import _RC, SERIES_COLORS, TEXT_SECONDARY, Run
from xfsi_solver.tools.newton_log import read_newton_log

ROOT = Path("output/newton_refinement")
DT_ROOT = Path("output/newton_dt")
DTS = [0.005, 0.0025, 0.00125, 0.000625]
T_WINDOW_END = 16.5525 + 0.12 + 1e-9
LEVELS = [0, 1, 2]
H = {level: 2.0**-level for level in LEVELS}  # mesh size relative to ref0
LEVEL_LABELS = {0: "ref0 (h)", 1: "ref1 (h/2)", 2: "ref2 (h/4)"}
MODES_ROOT = Path("output/newton_modes/analysis")
# The start of the beam at the cylinder, and its length, the arc length of the interface at the lower tip corner
BEAM_X0 = 0.2 + np.sqrt(0.05**2 - 0.01**2)
BEAM_LENGTH = 0.6 - BEAM_X0
# Ratios only between residuals above FLOOR * ||R_0|| of their step, clear of round-off.
FLOOR = 1.0e-9

figure_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "plots"
figure_dir.mkdir(parents=True, exist_ok=True)


def run(kind: str, level: int) -> Run | None:
    """The run, or None if it has no log yet, or too few steps to read, while it starts."""
    run_dir = ROOT / f"{kind}_ref{level}"
    if not (run_dir / f"{kind}_ref{level}_newton.txt").exists():
        return None
    try:
        return Run(f"{kind}, ref{level}", run_dir)
    except ValueError:
        return None


def ratios(r: Run, first: bool) -> np.ndarray:
    """Contraction ||R_(k+1)|| / ||R_k|| of every step, in the first iteration or the later ones."""
    out = []
    for step in r.steps:
        res = step["residual"]
        ks = [0] if first else range(1, len(res) - 1)
        out += [res[k + 1] / res[k] for k in ks if k + 1 < len(res) and res[k + 1] > FLOOR * res[0]]
    return np.array(out)


def step_rates(r: Run) -> np.ndarray:
    """Median contraction of the later iterations of each step, NaN for a step without any."""
    rates = []
    for step in r.steps:
        res = step["residual"]
        rs = [res[k + 1] / res[k] for k in range(1, len(res) - 1) if res[k + 1] > FLOOR * res[0]]
        rates.append(np.median(rs) if rs else np.nan)
    return np.array(rates)


def plot_rates_vs_h(path: Path) -> None:
    fig, (ax_rate, ax_its) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    series = [("j0_pred", False, "$J_0$, predicted: later iterations", SERIES_COLORS[0]),
              ("j0_nopred", False, "$J_0$, not predicted: later iterations", SERIES_COLORS[1]),
              ("j0_pred", True, "$J_0$, predicted: first iteration", SERIES_COLORS[2])]  # fmt: skip
    for kind, first, label, color in series:
        hs, med, lo, hi = [], [], [], []
        for level in LEVELS:
            r = run(kind, level)
            if r is None:
                continue
            values = ratios(r, first)
            hs.append(H[level])
            q = np.percentile(values, [25, 50, 75])
            lo.append(q[1] - q[0])
            med.append(q[1])
            hi.append(q[2] - q[1])
        ax_rate.errorbar(hs, med, yerr=[lo, hi], color=color, marker="o", markersize=5, capsize=3, label=label)
    ax_rate.set_xscale("log", base=2)
    ax_rate.set_yscale("log")
    ax_rate.set_xticks([H[level] for level in LEVELS], ["h", "h/2", "h/4"])
    ax_rate.invert_xaxis()
    ax_rate.set_xlabel("mesh size")
    ax_rate.set_ylabel(r"contraction $\|R_{k+1}\| / \|R_k\|$")
    ax_rate.set_title("Contraction per Newton iteration: median and IQR", loc="left", fontsize=9)
    ax_rate.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8)

    its_series = [("j0_pred", "$J_0$, predicted", SERIES_COLORS[0]),
                  ("j0_nopred", "$J_0$, not predicted", SERIES_COLORS[1]),
                  ("full_pred", "full Jacobian, predicted", SERIES_COLORS[3]),
                  ("j0_reuse", r"$J_0$, predicted, reused ($\gamma$=0.2)", SERIES_COLORS[4])]  # fmt: skip
    for kind, label, color in its_series:
        hs, its = [], []
        for level in LEVELS:
            r = run(kind, level)
            if r is not None:
                hs.append(H[level])
                its.append(r.iterations.mean())
        ax_its.plot(hs, its, color=color, marker="o", markersize=5, label=label)
    ax_its.set_xscale("log", base=2)
    ax_its.set_xticks([H[level] for level in LEVELS], ["h", "h/2", "h/4"])
    ax_its.invert_xaxis()
    ax_its.set_xlabel("mesh size")
    ax_its.set_ylabel("Newton iterations per step")
    ax_its.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_its.set_title("Mean Newton iterations per step (tight tolerance except reuse)", loc="left", fontsize=9)
    ax_its.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_residual_histories(path: Path) -> None:
    kinds = [("j0_pred", "$J_0$, predicted"), ("full_pred", "full Jacobian, predicted")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(9.0, 3.6), sharey=True)
    for ax, (kind, title) in zip(axes, kinds, strict=True):
        max_it = 0
        for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
            r = run(kind, level)
            if r is None:
                continue
            n = int(r.iterations.max())
            values = r.residuals(n) / r.residuals(n)[:, :1]
            reached = np.sum(~np.isnan(values), axis=0) >= len(values) / 2
            k = np.arange(n + 1)[reached]
            max_it = max(max_it, k[-1])
            lower, median, upper = (np.nanpercentile(values[:, reached], q, axis=0) for q in (25, 50, 75))
            ax.fill_between(k, lower, upper, color=color, alpha=0.18, linewidth=0)
            ax.plot(k, median, color=color, marker="o", markersize=4, label=LEVEL_LABELS[level])
        ax.set_yscale("log")
        ax.set_xticks(range(max_it + 1))
        ax.set_xlabel("Newton iteration $k$")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}: residual relative to the first", loc="left", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel(r"$\|R_k\| / \|R_0\|$")
    fig.savefig(path)
    plt.close(fig)


def plot_rate_over_period(path: Path) -> None:
    fig, (ax_disp, ax_rate) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True, height_ratios=[1, 1.6])
    for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
        r = run("j0_pred", level)
        if r is None:
            continue
        ax_disp.plot(r.qoi[1:, 0], r.qoi[1:, 4], color=color, linewidth=1.2, label=LEVEL_LABELS[level])
        ax_rate.plot(r.times, step_rates(r), color=color, linewidth=1.0, label=LEVEL_LABELS[level])
    ax_disp.set_ylabel("tip $y$-disp. [m]")
    ax_disp.set_title(
        "Tip displacement, and median contraction of the later iterations of each step ($J_0$, predicted)",
        loc="left",
        fontsize=9,
    )
    ax_rate.set_yscale("log")
    ax_rate.set_ylabel("contraction")
    ax_rate.set_xlabel("$t$ [s]")
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_reuse_iterations(path: Path) -> None:
    fig, (ax_its, ax_asm) = plt.subplots(2, 1, figsize=(6.4, 3.8), sharex=True, height_ratios=[1.6, 0.6])
    plotted = []
    for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
        r = run("j0_reuse", level)
        if r is None:
            continue
        plotted.append(level)
        ax_its.step(r.times, r.iterations, where="mid", color=color, linewidth=1.0,
                    label=f"{LEVEL_LABELS[level]} (mean {r.iterations.mean():.2f})")  # fmt: skip
        assembly_times = [t for t, step in zip(r.times, r.steps, strict=True) if np.any(step["assembled"])]
        ax_asm.eventplot(assembly_times, lineoffsets=len(plotted) - 1, linelengths=0.7, linewidths=1.0, colors=color)
    ax_its.set_ylabel("Newton iterations")
    ax_its.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_its.set_title(r"Newton iterations per step with reuse ($J_0$, predicted, $\gamma$=0.2)", loc="left", fontsize=9)
    ax_asm.set_yticks(range(len(plotted)), [""] * len(plotted))
    ax_asm.set_ylabel("assembly", rotation=0, ha="right", va="center", color=TEXT_SECONDARY)
    ax_asm.grid(axis="y", visible=False)
    ax_asm.set_xlabel("$t$ [s]")
    handles, labels = ax_its.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def spectrum(name: str) -> np.ndarray | None:
    path = DT_ROOT / name / f"{name}_spectrum.txt"
    return np.genfromtxt(path, names=True) if path.exists() else None


def window_ratios(path: Path, first: bool) -> np.ndarray:
    """The contractions of the steps of a Newton log in the window, in the first iteration or the later ones."""
    rows = read_newton_log(path)
    rows = rows[rows["t"] <= T_WINDOW_END]
    out = []
    for t in np.unique(rows["t"]):
        res = rows[rows["t"] == t]["residual"]
        ks = [0] if first else range(1, len(res) - 1)
        out += [res[k + 1] / res[k] for k in ks if k + 1 < len(res) and res[k + 1] > FLOOR * res[0]]
    return np.array(out)


def j0_log(level: int, dt: float) -> Path:
    if dt == 0.0025:  # the first 0.12 s of the period of the refinement study
        return ROOT / f"j0_pred_ref{level}" / f"j0_pred_ref{level}_newton.txt"
    return DT_ROOT / f"j0_ref{level}_dt{dt:g}" / f"j0_ref{level}_dt{dt:g}_newton.txt"


def plot_rho_vs_dt(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    def series(values_by_dt, label, color, marker="o", linestyle="-"):
        dts = [dt for dt in DTS if values_by_dt.get(dt) is not None and len(values_by_dt[dt])]
        q = np.array([np.percentile(values_by_dt[dt], [25, 50, 75]) for dt in dts])
        ax.errorbar(dts, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=color, marker=marker,
                    linestyle=linestyle, markersize=5, capsize=3, label=label)  # fmt: skip

    def spectral(prefix):
        return {dt: (lambda a: None if a is None else a["abs_1"])(spectrum(f"{prefix}_dt{dt:g}")) for dt in DTS}

    series(spectral("spec_ref0"), r"$\rho(J_0^{-1}N)$", SERIES_COLORS[0])
    series(spectral("split_momentum_ref0"), r"$\rho$, only the momentum part left out", SERIES_COLORS[3], "s", "--")
    series(spectral("split_continuity_ref0"), r"$\rho$, only the continuity part left out", SERIES_COLORS[4], "s", "--")
    measured = {dt: window_ratios(j0_log(0, dt), False) if j0_log(0, dt).exists() else None for dt in DTS}
    first = {dt: window_ratios(j0_log(0, dt), True) if j0_log(0, dt).exists() else None for dt in DTS}
    series(measured, "$J_0$, predicted: later iterations (measured)", SERIES_COLORS[1], "^", ":")
    series(first, "$J_0$, predicted: first iteration (measured)", SERIES_COLORS[2], "v", ":")
    ref = np.array([DTS[0], DTS[-1]])
    ax.plot(ref, 1.2e-2 * ref / DTS[0], color=TEXT_SECONDARY, linewidth=1.0, linestyle="-.", label=r"$\propto$ dt")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(DTS, [f"{dt:g}" for dt in DTS])
    ax.minorticks_off()
    ax.set_xlabel("time step dt")
    ax.set_ylabel("contraction")
    ax.set_title("Asymptotic and measured contraction against dt on h: median and IQR", loc="left", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncols=2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_rho_over_window(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for dt, color in zip(DTS, SERIES_COLORS, strict=False):
        a = spectrum(f"spec_ref0_dt{dt:g}")
        if a is not None:
            ax.plot(a["t"], a["abs_1"], color=color, marker="o", markersize=3, linewidth=1.2, label=f"dt = {dt:g}")
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$\rho(J_0^{-1}N)$")
    ax.set_title(r"Spectral radius of $J_0^{-1}N$ over the window on h, from the fast phase to the turning point",
                 loc="left", fontsize=9)  # fmt: skip
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncols=4, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_rho_vs_h(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    # h/4 at dt = 0.000625 covers only the first 0.04 s, so the line refined in both is compared there.
    lines = [
        ("h refined, dt = 0.0025", [(level, 0.0025) for level in LEVELS], SERIES_COLORS[0], 0.12),
        (
            "h and dt refined together (first 0.04 s)",
            [(0, 0.0025), (1, 0.00125), (2, 0.000625)],
            SERIES_COLORS[1],
            0.04,
        ),
        ("dt refined, on h", [(0, 0.0025), (0, 0.00125), (0, 0.000625)], SERIES_COLORS[2], 0.12),
    ]
    for label, points, color, window in lines:
        xs, q = [], []
        for i, (level, dt) in enumerate(points):
            a = spectrum(f"spec_ref{level}_dt{dt:g}")
            if a is not None:
                a = a[a["t"] <= 16.5525 + window + 1e-9]
                xs.append(2.0**-i)
                q.append(np.percentile(a["abs_1"], [25, 50, 75]))
        if xs:
            q = np.array(q)
            ax.errorbar(xs, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=color, marker="o",
                        markersize=5, capsize=3, label=label)  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 0.5, 0.25], ["1", "1/2", "1/4"])
    ax.minorticks_off()
    # The values lie within one decade, so the log axis gets its labels explicitly.
    ticks = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3]
    ax.set_yticks(ticks, [f"{tick:.0e}".replace("e-0", "e-") for tick in ticks])
    ax.invert_xaxis()
    ax.set_xlabel("refinement factor (h, dt or both)")
    ax.set_ylabel(r"$\rho(J_0^{-1}N)$")
    ax.set_title("Spectral radius under refinement in h, in dt and in both: median and IQR", loc="left", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def modes(name: str) -> list[dict]:
    """The measured modes of a run of output/newton_modes/, by sample time and then by index."""
    out = []
    for path in sorted(MODES_ROOT.glob(f"{name}_t*_mode*.npz")):
        d = dict(np.load(path))
        d["index"] = int(path.stem.rsplit("mode", 1)[1])
        d["eigenvalue"] = complex(d["eigenvalue"])
        out.append(d)
    return out


def families(name: str) -> dict[str, list[dict]]:
    """In every sample, the complex pair of largest modulus (the tip pair) and the real mode of largest
    modulus whose interface displacement is at least 40% net change of the area of the beam (the volume mode)."""
    by_t = {}
    for d in modes(name):
        by_t.setdefault(float(d["t"]), []).append(d)
    out = {"tip": [], "volume": []}
    for ds in by_t.values():
        pairs = [d for d in ds if abs(d["eigenvalue"].imag) > 0.05 * abs(d["eigenvalue"])]
        volume = [
            d
            for d in ds
            if abs(d["eigenvalue"].imag) <= 0.05 * abs(d["eigenvalue"])
            and abs(d["flux_u_I"].real) >= 0.4 * d["flux_abs_u_I"]
        ]
        for key, candidates in (("tip", pairs), ("volume", volume)):
            if candidates:
                out[key].append(max(candidates, key=lambda d: abs(d["eigenvalue"])))
    return out


FAMILY_LABELS = {"tip": "tip pair (complex)", "volume": "volume mode (real)"}


def plot_mode_families(path: Path) -> None:
    fig, (ax_dt, ax_h) = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)

    def series(ax, xs, names, key, color, marker):
        points = [(x, [abs(d["eigenvalue"]) for d in families(name)[key]]) for x, name in zip(xs, names, strict=True)]
        points = [(x, v) for x, v in points if v]
        if points:
            q = np.array([np.percentile(v, [25, 50, 75]) for _, v in points])
            ax.errorbar([x for x, _ in points], q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=color,
                        marker=marker, markersize=5, capsize=3, label=FAMILY_LABELS[key])  # fmt: skip

    for key, color, marker in (("tip", SERIES_COLORS[0], "o"), ("volume", SERIES_COLORS[1], "s")):
        series(ax_dt, DTS, [f"modes_ref0_dt{dt:g}" for dt in DTS], key, color, marker)
        series(ax_h, [H[level] for level in LEVELS], [f"modes_ref{level}_dt0.0025" for level in LEVELS], key, color,
               marker)  # fmt: skip
    for ax in (ax_dt, ax_h):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.invert_xaxis()
    ax_dt.set_xticks(DTS, [f"{dt:g}" for dt in DTS])
    ax_h.set_xticks([1, 0.5, 0.25], ["h", "h/2", "h/4"])
    ticks = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3]
    ax_dt.set_yticks(ticks, [f"{tick:.0e}".replace("e-0", "e-") for tick in ticks])
    ax_dt.set_xlabel("time step dt, on h")
    ax_h.set_xlabel("mesh, at dt = 0.0025")
    ax_dt.set_ylabel(r"$|\lambda|$ of $J_0^{-1}N$")
    ax_dt.set_title("The two families of slow modes: median and IQR over 4 samples", loc="left", fontsize=9)
    ax_dt.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncols=2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def first_sample(name: str, key: str) -> dict | None:
    found = [d for d in families(name)[key] if "field_x" in d]
    return found[0] if found else None


def plot_mode_fields(path: Path) -> None:
    """dp over the channel, and dp and |dv| near the tip, of the two families, on h at dt = 0.0025."""
    name = "modes_ref0_dt0.0025"
    zoom = (0.45, 0.75, 0.12, 0.28)
    fig, axes = plt.subplots(3, 2, figsize=(9.8, 6.6), layout="constrained",
                             gridspec_kw={"height_ratios": [1.0, 1.3, 1.3]})  # fmt: skip
    for column, key in enumerate(("tip", "volume")):
        d = first_sample(name, key)
        if d is None:
            continue
        x = d["field_x"]
        tri = Triangulation(x[:, 0], x[:, 1], d["field_triangles"])
        p, speed = d["field_p"], np.hypot(*d["field_v"].T)
        in_zoom = (x[:, 0] > zoom[0]) & (x[:, 0] < zoom[1]) & (x[:, 1] > zoom[2]) & (x[:, 1] < zoom[3])
        lam = d["eigenvalue"]
        axes[0, column].set_title(f"{FAMILY_LABELS[key]}: $\\lambda$ = {lam.real:.2e} {lam.imag:+.2e}i", loc="left",
                                  fontsize=9)  # fmt: skip
        for row, values, limit in ((0, p, np.percentile(np.abs(p), 99.5)), (1, p, np.abs(p[in_zoom]).max())):
            image = axes[row, column].tripcolor(tri, values, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                                shading="gouraud", rasterized=True)  # fmt: skip
            fig.colorbar(image, ax=axes[row, column], label=r"$\delta p$", shrink=0.9)
        low, high = np.percentile(speed[in_zoom], [2, 100])
        image = axes[2, column].tripcolor(tri, np.clip(speed, low, high), cmap="Blues", norm=LogNorm(low, high),
                                          shading="gouraud", rasterized=True)  # fmt: skip
        fig.colorbar(image, ax=axes[2, column], label=r"$|\delta v|$", shrink=0.9)
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes[1:].ravel():
        ax.set_xlim(*zoom[:2])
        ax.set_ylim(*zoom[2:])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mode_interface(path: Path) -> None:
    """|du_I| along the interface of the two families on the three meshes, at dt = 0.0025."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, key in zip(axes, ("tip", "volume"), strict=True):
        for level, color in zip(LEVELS, SERIES_COLORS, strict=False):
            found = families(f"modes_ref{level}_dt0.0025")[key]
            if found:
                d = found[0]
                magnitude = np.sqrt(np.abs(d["trace_u_n"]) ** 2 + np.abs(d["trace_u_t"]) ** 2)
                ax.plot(d["trace_s"] / BEAM_LENGTH, magnitude, color=color, linewidth=1.0, label=LEVEL_LABELS[level])
        for corner in (BEAM_LENGTH, BEAM_LENGTH + 0.02):
            ax.axvline(corner / BEAM_LENGTH, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(3e-2, 30)
        ax.set_xlabel("arc length / beam length: bottom, tip, top")
        ax.set_title(FAMILY_LABELS[key], loc="left", fontsize=9)
    axes[0].set_ylabel(r"$|\delta u_I|$, with $\|\delta u_I\|_{L^2} = 1$")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncols=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


with plt.rc_context(_RC):
    for suffix in ("svg", "png"):
        plot_rates_vs_h(figure_dir / f"rates_vs_h.{suffix}")
        plot_residual_histories(figure_dir / f"residual_histories.{suffix}")
        plot_rate_over_period(figure_dir / f"rate_over_period.{suffix}")
        plot_reuse_iterations(figure_dir / f"reuse_iterations.{suffix}")
        plot_rho_vs_dt(figure_dir / f"rho_vs_dt.{suffix}")
        plot_rho_over_window(figure_dir / f"rho_over_window.{suffix}")
        plot_rho_vs_h(figure_dir / f"rho_vs_h.{suffix}")
        plot_mode_families(figure_dir / f"mode_families.{suffix}")
        plot_mode_fields(figure_dir / f"mode_fields.{suffix}")
        plot_mode_interface(figure_dir / f"mode_interface.{suffix}")
