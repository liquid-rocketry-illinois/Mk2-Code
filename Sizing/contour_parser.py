import os
import sys

import numpy as np
import pandas as pd

"""
The purpose of this script is to turn the x-r contour table written by dxf_reader.py into every
geometric quantity the rest of the analysis needs - first and foremost the area profile A(x),
then the throat, the area ratios, the chamber dimensions, the volumes and the wetted areas.

INPUTS
------
path        : path to the contour CSV (columns x and r), defaults to engine_contour.csv here
n_stations  : if given, resample the contour onto this many uniformly spaced stations. The grid
              is anchored on the throat so A* survives the resample exactly; up to one step is
              lost off each end. Leave as None to keep the drawing's own vertices.
scale       : multiplies x and r, for a contour drawn in something other than meters

OUTPUTS
-------
A dict. Per-station arrays, all the same length and ordered by increasing x:
x, r, A, D, dr_dx, perimeter, area_ratio (A/A*), branch ("sub"/"sup", for the isentropic solve)
Scalars, all SI:
r_t, D_t, A_t, x_t, i_t                     : throat
r_c, D_c, A_c                               : chamber barrel
r_e, D_e, A_e                               : exit
contraction_ratio, expansion_ratio          : A_c/A_t, A_e/A_t
L_total, L_barrel, L_converging, L_chamber, L_diverging
L_star                                      : chamber volume / A_t
V_chamber, V_nozzle, V_total
S_chamber, S_nozzle, S_total                : wetted (gas side) surface areas
theta_conv, theta_div_max, theta_exit       : half angles [deg]
Rc_throat_up, Rc_throat_down                : throat radii of curvature, the Bartz input
percent_bell                                : L_diverging against a 15 deg cone to the same area ratio

ASSUMPTIONS
-----------
Axisymmetric contour
The throat is the single minimum-radius vertex. A contour with a flat throat section reports
the first vertex of it.
Units are whatever the CSV is in
"""

FLAT_SLOPE_TOL = 1e-3  # |dr/dx| under this counts as a cylindrical wall (0.06 deg)


def read_contour(path, scale=1.0):
    """The CSV as strictly-increasing x and matching r, both scaled."""
    df = pd.read_csv(path)
    missing = {"x", "r"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

    df = df[["x", "r"]].astype(float).sort_values("x")
    df = df[~df.x.round(12).duplicated()]  # the DXF repeats the throat vertex, see dxf_reader
    if len(df) < 3:
        raise ValueError(f"{path} has {len(df)} usable stations, need at least 3")

    return df.x.values * scale, df.r.values * scale


def resample(x, r, n, anchor):
    """About n uniformly spaced stations, with nodes landing exactly on `anchor` and on the exit.

    A plain linspace over the span would drop the throat between two nodes and interpolate a
    slightly-too-large A* into the grid, which then biases every area ratio downstream of it. So
    the grid is anchored on the throat, and the step is then nudged to divide the diverging
    length a whole number of times so the exit lands on a node too - between them that pins A*
    and A_e, and with them the contraction and expansion ratios, to their drawn values.

    What gives instead is the inlet, which is clipped by up to one step. That is the harmless
    end to lose: it sits inside the constant-radius barrel, so it costs a sliver of chamber
    length and volume and changes no radius, no area and no ratio. The station count comes back
    within a step or two of n for the same reason.
    """
    dx = (x[-1] - x[0]) / (n - 1)
    k_hi = max(1, round((x[-1] - anchor) / dx))  # exit lands on a node
    dx = (x[-1] - anchor) / k_hi
    k_lo = int(np.ceil((x[0] - anchor) / dx))
    grid = anchor + np.arange(k_lo, k_hi + 1) * dx
    grid[-1] = x[-1]  # kill the float drift so the exit radius is read, not extrapolated
    return grid, np.interp(grid, x, r)


def _barrel(x, r, i_t):
    """(start, end) node indices of the longest constant-radius run upstream of the throat.

    Returns (0, 0) - an empty barrel - when the chamber tapers the whole way, which is what a
    purely conical chamber looks like.
    """
    flat = np.abs(np.diff(r[: i_t + 1])) <= FLAT_SLOPE_TOL * np.abs(np.diff(x[: i_t + 1]))
    best = (0, 0)
    start = None
    for i, is_flat in enumerate(np.append(flat, False)):
        if is_flat and start is None:
            start = i
        elif not is_flat and start is not None:
            if x[i] - x[start] > x[best[1]] - x[best[0]]:
                best = (start, i)
            start = None
    return best


def _throat_curvature(x, r, i_t, side, window=0.1):
    """Radius of curvature of the wall on one side of the throat - the R_t that Bartz wants.

    Fits r = r_t + a*(x - x_t)^2 to the vertices within `window` throat radii of the throat and
    reads the curvature off it: R = 1/(2a). A least-squares circle fit would be tighter, but the
    wall right at a throat is a shallow arc where the parabola and the circle agree to well
    inside the tolerance the contour was drawn to.

    The window has to stay inside the throat arcs or the fit averages in whatever the contour
    does after them. On a Rao bell the downstream arc turns into the parabola within ~0.2 r_t of
    the throat, and this contour shows it: at a 1.0 r_t window the downstream fit reads 0.94 r_t,
    while from 0.15 r_t down it settles on 0.38 r_t - the arc that was actually drawn. Small it
    is, then, widened only if a side is too sparse to fit.
    """
    x_t, r_t = x[i_t], r[i_t]
    while True:
        sel = np.abs(x - x_t) <= window * r_t
        sel &= (x <= x_t) if side == "up" else (x >= x_t)
        if sel.sum() >= 5:
            break
        window *= 1.5
        if window > 1.0:  # a contour this coarse near the throat has no arc to find
            return np.nan

    dx = x[sel] - x_t
    dr = r[sel] - r_t
    a = float(np.sum(dx**2 * dr) / np.sum(dx**4))  # least squares through (x_t, r_t)
    if a <= 0:
        return np.nan

    # A sharp corner has no arc to measure, but a parabola fitted to two straight walls still
    # comes back with a finite, innocent-looking radius - and that number would go straight into
    # Bartz. So fit the wedge r = r_t + s*|dx| as well and hand back nan unless the parabola is
    # the better story, which it only is if the contour really was drawn with a throat arc.
    s = float(np.sum(np.abs(dx) * dr) / np.sum(dx**2))
    res_arc = np.sum((dr - a * dx**2) ** 2)
    res_wedge = np.sum((dr - s * np.abs(dx)) ** 2)
    return 1.0 / (2.0 * a) if res_arc < res_wedge else np.nan


def _frustums(x, r):
    """Per-segment volume and lateral surface area, exact for a polyline of revolution."""
    r1, r2 = r[:-1], r[1:]
    dx = np.diff(x)
    volume = np.pi / 3.0 * (r1**2 + r1 * r2 + r2**2) * dx
    slant = np.hypot(dx, np.diff(r))
    area = np.pi * (r1 + r2) * slant
    return volume, area


def parse_contour(path, n_stations=None, scale=1.0):
    """Every key geometry of the engine, from the contour CSV. See the module docstring."""
    x, r = read_contour(path, scale)
    i_t = int(np.argmin(r))

    if n_stations is not None:
        x, r = resample(x, r, n_stations, anchor=x[i_t])
        i_t = int(np.argmin(r))

    A = np.pi * r**2
    r_t, A_t = r[i_t], A[i_t]

    # the station arrays the flow and channel codes consume
    dr_dx = np.gradient(r, x)
    branch = np.where(np.arange(len(x)) <= i_t, "sub", "sup")

    # chamber barrel: the constant-radius run feeding the converging section
    b0, b1 = _barrel(x, r, i_t)
    has_barrel = b1 > b0
    r_c = float(np.mean(r[b0 : b1 + 1])) if has_barrel else float(r[0])
    A_c = np.pi * r_c**2

    # volumes and wetted areas, split at the throat vertex
    vol, surf = _frustums(x, r)
    V_chamber, V_nozzle = vol[:i_t].sum(), vol[i_t:].sum()
    S_chamber, S_nozzle = surf[:i_t].sum(), surf[i_t:].sum()

    # half angles off the segments themselves rather than the smoothed gradient, so the corner
    # at the end of the barrel stays a corner
    seg_slope = np.diff(r) / np.diff(x)
    conv, div = seg_slope[:i_t], seg_slope[i_t:]
    theta_conv = np.degrees(np.arctan(-conv.min())) if conv.size else np.nan
    theta_div_max = np.degrees(np.arctan(div.max())) if div.size else np.nan
    theta_exit = np.degrees(np.arctan(div[-1])) if div.size else np.nan

    L_conical_15 = (r[-1] - r_t) / np.tan(np.radians(15.0))

    return {
        # per-station
        "x": x, "r": r, "A": A, "D": 2.0 * r,
        "dr_dx": dr_dx, "perimeter": 2.0 * np.pi * r,
        "area_ratio": A / A_t, "branch": branch,
        # throat
        "i_t": i_t, "x_t": float(x[i_t]), "r_t": float(r_t),
        "D_t": float(2.0 * r_t), "A_t": float(A_t),
        # chamber
        "has_barrel": bool(has_barrel), "r_c": r_c, "D_c": 2.0 * r_c, "A_c": float(A_c),
        # exit
        "r_e": float(r[-1]), "D_e": float(2.0 * r[-1]), "A_e": float(A[-1]),
        # ratios
        "contraction_ratio": float(A_c / A_t),
        "expansion_ratio": float(A[-1] / A_t),
        # lengths
        "L_total": float(x[-1] - x[0]),
        "L_barrel": float(x[b1] - x[b0]) if has_barrel else 0.0,
        "L_converging": float(x[i_t] - x[b1]) if has_barrel else float(x[i_t] - x[0]),
        "L_chamber": float(x[i_t] - x[0]),
        "L_diverging": float(x[-1] - x[i_t]),
        "L_star": float(V_chamber / A_t),
        # volumes and wetted areas
        "V_chamber": float(V_chamber), "V_nozzle": float(V_nozzle),
        "V_total": float(V_chamber + V_nozzle),
        "S_chamber": float(S_chamber), "S_nozzle": float(S_nozzle),
        "S_total": float(S_chamber + S_nozzle),
        # angles and curvature
        "theta_conv": float(theta_conv),
        "theta_div_max": float(theta_div_max),
        "theta_exit": float(theta_exit),
        "Rc_throat_up": _throat_curvature(x, r, i_t, "up"),
        "Rc_throat_down": _throat_curvature(x, r, i_t, "down"),
        "percent_bell": float(100.0 * (x[-1] - x[i_t]) / L_conical_15),
    }


def summarize(geo, path=""):
    """Print the geometry in mm, cm^2 and cm^3 - the units a contour this size gets discussed in."""
    mm, cm2, cm3 = 1e3, 1e4, 1e6

    print("=" * 66)
    print(f"ENGINE GEOMETRY   {os.path.basename(path)}")
    print("=" * 66)
    print(f"  stations {len(geo['x']):>4}   x {geo['x'][0] * mm:.2f} .. {geo['x'][-1] * mm:.2f} mm")

    print("\n--- STATIONS ---")
    for label, a_key, r_key in (("chamber", "A_c", "r_c"), ("throat", "A_t", "r_t"),
                                ("exit", "A_e", "r_e")):
        print(f"  {label:<9} r {geo[r_key] * mm:8.3f} mm   D {2 * geo[r_key] * mm:8.3f} mm   "
              f"A {geo[a_key] * cm2:8.3f} cm^2")
    if not geo["has_barrel"]:
        print("  ! no constant-radius barrel found - chamber values are the first station")

    print("\n--- RATIOS ---")
    print(f"  contraction  A_c/A_t   {geo['contraction_ratio']:8.3f}")
    print(f"  expansion    A_e/A_t   {geo['expansion_ratio']:8.3f}")
    print(f"  L*                     {geo['L_star'] * mm:8.1f} mm")
    print(f"  percent bell           {geo['percent_bell']:8.1f} %")

    print("\n--- LENGTHS [mm] ---")
    for label, key in (("barrel", "L_barrel"), ("converging", "L_converging"),
                       ("chamber (to throat)", "L_chamber"), ("diverging", "L_diverging"),
                       ("total", "L_total")):
        print(f"  {label:<22} {geo[key] * mm:8.2f}")

    print("\n--- ANGLES [deg] / THROAT CURVATURE [mm] ---")
    print(f"  convergence half angle {geo['theta_conv']:8.2f}")
    print(f"  divergence, max        {geo['theta_div_max']:8.2f}")
    print(f"  divergence, at exit    {geo['theta_exit']:8.2f}")
    print(f"  throat R, upstream     {geo['Rc_throat_up'] * mm:8.3f}")
    print(f"  throat R, downstream   {geo['Rc_throat_down'] * mm:8.3f}")

    print("\n--- VOLUME [cm^3] / WETTED AREA [cm^2] ---")
    print(f"  chamber   V {geo['V_chamber'] * cm3:9.2f}   S {geo['S_chamber'] * cm2:9.2f}")
    print(f"  nozzle    V {geo['V_nozzle'] * cm3:9.2f}   S {geo['S_nozzle'] * cm2:9.2f}")
    print(f"  total     V {geo['V_total'] * cm3:9.2f}   S {geo['S_total'] * cm2:9.2f}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_contour.csv")
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")

    geo = parse_contour(path)
    summarize(geo, path)
    return geo


if __name__ == "__main__":
    main()
