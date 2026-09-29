"""Edit SIZING_INPUTS and INPUTS below, then run: python run_contour.py"""

import argparse
from pathlib import Path


# LOX / IPA70 sizing inputs from the supplied running script.
# IPA70 is the existing 70% isopropanol / 30% water fuel card (by mass).
PSI_TO_PA = 6894.76
SIZING_INPUTS = {
    "m_dot": 2.0,                          # Total propellant mass flow [kg/s]
    "ofratio": 1.0,                        # Oxidizer / fuel mass ratio
    "eta_c_star": 0.90,                    # C* efficiency for throat sizing
    "exit_pressure": 101325.0,             # Target nozzle exit pressure [Pa]
    "chamber_pressure": 464 * PSI_TO_PA,    # Chamber pressure [Pa] (464 psi)
}

# Geometry inputs from the supplied running script.
# All lengths are in meters and all angles are in degrees.
# Rt and eps are calculated using RocketCEA rather than entered here.
INPUTS = {
    "theta_n_deg": 23.0,          # Bell wall angle at its start [deg]
    "theta_e_deg": 13.0,          # Bell wall angle at the exit [deg]
    "contraction_ratio": 9.0,     # Chamber area / throat area
    "L_star": 1.2,                # Chamber volume / throat area [m]
    "conv_half_angle_deg": 45.0,  # Converging section half-angle [deg]
    "length_pct": 0.8,            # Fraction of reference cone length (0.8 = 80%)
    "cone_half_angle_deg": 15.0,  # Reference cone half-angle [deg]
    "Ru_over_Rt": 1.5,            # Upstream throat fillet radius / throat radius
    "Rd_over_Rt": 0.382,          # Downstream throat fillet radius / throat radius
    "stations_bell": 100,         # Number of sampled points along the bell
    "station_throat_up": 100,     # Number of upstream throat arc points
    "stations_throat_down": 100,  # Downstream arc and chamber sampling count
}

# Files are saved beside this script, inside the output folder.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CSV_POINTS = 500                  # Approximate count; export includes x=0
DXF_UNITS = "mm"                 # "mm", "m", or "in" (CSV always uses meters)
DXF_MIRROR = False               # Upper wall only, matching the supplied script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save output files without opening the plot window.",
    )
    args = parser.parse_args()

    # Select a noninteractive backend before importing the plotting functions.
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from Enginecontourfunc import (
        CEA_contour_numbers, throat_goat,
        export_csv, export_dxf_analytic, full_engine_contour, plot_full_contour,
    )

    eps, c_star = CEA_contour_numbers(
        p_c=SIZING_INPUTS["chamber_pressure"],
        r_of=SIZING_INPUTS["ofratio"],
        p_e=SIZING_INPUTS["exit_pressure"],
    )
    c_star_adjusted = SIZING_INPUTS["eta_c_star"] * c_star
    area_throat, Rt = throat_goat(
        SIZING_INPUTS["m_dot"], c_star_adjusted, SIZING_INPUTS["chamber_pressure"],
    )
    print(f"CEA expansion ratio (Ae/At): {eps:.6f}")
    print(f"CEA ideal C*: {c_star:.2f} m/s")
    print(f"Adjusted C*: {c_star_adjusted:.2f} m/s "
          f"(efficiency = {SIZING_INPUTS['eta_c_star']:.3f})")
    print(f"Throat area: {area_throat:.8f} m^2")

    x, radius, info = full_engine_contour(Rt=Rt, eps=eps, **INPUTS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plot_full_contour(x, radius, info, OUTPUT_DIR / "engine_contour.png")
    export_csv(x, radius, OUTPUT_DIR / "engine_contour.csv", n_points=CSV_POINTS)
    export_dxf_analytic(
        info,
        x_chamber_start=info["x_chamber_start"],
        conv_half_angle_deg=INPUTS["conv_half_angle_deg"],
        theta_n_deg=INPUTS["theta_n_deg"],
        filepath=OUTPUT_DIR / "engine_contour.dxf",
        units=DXF_UNITS,
        mirror=DXF_MIRROR,
    )

    print(f"\nThroat diameter: {2 * info['Rt'] * 1000:.2f} mm")
    print(f"Chamber diameter: {2 * info['Rc'] * 1000:.2f} mm")
    print(f"Exit diameter: {2 * info['Re'] * 1000:.2f} mm")
    print(f"Straight chamber length: {info['L_c'] * 1000:.2f} mm")
    print(f"Total contour length: {info['total_length'] * 1000:.2f} mm")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
