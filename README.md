# Mk2-Code

Generate an engine contour by editing the `SIZING_INPUTS` and `INPUTS`
dictionaries at the top of `run_contour.py`, then running it from your IDE or
terminal:

```powershell
python -m pip install -r requirements.txt
python run_contour.py
```

The defaults match the supplied running script: LOX/IPA70 (70% isopropanol and
30% water by mass), 2 kg/s total mass flow, O/F = 1, 464 psi chamber pressure,
101325 Pa target exit pressure, and 0.90 C* efficiency. RocketCEA calculates
`eps` (exit area / throat area) from the chamber-to-exit pressure ratio and
the ideal C*. Throat sizing uses `At = m_dot * eta_c_star * c_star / Pc` and
`Rt = sqrt(At / pi)`.

Geometry defaults include `L_star=1.2` m, contraction ratio 9, bell angles
23 and 13 degrees, and 100 samples per specified section. Lengths are in meters
and angles are in degrees. `contraction_ratio` is chamber area / throat area.
`L_star` is chamber volume divided by throat area, not the straight chamber
length. `length_pct=0.8` means 80% of the reference cone length.

Each run saves these files in `output/` beside the script, replacing the previous
outputs, and displays the contour plot:

- `engine_contour.png`: mirrored longitudinal contour.
- `engine_contour.csv`: axial position `x` and upper-wall radius `r`, in meters,
  with the throat at `x=0`. The existing exporter uses uniform spacing and may
  extend the end samples by less than one spacing to include the throat exactly.
- `engine_contour.dxf`: analytic CAD contour, in millimeters by default, with
  the upper wall only (`DXF_MIRROR=False`, matching the supplied script).

Use `python run_contour.py --no-show` to save files without a plot window.
The terminal also prints the diameters and lengths. If the function reports that
`L_star` is too small, increase it or adjust the geometry.

`Enginecontourfunc.py` contains the reusable geometry and export functions;
running that file directly executes its coolant-channel demo. The runner uses
`CEA_contour_numbers(...)` to request only expansion ratio and C* from RocketCEA.
It does not run heat transfer, coolant-channel sizing, or performance calculations.
Ambient pressure, coolant properties, wall thickness, fins, channel heights,
and injector stiffness are omitted because they do not enter this contour model.
