import numpy as np

"""
The purpose of this script is to get our geometric quantities associated with the regen channels as a function of x (as they change in size along the engine contour)

INPUTS
----------
A, x            : equal-length sequences - area [m^2] at each axial station [m]
n_channels      : the total number of channels (radially)
wall_th         : Thickness of the bottom wall (from hot gas to coolant) [m] - a scalar, or an
                  array with one value per station
fin_width       : the fin width - should be found from DFM for 3D printing [m] - a scalar, or an
                  array with one value per station

OUTPUTS
-------
As functions of x:
w_c     : channel width
P_fin   : fin perimeter for h_fin calculations
L_fin   : fin length
Ac_fin  : fin cross-sectional area

ASSUMPTIONS
-----------
Channels modeled as rectangular
n_channels remains constant over x
"""

def calculate_geometry(A, x, n_channels, wall_th, fin_width):
    dx = x[1] - x[0]
    A = np.asarray(A, float)
    r = np.sqrt(A/np.pi)
    wall_th = np.broadcast_to(np.asarray(wall_th, float), r.shape)
    fin_width = np.broadcast_to(np.asarray(fin_width, float), r.shape)
    outer_circumference = 2*np.pi*(r + wall_th)
    """
    Essentially, we have the "outer circumference" which is just the contour plus the local wall thickness, and that length is split between channels and fins. Whatever the fins (of local width fin_width) don't take up is shared equally by the channels, which gives the channel width as a function of x.
    """
    w_c = (outer_circumference - n_channels * fin_width) / n_channels

    if np.any(w_c <= 0):
        i = int(np.argmin(w_c))
        raise ValueError(f"{n_channels} fins of {fin_width[i]*1e3:.2f} mm don't fit around the circumference at "
                         f"x = {np.asarray(x)[i]*1e3:.1f} mm - reduce n_channels or fin_width, or raise the wall thickness there")
    P_fin = 2*dx + 2*fin_width
    Ac_fin = dx*fin_width
    
    return w_c, P_fin, Ac_fin

if __name__ == "__main__":
    # example: 5:1 contraction to a 4:1 expansion, hot gas
    x = [00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    A = [50e-4, 30e-4, 15e-4, 11e-4, 10e-4, 12e-4, 18e-4, 28e-4, 40e-4]
 
    n_channels = 20
    wall_th = 0.0025        # m
    fin_width = 0.0002  # m
 
    w_c, P_fin, Ac_fin = calculate_geometry(A, x, n_channels, wall_th, fin_width)