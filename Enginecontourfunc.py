##Libraries
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
import csv
from rocketcea.cea_obj import add_new_fuel
from rocketcea.cea_obj_w_units import CEA_Obj
 
###Functions
 
##C_star
def c_star_func(gamma, R_spec, Temp_chamber):
    c_star = np.sqrt(gamma*R_spec*Temp_chamber) / (gamma * np.sqrt((2 / (gamma + 1))**((gamma + 1) / (gamma - 1))))
    return c_star
 
##Throat Area & Radius
def throat_goat(m_dot, c_star, chamber_pressure):
    area_throat = m_dot * c_star / chamber_pressure
    Rt = np.sqrt(area_throat/np.pi)
    return area_throat, Rt
 
##Converging Section Arc
def conv_arc(Rc, Tx, Ty, conv_half_angle_deg, stations):
    conv_half_angle_rad = np.radians(conv_half_angle_deg)
    m2 = -np.tan(conv_half_angle_rad)
    R = (Rc - Ty)/(1 - np.cos(conv_half_angle_rad))
    k = Ty - Rc + R
    x1 = Tx +m2 * k
    ox, oy = x1, Rc - R
    ang1 = np.radians(90)
    ang2 = np.radians(90 - conv_half_angle_deg)
    ang= np.linspace(ang1, ang2, stations)
    x = ox + R * np.cos(ang)
    y = oy + R * np.sin(ang)
    return x,y,dict(R=R,x1=x1)
 
##Chmaber plus arc geometry
def chamber_geometry(Rt, Ru, contraction_ratio, L_star, Tx, Ty, conv_half_angle_deg, stations):
    Rc = Rt * np.sqrt(contraction_ratio)
    x_arc, y_arc, arc_info = conv_arc(Rc, Tx, Ty, conv_half_angle_deg, stations)
    x1 = arc_info['x1']
    trapz = getattr(np, "trapezoid", None) or np.trapz

    # Converging fillet volume (cylinder -> Tx,Ty)
    Vol_conv = abs(trapz(np.pi * y_arc**2, x_arc))

    # Throat-entrance fillet volume (Tx,Ty -> throat at x=0, Rt) -- previously missing
    conv_half_angle_rad = np.radians(conv_half_angle_deg)
    ang_start = -(np.pi/2 + conv_half_angle_rad)
    ang_end = -np.pi/2
    ang_up = np.linspace(ang_start, ang_end, stations)
    x_up = Ru * np.cos(ang_up)
    y_up = Ru * np.sin(ang_up) + Ru + Rt
    Vol_throat_up = abs(trapz(np.pi * y_up**2, x_up))

    Vol_chamber = Vol_conv + Vol_throat_up

    At = np.pi * Rt**2
    V_total = L_star * At
    Ac = np.pi * Rc**2
    L_c = (V_total - Vol_chamber) / Ac
    if L_c <= 0:
        raise ValueError("L_star is too small for the given geometry. Increase L_star or adjust other parameters.")

    return dict(Rc=Rc, L_c=L_c, x_arc=x_arc, y_arc=y_arc, x1=x1,
                R_conv=arc_info['R'], Vol_chamber=Vol_chamber,
                Vol_conv=Vol_conv, Vol_throat_up=Vol_throat_up, V_total=V_total)

##Bell
def rao_bell(Rt, eps, theta_n_deg, theta_e_deg, length_pct, Ru_over_Rt, Rd_over_Rt, conv_half_angle_deg, cone_half_angle_deg, station_throat_up, stations_throat_down, stations_bell):
    theta_n = np.radians(theta_n_deg)
    theta_e = np.radians(theta_e_deg)
    alpha = np.radians(cone_half_angle_deg)
    conv_angle = np.radians(conv_half_angle_deg)
    Re = Rt * np.sqrt(eps)
    Ru = Ru_over_Rt * Rt
    Rd = Rd_over_Rt * Rt
    ang_up_start = -(np.pi / 2 + conv_angle)
    ang_up_end = -np.pi / 2
    ang_up = np.linspace(ang_up_start, ang_up_end, station_throat_up)
    x_up = Ru * np.cos(ang_up)
    y_up = Ru * np.sin(ang_up) +Ru +Rt
    Tx, Ty = x_up[0], y_up[0]
    ang_down_start = -np.pi/2
    ang_down_end = theta_n - np.pi/2
    ang_down = np.linspace(ang_down_start, ang_down_end, stations_throat_down)
    x_down = Rd * np.cos(ang_down)
    y_down = Rd * np.sin(ang_down) +Rd +Rt
    Nx, Ny = x_down[-1], y_down[-1]
    x_throat = np.concatenate([x_up, x_down[1:]])
    y_throat = np.concatenate([y_up, y_down[1:]])
    L = length_pct * (Rt * (np.sqrt(eps) -1))/np.tan(alpha)
    Ex, Ey = L, Re
    m1, m2 = np.tan(theta_n), np.tan(theta_e)
    C1 = Ny-m1*Nx
    C2 = Ey - m2*Ex
    Qx = (C2 - C1)/(m1 - m2)
    Qy = m1 * Qx +C1
    t= np.linspace(0,1, stations_bell)
    x_bell = (1-t)**2 *Nx+2*t*(1-t) *Qx+t**2*Ex
    y_bell = (1-t)**2 *Ny+2*t*(1-t) *Qy+t**2*Ey
    x = np.concatenate([x_throat, x_bell[1:]])
    y = np.concatenate([y_throat, y_bell[1:]])
    info = dict(Rt=Rt, Re=Re, Ru=Ru, Rd=Rd, L=L,T=(Tx, Ty), N=(Nx, Ny), Q=(Qx, Qy), E=(Ex, Ey))
    return x, y, info
 
##Engine Contour
def full_engine_contour(Rt, eps, theta_n_deg, theta_e_deg, contraction_ratio, L_star, conv_half_angle_deg, length_pct, cone_half_angle_deg, Ru_over_Rt, Rd_over_Rt, stations_bell, station_throat_up, stations_throat_down):
    x_bell_part, y_bell_part, bell_info = rao_bell(Rt, eps, theta_n_deg, theta_e_deg, length_pct, Ru_over_Rt, Rd_over_Rt, conv_half_angle_deg, cone_half_angle_deg, station_throat_up, stations_throat_down, stations_bell)
    Tx, Ty = bell_info["T"]
    Ru = bell_info["Ru"]
    chamber = chamber_geometry(Rt, Ru, contraction_ratio, L_star, Tx, Ty, conv_half_angle_deg, stations_throat_down)
    Rc, L_c = chamber["Rc"], chamber["L_c"]
    x_arc, y_arc, x1 = chamber["x_arc"], chamber["y_arc"], chamber["x1"]
    x_chamber_start = x1 - L_c
    x_chamber = np.linspace(x_chamber_start, x1, stations_throat_down)
    y_chamber = np.full_like(x_chamber, Rc)
    x = np.concatenate([x_chamber, x_arc[1:], x_bell_part])
    y = np.concatenate([y_chamber, y_arc[1:], y_bell_part])
    info = dict(chamber, **bell_info, x_chamber_start=x_chamber_start, total_length=bell_info["E"][0] - x_chamber_start)
    return x, y, info
 
#plot
def plot_full_contour(x, y, info, savepath="engine_contour.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y, color="tab:blue", lw=2)
    ax.plot(x, -y, color="tab:blue", lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(0, info["Rt"], "ko", ms=4)
    ax.plot(*info["E"], "ko", ms=4)
    ax.plot(info["x_chamber_start"], info["Rc"], "ko", ms=4)
    ax.set_xlabel("Axial distance [m]")
    ax.set_ylabel("Radius [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"Saved plot to {savepath}")
    return fig
 
#CSV export
def export_csv(x, y, filepath="engine_contour.csv", dx=None, n_points=500):
    
    order = np.argsort(x, kind="stable")
    x_sorted, y_sorted = x[order], y[order]
    x_min, x_max = x_sorted[0], x_sorted[-1]
 
    if dx is None:
        dx = (x_max - x_min) / (n_points - 1)
 
    n_neg = int(np.ceil(-x_min / dx))   # steps needed from 0 down to x_min
    n_pos = int(np.ceil(x_max / dx))    # steps needed from 0 up to x_max
 
    x_uniform = np.arange(-n_neg, n_pos + 1) * dx   # x=0 guaranteed at index n_neg
    y_uniform = np.interp(x_uniform, x_sorted, y_sorted)
 
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "r"])
        for xi, yi in zip(x_uniform, y_uniform):
            writer.writerow([f"{xi:.8f}", f"{yi:.8f}"])
 
    actual_dx = x_uniform[1] - x_uniform[0]
    print(f"Saved CSV to {filepath}  ({len(x_uniform)} points, constant dx={actual_dx:.8f} m, "
          f"origin at throat [index {n_neg}], x range [{x_uniform.min():.5f}, {x_uniform.max():.5f}] m)")
 
 
#DXF export
def export_dxf_analytic(info, x_chamber_start, conv_half_angle_deg, theta_n_deg, filepath="engine_contour.dxf", mirror=True, units="mm"):
    # `scale` converts every coordinate written into an entity so the header
    # flag ($INSUNITS) and the actual numbers agree. All upstream geometry
    # (info dict, x_chamber_start) is computed in meters, so scale=1000.0
    # for units="mm" turns 0.1995 m into 199.5 (mm) in the file.
    scale = {"mm": 1000.0, "in": 1000.0 / 25.4, "m": 1.0}.get(units, 1.0)

    Rc, x1, R_conv = info["Rc"] * scale, info["x1"] * scale, info["R_conv"] * scale
    Rt, Ru, Rd = info["Rt"] * scale, info["Ru"] * scale, info["Rd"] * scale
    N = (info["N"][0] * scale, info["N"][1] * scale)
    Q = (info["Q"][0] * scale, info["Q"][1] * scale)
    E = (info["E"][0] * scale, info["E"][1] * scale)
    x_chamber_start = x_chamber_start * scale

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = {"mm": 4, "in": 1, "m": 6}.get(units, 0)
    msp = doc.modelspace()
    doc.layers.add(name="WALL_UPPER", color=5)
    doc.layers.add(name="CENTERLINE", color=1)

    def add_wall_geometry(layer, sign):
        msp.add_line((x_chamber_start, sign * Rc), (x1, sign * Rc),
                     dxfattribs={"layer": layer})

        ang_lo, ang_hi = 90.0 - conv_half_angle_deg, 90.0
        if sign > 0:
            msp.add_arc(center=(x1, Rc - R_conv), radius=R_conv,
                        start_angle=ang_lo, end_angle=ang_hi,
                        dxfattribs={"layer": layer})
        else:
            msp.add_arc(center=(x1, -(Rc - R_conv)), radius=R_conv,
                        start_angle=-ang_hi, end_angle=-ang_lo,
                        dxfattribs={"layer": layer})

        ang_lo, ang_hi = -(90.0 + conv_half_angle_deg), -90.0
        if sign > 0:
            msp.add_arc(center=(0, Ru + Rt), radius=Ru,
                        start_angle=ang_lo, end_angle=ang_hi,
                        dxfattribs={"layer": layer})
        else:
            msp.add_arc(center=(0, -(Ru + Rt)), radius=Ru,
                        start_angle=-ang_hi, end_angle=-ang_lo,
                        dxfattribs={"layer": layer})

        ang_lo, ang_hi = -90.0, theta_n_deg - 90.0
        if sign > 0:
            msp.add_arc(center=(0, Rd + Rt), radius=Rd,
                        start_angle=ang_lo, end_angle=ang_hi,
                        dxfattribs={"layer": layer})
        else:
            msp.add_arc(center=(0, -(Rd + Rt)), radius=Rd,
                        start_angle=-ang_hi, end_angle=-ang_lo,
                        dxfattribs={"layer": layer})

        sp = msp.add_spline(dxfattribs={"layer": layer})
        sp.control_points = [(N[0], sign * N[1], 0),
                              (Q[0], sign * Q[1], 0),
                              (E[0], sign * E[1], 0)]
        sp.dxf.degree = 2
        sp.knots = [0, 0, 0, 1, 1, 1]

    add_wall_geometry("WALL_UPPER", +1)

    if mirror:
        doc.layers.add(name="WALL_LOWER", color=5)
        add_wall_geometry("WALL_LOWER", -1)

    msp.add_line((x_chamber_start, 0), (E[0], 0), dxfattribs={"layer": "CENTERLINE"})

    doc.saveas(filepath)
    print(f"Saved DXF to {filepath}  (units label: {units}, $INSUNITS={doc.header['$INSUNITS']}, scale={scale})")
    print("Entities: 1 LINE (chamber) + 3 ARC (converging, throat-up, throat-down) "
          "+ 1 SPLINE (bell) per wall side -- no sampled polyline.")
card = """
    fuel C3H8O-2propanol C 3 H 8 O 1    wt%=70.0
    h,cal=-65133.     t(k)=298.15   rho=0.786
    fuel water H 2.0 O 1.0  wt%=30.0
    h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
    """
add_new_fuel('IPA70', card)

def CEA_numbers(p_c,r_of,p_e,mdot):
    rcea = CEA_Obj(oxName = 'LOX',fuelName = 'IPA70',cstar_units='m/s',pressure_units='Pa',temperature_units='K',sonic_velocity_units='m/s',specific_heat_units='kJ/kg-K')
    eps = rcea.get_eps_at_PcOvPe(Pc=p_c,MR=r_of,PcOvPe=p_c/p_e, frozen=1, frozenAtThroat=1)
    print(f'Expansion Ratio {eps}')
    t_c = rcea.get_Temperatures(Pc=p_c, MR=r_of,eps=eps, frozen=1, frozenAtThroat=1)[0]
    print(f'Chamber Temp {t_c} K')
    Cp = rcea.get_Chamber_Cp(Pc=p_c, MR=r_of,eps=eps)
    print(f'Specific Heat Chamber {Cp}')
    gamma = rcea.get_Chamber_MolWt_gamma(Pc=p_c, MR=r_of,eps=eps)[1]
    gamma2 = rcea.get_Throat_MolWt_gamma(Pc=p_c, MR=r_of,eps=eps, frozen=1)[1]
    MW, gamma3 = rcea.get_exit_MolWt_gamma(Pc=p_c, MR=r_of,eps=eps, frozen=1, frozenAtThroat=1)
    print(f"gamma throat {gamma2}")
    ISP = rcea.get_Isp(Pc=p_c, MR=r_of,eps=eps, frozen=1, frozenAtThroat=1)
    print(f'ISP normalized {ISP}')
    ISPU = ISP*9.81
    print(f'ISP unnormalized {ISPU}')
    M_e = rcea.get_MachNumber(Pc=p_c,MR=r_of,eps=eps, frozen=1, frozenAtThroat=1)
    print(f'Exit Mach {M_e}')
    Thrust = ISPU*mdot
    print(f'Thrust {Thrust} N')
    cstar = rcea.get_Cstar(p_c, r_of)
    print(f'Cstar {cstar}')
    MW_chamber = rcea.get_Chamber_MolWt_gamma(Pc=p_c, MR=r_of, eps=eps)[0]
    print(f'MW Chamber {MW_chamber} lbm/lbmole')

    return([eps,t_c,Cp,gamma,gamma2,gamma3,ISPU,ISP,M_e,Thrust,cstar,MW_chamber])
