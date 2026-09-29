###Running Script - 5kN ablative engine
import matplotlib.pyplot as plt
from Enginecontourfunc import *

##Useful english to SI conversions
psi_to_pa = 6894.76
in_to_m = 0.0254
units = "mm" #for dxf only

##Other Inputs
tol_pct = 10E-7 #tol
g0 = 9.8 #m/s

##Engine Requirements (nominal value, +- tolerance)
thrust_req = 5000 #N
thrust_tol = 500 #N
burn_time = 10 #s
burn_time_tol = 1 #s
m_dot_tol = 0.2 #kg/s
ofratio_tol = 0.13
chamber_pressure_tol = 15*psi_to_pa #Pa
ipa_tank_MEOP = 514*psi_to_pa #Pa
lox_tank_MEOP = 514*psi_to_pa #Pa
ipa_inj_inlet_pressure = 400*psi_to_pa #Pa
lox_inj_inlet_pressure = 400*psi_to_pa #Pa
inj_inlet_pressure_tol = 20*psi_to_pa #Pa
vehicle_TW = 1.3 #thrust to weight of the vehicle
max_engine_mass = 20 #kg, engine + injector - not checked by this script

##Sizing Params
m_dot = 2 #kg/s
ofratio = 1.3
eta_c_star = 0.90 #C* efficiency used for throat sizing
exit_pressure = 101325 #Pa = 1atm
ambient_pressure = 101325 #Pa = 1atm
chamber_pressure = 300*psi_to_pa #Pa

#calced CEA params
CEA = CEA_numbers(chamber_pressure,ofratio,exit_pressure,m_dot)
[eps,chamber_temp,cp,gamma,gamma2,gamma3,ISPU,ISP,exit_mach,Thrust,c_star,MW_chamber] = CEA
c_star_adjusted = eta_c_star * c_star
L_star = 1.2 #m

##Design Inputs
theta_n_deg = 23 #deg from rao tables
theta_e_deg = 13 #from Rao tables
contraction_ratio = 9 #free design choice, how stubby you want this bitch
conv_half_angle_deg = 45 #deg, wall angle between the start of thraot and entry arc
length_pct = 0.8 #current agreed upon values
cone_half_angle_deg = 15 #deg
Ru_over_Rt = 1.5 #how gently the converign section blends to throat
Rd_over_Rt = 0.382 #curvature post throat, cant realy change this cause Rao said so
stations_bell = 100 #fidelty of bell
station_throat_up = 100 #upward section of throat before proper bell
stations_throat_down = 100 #fidelity downstream of throat

##Calc Contour
area_throat, Rt = throat_goat(m_dot, c_star_adjusted, chamber_pressure)

x,y,info = full_engine_contour(Rt, eps, theta_n_deg,theta_e_deg, contraction_ratio, L_star, conv_half_angle_deg, length_pct, cone_half_angle_deg, Ru_over_Rt, Rd_over_Rt, stations_bell, station_throat_up, stations_throat_down)

# Performance: retain ideal CEA vacuum Isp; efficiency adjusts geometry only.
# The ambient pressure correction assumes attached nozzle flow.
area_exit = eps * area_throat
vacuum_isp = ISP
vacuum_thrust = vacuum_isp * m_dot * g0
sea_level_thrust = vacuum_thrust - area_exit * ambient_pressure
sea_level_isp = sea_level_thrust / (m_dot * g0)
print(f"Adjusted C*: {c_star_adjusted:.2f} m/s (efficiency = {eta_c_star:.3f})")
print(f"CEA ideal vacuum thrust: {vacuum_thrust:.2f} N")
print(f"CEA ideal vacuum Isp: {vacuum_isp:.2f} s")
print(f"Sea-level thrust: {sea_level_thrust:.2f} N")
print(f"Sea-level Isp: {sea_level_isp:.2f} s")

# Delivered estimate: C* efficiency applied to the vacuum thrust (F = Cf*Pc*At with At sized on eta*C*)
delivered_sea_level_thrust = eta_c_star * vacuum_thrust - area_exit * ambient_pressure
print(f"Delivered sea-level thrust (C* eff. applied): {delivered_sea_level_thrust:.2f} N")

##Derived Sizing
m_dot_fuel = m_dot / (1 + ofratio) #kg/s IPA
m_dot_ox = m_dot - m_dot_fuel #kg/s LOX
m_fuel = m_dot_fuel * burn_time #kg, usable IPA
m_ox = m_dot_ox * burn_time #kg, usable LOX
ipa_inj_stiffness = (ipa_inj_inlet_pressure - chamber_pressure) / chamber_pressure
lox_inj_stiffness = (lox_inj_inlet_pressure - chamber_pressure) / chamber_pressure
ipa_feed_dp = ipa_tank_MEOP - ipa_inj_inlet_pressure #Pa, max allowable tank to injector loss
lox_feed_dp = lox_tank_MEOP - lox_inj_inlet_pressure #Pa
max_vehicle_mass = thrust_req / (vehicle_TW * g0) #kg, liftoff mass at nominal thrust

print("\n--- DERIVED SIZING ---")
print(f"IPA mdot: {m_dot_fuel:.3f} kg/s   LOX mdot: {m_dot_ox:.3f} kg/s")
print(f"Usable IPA: {m_fuel:.2f} kg   Usable LOX: {m_ox:.2f} kg   ({burn_time} s burn)")
print(f"Injector stiffness: IPA {ipa_inj_stiffness*100:.1f} %   LOX {lox_inj_stiffness*100:.1f} %")
print(f"Max feed loss (MEOP - inj inlet): IPA {ipa_feed_dp/psi_to_pa:.1f} psi   LOX {lox_feed_dp/psi_to_pa:.1f} psi")
print(f"Max vehicle liftoff mass for T/W = {vehicle_TW}: {max_vehicle_mass:.1f} kg")
print(f"Throat dia: {2*Rt*1000:.2f} mm   Exit dia: {2*info['Re']*1000:.2f} mm   Chamber dia: {2*info['Rc']*1000:.2f} mm")
print(f"Total engine length: {info['total_length']*1000:.1f} mm")

##Requirement Checks
def check(name, value, nominal, tol, unit=""):
    ok = abs(value - nominal) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.2f} {unit} (req {nominal:.2f} +- {tol:.2f} {unit})")
    return ok

print("\n--- REQUIREMENT CHECKS ---")
check("Sea-level thrust (ideal)", sea_level_thrust, thrust_req, thrust_tol, "N")
check("Sea-level thrust (delivered)", delivered_sea_level_thrust, thrust_req, thrust_tol, "N")
print(f"  [NOTE] Engine + injector mass <= {max_engine_mass} kg is not evaluated by this script")

#Plot Contour
fig = plot_full_contour(x, y, info)
plt.show()

#csv
export_csv(x, y, filepath="engine_contour.csv")

#DXF
export_dxf_analytic(info, info["x_chamber_start"], conv_half_angle_deg, theta_n_deg, mirror=False, units="mm")
