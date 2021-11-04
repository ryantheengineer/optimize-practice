from gekko import GEKKO
import numpy as np
from scipy.interpolate import interp1d

m = GEKKO(remote=True)

##### LOCAL FUNCTIONS #####
def NPV(horsepower,i,n):
    # Calculates the total cost of operatoins using net present value methods,
    # with interest calculated once per year (n = 7).

    # Capital cost of grinder and pump
    capital_cost = 300*horsepower + 200*horsepower

    power_cost = horsepower*8*300*0.07  # power cost, $/year

    totalcost = 0
    for t in range(0,n):
        if t == 0:
            Rt = capital_cost + power_cost
        else:
            Rt = power_cost

        totalcost += Rt/((1+i)**t)

    return totalcost

def dragReynolds(CdRpsq_calculated):
    # Interpolates the empirical relationship between Cd and CdRp^2 using
    # provided data

    CdRpsq =
    Cd_vector =
    CdRpsq_calculated =

    Cd_interpolated = interp1d(CdRpsq, Cd_vector, CdRpsq_calculated)

    return Cd_interpolated



# Design variables
V,D,d = [m.Var() for i in range(3)]

# Initial values
V.value = 14.95
D.value = 0.4
d.value = 0.005

# Lower bounds
V.lower = 0.01
D.lower = 0.1
d.lower = 0.0005

# Upper bounds
V.upper = 30.0
D.upper = 0.5
d.upper = 0.007

# Other analysis variables
L = m.Const(value=(15*5280))        # length of pipeline, feet
W = m.Const(value=12.67)            # flowrate of limestone, lbm/s
a = m.Const(value=0.01)             # avg lump size of limestone before grinding, ft
g = m.Const(value=32.17)            # acceleration due to gravity, ft/s^2
rho_w = m.Const(value=62.4)         # density of water, lbm/ft^3
gamma = m.Const(value=168.5)        # limestone density
S = m.Const(value=(gamma/rho_w))     # limestone specific gravity
mu = m.Const(value=7.392*10**-4)          # viscosity of water lbm/(ft*s)
gc = m.Const(value=32.17)                 # conversion factor between lbf and lbm

# Analysis functions
C = 4*W/(np.pi*gamma*V*(D**2))
Area = (np.pi/4)*D**2
rho = rho_w + C*(gamma-rho_w)
Pg = (218*W*((1/(m.sqrt(d))) - (1/(m.sqrt(a)))))/550
CdRpsq_calculated = 4*g*rho_w*(d**3)*((gamma-rho_w)/(3*mu**2))
Cd = dragReynolds(CdRpsq_calculated)
Rw = (rho_w*V*D)/mu
fw = fw_function(Rw)
F = fw*((rho_w/rho) + 150*C*(rho_w/rho)*((g*D*(S-1))/((V**2)*m.sqrt(Cd)))**1.5)
delta_p = (F*rho*L*V**2)/(2*D*gc)
Qslurry = Area*V
Pf = (delta_p*Qslurry)/550     # pump power, hp
Vc = ((40*g*C*(S-1)*D)/m.sqrt(Cd))**0.5
horsepower = Pf + Pg
cost = NPV(horsepower,0.07,7) # develop cost function here

# Constraints
m.Equation(1.1*Vc<=V)
m.Equation(C<=0.4)

# Objective function
m.Minimize(cost)        # Minimize the total cost

# Set global options
m.options.IMODE = 3 # steady state optimization

# Solve simulation
m.solve() # solve on public server

# Results
print('')
print('V: ' + str(V.value))
print('D: ' + str(D.value))
print('d: ' + str(d.value))
