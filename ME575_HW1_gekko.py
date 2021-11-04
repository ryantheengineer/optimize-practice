from gekko import GEKKO
import numpy as np

m = GEKKO(remote=True)

# Design variables
d,D,n,hf = [m.Var() for i in range(4)]

# Initial values
d.value = 0.20
D.value = 0.210
n.value = 90.0
hf.value = 0.020

# Upper bounds
d.upper = 0.2
D.upper = 1.0
n.upper = 100.0
hf.upper = 20.0

# Lower bounds
d.lower = 0.01
D.lower = 0.2
n.lower = 1.0
hf.lower = 0.01

# Other analysis variables
h0 = m.Const(value=1.0)                 # preload height, in
delta0 = m.Const(value=0.4)             # deflection, in
hdef = m.Const(value=(h0 - delta0))     # deflection height, in
G = m.Const(value=12*10**6)             # psi
Se = m.Const(value=45000)               # psi
w = m.Const(value=0.18)
Sf = m.Const(value=1.5)
Q = m.Const(value=150000)               # psi
delta_f = m.Const(value=0)              # in

# Analysis functions
k = (G*d**4)/(8*(D**3)*n)            # spring stiffness
delta_p = hf-h0                    # deflection at preload
delta_def = delta0 + (hf-h0)       # greatest working deflection
hs = n*d                           # solid height
delta_s = hf - hs                  # deflection at solid height
F_f = k*delta_f                    # full height force (zero)
F_p = k*delta_p                    # preload force
F_def = k*delta_def                # force at full deflection
F_s = k*delta_s                    # force at solid height
K = ((4*D)-d)/(4*(D-d))+0.62*(d/D) # Wahl factor
tau_f = ((8*F_f*D)/(np.pi*d**3))*K     # stress at full height (zero)
tau_p = ((8*F_p*D)/(np.pi*d**3))*K     # stress at preload height
tau_def = ((8*F_def*D)/(np.pi*d**3))*K # stress at full deflection
tau_s = ((8*F_s*D)/(np.pi*d**3))*K     # stress at solid height
tau_max = tau_def                  # max stress (assumed at max deflection)
tau_min = tau_p                    # min stress (assumed at preload deflection)
tau_m = (tau_max + tau_min)/2      # mean stress
tau_a = (tau_max - tau_min)/2      # alternating stress
Sy = 0.44*(Q/(d**w))                # yield stress
dratio = D/d                       # ratio of diameters
dsum = D + d                       # sum of diameters
clash = hdef - hs                  # clash allowance
Sefratio = Se/Sf                   # endurance limit to safety factor
Syfratio = Sy/Sf                   # yield strength to safety factor
tau_amsum = tau_a + tau_m          # sum of alternating and mean stresses

# Constraints
m.Equation(tau_a<=Sefratio)
m.Equation((tau_amsum<=Syfratio))
m.Equation(dratio<=16.0)
m.Equation(dratio>=4.0)
m.Equation(dsum<=0.75)
m.Equation(clash>=0.05)
m.Equation(tau_s<=Sy)

# Objective function
m.Maximize(F_p)

# Set global options
m.options.IMODE = 3 #steady state optimization

# Solve simulation
m.solve()

# Results
print('')
print('d: ' + str(d.value))
print('D: ' + str(D.value))
print('n: ' + str(n.value))
print('hf: ' + str(hf.value))
