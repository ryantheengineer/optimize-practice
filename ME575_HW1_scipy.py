# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:42:01 2022

@author: Ryan.Larson
"""

# from gekko import GEKKO
import scipy.optimize as opt
import numpy as np

# m = GEKKO(remote=True)


Nfeval = 1

def objective(x):    
    d = x[0]
    D = x[1]
    n = x[2]
    hf = x[3]
    
    h0 = 1.0                 # preload height, in
    delta0 = 0.4             # deflection, in
    hdef = h0 - delta0     # deflection height, in
    G = 12*10**6             # psi
    Se = 45000               # psi
    w = 0.18
    Sf = 1.5
    Q = 150000               # psi
    delta_f = 0              # in
    
    k = (G*d**4)/(8*(D**3)*n)            # spring stiffness
    delta_p = hf-h0                    # deflection at preload
    F_p = k*delta_p                    # preload force
    
    f = -F_p    # Maximize F_p

    return f


def get_tau_a(x):
    d = x[0]
    D = x[1]
    n = x[2]
    hf = x[3]
    
    h0 = 1.0                 # preload height, in
    delta0 = 0.4             # deflection, in
    hdef = h0 - delta0     # deflection height, in
    G = 12*10**6             # psi
    Se = 45000               # psi
    w = 0.18
    Sf = 1.5
    Q = 150000               # psi
    delta_f = 0              # in
    
    k = (G*d**4)/(8*(D**3)*n)            # spring stiffness
    delta_p = hf-h0                    # deflection at preload
    delta_def = delta0 + (hf-h0)       # greatest working deflection
    F_p = k*delta_p                    # preload force
    F_def = k*delta_def                # force at full deflection
    K = ((4*D)-d)/(4*(D-d))+0.62*(d/D) # Wahl factor
    tau_p = ((8*F_p*D)/(np.pi*d**3))*K     # stress at preload height
    tau_def = ((8*F_def*D)/(np.pi*d**3))*K # stress at full deflection
    tau_max = tau_def                  # max stress (assumed at max deflection)
    tau_min = tau_p                    # min stress (assumed at preload deflection)
    tau_a = (tau_max - tau_min)/2      # alternating stress
    
    return tau_a


def get_Sefratio(x):
    d = x[0]
    D = x[1]
    n = x[2]
    hf = x[3]
    
    h0 = 1.0                 # preload height, in
    delta0 = 0.4             # deflection, in
    hdef = h0 - delta0     # deflection height, in
    G = 12*10**6             # psi
    Se = 45000               # psi
    Sf = 1.5

    Sefratio = Se/Sf                   # endurance limit to safety factor
    
    return Sefratio


def get_Syfratio(x):
    d = x[0]
    
    w = 0.18
    Sf = 1.5
    Q = 150000               # psi
    Sy = 0.44*(Q/(d**w))                # yield stress
    Syfratio = Sy/Sf                   # yield strength to safety factor
    
    return Syfratio
    
    
def get_tau_amsum(x):
    d = x[0]
    D = x[1]
    n = x[2]
    hf = x[3]
    
    h0 = 1.0                 # preload height, in
    delta0 = 0.4             # deflection, in
    hdef = h0 - delta0     # deflection height, in
    G = 12*10**6             # psi
    
    k = (G*d**4)/(8*(D**3)*n)            # spring stiffness
    delta_p = hf-h0                    # deflection at preload
    delta_def = delta0 + (hf-h0)       # greatest working deflection
    F_p = k*delta_p                    # preload force
    F_def = k*delta_def                # force at full deflection
    K = ((4*D)-d)/(4*(D-d))+0.62*(d/D) # Wahl factor
    tau_p = ((8*F_p*D)/(np.pi*d**3))*K     # stress at preload height
    tau_def = ((8*F_def*D)/(np.pi*d**3))*K # stress at full deflection
    tau_max = tau_def                  # max stress (assumed at max deflection)
    tau_min = tau_p                    # min stress (assumed at preload deflection)
    tau_m = (tau_max + tau_min)/2      # mean stress
    tau_a = (tau_max - tau_min)/2      # alternating stress
    tau_amsum = tau_a + tau_m          # sum of alternating and mean stresses
    
    return tau_amsum


def get_dratio(x):
    d = x[0]
    D = x[1]
    
    dratio = D/d                       # ratio of diameters
    
    return dratio


def get_dsum(x):
    d = x[0]
    D = x[1]

    dsum = D + d                       # sum of diameters
    
    return dsum


def get_clash(x):
    d = x[0]
    n = x[2]
    
    h0 = 1.0                 # preload height, in
    delta0 = 0.4             # deflection, in
    hdef = h0 - delta0     # deflection height, in

    hs = n*d                           # solid height

    clash = hdef - hs                  # clash allowance

    
    return clash


def get_Sy(x):
    d = x[0]

    w = 0.18

    Q = 150000               # psi

    Sy = 0.44*(Q/(d**w))                # yield stress
    
    return Sy


def get_tau_s(x):
    d = x[0]
    D = x[1]
    n = x[2]
    hf = x[3]
    
    # h0 = 1.0                 # preload height, in
    # delta0 = 0.4             # deflection, in
    # hdef = h0 - delta0     # deflection height, in
    G = 12*10**6             # psi
    # Se = 45000               # psi
    # w = 0.18
    # Sf = 1.5
    # Q = 150000               # psi
    # # delta_f = 0              # in
    
    k = (G*d**4)/(8*(D**3)*n)            # spring stiffness
    # # delta_p = hf-h0                    # deflection at preload
    # delta_def = delta0 + (hf-h0)       # greatest working deflection
    hs = n*d                           # solid height
    delta_s = hf - hs                  # deflection at solid height
    # # # F_f = k*delta_f                    # full height force (zero)
    # # F_p = k*delta_p                    # preload force
    # F_def = k*delta_def                # force at full deflection
    F_s = k*delta_s                    # force at solid height
    K = ((4*D)-d)/(4*(D-d))+0.62*(d/D) # Wahl factor
    # # # # tau_f = ((8*F_f*D)/(np.pi*d**3))*K     # stress at full height (zero)
    # tau_p = ((8*F_p*D)/(np.pi*d**3))*K     # stress at preload height
    # tau_def = ((8*F_def*D)/(np.pi*d**3))*K # stress at full deflection
    tau_s = ((8*F_s*D)/(np.pi*d**3))*K     # stress at solid height
    # tau_max = tau_def                  # max stress (assumed at max deflection)
    # tau_min = tau_p                    # min stress (assumed at preload deflection)
    # tau_m = (tau_max + tau_min)/2      # mean stress
    # tau_a = (tau_max - tau_min)/2      # alternating stress
    # Sy = 0.44*(Q/(d**w))                # yield stress
    # dratio = D/d                       # ratio of diameters
    # dsum = D + d                       # sum of diameters
    # clash = hdef - hs                  # clash allowance
    # Sefratio = Se/Sf                   # endurance limit to safety factor
    # Syfratio = Sy/Sf                   # yield strength to safety factor
    # tau_amsum = tau_a + tau_m          # sum of alternating and mean stresses
    
    return tau_s


def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], objective(Xi)))
    Nfeval += 1
    

if __name__ == "__main__":
        
    x0 = [0.20, 0.210, 90.0, 0.020]
    
    
    bnds = ((0.01, 0.2), (0.2, 1.0), (1.0, 100.0), (0.01, 20.0))
    
    cons = ({"type": "ineq",
             "fun": lambda x: get_Sefratio(x) - get_tau_a(x)},
            {"type": "ineq",
             "fun": lambda x: get_Syfratio(x) - get_tau_amsum(x)},
            {"type": "ineq",
             "fun": lambda x: 16.0 - get_dratio(x)},
            {"type": "ineq",
             "fun": lambda x: get_dratio(x) - 4.0},
            {"type": "ineq",
             "fun": lambda x: 0.75 - get_dsum(x)},
            {"type": "ineq",
             "fun": lambda x: get_clash(x) - 0.05},
            {"type": "ineq",
             "fun": lambda x: get_Sy(x) - get_tau_s(x)})
    
    
    result = opt.minimize(objective, x0, callback=callbackF, constraints=cons, bounds=bnds)
    print(result)
    
    d_opt = result.x[0]
    D_opt = result.x[1]
    n_opt = result.x[2]
    hf_opt = result.x[3]
    F_p_opt = np.abs(result.fun)    # Absolute value because the objective is negative (since we're maximizing using the minimize function)
    
    print("\nOptimum:")
    print("d: {}".format(d_opt))
    print("D: {}".format(D_opt))
    print("n: {}".format(n_opt))
    print("hf: {}".format(hf_opt))
    print("\nF_p: {}".format(F_p_opt))
    