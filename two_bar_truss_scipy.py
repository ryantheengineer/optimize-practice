# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 07:49:26 2022

@author: Ryan.Larson
"""

import numpy as np
import scipy.optimize as opt

Nfeval = 1


def objective(parameters):
    # Declare model parameters
    width = 60.
    thickness = 0.15
    density = 0.3
    # modulus = 300000.
    # load = 66.
    
    # Read in design variables
    height = parameters[0]
    diameter = parameters[1]
    
    # Intermediate variables with explicit equations
    leng = np.sqrt((width/2)**2 + height**2)
    area = np.pi * diameter * thickness
    # iovera = (diameter**2 + thickness**2)/8
    # stress = (load * leng / (2*area*height))
    # buckling = np.pi**2 * modulus * iovera / (leng**2)
    # deflection = load * leng**3 / (2 * modulus * area * height**2)
    
    weight = 2 * density * area * leng
    
    return weight


def get_stress(parameters):
    # Declare model parameters
    width = 60.
    thickness = 0.15
    # density = 0.3
    # modulus = 300000.
    load = 66.
    
    # Read in design variables
    height = parameters[0]
    diameter = parameters[1]
    
    leng = np.sqrt((width/2)**2 + height**2)
    area = np.pi * diameter * thickness
    
    stress = (load * leng / (2*area*height))
    
    return stress


def get_buckling(parameters):
    # Declare model parameters
    width = 60.
    thickness = 0.15
    # density = 0.3
    modulus = 300000.
    # load = 66.
    
    # Read in design variables
    height = parameters[0]
    diameter = parameters[1]
    
    leng = np.sqrt((width/2)**2 + height**2)
    # area = np.pi * diameter * thickness
    iovera = (diameter**2 + thickness**2)/8
    # stress = (load * leng / (2*area*height))
    buckling = np.pi**2 * modulus * iovera / (leng**2)
    
    return buckling


def get_deflection(parameters):
    # Declare model parameters
    width = 60.
    thickness = 0.15
    # density = 0.3
    modulus = 300000.
    load = 66.
    
    # Read in design variables
    height = parameters[0]
    diameter = parameters[1]
    
    # Intermediate variables with explicit equations
    leng = np.sqrt((width/2)**2 + height**2)
    area = np.pi * diameter * thickness
    # iovera = (diameter**2 + thickness**2)/8
    # stress = (load * leng / (2*area*height))
    # buckling = np.pi**2 * modulus * iovera / (leng**2)
    deflection = load * leng**3 / (2 * modulus * area * height**2)
    
    return deflection


def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(Nfeval, Xi[0], Xi[1], objective(Xi)))
    Nfeval += 1



if __name__ == "__main__":
    
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', ' X1', ' X2', 'f(X)'))
    x0 = [30.0, 3.0]
    bnds = ((10.0, 50.0), (1.0, 4.0))   # Lower and upper bounds on height and diameter
    
    cons = ({"type": "ineq",
            "fun": lambda x: 24 - objective(x)},    # weight < 24
            {"type": "ineq",
             "fun": lambda x: 100 - get_stress(x)}, # stress < 100
            {"type": "ineq",
             "fun": lambda x: get_buckling(x) - get_stress(x)}, # stress < buckling
            {"type": "ineq",
             "fun": lambda x: 0.25 - get_deflection(x)} # deflection < 0.25
            )
    
    result = opt.minimize(objective, x0, callback=callbackF, constraints=cons, bounds=bnds)
    print(result)
    
    height_opt = result.x[0]
    diameter_opt = result.x[1]
    weight_opt = result.fun
    print("\nOptimum:")
    print("Height: {}".format(height_opt))
    print("Diameter: {}".format(diameter_opt))
    print("Weight: {}".format(weight_opt))