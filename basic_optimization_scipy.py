# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:00:29 2022

@author: Ryan.Larson
"""

import numpy as np
import scipy.optimize as opt

objective = np.poly1d([1.0, -2.0, 0.0])
# print(objective)
cons = ({'type': 'ineq',
         'fun' : lambda x: np.array([x[0] - 4])}) # x > 4


x0 = 3.0
results = opt.minimize(objective,x0,
                       constraints = cons,
                       options = {'disp':True})
print("Solution: x=%f" % results.x)

import matplotlib.pyplot as plt
x = np.linspace(-3,5,100)
plt.plot(x,objective(x))
plt.plot(results.x,objective(results.x),'ro')
plt.show()