# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:08:48 2025

@author: ryanl
"""

import numpy as np, math
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.loadtxt('CRM_MULTISCALE_DATA.txt')
X = data[:, :12]
CL = data[:, 12]
CD = data[:, 13]
n, d = X.shape

tree = KDTree(X)
k_neighbors = 20
grads = np.zeros((n, d))
rms_res = np.zeros(n)
nbr_indices = np.zeros((n, k_neighbors), dtype=int)

for i in range(n):
    dist, idx = tree.query(X[i:i+1], k=k_neighbors)
    idx = idx.flatten()
    nbr_indices[i,:] = idx
    Xnbr = X[idx] - X[i]
    # y = CD[idx]
    y = CL[idx]
    lr = LinearRegression()
    lr.fit(Xnbr, y)
    grads[i,:] = lr.coef_
    pred = lr.predict(Xnbr)
    rms_res[i] = np.sqrt(np.mean((pred - y)**2))

# gradient norms
grad_norms = np.linalg.norm(grads, axis=1)
mean_grad_norm = float(np.mean(grad_norms))
median_grad_norm = float(np.median(grad_norms))
pct_grad_norm_gt = float(100.0*np.mean(grad_norms > 0.001))  # arbitrary thresh

# angles between neighbors (first 4 neighbors)
angles = []
for i in range(n):
    g0 = grads[i]
    for j_idx in nbr_indices[i,1:5]:
        g1 = grads[j_idx]
        na = np.linalg.norm(g0); nb = np.linalg.norm(g1)
        if na==0 or nb==0:
            continue
        cos = np.dot(g0,g1)/(na*nb)
        cos = max(-1.0, min(1.0, cos))
        angles.append(math.degrees(math.acos(cos)))
angles = np.array(angles)

# stats
frac_rms_gt_tol = float(100.0*np.mean(rms_res > 0.008))
frac_angles_gt15 = float(100.0*np.mean(angles > 15))
frac_angles_gt30 = float(100.0*np.mean(angles > 30))
frac_angles_gt45 = float(100.0*np.mean(angles > 45))

# relative prediction error factor for unit step based on angle: 1-cos(theta)
rel_err_mean = float(np.mean(1 - np.cos(np.deg2rad(angles))))
rel_err_median = float(np.median(1 - np.cos(np.deg2rad(angles))))
rel_err_95 = float(np.percentile(1 - np.cos(np.deg2rad(angles)),95))

result = {
    "mean_grad_norm": mean_grad_norm,
    "median_grad_norm": median_grad_norm,
    "pct_grad_norm_gt_0.001": pct_grad_norm_gt,
    "fraction_rms_gt_0.008_percent": frac_rms_gt_tol,
    "mean_angle_deg": float(np.nanmean(angles)),
    "median_angle_deg": float(np.nanmedian(angles)),
    "angle_95pct": float(np.nanpercentile(angles,95)),
    "frac_angles_gt15": frac_angles_gt15,
    "frac_angles_gt30": frac_angles_gt30,
    "frac_angles_gt45": frac_angles_gt45,
    "rel_err_mean": rel_err_mean,
    "rel_err_median": rel_err_median,
    "rel_err_95pct": rel_err_95
}
