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
k_neighbors = 20

# --- Function to compute gradients, rms, neighbor indices ---
def compute_local_gradients(y):
    tree = KDTree(X)
    grads = np.zeros((n, d))
    rms_res = np.zeros(n)
    nbr_indices = np.zeros((n, k_neighbors), dtype=int)
    distances = np.zeros((n, k_neighbors))
    for i in range(n):
        dist, idx = tree.query(X[i:i+1], k=k_neighbors)
        idx = idx.flatten()
        nbr_indices[i,:] = idx
        distances[i,:] = dist.flatten()
        Xnbr = X[idx] - X[i]
        ynbr = y[idx]
        lr = LinearRegression()
        lr.fit(Xnbr, ynbr)
        grads[i,:] = lr.coef_
        pred = lr.predict(Xnbr)
        rms_res[i] = np.sqrt(np.mean((pred - ynbr)**2))
    return grads, rms_res, nbr_indices, distances

# --- Compute for CL and CD ---
grads_CL, rms_CL, nbr_CL, dist_CL = compute_local_gradients(CL)
grads_CD, rms_CD, nbr_CD, dist_CD = compute_local_gradients(CD)

# --- Compute neighbor gradient angles ---
def compute_angles(grads, nbr_indices):
    angles = []
    for i in range(len(grads)):
        g0 = grads[i]
        for j_idx in nbr_indices[i,1:5]:  # first 4 neighbors (skip self)
            g1 = grads[j_idx]
            na, nb = np.linalg.norm(g0), np.linalg.norm(g1)
            if na == 0 or nb == 0:
                continue
            cos = np.dot(g0, g1) / (na * nb)
            cos = np.clip(cos, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos)))
    return np.array(angles)

angles_CL = compute_angles(grads_CL, nbr_CL)
angles_CD = compute_angles(grads_CD, nbr_CD)

# --- Compute neighbor distance vs angle ---
neighbor_distances_CL = dist_CL[:,1:5].flatten()
neighbor_distances_CD = dist_CD[:,1:5].flatten()
angle_pairs_CL = angles_CL[:len(neighbor_distances_CL)]
angle_pairs_CD = angles_CD[:len(neighbor_distances_CD)]

# --- Plotting 2x2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(14,10))

# RMS residuals
axes[0,0].hist(rms_CL, bins=30, alpha=0.6, label='CL', color='salmon', edgecolor='k')
axes[0,0].hist(rms_CD, bins=30, alpha=0.6, label='CD', color='skyblue', edgecolor='k')
axes[0,0].set_xlabel('RMS residual')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Local Linear Fit Residuals')
axes[0,0].legend()

# Gradient magnitudes
axes[0,1].hist(np.linalg.norm(grads_CL, axis=1), bins=30, alpha=0.6, label='CL', color='salmon', edgecolor='k')
axes[0,1].hist(np.linalg.norm(grads_CD, axis=1), bins=30, alpha=0.6, label='CD', color='skyblue', edgecolor='k')
axes[0,1].set_xlabel('Gradient Norm')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Distribution of Local Gradient Magnitudes')
axes[0,1].legend()

# Neighbor gradient angles
axes[1,0].hist(angles_CL, bins=30, alpha=0.6, label='CL', color='salmon', edgecolor='k')
axes[1,0].hist(angles_CD, bins=30, alpha=0.6, label='CD', color='skyblue', edgecolor='k')
axes[1,0].set_xlabel('Angle Between Neighbor Gradients (deg)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Local Gradient Direction Variability')
axes[1,0].legend()

# Gradient angle vs neighbor distance
axes[1,1].scatter(neighbor_distances_CL, angle_pairs_CL, alpha=0.5, label='CL', color='salmon', edgecolor='k')
axes[1,1].scatter(neighbor_distances_CD, angle_pairs_CD, alpha=0.5, label='CD', color='skyblue', edgecolor='k')
axes[1,1].set_xlabel('Neighbor Distance in Design Space')
axes[1,1].set_ylabel('Angle Between Gradients (deg)')
axes[1,1].set_title('Gradient Angle vs Neighbor Distance')
axes[1,1].legend()

plt.tight_layout()
plt.show()



# tree = KDTree(X)
# k_neighbors = 20
# grads = np.zeros((n, d))
# rms_res = np.zeros(n)
# nbr_indices = np.zeros((n, k_neighbors), dtype=int)

# for i in range(n):
#     dist, idx = tree.query(X[i:i+1], k=k_neighbors)
#     idx = idx.flatten()
#     nbr_indices[i,:] = idx
#     Xnbr = X[idx] - X[i]
#     # y = CD[idx]
#     y = CL[idx]
#     lr = LinearRegression()
#     lr.fit(Xnbr, y)
#     grads[i,:] = lr.coef_
#     pred = lr.predict(Xnbr)
#     rms_res[i] = np.sqrt(np.mean((pred - y)**2))

# # gradient norms
# grad_norms = np.linalg.norm(grads, axis=1)
# mean_grad_norm = float(np.mean(grad_norms))
# median_grad_norm = float(np.median(grad_norms))
# pct_grad_norm_gt = float(100.0*np.mean(grad_norms > 0.001))  # arbitrary thresh

# # angles between neighbors (first 4 neighbors)
# angles = []
# for i in range(n):
#     g0 = grads[i]
#     for j_idx in nbr_indices[i,1:5]:
#         g1 = grads[j_idx]
#         na = np.linalg.norm(g0); nb = np.linalg.norm(g1)
#         if na==0 or nb==0:
#             continue
#         cos = np.dot(g0,g1)/(na*nb)
#         cos = max(-1.0, min(1.0, cos))
#         angles.append(math.degrees(math.acos(cos)))
# angles = np.array(angles)

# # stats
# frac_rms_gt_tol = float(100.0*np.mean(rms_res > 0.008))
# frac_angles_gt15 = float(100.0*np.mean(angles > 15))
# frac_angles_gt30 = float(100.0*np.mean(angles > 30))
# frac_angles_gt45 = float(100.0*np.mean(angles > 45))

# # relative prediction error factor for unit step based on angle: 1-cos(theta)
# rel_err_mean = float(np.mean(1 - np.cos(np.deg2rad(angles))))
# rel_err_median = float(np.median(1 - np.cos(np.deg2rad(angles))))
# rel_err_95 = float(np.percentile(1 - np.cos(np.deg2rad(angles)),95))

# result = {
#     "mean_grad_norm": mean_grad_norm,
#     "median_grad_norm": median_grad_norm,
#     "pct_grad_norm_gt_0.001": pct_grad_norm_gt,
#     "fraction_rms_gt_0.008_percent": frac_rms_gt_tol,
#     "mean_angle_deg": float(np.nanmean(angles)),
#     "median_angle_deg": float(np.nanmedian(angles)),
#     "angle_95pct": float(np.nanpercentile(angles,95)),
#     "frac_angles_gt15": frac_angles_gt15,
#     "frac_angles_gt30": frac_angles_gt30,
#     "frac_angles_gt45": frac_angles_gt45,
#     "rel_err_mean": rel_err_mean,
#     "rel_err_median": rel_err_median,
#     "rel_err_95pct": rel_err_95
# }
