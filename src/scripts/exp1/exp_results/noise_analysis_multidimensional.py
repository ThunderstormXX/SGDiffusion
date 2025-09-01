import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = "./setup1"
SAVE_DIR = f"{BASE_DIR}/hessian_analysis"
LR_TO_USE = "0.01"
PATH = os.path.join(BASE_DIR, f"weights_lr{LR_TO_USE}.pt")
HESS_PATH = os.path.join(BASE_DIR, "hessians_traj_lr0.01.pt")

# --- диапазон параметров ---
PARAM_RANGE = slice(0, 10)

W = torch.load(PATH, map_location="cpu").detach().float().numpy()
N, T, D = W.shape
W = W[:,:,PARAM_RANGE]
_, _, D_sub = W.shape

def estimate_hurst_sigma(trajs):
    mean_traj = trajs.mean(axis=0, keepdims=True)
    detrended = trajs - mean_traj
    t = np.arange(1, detrended.shape[1]+1)
    var_t = detrended.var(axis=0)
    mask = var_t > 0
    log_t, log_var = np.log(t[mask]), np.log(var_t[mask])
    b, a = np.polyfit(log_t, log_var, deg=1)
    H_hat = b / 2
    sigma_hat = np.exp(a / 2)
    return sigma_hat, H_hat

# --- Визуализация траекторий 3D (до/после детрендинга) ---
rng = np.random.default_rng(42)
idx = rng.choice(D_sub, size=3, replace=False)
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121, projection='3d')
for n in range(min(50, N)):
    ax.plot(W[n,:,idx[0]], W[n,:,idx[1]], W[n,:,idx[2]], alpha=0.3)
ax.set_title("Raw trajectories (3 params)")
mean_traj = W.mean(axis=0, keepdims=True)
W_detr = W - mean_traj
ax2 = fig.add_subplot(122, projection='3d')
for n in range(min(50, N)):
    ax2.plot(W_detr[n,:,idx[0]], W_detr[n,:,idx[1]], W_detr[n,:,idx[2]], alpha=0.3)
ax2.set_title("Detrended trajectories (3 params)")
plt.savefig(os.path.join(SAVE_DIR, "trajectories_3d.png"))
plt.close()

# --- Оценка H и σ ---
Hs, Sigmas = [], []
for d in range(D_sub):
    sigma_hat, H_hat = estimate_hurst_sigma(W[:,:,d])
    Hs.append(H_hat); Sigmas.append(sigma_hat)
Hs, Sigmas = np.array(Hs), np.array(Sigmas)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(np.sort(Hs), ".k"); plt.yscale("log")
plt.title("Sorted Hurst exponents (log)"); plt.xlabel("Parameter rank"); plt.ylabel("H")
plt.subplot(1,2,2); plt.plot(np.sort(Sigmas), ".r"); plt.yscale("log")
plt.title("Sorted Sigma (log)"); plt.xlabel("Parameter rank"); plt.ylabel("Sigma")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hurst_sigma_sorted.png"))
plt.close()

# --- Матрица корреляций ---
increments = np.diff(W, axis=1)
increments_flat = increments.reshape(-1, D_sub)
corr_matrix = np.corrcoef(increments_flat, rowvar=False)

# --- Гессиан ---
H_traj = torch.load(HESS_PATH, map_location="cpu").detach().float().numpy()
H_mean = H_traj.mean(axis=0)
H_sub = H_mean[PARAM_RANGE, :][:, PARAM_RANGE]

# --- бинаризация ---
H_bin = (np.abs(H_sub) > 1e-5).astype(float)

# 4. Визуализация: корреляция vs гессиан
fig, axs = plt.subplots(1,2, figsize=(14,6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=axs[0], cbar_kws={'shrink':0.7})
axs[0].set_title("Correlation matrix of increments")

sns.heatmap(H_bin, cmap="Greys", ax=axs[1], cbar=False)
axs[1].set_title("Mean Hessian (binary >1e-5)")
plt.savefig(os.path.join(SAVE_DIR, "corr_vs_hessian.png"))
plt.close()


# --- PCA ---
pca = PCA(n_components=min(10, D_sub))
pca_increments = pca.fit_transform(increments_flat)

Hs_pca, Sigmas_pca = [], []
for k in range(pca_increments.shape[1]):
    comp_series = pca_increments[:,k].reshape(N, T-1)
    sigma_hat, H_hat = estimate_hurst_sigma(comp_series)
    Hs_pca.append(H_hat); Sigmas_pca.append(sigma_hat)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.bar(range(1,len(Hs_pca)+1), Hs_pca)
plt.title("Hurst exponents (PCA)"); plt.xlabel("PCA comp"); plt.ylabel("H")
plt.subplot(1,2,2); plt.bar(range(1,len(Sigmas_pca)+1), Sigmas_pca)
plt.title("Sigma (PCA)"); plt.xlabel("PCA comp"); plt.ylabel("Sigma")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hurst_sigma_pca.png"))
plt.close()

plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.title("Explained variance PCA")
plt.xlabel("Num components"); plt.ylabel("Cumulative explained var")
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "pca_explained_variance.png"))
plt.close()
