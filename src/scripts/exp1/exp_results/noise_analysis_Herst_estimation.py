import numpy as np
import torch
import matplotlib.pyplot as plt
import os

BASE_DIR = "./setup1"
LR_TO_USE = "0.01"
PATH = os.path.join(BASE_DIR, f"weights_lr{LR_TO_USE}.pt")

W = torch.load(PATH, map_location="cpu").detach().float().numpy()  # [N, T, D]
N, T, D = W.shape

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

Hs, Sigmas = [], []
for d in range(D):
    sigma_hat, H_hat = estimate_hurst_sigma(W[:,:,d])
    Hs.append(H_hat)
    Sigmas.append(sigma_hat)

Hs, Sigmas = np.sort(Hs), np.sort(Sigmas)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(Hs, ".k", alpha=0.7)
plt.yscale("log")
plt.title("Sorted Hurst exponents (log scale)")
plt.xlabel("Parameter rank"); plt.ylabel("H")

plt.subplot(1,2,2)
plt.plot(Sigmas, ".r", alpha=0.7)
plt.yscale("log")
plt.title("Sorted Sigma (log scale)")
plt.xlabel("Parameter rank"); plt.ylabel("Sigma")

plt.tight_layout(); plt.show()
