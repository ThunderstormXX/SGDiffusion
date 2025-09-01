import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

BASE_DIR=Path("./setup1")
SAVE_DIR=BASE_DIR/"Hessian_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def estimate_gaussian_params(X: np.ndarray):
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    return mu, Sigma

def est(trajs):
    t=np.arange(1,trajs.shape[1]+1)
    v=trajs.var(0); msk=v>0
    b,a=np.polyfit(np.log(t[msk]),np.log(v[msk]),1)
    return np.exp(a/2), b/2

def fbm_nd(Hs,sigmas,rho,n_paths,n_steps=1000):
    d=len(Hs)
    Ls=[]
    for H in Hs:
        C=np.fromfunction(lambda i,j:0.5*((i+1)**(2*H)+(j+1)**(2*H)-abs(i-j)**(2*H)),(n_steps,n_steps))
        Ls.append(np.linalg.cholesky(C+1e-12*np.eye(n_steps)))
    sims=[]
    for _ in range(n_paths):
        z=np.random.multivariate_normal(np.zeros(d),rho,size=n_steps).T
        coords=[sigmas[k]*Ls[k]@z[k] for k in range(d)]
        sims.append(np.vstack(coords).T)
    return np.array(sims)

# --- выбираем параметры ---
W=torch.load(BASE_DIR/"weights_lr0.01.pt",map_location="cpu").detach().float().numpy()
N,T,D=W.shape
params= np.arange(0,384)  # <--- тут задаём список параметров
X=W[:,:,params]

# --- детрендинг ---
mean_traj=X.mean(0,keepdims=1)
X_detr=X-mean_traj

# --- оценка гиперпараметров ---
sigmas,Hs=[],[]
for d in range(X_detr.shape[2]):
    s,H=est(X_detr[:,:,d]); sigmas.append(s); Hs.append(H)
sigmas, Hs=np.array(sigmas), np.array(Hs)
inc=np.diff(X_detr,1,1).reshape(-1,X_detr.shape[2])
rho=np.corrcoef(inc,rowvar=False)

# --- симуляции ---
sim=fbm_nd(Hs,sigmas,rho,N,1000)

# --- финальные точки ---
final_emp = X_detr[:,-1,:]
final_sim = sim[:,-1,:]

# --- гауссовские параметры ---
mu_emp, Sigma_emp = estimate_gaussian_params(final_emp)
mu_sim, Sigma_sim = estimate_gaussian_params(final_sim)

# --- относительная невязка ---
rel_err = np.linalg.norm(Sigma_emp - Sigma_sim, "fro") / np.linalg.norm(Sigma_emp, "fro")
print("Relative error (Frobenius norm):", rel_err)

# --- визуализация ---
labels=[f"p{p}" for p in params]
fig,axs=plt.subplots(1,2,figsize=(12,5))
sns.heatmap(Sigma_emp,ax=axs[0],cmap="coolwarm",
            xticklabels=labels,yticklabels=labels,cbar=True)
axs[0].set_title("Empirical covariance")
sns.heatmap(Sigma_sim,ax=axs[1],cmap="coolwarm",
            xticklabels=labels,yticklabels=labels,cbar=True)
axs[1].set_title("Simulated covariance")
plt.suptitle(f"Relative error = {rel_err:.3f}", fontsize=14)
plt.savefig(SAVE_DIR/"covariance_compare.png"); plt.close()
