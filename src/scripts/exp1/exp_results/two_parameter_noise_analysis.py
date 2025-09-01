import numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR=Path("./setup1")
SAVE_DIR=BASE_DIR/"Hessian_analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

W=torch.load(BASE_DIR/"weights_lr0.01.pt",map_location="cpu").detach().float().numpy()
N,T,D=W.shape; params=[1,7]; X=W[:,:,params]

# --- детрендинг ---
mean_traj=X.mean(0,keepdims=1)
X_detr=X-mean_traj

def est(trajs):
    t=np.arange(1,trajs.shape[1]+1)
    v=trajs.var(0); msk=v>0
    b,a=np.polyfit(np.log(t[msk]),np.log(v[msk]),1)
    return np.exp(a/2), b/2

s1,H1=est(X_detr[:,:,0]); s2,H2=est(X_detr[:,:,1])
inc=np.diff(X_detr,1,1).reshape(-1,2); rho=np.corrcoef(inc.T)[0,1]
print(f"H1={H1:.3f}, σ1={s1:.3f}, H2={H2:.3f}, σ2={s2:.3f}, ρ={rho:.3f}")

def fbm2d(H1,H2,s1,s2,rho,n_paths,n_steps=1000):
    C11=np.fromfunction(lambda i,j:0.5*((i+1)**(2*H1)+(j+1)**(2*H1)-abs(i-j)**(2*H1)),(n_steps,n_steps))
    C22=np.fromfunction(lambda i,j:0.5*((i+1)**(2*H2)+(j+1)**(2*H2)-abs(i-j)**(2*H2)),(n_steps,n_steps))
    L1=np.linalg.cholesky(C11+1e-12*np.eye(n_steps))
    L2=np.linalg.cholesky(C22+1e-12*np.eye(n_steps))
    sims=[]
    for _ in range(n_paths):
        z1=np.random.randn(n_steps); z2=rho*z1+np.sqrt(1-rho**2)*np.random.randn(n_steps)
        sims.append(np.vstack([s1*L1@z1,s2*L2@z2]).T)
    return np.array(sims)

sim=fbm2d(H1,H2,s1,s2,rho,N,1000)

plt.figure(figsize=(8,6))
for n in range(200):
    plt.plot(X_detr[n,:,0],X_detr[n,:,1],'k--',linewidth=0.5)
    plt.plot(sim[n,:,0],sim[n,:,1],'r-',linewidth=0.5)
plt.title("Real (black) vs Simulated (red) 2D trajectories")
plt.savefig(SAVE_DIR/"overlay_2d.png"); plt.close()
