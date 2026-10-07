#!/usr/bin/env python3
"""Finite-step effect in NanoGPT (6.6M parameters) on WikiText-2: discrete SGD vs. the
Langevin approximation, measured along the sharpest Hessian eigendirections.

SGD is started many times from a reference point w* and the variance of the iterates along
each top eigendirection of the mean Hessian is compared with two predictions computed from
quantities measured at w* (curvature lambda_i, gradient-noise variance d_i, curvature
variance Gamma_ii):
    Langevin:      eta d_i / (2 lambda_i - eta Gamma_ii)
    discrete SGD:  eta d_i / (2 lambda_i - eta (lambda_i^2 + Gamma_ii))

Stages (each writes into --out, is skipped if its output exists, and resumes
from its own checkpoint if it was interrupted):
  data      WikiText-2 (raw) -> GPT-2 BPE -> fixed set of 257-token sequences
  train     SGD, lr=0.05, B=32, sampling with replacement, fixed number of steps
  anneal    low-noise SGD with larger batches (the noise scales as lr / batch)
  refine    100 steps of full-gradient descent -> reference point w*
  refine_ref  for comparison only: full-gradient descent straight from the SGD end point
  lanczos   candidate top-K basis: Lanczos on a fixed subsample of the training set
  exact     worker (one per GPU): exact Hessian-vector products of the basis over the
            whole training set
  ritz      Rayleigh-Ritz on the exact products -> exact eigenvalues and residuals;
            then one exact subspace-iteration step and a second exact round
  plan      learning rates for the trajectories: eta = c / lambda_max
  stats     E[H_ii^2], d_i along those directions at w*, minibatch B=32
  traj      worker (any number in parallel): SGD trajectories from w*, round-robin
            over the learning rates
  drift     lambda_i and d_i re-measured at trajectory end points
  analyze   window variances, discrete vs Langevin predictions, plots, report

Design choices:
  * The training set is a finite set of sequences, so E_L, the full gradient
    and the mean Hessian are exact averages over the set SGD samples from.
  * Eigenvalues and eigenvector residuals come from exact Hessian-vector
    products over the whole training set.
  * E[H_ii^2] and d_i are measured at the SGD batch size (32) and enter the
    predictions directly: no constant is fitted to the trajectories.
  * Predictions are compared with the variance averaged over the same window,
    using the time-dependent formula, so unsaturated directions are handled.
  * Drift check: lambda_i and d_i are re-measured at trajectory end points.

The defaults are the configuration of the experiment reported in the paper;
see README.md and run_experiment.sh.
"""
import argparse
import glob
import json
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 50257
USE_TF32 = False
OOM_TRIES, OOM_WAIT = 10, 60
STEP_CHUNK = 32
T0 = time.time()


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f}m]", *args, flush=True)


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_pt(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def remove(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def project(U, x):
    """U @ x in full fp32: projections are never rounded to TF32."""
    set_tf32(False)
    out = U @ x
    set_tf32(USE_TF32)
    return out


def retry_oom(fn, what):
    """Run fn(); on CUDA OOM free the cache, wait and try again (the GPU is shared).

    fn must be safe to call again from scratch."""
    for attempt in range(1, OOM_TRIES + 1):
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as e:
            msg = str(e).split("\n")[0][:120]
        # Outside the except block, so the traceback no longer pins GPU tensors.
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        log(f"CUDA OOM in {what} (attempt {attempt}/{OOM_TRIES}, {free:.1f} GB free on the GPU): {msg}")
        time.sleep(OOM_WAIT)
    # The supervisor restarts the process on the GPU that has the most free memory by then.
    raise RuntimeError(f"CUDA OOM in {what}: gave up after {OOM_TRIES} attempts")


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, mlp_ratio, block_size):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        # Explicit attention (no fused kernel) so that double backward works for HVPs.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    """NanoGPT: d=64, 2 heads, 4 layers, context 256, MLP ratio 4, untied head (6.65M params)."""

    def __init__(self, d=64, n_head=2, n_layer=4, block_size=256, mlp_ratio=4):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, d)
        self.wpe = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(Block(d, n_head, mlp_ratio, block_size) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, 0.0, 0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(b.fc2.weight, 0.0, 0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx, targets):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))


def new_model(a):
    return GPT(a.d_model, a.n_head, a.n_layer, a.block_size, a.mlp_ratio).to(DEV)


def flatten_params_(model):
    """Make every parameter a view into one flat fp32 buffer and return the buffer."""
    ps = list(model.parameters())
    flat = torch.cat([p.detach().reshape(-1) for p in ps]).contiguous()
    o = 0
    for p in ps:
        n = p.numel()
        p.data = flat[o:o + n].view_as(p)
        o += n
    return flat


def split_like(vec, ps):
    out, o = [], 0
    for p in ps:
        n = p.numel()
        out.append(vec[o:o + n].view_as(p))
        o += n
    return out


def flat_grads(gs, ps):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(gs, ps)])


def load_model_flat(a, path):
    model = new_model(a)
    flat = flatten_params_(model)
    flat.copy_(torch.load(path, map_location=DEV))
    return model, flat


# ----------------------------------------------------------------------------- data and passes
WT2_URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
           "wikitext-2-raw-v1/{split}-00000-of-00001.parquet")


def out_path(a, name):
    return os.path.join(a.out, name)


def load_seqs(a, split):
    return torch.from_numpy(np.load(out_path(a, f"{split}_seqs.npy"))).long().to(DEV)


def rand_idx(n, batch, seed):
    """Minibatch indices (with replacement) that depend only on the seed -> resumable loops."""
    return torch.randint(n, (batch,), device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed))


def batch_loss(model, seqs, idx):
    s = seqs[idx]
    return model(s[:, :-1], s[:, 1:])


def loss_and_grads(model, seqs, idx):
    """Minibatch loss and gradient, accumulated over sub-chunks to keep the memory peak low
    (the mean of equal-size chunk gradients is exactly the minibatch gradient)."""
    ps = list(model.parameters())
    loss_tot, grads = 0.0, None
    for part in idx.split(STEP_CHUNK):
        loss = batch_loss(model, seqs, part) * (len(part) / len(idx))
        g = torch.autograd.grad(loss, ps)
        grads = list(g) if grads is None else [x.add_(y) for x, y in zip(grads, g)]
        loss_tot = loss_tot + loss.detach()
    return loss_tot, grads


def batch_grad(model, seqs, idx):
    return flat_grads(loss_and_grads(model, seqs, idx)[1], list(model.parameters()))


@torch.no_grad()
def full_loss(model, seqs, chunk):
    tot = 0.0
    for i in range(0, len(seqs), chunk):
        s = seqs[i:i + chunk]
        tot += model(s[:, :-1], s[:, 1:]).item() * len(s)
    return tot / len(seqs)


def full_loss_grad(model, seqs, chunk):
    """Exact loss and gradient of the average loss over the finite training set."""
    ps = list(model.parameters())
    N, tot, g = len(seqs), 0.0, None
    for i in range(0, N, chunk):
        s = seqs[i:i + chunk]
        loss = model(s[:, :-1], s[:, 1:]) * (len(s) / N)
        gf = flat_grads(torch.autograd.grad(loss, ps), ps)
        g = gf if g is None else g.add_(gf)
        tot += loss.item()
    return tot, g


def hv_rows(model, s, V, j0=0):
    """Yield (j, H_s V[j]) for j >= j0, H_s being the Hessian of the mean loss over the
    sequences s: one graph, one second backward per direction, no K x P temporary."""
    ps = list(model.parameters())
    g = torch.autograd.grad(model(s[:, :-1], s[:, 1:]), ps, create_graph=True)
    K = V.shape[0]
    for j in range(j0, K):
        dot = sum((gi * vi).sum() for gi, vi in zip(g, split_like(V[j], ps)))
        yield j, flat_grads(torch.autograd.grad(dot, ps, retain_graph=j < K - 1, allow_unused=True), ps)


def make_hvp(model, seqs, chunk):
    """v -> H v for the average loss over the given (fixed) sequences."""
    def hvp(v):
        out = torch.zeros_like(v)
        for i in range(0, len(seqs), chunk):
            s = seqs[i:i + chunk]
            for _, hv in hv_rows(model, s, v[None]):
                out.add_(hv, alpha=len(s) / len(seqs))
        return out
    return hvp


def quad_forms(model, seqs, idx, V, chunk, full):
    """Minibatch quadratic forms: Q[i, j] = v_i^T H_B v_j (full) or Q[i] = v_i^T H_B v_i."""
    K = V.shape[0]
    Q = torch.zeros((K, K) if full else (K,), dtype=torch.float64)
    for part in idx.split(chunk):
        w = len(part) / len(idx)
        for j, hv in hv_rows(model, seqs[part], V):
            if full:
                Q[:, j] += project(V, hv).double().cpu() * w
            else:
                Q[j] += (V[j] @ hv).item() * w
    return Q


def directions(a, P):
    """Rows: top-K Hessian directions, then n_rand random unit directions. Returns (U, K)."""
    V = torch.load(out_path(a, "spectrum.pt"))["V"]
    K = V.shape[0]
    R = torch.randn(a.n_rand, P, generator=torch.Generator().manual_seed(a.seed + 99))
    U = torch.empty(K + a.n_rand, P, device=DEV)
    U[:K] = V
    U[K:] = R / R.norm(dim=1, keepdim=True)
    return U, K


# ----------------------------------------------------------------------------- stages
def stage_data(a):
    if os.path.exists(out_path(a, "data.json")):
        return
    import pyarrow.parquet as pq
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    info = {}
    for split in ("train", "validation"):
        pq_path = out_path(a, f"wt2_{split}.parquet")
        if not os.path.exists(pq_path):
            urllib.request.urlretrieve(WT2_URL.format(split=split), pq_path)
        text = "".join(pq.read_table(pq_path).column("text").to_pylist())
        ids = np.array(enc.encode_ordinary(text), dtype=np.int32)
        n = (len(ids) - 1) // a.block_size
        seqs = np.stack([ids[i * a.block_size: i * a.block_size + a.block_size + 1] for i in range(n)])
        np.save(out_path(a, f"{split}_seqs.npy"), seqs)
        info[split] = dict(tokens=int(len(ids)), sequences=int(n))
        log(f"data {split}: {len(ids):,} tokens -> {n:,} sequences of {a.block_size}+1")
    save_json(out_path(a, "data.json"), info)


def stage_train(a):
    if os.path.exists(out_path(a, "w_sgd.pt")):
        return
    ck = out_path(a, "train_ckpt.pt")
    torch.manual_seed(a.seed)
    model = new_model(a)
    flat = flatten_params_(model)
    ps = list(model.parameters())
    tr, va = load_seqs(a, "train"), load_seqs(a, "validation")
    step, hist = 0, []
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV)
        flat.copy_(c["flat"])
        step, hist = c["step"], c["hist"]
        log(f"train: resuming from step {step}")
    log(f"model parameters: {flat.numel():,}")
    gen = torch.Generator(device=DEV)
    acc = torch.zeros((), device=DEV)
    while step < a.max_steps:
        gen.manual_seed(a.seed * 1_000_003 + 17 + step // a.window)   # one seed per window -> resumable
        for _ in range(a.window):
            idx = torch.randint(len(tr), (a.batch,), device=DEV, generator=gen)
            loss, grads = retry_oom(lambda: loss_and_grads(model, tr, idx), "train")
            with torch.no_grad():
                torch._foreach_add_(ps, grads, alpha=-a.lr)
            acc += loss
        step += a.window
        cur = acc.item() / a.window
        acc.zero_()
        if not math.isfinite(cur):
            raise RuntimeError(f"training diverged at step {step}")
        val = full_loss(model, va, a.chunk) if step % a.val_every == 0 else float("nan")
        hist.append(dict(step=step, train_window=cur, val=val))
        log(f"train step {step:6d}  window loss {cur:.4f}  val {val:.4f}")
        if step % a.ckpt_steps == 0:
            save_pt(dict(flat=flat, step=step, hist=hist), ck)
    save_json(out_path(a, "train.json"), dict(steps=step, lr=a.lr, val_final=full_loss(model, va, a.chunk),
                                              history=hist))
    diagnose(a, model, tr, "after SGD")
    save_pt(flat.detach().clone(), out_path(a, "w_sgd.pt"))
    remove(ck)


def top_eigs(model, seqs, chunk, iters, seed):
    """Three largest Hessian eigenvalues over the given sequences (short Lanczos, full reorthogonalization)."""
    hvp = make_hvp(model, seqs, chunk)
    P = sum(p.numel() for p in model.parameters())
    Q = torch.empty(iters + 1, P, device=DEV)
    q = torch.randn(P, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed))
    Q[0] = q / q.norm()
    alpha, beta = np.zeros(iters), np.zeros(iters)
    for j in range(iters):
        w = retry_oom(lambda: hvp(Q[j]), "diagnose")
        alpha[j] = torch.dot(w, Q[j]).item()
        for _ in range(2):
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        beta[j] = w.norm().item()
        Q[j + 1] = w / beta[j]
    theta = np.linalg.eigvalsh(np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1))
    return theta[::-1][:3].tolist()


def diagnose(a, model, tr, tag):
    """Full loss, |full gradient| and the top Hessian eigenvalues at the current point;
    one record per tag in finish_log.json. Shows what each finishing phase does to the
    distance from a critical point and to the sharpness."""
    loss, g = retry_oom(lambda: full_loss_grad(model, tr, a.chunk), "diagnose")
    perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + 5))
    lam = top_eigs(model, tr[perm[:a.diag_seqs].to(DEV)], a.hvp_chunk, a.diag_iters, a.seed + 11)
    path = out_path(a, "finish_log.json")
    hist = [h for h in (load_json(path) if os.path.exists(path) else []) if h["tag"] != tag]
    save_json(path, hist + [dict(tag=tag, loss=loss, grad_norm=g.norm().item(), lam_top=lam)])
    log(f"diagnose [{tag}]: full loss {loss:.5f}  |grad| {g.norm().item():.4e}  "
        f"lambda_max ~ {lam[0]:.2f} (next {lam[1]:.2f}, {lam[2]:.2f})")


def anneal_spec(a):
    """'0.4:256:3000,0.4:1024:2000' -> [(lr, batch, steps), ...] with lr = multiplier * training lr.
    The finishing steps are scaled to the training lr because the sharpness reached in training
    follows 2 / lr, so a larger step afterwards would be unstable."""
    return [(float(m) * a.lr, int(b), int(n)) for m, b, n in (x.split(":") for x in a.anneal.split(","))] if a.anneal else []


def stage_anneal(a):
    """Low-noise SGD before the final GD steps: same update rule, larger batches
    (the stationary noise scales as lr / batch). It relaxes the many directions of
    moderate curvature that 100 full-gradient steps are too few for."""
    if os.path.exists(out_path(a, "w_anneal.pt")):
        return
    ck = out_path(a, "anneal_ckpt.pt")
    model, flat = load_model_flat(a, out_path(a, "w_sgd.pt"))
    ps = list(model.parameters())
    tr = load_seqs(a, "train")
    ph0, st0 = 0, 0
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV)
        flat.copy_(c["flat"])
        ph0, st0 = c["phase"], c["step"]
        log(f"anneal: resuming at phase {ph0 + 1}, step {st0}")
    gen = torch.Generator(device=DEV)
    for ph, (lr, batch, steps) in enumerate(anneal_spec(a)):
        if ph < ph0:
            continue
        step = st0 if ph == ph0 else 0
        while step < steps:
            gen.manual_seed(a.seed * 1_000_003 + 5_000_000 + ph * 100_000 + step // a.anneal_block)
            n, acc = min(a.anneal_block, steps - step), 0.0
            for _ in range(n):
                idx = torch.randint(len(tr), (batch,), device=DEV, generator=gen)
                loss, grads = retry_oom(lambda: loss_and_grads(model, tr, idx), "anneal")
                with torch.no_grad():
                    torch._foreach_add_(ps, grads, alpha=-lr)
                acc = acc + loss
            step += n
            log(f"anneal phase {ph + 1} (lr {lr:g}, B {batch}): step {step}/{steps}  mean loss {acc.item() / n:.4f}")
            save_pt(dict(flat=flat, phase=ph, step=step), ck)
        diagnose(a, model, tr, f"after low-noise SGD phase {ph + 1} (lr {lr:g}, B {batch})")
        save_pt(dict(flat=flat, phase=ph + 1, step=0), ck)
    save_pt(flat.detach().clone(), out_path(a, "w_anneal.pt"))
    remove(ck)


def stage_refine_ref(a):
    """For comparison only: finishing with full GD alone, straight from the SGD end point.
    Records the same diagnostics; its weights are not used."""
    if os.path.exists(out_path(a, "refine_ref.json")):
        return
    model, flat = load_model_flat(a, out_path(a, "w_sgd.pt"))
    tr = load_seqs(a, "train")
    for _ in range(a.refine_steps):
        _, g = retry_oom(lambda: full_loss_grad(model, tr, a.chunk), "refine_ref")
        flat.add_(g, alpha=-a.refine_lr_mult * a.lr)
    diagnose(a, model, tr, "GD-only finishing, from the SGD end point")
    save_json(out_path(a, "refine_ref.json"), dict(done=True))


def stage_refine(a):
    if os.path.exists(out_path(a, "w_star.pt")):
        return
    ck = out_path(a, "refine_ckpt.pt")
    src = out_path(a, "w_anneal.pt")
    model, flat = load_model_flat(a, src if os.path.exists(src) else out_path(a, "w_sgd.pt"))
    tr, va = load_seqs(a, "train"), load_seqs(a, "validation")
    k0, hist = 0, []
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV)
        flat.copy_(c["flat"])
        k0, hist = c["k"], c["hist"][:-1]
        log(f"refine: resuming from step {k0}")
    for k in range(k0, a.refine_steps + 1):
        loss, g = retry_oom(lambda: full_loss_grad(model, tr, a.chunk), "refine")
        if k % 10 == 0 or k == a.refine_steps:
            hist.append(dict(step=k, loss=loss, grad_norm=g.norm().item()))
            log(f"refine step {k:4d}  full loss {loss:.6f}  |grad| {g.norm().item():.4e}")
            save_pt(dict(flat=flat, k=k, hist=hist), ck)        # state before step k is applied
        if k == a.refine_steps:
            break
        flat.add_(g, alpha=-a.refine_lr_mult * a.lr)
    save_pt(g, out_path(a, "grad_star.pt"))
    save_json(out_path(a, "refine.json"), dict(
        history=hist, full_loss=loss, val_loss=full_loss(model, va, a.chunk),
        grad_norm=g.norm().item(), w_norm=flat.norm().item()))
    diagnose(a, model, tr, "after final GD (w*)")
    save_pt(flat.detach().clone(), out_path(a, "w_star.pt"))
    remove(ck)


def lanczos_subsample(a, model, tr, P, k_sub):
    """Lanczos with full reorthogonalization on a fixed subsample; checkpointed every iteration."""
    perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + 5))
    sub = tr[perm[:a.hvp_seqs].to(DEV)] if a.hvp_seqs else tr
    hvp = make_hvp(model, sub, a.hvp_chunk)
    m = a.lanczos_iters
    qfile, sfile = out_path(a, "lanczos_Q.bin"), out_path(a, "lanczos_state.npz")
    Q = torch.empty(m + 1, P)                 # Lanczos basis, kept on the CPU and appended to qfile
    alpha, beta, j0 = np.zeros(m), np.zeros(m), 0
    if os.path.exists(sfile) and os.path.exists(qfile):
        try:
            s = np.load(sfile)
            if len(s["alpha"]) == m and int(s["n_sub"]) == len(sub):
                n_done = int(s["n_done"])
                with open(qfile, "r+b") as f:
                    f.truncate((n_done + 1) * P * 4)
                Q[:n_done + 1] = torch.from_numpy(np.fromfile(qfile, dtype=np.float32).reshape(n_done + 1, P))
                j0, alpha, beta = n_done, s["alpha"].copy(), s["beta"].copy()
                log(f"lanczos: resuming from iteration {j0}")
        except (OSError, ValueError, KeyError) as e:
            log(f"lanczos: checkpoint unreadable ({e}); starting over")
    if j0 == 0:
        q = torch.randn(P, generator=torch.Generator().manual_seed(a.seed + 6))
        Q[0] = q / q.norm()
        with open(qfile, "wb") as f:
            f.write(Q[0].numpy().tobytes())
    log(f"lanczos: {m} iterations, HVP over {len(sub):,} fixed sequences")
    t_start = time.time()
    for j in range(j0, m):
        w = retry_oom(lambda: hvp(Q[j].to(DEV)).cpu(), "lanczos")
        alpha[j] = torch.dot(w.double(), Q[j].double()).item()
        for _ in range(2):                    # full reorthogonalization, twice
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        beta[j] = w.norm().item()
        Q[j + 1] = w / beta[j]
        with open(qfile, "ab") as f:
            f.write(Q[j + 1].numpy().tobytes())
        np.savez(sfile + ".tmp.npz", n_done=j + 1, alpha=alpha, beta=beta, n_sub=len(sub))
        os.replace(sfile + ".tmp.npz", sfile)
        if (j + 1) % 20 == 0:
            per = (time.time() - t_start) / (j + 1 - j0)
            log(f"lanczos {j + 1}/{m}  {per:.1f}s per iteration  ETA {per * (m - j - 1) / 60:.0f} min")
    theta, S = np.linalg.eigh(np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1))
    top = np.argsort(theta)[::-1][:k_sub]
    V = torch.from_numpy(S[:, top].T.astype(np.float32).copy()) @ Q[:m]
    return V / V.norm(dim=1, keepdim=True), theta[top]


def basis_sig(V):
    return float(V[:, ::997].double().sum())


def stage_lanczos(a):
    """Candidate basis for the exact passes: Ritz vectors of Lanczos on a fixed subsample."""
    if os.path.exists(out_path(a, "spectrum.pt")) or os.path.exists(out_path(a, "basis.pt")):
        return
    model, flat = load_model_flat(a, out_path(a, "w_star.pt"))
    V, lam_sub = lanczos_subsample(a, model, load_seqs(a, "train"), flat.numel(), a.k + a.k_extra)
    save_pt(dict(V=V, round=1, sig=basis_sig(V), lam_sub=torch.tensor(lam_sub.copy())), out_path(a, "basis.pt"))
    remove(out_path(a, "lanczos_Q.bin"), out_path(a, "lanczos_state.npz"))


def stage_exact(a):
    """Worker: rows of W = E_L[H] V over the whole training set for one slice of the basis.
    Run one worker per GPU with --part p --parts n; checkpointed."""
    if os.path.exists(out_path(a, "spectrum.pt")):
        return
    b = torch.load(out_path(a, "basis.pt"))
    rows = torch.arange(b["V"].shape[0]).chunk(a.parts)[a.part]
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    path = out_path(a, f"exact_r{b['round']}_p{a.part}of{a.parts}.pt")
    model, flat = load_model_flat(a, out_path(a, "w_star.pt"))
    tr = load_seqs(a, "train")
    if a.exact_seqs:
        tr = tr[:a.exact_seqs]
    N, chunk = len(tr), a.exact_chunk
    V = b["V"][lo:hi].to(DEV)
    W, i0 = torch.zeros_like(V), 0
    if os.path.exists(path):
        c = torch.load(path, map_location=DEV)
        if c["sig"] == b["sig"] and c["chunk"] == chunk and c["n"] == N and c["W"].shape == V.shape:
            W, i0 = c["W"], c["i"]
            log(f"exact round {b['round']} part {a.part}: resuming from sequence {i0}")
    log(f"exact round {b['round']} part {a.part}/{a.parts}: directions {lo}..{hi - 1}, {N} sequences")
    t_start, n_chunks = time.time(), 0
    for i in range(i0, N, chunk):
        s = tr[i:i + chunk]
        state = [0]                           # directions already added for this chunk; survives OOM retries

        def add_chunk():
            for j, hv in hv_rows(model, s, V, state[0]):
                W[j].add_(hv, alpha=len(s) / N)
                state[0] = j + 1

        retry_oom(add_chunk, "exact pass")
        n_chunks += 1
        last = i + chunk >= N
        if n_chunks % a.exact_ckpt == 0 or last:
            save_pt(dict(W=W, i=i + chunk, n=N, chunk=chunk, sig=b["sig"], lo=lo, hi=hi), path)
        if n_chunks % 50 == 0 or last:
            per = (time.time() - t_start) / n_chunks
            log(f"exact round {b['round']} part {a.part}: {min(i + chunk, N)}/{N} sequences  "
                f"{per:.2f}s per chunk  ETA {per * max(0, (N - i - chunk) / chunk) / 60:.0f} min")


def rayleigh_ritz(V, W):
    """Best eigenpairs of H inside span(V) given W = H V (CPU): Ritz values (descending),
    rotated V and W, relative residuals |H v - theta v| / |theta|."""
    M = (V @ W.T).double()
    theta, S = torch.linalg.eigh((M + M.T) / 2)
    order = torch.argsort(theta, descending=True)
    theta, S = theta[order], S[:, order].float()
    V2, W2 = S.T @ V, S.T @ W
    resid = (W2 - theta[:, None].float() * V2).norm(dim=1) / theta.abs().float()
    return theta, V2, W2, resid


def orthonormalize(W):
    """Rows of W -> orthonormal rows with the same span (Cholesky of the Gram matrix, twice)."""
    for _ in range(2):
        L = torch.linalg.cholesky((W @ W.T).double())
        W = torch.linalg.solve_triangular(L, torch.eye(len(L), dtype=L.dtype), upper=False).float() @ W
    return W


def stage_ritz(a):
    """Combine the exact-pass parts: Rayleigh-Ritz, residuals, then either start another
    round from H V (one exact subspace-iteration step) or write the final spectrum."""
    if os.path.exists(out_path(a, "spectrum.pt")):
        return
    b = torch.load(out_path(a, "basis.pt"))
    V, r = b["V"], b["round"]
    W, covered = torch.empty_like(V), 0
    parts = glob.glob(out_path(a, f"exact_r{r}_p*.pt"))
    for f in parts:
        c = torch.load(f, map_location="cpu")
        if c["sig"] == b["sig"] and c["i"] >= c["n"]:
            W[c["lo"]:c["hi"]] = c["W"]
            covered += c["hi"] - c["lo"]
    if covered != V.shape[0]:
        raise RuntimeError(f"exact round {r} incomplete: {covered}/{V.shape[0]} directions")
    theta, V2, W2, resid = rayleigh_ritz(V, W)
    res = resid[:a.k].numpy()
    hist_path = out_path(a, "spectrum_rounds.json")
    hist = load_json(hist_path) if os.path.exists(hist_path) else []
    hist = [h for h in hist if h["n"] < r] + [dict(
        n=r, resid_median=float(np.median(res)), resid_q90=float(np.quantile(res, 0.9)),
        resid_max=float(res.max()), lam_top=theta[0].item(), lam_k=theta[a.k - 1].item())]
    save_json(hist_path, hist)
    log(f"exact round {r}: lambda {theta[0].item():.2f} .. {theta[a.k - 1].item():.2f}; relative residual "
        f"median {np.median(res):.4f}, q90 {np.quantile(res, 0.9):.4f}, max {res.max():.4f}")
    if r < a.max_passes and np.quantile(res, 0.9) > a.resid_tol:
        # One exact subspace-iteration step: an admixture from a direction with curvature
        # lambda_f shrinks by lambda_f / lambda_i, so flat-direction contamination is removed.
        Vn = orthonormalize(W2)
        save_pt(dict(V=Vn, round=r + 1, sig=basis_sig(Vn), lam_sub=b["lam_sub"]), out_path(a, "basis.pt"))
        remove(*parts)
        return
    save_json(out_path(a, "spectrum.json"), dict(
        rounds=hist, lanczos_iters=a.lanczos_iters, hvp_seqs=a.hvp_seqs,
        lam=theta[:a.k].tolist(), resid=resid[:a.k].tolist(), lam_lanczos_sub=b["lam_sub"][:a.k].tolist()))
    save_pt(dict(lam=theta[:a.k], V=V2[:a.k].contiguous(), resid=resid[:a.k]), out_path(a, "spectrum.pt"))
    remove(out_path(a, "basis.pt"), *parts)


def stage_stats(a):
    if os.path.exists(out_path(a, "stats.json")):
        return
    model, flat = load_model_flat(a, out_path(a, "w_star.pt"))
    tr = load_seqs(a, "train")
    U, K = directions(a, flat.numel())
    V = U[:K]
    part = out_path(a, "stats_part.npz")
    G = np.zeros((a.m_grad, U.shape[0]))
    Hq = np.zeros((a.m_hess, K, K))                 # Hq[m, i, j] = v_i^T H_B v_j
    ng = nh = 0
    if os.path.exists(part):
        z = np.load(part)
        if z["G"].shape == G.shape and z["Hq"].shape == Hq.shape:
            G, Hq, ng, nh = z["G"], z["Hq"], int(z["ng"]), int(z["nh"])
            log(f"stats: resuming at {ng} gradient and {nh} Hessian batches")

    def save_part():
        np.savez(part + ".tmp.npz", G=G, Hq=Hq, ng=ng, nh=nh)
        os.replace(part + ".tmp.npz", part)

    base = a.seed * 1_000_003
    for m in range(ng, a.m_grad):
        idx = rand_idx(len(tr), a.batch, base + 7_000_000 + m)
        G[m] = retry_oom(lambda: project(U, batch_grad(model, tr, idx)).double().cpu().numpy(), "stats")
        ng = m + 1
        if ng % 1000 == 0 or ng == a.m_grad:
            save_part()
            log(f"stats: gradient batches {ng}/{a.m_grad}")
    for m in range(nh, a.m_hess):
        idx = rand_idx(len(tr), a.batch, base + 8_000_000 + m)
        Hq[m] = retry_oom(lambda: quad_forms(model, tr, idx, V, a.hvp_chunk, full=True).numpy(), "stats")
        nh = m + 1
        if nh % 5 == 0 or nh == a.m_hess:
            save_part()
            log(f"stats: Hessian batches {nh}/{a.m_hess}")

    g_star = torch.load(out_path(a, "grad_star.pt"), map_location=DEV)
    sp = load_json(out_path(a, "spectrum.json"))
    lam = np.array(sp["lam"])                       # exact Ritz values over the whole training set
    np.savez(out_path(a, "stats_raw.npz"), G=G, Hq=Hq, g_star_proj=project(U, g_star).double().cpu().numpy())
    d = G.var(axis=0, ddof=1)                       # gradient-noise variance per direction
    diag = np.einsum("mii->mi", Hq)
    Gamma = diag.var(axis=0, ddof=1)                # Var(v_i^T H_B v_i) at B = batch
    Kcov = np.cov(G[:, :K].T)
    H2 = (Hq ** 2).mean(0)
    top = min(20, K)
    summary = dict(
        batch=a.batch, m_grad=a.m_grad, m_hess=a.m_hess, K=K, n_rand=a.n_rand,
        lam=lam.tolist(), resid=sp["resid"],
        lam_mc=diag.mean(0).tolist(), lam_mc_se=(diag.std(0, ddof=1) / math.sqrt(a.m_hess)).tolist(),
        Gamma=Gamma.tolist(), EH2=(lam ** 2 + Gamma).tolist(),
        d=d.tolist(), d_se=(d * math.sqrt(2 / (a.m_grad - 1))).tolist(),
        grad_mean_proj=G.mean(0).tolist(),
        gamma_ls_top20=float((d[:top] * lam[:top]).sum() / (lam[:top] ** 2).sum()),
        gamma_ls_all=float((d[:K] * lam).sum() / (lam ** 2).sum()),
        grad_cov_offdiag_ratio=float(np.abs(Kcov - np.diag(np.diag(Kcov))).sum() / np.diag(Kcov).sum()),
        hess_second_moment_offdiag_ratio=float((H2.sum() - np.trace(H2)) / np.trace(H2)),
    )
    save_json(out_path(a, "stats.json"), summary)
    remove(part)
    log(f"stats: lambda in [{lam.min():.2f}, {lam.max():.2f}]  Gamma/lam^2 median {np.median(Gamma / lam ** 2):.2e}  "
        f"max |lam_mc/lam-1| {np.abs(diag.mean(0) / lam - 1).max():.3f}  gamma_LS(top20) {summary['gamma_ls_top20']:.3e}")


def stage_plan(a):
    """Learning rates for the trajectories, chosen from the measured spectrum:
    eta = c / lambda_max, so that eta * lambda_max spans a fixed grid below the stability
    limit (2) wherever training happened to end."""
    path = out_path(a, "traj_spec.json")
    if os.path.exists(path):
        return
    lam_max = load_json(out_path(a, "spectrum.json"))["lam"][0]
    cs = [float(x) for x in a.c_grid.split(",")]
    etas = [float(f"{c / lam_max:.2g}") for c in cs]
    save_json(path, dict(lam_max=lam_max, c=cs, eta=etas, steps=[int(x) for x in a.c_steps.split(",")]))
    log(f"plan: lambda_max = {lam_max:.2f} -> eta = {etas}  (eta*lambda_max = {cs})")


def traj_spec(a):
    """[(eta, steps), ...]: from traj_spec.json ('auto') or from '0.001:500,0.005:300'."""
    if a.traj_spec == "auto":
        s = load_json(out_path(a, "traj_spec.json"))
        return list(zip(s["eta"], s["steps"]))
    return [(float(e), int(s)) for e, s in (x.split(":") for x in a.traj_spec.split(","))]


def stage_traj(a):
    """Worker: SGD trajectories from w*, round-robin over the learning rates, so that at any
    moment all learning rates have (almost) the same number of trajectories. Each worker
    (--worker w) owns its own files and seeds; any number of workers can run in parallel."""
    model, flat = load_model_flat(a, out_path(a, "w_star.pt"))
    w_star = flat.detach().clone()
    ps = list(model.parameters())
    tr = load_seqs(a, "train")
    U, _ = directions(a, flat.numel())
    spec = traj_spec(a)
    store = {}
    for eta, steps in spec:
        path = out_path(a, f"traj_eta{eta:g}_w{a.worker}.npz")
        s = dict(path=path, proj=np.zeros((a.n_local, steps + 1, U.shape[0]), dtype=np.float32),
                 loss=np.zeros((a.n_local, steps), dtype=np.float32), done=0, dirty=False)
        if os.path.exists(path):
            z = np.load(path)
            if z["proj"].shape[1:] == s["proj"].shape[1:]:
                s["done"] = min(int(z["done"]), a.n_local)
                s["proj"][:s["done"]], s["loss"][:s["done"]] = z["proj"][:s["done"]], z["loss"][:s["done"]]
            else:
                log(f"traj eta={eta:g}: saved file has a different shape, starting over")
        store[eta] = s

    def save_all():
        for eta, s in store.items():
            if s["dirty"]:
                np.savez(s["path"] + ".tmp.npz", proj=s["proj"][:s["done"]], loss=s["loss"][:s["done"]],
                         done=s["done"], eta=eta)
                os.replace(s["path"] + ".tmp.npz", s["path"])
                s["dirty"] = False

    log(f"traj worker {a.worker}: target {a.n_local} per learning rate, "
        f"already done {[store[e]['done'] for e, _ in spec]}")
    t_start, n_run = time.time(), 0
    for t in range(a.n_local):
        for eta, steps in spec:
            s = store[eta]
            if s["done"] > t:
                continue
            flat.copy_(w_star)
            gen = torch.Generator(device=DEV).manual_seed(
                a.seed * 1_000_003 + int(round(eta * 1e6)) * 10_007 + a.worker * 1_000_000 + t)
            P_t = torch.zeros(steps + 1, U.shape[0], device=DEV)
            L_t = torch.zeros(steps, device=DEV)
            for n in range(steps):
                idx = torch.randint(len(tr), (a.batch,), device=DEV, generator=gen)
                loss, grads = retry_oom(lambda: loss_and_grads(model, tr, idx), "traj")
                with torch.no_grad():
                    torch._foreach_add_(ps, grads, alpha=-eta)
                    P_t[n + 1] = project(U, flat - w_star)
                    L_t[n] = loss
            s["proj"][t], s["loss"][t] = P_t.cpu().numpy(), L_t.cpu().numpy()
            s["done"], s["dirty"] = t + 1, True
            n_run += 1
            if a.worker == 0 and t < a.n_end:
                save_pt(flat.detach().cpu().clone(), out_path(a, f"traj_end_eta{eta:g}_{t}.pt"))
        if (t + 1) % a.save_every == 0 and n_run:
            save_all()
            per = (time.time() - t_start) / n_run * len(spec)
            log(f"traj worker {a.worker}: {t + 1}/{a.n_local} per learning rate  {per:.1f}s per round  "
                f"ETA {per * (a.n_local - t - 1) / 60:.0f} min")
    save_all()
    log(f"traj worker {a.worker}: finished, done {[store[e]['done'] for e, _ in spec]}")


def stage_drift(a):
    if os.path.exists(out_path(a, "drift.json")):
        return
    model, flat = load_model_flat(a, out_path(a, "w_star.pt"))
    w_star = flat.detach().clone()
    tr = load_seqs(a, "train")
    U, K = directions(a, flat.numel())
    V = U[:K]
    st = load_json(out_path(a, "stats.json"))
    lam0, d0 = np.array(st["lam"]), np.array(st["d"])
    part = out_path(a, "drift_part.json")
    out = load_json(part) if os.path.exists(part) else {}
    base = a.seed * 1_000_003 + 9_000_000
    for f in sorted(glob.glob(out_path(a, "traj_end_eta*_*.pt"))):
        eta_s, e = os.path.basename(f)[len("traj_end_eta"):-3].rsplit("_", 1)
        eta, e = float(eta_s), int(e)
        rows = out.setdefault(eta_s, [])
        w_end = torch.load(f)
        if any(r["idx"] == e for r in rows) or not torch.isfinite(w_end).all():
            continue
        flat.copy_(w_end.to(DEV))
        sd = base + int(round(eta * 1e6)) * 101 + e * 100_000
        G = np.stack([retry_oom(lambda: project(U, batch_grad(model, tr, rand_idx(
            len(tr), a.batch, sd + m))).double().cpu().numpy(), "drift") for m in range(a.m_grad_end)])
        q = sum(retry_oom(lambda: quad_forms(model, tr, rand_idx(len(tr), a.batch, sd + 50_000 + m),
                                             V, a.hvp_chunk, full=False), "drift")
                for m in range(a.m_hess_end)).numpy() / a.m_hess_end
        d_ratio, lam_ratio = G.var(axis=0, ddof=1) / d0, q / lam0
        # plateau ~ eta d / (2 lam - eta lam^2): its relative change if the end point is used
        plateau_ratio = d_ratio[:K] * (2 * lam0 - eta * lam0 ** 2) / (2 * q - eta * q ** 2)
        rows.append(dict(idx=e, dist=(flat - w_star).norm().item(),
                         d_ratio_sharp_mean=float(d_ratio[:K].mean()),
                         d_ratio_random_mean=float(d_ratio[K:].mean()),
                         lam_ratio_mean=float(lam_ratio.mean()),
                         lam_ratio_range=[float(lam_ratio.min()), float(lam_ratio.max())],
                         plateau_change_mean=float(plateau_ratio.mean() - 1)))
        save_json(part, out)
        log(f"drift eta={eta_s} #{e}: |w-w*|={rows[-1]['dist']:.3f}  d ratio {rows[-1]['d_ratio_sharp_mean']:.3f}  "
            f"lambda ratio {rows[-1]['lam_ratio_mean']:.3f}  plateau change {rows[-1]['plateau_change_mean']:+.3f}")
    flat.copy_(w_star)
    save_json(out_path(a, "drift.json"), out)
    remove(part)


def stage_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = load_json(out_path(a, "stats.json"))
    sp = load_json(out_path(a, "spectrum.json"))
    lam = np.array(st["lam"])
    K = len(lam)
    Gam = np.array(st["Gamma"])
    EH2 = lam ** 2 + Gam
    d = np.array(st["d"])
    dK, dR = d[:K], d[K:]
    ntop = min(20, K)
    rng = np.random.default_rng(a.seed)

    parts = {}
    for f in glob.glob(out_path(a, "traj_eta*_w*.npz")):
        if f.endswith(".tmp.npz"):          # a worker's file in the middle of being replaced
            continue
        z = np.load(f)
        if int(z["done"]):
            parts.setdefault(float(z["eta"]), []).append(z["proj"][:int(z["done"])].astype(np.float64))
    runs, dropped = {}, {}
    for eta, ps_ in parts.items():
        p = np.concatenate(ps_)
        ok = np.isfinite(p).all(axis=(1, 2))
        ok[ok] &= np.abs(p[ok]).max(axis=(1, 2)) < 1e3
        if ok.sum() >= 2:
            runs[eta], dropped[eta] = p[ok], int((~ok).sum())
    etas = sorted(runs)
    eta0 = etas[0]

    def window_pred(eta, dvec, T1, W):
        """Window-mean of the discrete (Eq. key) and Langevin variance formulas."""
        n = np.arange(T1 - W, T1)[:, None]
        a_i = 1 - 2 * eta * lam + eta ** 2 * EH2
        disc = (eta ** 2 * dvec * (1 - a_i ** n) / (1 - a_i)).mean(0)
        kl = 2 * lam - eta * Gam
        lang = (eta * dvec / kl * (1 - np.exp(-kl * eta * n))).mean(0)
        return disc, lang

    def bars(c, lo, hi):   # error bars; a bootstrap percentile can fall on the wrong side of c
        return [np.maximum(c - lo, 0), np.maximum(hi - c, 0)]

    res = {}
    for eta in etas:
        proj = runs[eta]
        N, T1, _ = proj.shape
        W = min(a.plateau_window, T1 - 1)
        var = proj.var(axis=0, ddof=1)
        tail = proj[:, -W:, :K]
        boots = np.empty((a.n_boot, K))
        for b in range(a.n_boot):
            boots[b] = tail[rng.integers(0, N, N)].var(axis=0, ddof=1).mean(0)
        disc, lang = window_pred(eta, dK, T1, W)
        h = W // 2
        res[eta] = dict(N=N, T1=T1, W=W, var=var, emp=var[-W:, :K].mean(0), boots=boots,
                        lo=np.percentile(boots, 5, 0), hi=np.percentile(boots, 95, 0),
                        disc=disc, lang=lang,
                        creep=float((var[-h:, :K].mean(0) / var[-W:-h, :K].mean(0)).mean()))

    # One-constant calibration (d_i = gamma * lam_i): gamma from the top-20 plateau at the smallest eta.
    gamma_hat = 2 / eta0 * res[eta0]["emp"][:ntop].mean()
    for eta in etas:
        r = res[eta]
        r["disc_g"], r["lang_g"] = window_pred(eta, gamma_hat * lam, r["T1"], r["W"])

    def metrics(r, key, sl=slice(None)):
        p, e = r[key][sl], r["emp"][sl]
        return dict(mean_ratio_pred_over_emp=float(np.mean(p / e)),
                    median_rel_err=float(np.median(np.abs(p - e) / e)),
                    frac_in_ci=float(np.mean((p >= r["lo"][sl]) & (p <= r["hi"][sl]))))

    def slope(y):  # least squares through the origin of y vs lambda
        return (y * lam).sum(-1) / (lam ** 2).sum()

    keys = ("disc", "lang", "disc_g", "lang_g")
    drift = load_json(out_path(a, "drift.json")) if os.path.exists(out_path(a, "drift.json")) else {}
    summary = dict(eta0=eta0, gamma_hat=gamma_hat, gamma_ls_top20=st["gamma_ls_top20"], K=K,
                   lam_range=[float(lam.min()), float(lam.max())], spectrum_rounds=sp["rounds"],
                   Gamma_over_lam2_median=float(np.median(Gam / lam ** 2)), per_eta={})
    r0 = res[eta0]
    for eta in etas:
        r = res[eta]
        mb = r["boots"][:, :ntop].mean(1)
        e = dict(N=r["N"], dropped=dropped[eta], T=r["T1"] - 1, window=r["W"],
                 creep_second_half_over_first=r["creep"],
                 top20_mean=float(r["emp"][:ntop].mean()),
                 top20_mean_ci=[float(np.percentile(mb, 5)), float(np.percentile(mb, 95))],
                 metrics_all={k: metrics(r, k) for k in keys},
                 metrics_top20={k: metrics(r, k, slice(0, ntop)) for k in keys})
        if eta != eta0:
            # gamma-free test: growth of (window variance / eta) relative to eta0, per direction
            R_emp = (r["emp"] / eta) / (r0["emp"] / eta0)
            R_b = (r["boots"] / eta) / (r0["boots"] / eta0)
            R_disc = (r["disc"] / eta) / (r0["disc"] / eta0)
            R_lang = (r["lang"] / eta) / (r0["lang"] / eta0)
            s_emp, s_b = slope(R_emp - 1), slope(R_b - 1)
            s_disc, s_lang = slope(R_disc - 1), slope(R_lang - 1)
            r.update(R_emp=R_emp, R_lo=np.percentile(R_b, 5, 0), R_hi=np.percentile(R_b, 95, 0),
                     R_disc=R_disc, R_lang=R_lang)
            e["ratio_test"] = dict(
                top20_empirical=float(R_emp[:ntop].mean()),
                top20_empirical_ci=[float(np.percentile(R_b[:, :ntop].mean(1), q)) for q in (5, 95)],
                top20_discrete=float(R_disc[:ntop].mean()), top20_langevin=float(R_lang[:ntop].mean()),
                slope_empirical=float(s_emp), slope_ci=[float(np.percentile(s_b, q)) for q in (5, 95)],
                slope_discrete=float(s_disc), slope_langevin=float(s_lang),
                effect_fraction_of_discrete=float((s_emp - s_lang) / (s_disc - s_lang)),
                effect_fraction_ci=[float(np.percentile((s_b - s_lang) / (s_disc - s_lang), q)) for q in (5, 95)])
        n = np.arange(r["T1"])
        vr = r["var"][:, K:]
        e["random_dirs_slope_over_eta2_d_median"] = float(np.median(
            (vr * n[:, None]).sum(0) / (n ** 2).sum() / (eta ** 2 * dR)))
        # The gradient noise along the sharp directions is larger away from w* (the term
        # sum_k Gamma_ik Pi_kk that Eq. key drops). Its measured value at trajectory end points
        # gives a corrected discrete prediction.
        dr = [row["d_ratio_sharp_mean"] for row in drift.get(f"{eta:g}", [])]
        if dr:
            c = float(np.mean(dr))
            e["end_point_noise"] = dict(
                d_ratio=c, n_end_points=len(dr),
                discrete_pred_over_emp_all=float(np.mean(c * r["disc"] / r["emp"])),
                discrete_pred_over_emp_top20=float(np.mean(c * r["disc"][:ntop] / r["emp"][:ntop])),
                langevin_pred_over_emp_all=float(np.mean(c * r["lang"] / r["emp"])))
        summary["per_eta"][f"{eta:g}"] = e
    if drift:
        summary["drift"] = drift
    if os.path.exists(out_path(a, "finish_log.json")):
        summary["finish_log"] = load_json(out_path(a, "finish_log.json"))
    save_json(out_path(a, "summary.json"), summary)

    # ---- figures
    fig, axs = plt.subplots(1, len(etas), figsize=(5 * len(etas), 4), squeeze=False)
    for ax, eta in zip(axs[0], etas):
        r = res[eta]
        ax.errorbar(lam, r["emp"], yerr=bars(r["emp"], r["lo"], r["hi"]),
                    fmt="o", color="k", ms=3, capsize=1.5, label="empirical (90% CI)")
        ax.plot(lam, r["disc"], "x", color="C0", label="discrete, measured $d_i$")
        ax.plot(lam, r["lang"], "D", mfc="none", ms=4, color="C3", label="Langevin, measured $d_i$")
        ax.plot(lam, r["disc_g"], "+", color="C2", label=r"discrete, $d_i=\hat\gamma\lambda_i$")
        ax.set_title(f"$\\eta$={eta:g}, N={r['N']}, window {r['W']} steps")
        ax.set_xlabel(r"$\lambda_i$")
    axs[0][0].set_ylabel(r"window-mean variance $\hat\Pi_{ii}$")
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path(a, "fig_plateau_vs_lambda.png"), dpi=150)

    others = [e for e in etas if e != eta0]
    if others:
        fig, axs = plt.subplots(1, len(others), figsize=(5 * len(others), 4), squeeze=False)
        for ax, eta in zip(axs[0], others):
            r = res[eta]
            o = np.argsort(lam)
            ax.errorbar(lam, r["R_emp"] - 1, yerr=bars(r["R_emp"], r["R_lo"], r["R_hi"]),
                        fmt="o", color="k", ms=3, capsize=1.5, label="empirical (90% CI)")
            ax.plot(lam[o], r["R_disc"][o] - 1, "-", color="C0", label="discrete (Eq. key)")
            ax.plot(lam[o], r["R_lang"][o] - 1, "--", color="C3", label="Langevin")
            rt = summary["per_eta"][f"{eta:g}"]["ratio_test"]
            ax.set_title(f"$\\eta$={eta:g} vs {eta0:g}: effect = {rt['effect_fraction_of_discrete']:.2f} "
                         f"[{rt['effect_fraction_ci'][0]:.2f}, {rt['effect_fraction_ci'][1]:.2f}] of discrete",
                         fontsize=9)
            ax.set_xlabel(r"$\lambda_i$")
        axs[0][0].set_ylabel(r"$R_i-1$, $R_i=\frac{\hat\Pi_{ii}(\eta)/\eta}{\hat\Pi_{ii}(\eta_0)/\eta_0}$")
        axs[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path(a, "fig_dose_response.png"), dpi=150)

    show = [0, K // 2, K - 1]
    fig, axs = plt.subplots(len(etas), len(show), figsize=(4 * len(show), 3 * len(etas)), squeeze=False)
    for row, eta in zip(axs, etas):
        r = res[eta]
        n = np.arange(r["T1"])
        for ax, i in zip(row, show):
            a_i = 1 - 2 * eta * lam[i] + eta ** 2 * EH2[i]
            kl = 2 * lam[i] - eta * Gam[i]
            ax.plot(n, r["var"][:, i], color="k", lw=1, label="empirical")
            ax.plot(n, eta ** 2 * dK[i] * (1 - a_i ** n) / (1 - a_i), "--", color="C0", label="discrete")
            ax.plot(n, eta * dK[i] / kl * (1 - np.exp(-kl * eta * n)), ":", color="C3", label="Langevin")
            ax.set_title(f"$\\eta$={eta:g}, $\\lambda_{{{i + 1}}}$={lam[i]:.1f}", fontsize=9)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path(a, "fig_dynamics.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    e_arr = np.array(etas)
    mp = np.array([res[e]["emp"][:ntop].mean() for e in etas])
    ci = np.array([summary["per_eta"][f"{e:g}"]["top20_mean_ci"] for e in etas])
    ax.errorbar(e_arr, mp, yerr=bars(mp, ci[:, 0], ci[:, 1]), fmt="o", color="k", label="empirical, top 20")
    eg = np.linspace(0, e_arr.max() * 1.05, 100)
    ax.plot(eg, 0.5 * gamma_hat * eg, ":", color="gray", label=r"$\frac{1}{2}\eta\hat\gamma$")
    ax.plot(e_arr, [res[e]["disc"][:ntop].mean() for e in etas], "x-", color="C0", label="discrete")
    ax.plot(e_arr, [res[e]["lang"][:ntop].mean() for e in etas], "D-", mfc="none", color="C3", label="Langevin")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("window-mean variance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path(a, "fig_eta_scaling.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    for eta in etas:
        n = np.arange(res[eta]["T1"])
        ax.plot(n, res[eta]["var"][:, K:].mean(1) / (eta ** 2 * dR.mean()), label=f"$\\eta$={eta:g}")
    n = np.arange(max(res[e]["T1"] for e in etas))
    ax.plot(n, n, "k--", lw=1, label=r"$\Pi_n=\eta^2 d\,n$")
    ax.set_xlabel("step n")
    ax.set_ylabel(r"$\hat\Pi_n/(\eta^2 d)$, random directions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path(a, "fig_random_dirs.png"), dpi=150)
    plt.close("all")

    lines = [f"K={K} directions, exact lambda in [{lam.min():.2f}, {lam.max():.2f}];  "
             f"Gamma/lam^2 median {np.median(Gam / lam ** 2):.2e}",
             "spectrum: " + "; ".join(f"round {p['n']}: residual median {p['resid_median']:.4f} "
                                      f"q90 {p['resid_q90']:.4f} max {p['resid_max']:.4f}" for p in sp["rounds"]),
             f"gamma_hat(top20 plateau, eta={eta0:g}) = {gamma_hat:.3e},  gamma_LS(top20) = "
             f"{st['gamma_ls_top20']:.3e}"]
    lines += [f"finishing | {h['tag']}: full loss {h['loss']:.4f}  |grad| {h['grad_norm']:.3e}  "
              f"lambda_max ~ {h['lam_top'][0]:.1f}" for h in summary.get("finish_log", [])] + [""]
    for eta in etas:
        e = summary["per_eta"][f"{eta:g}"]
        lines.append(f"eta={eta:g} (eta*lambda_max={eta * lam.max():.2f})  N={e['N']} (dropped {e['dropped']})  "
                     f"T={e['T']}  window={e['window']}  "
                     f"top-20 mean variance {e['top20_mean']:.3e} [{e['top20_mean_ci'][0]:.3e}, "
                     f"{e['top20_mean_ci'][1]:.3e}]  creep(2nd/1st half) {e['creep_second_half_over_first']:.3f}")
        for k, name in (("disc", "discrete (measured d)"), ("lang", "Langevin (measured d)"),
                        ("disc_g", "discrete (gamma)"), ("lang_g", "Langevin (gamma)")):
            m, m20 = e["metrics_all"][k], e["metrics_top20"][k]
            lines.append(f"    {name:24s} pred/emp all {m['mean_ratio_pred_over_emp']:.3f} top20 "
                         f"{m20['mean_ratio_pred_over_emp']:.3f} | median rel err {m['median_rel_err']:.3f} "
                         f"| in 90% CI {m['frac_in_ci']:.2f}")
        if "ratio_test" in e:
            rt = e["ratio_test"]
            lines.append(f"    ratio test top20: empirical {rt['top20_empirical']:.3f} "
                         f"[{rt['top20_empirical_ci'][0]:.3f}, {rt['top20_empirical_ci'][1]:.3f}]  "
                         f"discrete {rt['top20_discrete']:.3f}  Langevin {rt['top20_langevin']:.3f}")
            lines.append(f"    dose-response slope of (R-1) vs lambda: empirical {rt['slope_empirical']:.2e} "
                         f"[{rt['slope_ci'][0]:.2e}, {rt['slope_ci'][1]:.2e}]  discrete {rt['slope_discrete']:.2e}  "
                         f"Langevin {rt['slope_langevin']:.2e}  ->  observed effect = "
                         f"{rt['effect_fraction_of_discrete']:.2f} [{rt['effect_fraction_ci'][0]:.2f}, "
                         f"{rt['effect_fraction_ci'][1]:.2f}] of the discrete prediction")
        if "end_point_noise" in e:
            en = e["end_point_noise"]
            lines.append(f"    with the noise measured at end points (d ratio {en['d_ratio']:.3f}, "
                         f"{en['n_end_points']} points): discrete pred/emp all {en['discrete_pred_over_emp_all']:.3f} "
                         f"top20 {en['discrete_pred_over_emp_top20']:.3f} | Langevin all "
                         f"{en['langevin_pred_over_emp_all']:.3f}")
        lines.append(f"    random dirs: slope/(eta^2 d) median {e['random_dirs_slope_over_eta2_d_median']:.3f}")
    for eta, rows in summary.get("drift", {}).items():
        for row in rows:
            lines.append(f"drift eta={eta}: |w_T-w*|={row['dist']:.3f}  d ratio {row['d_ratio_sharp_mean']:.3f}  "
                         f"lambda ratio {row['lam_ratio_mean']:.3f}  implied plateau change "
                         f"{row['plateau_change_mean']:+.3f}")
    report = "\n".join(lines)
    with open(out_path(a, "report.txt"), "w") as f:
        f.write(report + "\n")
    print(report, flush=True)


STAGES = dict(data=stage_data, train=stage_train, anneal=stage_anneal, refine=stage_refine,
              refine_ref=stage_refine_ref, lanczos=stage_lanczos, plan=stage_plan,
              exact=stage_exact, ritz=stage_ritz, stats=stage_stats, traj=stage_traj,
              drift=stage_drift, analyze=stage_analyze)


def main():
    global USE_TF32, OOM_TRIES, OOM_WAIT, STEP_CHUNK
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/run1")
    p.add_argument("--stages", required=True, help="comma-separated, from: " + ",".join(STAGES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", type=int, default=1, help="TF32 matmuls (projections always fp32)")
    p.add_argument("--oom_tries", type=int, default=10)
    p.add_argument("--oom_wait", type=int, default=60, help="seconds to wait after a CUDA OOM")
    # model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--mlp_ratio", type=int, default=4)
    # training: plain SGD with a constant learning rate and a fixed number of steps
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--val_every", type=int, default=1000, help="validation loss every this many steps")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--step_chunk", type=int, default=32, help="sequences per backward pass inside a step")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--ckpt_steps", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=32, help="sequences per chunk for full-data passes")
    # low-noise SGD (lr_multiplier:batch:steps phases), then full-gradient steps
    p.add_argument("--anneal", default="0.4:256:3000,0.4:1024:2000",
                   help="low-noise phases, lr_multiplier:batch:steps (multiplier of the training lr)")
    p.add_argument("--anneal_block", type=int, default=100, help="steps between checkpoints")
    p.add_argument("--diag_seqs", type=int, default=512, help="sequences for the lambda_max diagnostic")
    p.add_argument("--diag_iters", type=int, default=30)
    p.add_argument("--refine_steps", type=int, default=100)
    p.add_argument("--refine_lr_mult", type=float, default=0.2, help="GD lr = this * training lr")
    # spectrum: Lanczos on a subsample, then exact passes over the whole training set
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--k_extra", type=int, default=10, help="buffer vectors carried through the exact passes")
    p.add_argument("--lanczos_iters", type=int, default=500)
    p.add_argument("--hvp_chunk", type=int, default=32)
    p.add_argument("--hvp_seqs", type=int, default=1024, help="fixed subsample for Lanczos; 0 = all")
    p.add_argument("--exact_chunk", type=int, default=32)
    p.add_argument("--exact_ckpt", type=int, default=30, help="chunks between exact-pass checkpoints")
    p.add_argument("--exact_seqs", type=int, default=0, help="testing only: limit the exact pass")
    p.add_argument("--part", type=int, default=0, help="exact stage: which slice of the basis")
    p.add_argument("--parts", type=int, default=1, help="exact stage: number of slices (one per GPU)")
    p.add_argument("--resid_tol", type=float, default=0.01, help="q90 residual above this -> another round")
    p.add_argument("--max_passes", type=int, default=3)
    # noise statistics at w*
    p.add_argument("--m_grad", type=int, default=8000)
    p.add_argument("--m_hess", type=int, default=100)
    p.add_argument("--n_rand", type=int, default=20, help="random unit directions as a flat-ish control")
    # trajectories
    p.add_argument("--traj_spec", default="auto",
                   help="auto (eta = c / lambda_max, see --c_grid) or explicit eta:steps pairs")
    p.add_argument("--c_grid", default="0.05,0.1,0.25,0.5,1.0,1.5", help="values of eta * lambda_max")
    p.add_argument("--c_steps", default="500,500,300,300,300,300", help="trajectory length for each of them")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n_local", type=int, default=240, help="trajectories per learning rate for this worker")
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--n_end", type=int, default=4, help="trajectory end points kept for the drift check")
    # drift check at end points
    p.add_argument("--m_grad_end", type=int, default=1000)
    p.add_argument("--m_hess_end", type=int, default=10)
    # analysis
    p.add_argument("--plateau_window", type=int, default=200)
    p.add_argument("--n_boot", type=int, default=1000)
    a = p.parse_args()
    USE_TF32, OOM_TRIES, OOM_WAIT, STEP_CHUNK = bool(a.tf32), a.oom_tries, a.oom_wait, a.step_chunk
    set_tf32(USE_TF32)
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    stages = a.stages.split(",")
    log(f"device {DEV} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}), tf32 {USE_TF32}, "
        f"stages {stages}, out {a.out}")
    for s in stages:
        log(f"=== stage {s}")
        STAGES[s](a)
    log("done")


if __name__ == "__main__":
    main()
