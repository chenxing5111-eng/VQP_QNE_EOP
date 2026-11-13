# -*- coding: utf-8 -*-
"""
Converged-oriented DIRECT E_p reference (no VQA, no QNE)
===========================================================
Original paper-code baseline: exact spectral purification + SU(4) / SU(16)
auxiliary unitary optimization. The original optimizer used finite multistart
Adam with 15 or 255 real parameters, and could stop at poor local minima.

This script uses the MATHEMATICALLY EQUIVALENT set of pure-state extensions:
    rho_AB = K K^dagger, K=(eigvecs)*sqrt(eigvals), shape=(4,4)
    Psi_AB,aux = K W^T,   W^dagger W = I_4.
W has shape D x 4, where D=dim(A')*dim(B')=d*d, d=2 or d=4.
Any such W extends to a D x D unitary; global phase does not affect entropy.
For d=2 this is equivalent to the full SU(4) search; for d=4 it is
 equivalent to the full SU(16) search, with only the 4 occupied columns.

Crucial numerical safeguards:
 - Use the SAME 100 Haar-random 1TN states as the supplied code (fixed seeds).
 - Exact purification throughout, not a learned approximation.
 - L-BFGS-B using Torch autograd and a Cholesky/QR-type orthogonalization.
 - Many random starts and best-solution perturbation refinements.
 - For d=4, warm start from the embedded optimized d=2 solution.
 - For d=4, include analytically valid E_p <= S(rho_A), S(rho_B) seeds.
 - Explicit verification that E_direct(d=4) <= E_direct(d=2), and
   E_direct(d=4) <= min(S_A,S_B), up to numeric tolerance.
 - Save per-state checkpoints, per-start diagnostics, and convergence status.

No nonconvex algorithm can PROVE global convergence. The code reports
approximate LOCAL stationarity, repeated-start stability, and exact physical
checks. Never claim the result is mathematically the global E_p without
an additional certificate.

Install:
    pip install numpy scipy pandas torch

Full experiment:
    python direct_ep_refined_reference.py

Quick test:
    python direct_ep_refined_reference.py --states 1:3 --restarts2 4 --restarts4 5 --maxiter 400 --polish 1

Run only problematic states (still computes dim2 first to warm-start dim4):
    python direct_ep_refined_reference.py --states 2,74,78

All outputs are saved in --outdir (default direct_reference_refined).
When settings change for the same ids, use --force to recompute them.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

# Small dense matrices: multithreading often adds significant overhead.
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
NUM_STATES = 100
STATE_SEED_BASE = 20260918
DOUBLE = torch.float64
CDOUBLE = torch.complex128


def parse_state_ids(spec):
    if spec == "all":
        return list(range(1, NUM_STATES + 1))
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("Use --states 1:100 or --states 1,2,3")
        lo, hi = map(int, parts)
        ids = list(range(lo, hi + 1))
    else:
        ids = sorted(set(int(v) for v in spec.split(",")))
    if not ids or any(i < 1 or i > NUM_STATES for i in ids):
        raise ValueError("State IDs must be within 1..100")
    return ids


def rho_for_state(state_id):
    """Exactly reproduce generate_1tn_random_state in the uploaded code."""
    rng = np.random.default_rng(STATE_SEED_BASE + int(state_id))
    z = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi_abc = z / np.linalg.norm(z)
    mat = psi_abc.reshape(4, 4)
    rho = mat @ mat.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    return rho


def entropy(rho):
    eigs = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real
    eigs = np.maximum(eigs, 0.)
    eigs /= eigs.sum()
    nz = eigs[eigs > 0]
    return float(-np.dot(nz, np.log2(nz)))


def theoretical_upper(rho):
    rt = rho.reshape(2, 2, 2, 2)
    rho_a = np.trace(rt, axis1=1, axis2=3)
    rho_b = np.trace(rt, axis1=0, axis2=2)
    return entropy(rho_a), entropy(rho_b)


def spectral_factor(rho):
    vals, vecs = np.linalg.eigh(rho)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx].real, 0.0)
    vals /= vals.sum()
    return vecs[:, idx] * np.sqrt(vals)[None, :]


def to_flat(y):
    return np.concatenate([y.real.ravel(), y.imag.ravel()]).astype(np.float64)


def from_flat(v, D):
    n = D * 4
    return (v[:n] + 1j * v[n:]).reshape(D, 4)


def orthonormalize(y):
    q, r = np.linalg.qr(y, mode="reduced")
    diag = np.diag(r)
    phase = np.where(np.abs(diag) > 1e-14, diag / np.maximum(np.abs(diag), 1e-14), 1)
    return q * phase[None, :]


def direct_entropy_of_y(K, y, d):
    """Pure state coefficients in [A,B,A',B'] order, trace B,B'."""
    psi_abaux = K @ y.T  # shape=(4,d*d)
    mm = psi_abaux.reshape(2, 2, d, d).transpose(0, 2, 1, 3)
    mm = mm.reshape(2 * d, 2 * d)  # rows=(A,A'), columns=(B,B')
    rr = mm @ mm.conj().T
    return entropy(rr)


def validate_purification(K, y, d):
    rho_orig = K @ K.conj().T
    psi = K @ y.T
    rho_back = psi @ psi.conj().T
    hs2 = np.linalg.norm(rho_back - rho_orig, "fro") ** 2
    ortho = np.linalg.norm(y.conj().T @ y - np.eye(4), "fro")
    return float(hs2), float(ortho)


def random_isometry(d, rng):
    D = d*d
    X = rng.standard_normal((D, 4)) + 1j * rng.standard_normal((D, 4))
    return orthonormalize(X)


def embed_dim2_into_dim4(y2):
    """Embed A' and B' qubits separately: |a,b> -> |a,b> with a,b in {0,1}."""
    y4 = np.zeros((16, 4), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            y4[4*a + b] = y2[2*a + b]
    return y4


def canonical_start(d):
    return np.eye(d*d, 4, dtype=np.complex128)


def upper_seeds_d4():
    """Explicit exact purifications giving S_A and S_B respectively."""
    y_a = np.zeros((16, 4), dtype=np.complex128)  # A'=0, B'=i
    y_b = np.zeros((16, 4), dtype=np.complex128)  # A'=i, B'=0
    for i in range(4):
        y_a[i, i] = 1.
        y_b[i * 4, i] = 1.
    return y_a, y_b


class DirectLoss:
    """Differentiable projection from unconstrained X onto 4-column isometries."""
    def __init__(self, K, d):
        self.K = torch.tensor(K, dtype=CDOUBLE)
        self.d = int(d)
        self.D = self.d * self.d
        self._eye = torch.eye(4, dtype=CDOUBLE)
        self.calls = 0

    def __call__(self, raw):
        self.calls += 1
        v = torch.tensor(raw, dtype=DOUBLE, requires_grad=True)
        n = self.D * 4
        X = torch.complex(v[:n].reshape(self.D, 4),
                          v[n:].reshape(self.D, 4))
        try:
            gram = X.conj().T @ X
            # Cholesky-based orthonormalization: Y = X @ upper^{-1}.
            upper = torch.linalg.cholesky(gram).conj().T
            Y = torch.linalg.solve_triangular(upper.T, X.T,
                                               upper=False, left=True).T
            psi = self.K @ Y.T
            mm = psi.reshape(2, 2, self.d, self.d).permute(0, 2, 1, 3)
            mm = mm.reshape(2*self.d, 2*self.d)
            vals = torch.linalg.eigvalsh(mm @ mm.conj().T).real
            vpos = torch.clamp(vals, min=0.0)
            # Smooth-enough finite precision extension at zero eigenvalues.
            loss = -(vpos * torch.log2(torch.clamp(vpos, min=1e-13))).sum()
            loss.backward()
            value = float(loss.detach())
            grad = v.grad.detach().numpy().astype(np.float64, copy=True)
            if np.isfinite(value) and np.all(np.isfinite(grad)):
                return value, grad
        except (RuntimeError, ValueError):
            pass
        # Safe fallback for invalid trial point (ill-conditioned X).
        xnp = from_flat(raw, self.D)
        penalty = 1e4 + float(np.linalg.norm(xnp.conj().T @ xnp - np.eye(4))**2)
        return penalty, np.zeros_like(raw, dtype=np.float64)


def run_one_start(K, d, y0, label, maxiter, ftol, gtol):
    loss = DirectLoss(K, d)
    initial = direct_entropy_of_y(K, y0, d)
    tic = time.perf_counter()
    res = minimize(
        loss, to_flat(y0), method="L-BFGS-B", jac=True,
        options=dict(maxiter=maxiter, ftol=ftol, gtol=gtol,
                     maxls=60, maxcor=20),
    )
    candidate = orthonormalize(from_flat(res.x, d*d))
    end = direct_entropy_of_y(K, candidate, d)
    # NEVER let numerical optimization undo a valid better starting candidate.
    if (not np.isfinite(end)) or end > initial:
        candidate = y0.copy()
        end = initial
    hs2, ortho = validate_purification(K, candidate, d)
    return candidate, dict(
        start=label, dim_ap=d, ep_initial=float(initial),
        ep_final=float(end), nit=int(res.nit),
        nfev=int(res.nfev), success=bool(res.success),
        stopping_message=str(res.message),
        gradient_norm=float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
        purification_hs2=hs2, orthogonality_error=ortho,
        seconds=float(time.perf_counter()-tic),
    )


def optimize_baseline(rho, state_id, d, rng, *, warm_y2=None,
                      random_starts=8, polish=3,
                      maxiter=1200, ftol=1e-13, gtol=1e-8):
    K = spectral_factor(rho)
    seeds = [("spectral_identity", canonical_start(d))]
    if d == 4:
        ya, yb = upper_seeds_d4()
        seeds += [("exact_SA_seed", ya), ("exact_SB_seed", yb)]
        if warm_y2 is not None:
            seeds.insert(0, ("embedded_dim2_best", embed_dim2_into_dim4(warm_y2)))
    for j in range(random_starts):
        seeds.append((f"random_{j+1}", random_isometry(d, rng)))

    # Valid initial value is retained even if optimizer fails to progress.
    best_label, best_y = min(seeds,
        key=lambda item: direct_entropy_of_y(K, item[1], d))
    best = direct_entropy_of_y(K, best_y, d)
    trace = []
    for label, y0 in seeds:
        yy, info = run_one_start(K,d,y0,label,maxiter,ftol,gtol)
        info["state_id"] = state_id
        trace.append(info)
        if info["ep_final"] < best - 1e-12:
            best, best_y, best_label = info["ep_final"], yy, label

    # Escape occasional local minima with perturb-and-refine from global best.
    for j in range(polish):
        sigma = [0.03, 0.12, 0.30][j % 3]
        noise = rng.normal(size=best_y.shape) + 1j*rng.normal(size=best_y.shape)
        y0 = orthonormalize(best_y + sigma * noise / np.sqrt(d*d))
        yy, info = run_one_start(K,d,y0,f"polish_{j+1}_sigma{sigma}",maxiter,ftol,gtol)
        info["state_id"] = state_id
        trace.append(info)
        if info["ep_final"] < best - 1e-12:
            best, best_y, best_label = info["ep_final"], yy, info["start"]

    best = direct_entropy_of_y(K, best_y, d)
    hs2, ortho = validate_purification(K,best_y,d)
    rec = dict(
        state_id=int(state_id), state_seed=STATE_SEED_BASE+state_id,
        dim_ap=int(d), dim_bp=int(d), aux_qubits=int(2*np.log2(d)),
        ep_direct=float(best), best_start=best_label,
        starts_attempted=int(len(trace)),
        starts_converged=int(sum(t["success"] for t in trace)),
        best_start_gradient_norm=float(min(t["gradient_norm"] for t in trace if t["start"] == best_label)),
        purification_hs2=hs2, orthogonality_error=ortho,
        best_trial_value_spread=float(max(t["ep_final"] for t in trace)-min(t["ep_final"] for t in trace)),
        median_trial_value=float(np.median([t["ep_final"] for t in trace])),
    )
    return rec, best_y, trace


def atomic_save_csv(records, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(temp, index=False, float_format="%.15g")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description="Strong independent direct E_p reference")
    parser.add_argument("--states", default="all", help="all, 1:100 or 2,74,78")
    parser.add_argument("--restarts2", type=int, default=8)
    parser.add_argument("--restarts4", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--polish", type=int, default=3)
    parser.add_argument("--ftol", type=float, default=1e-13)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument("--outdir", default="direct_reference_refined")
    parser.add_argument("--force", action="store_true", help="recompute existing state IDs")
    args = parser.parse_args()

    ids = parse_state_ids(args.states)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fp2 = out / "random_1tn_direct_refined_dim2.csv"
    fp4 = out / "random_1tn_direct_refined_dim4.csv"
    fpcomb = out / "random_1tn_direct_refined_combined.csv"
    fpruns = out / "random_1tn_direct_refined_all_starts.csv"

    previous = {}
    if fpcomb.exists() and not args.force:
        prev = pd.read_csv(fpcomb)
        for item in prev.to_dict("records"):
            previous[int(item["state_id"])] = item
    combined = {k:v for k,v in previous.items()}
    traces = []
    if fpruns.exists() and not args.force:
        traces = pd.read_csv(fpruns).to_dict("records")

    config = vars(args).copy()
    config.update(state_seed_base=STATE_SEED_BASE,
                  note="Local numerical reference, not mathematical global certificate")
    (out/"run_settings.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    print("Direct baseline: exact rho_AB, isometry equivalent to SU(4) and SU(16)")
    print("Same Haar ensemble as original; seed=20260918+state_id; logarithm base2")
    print(f"States: {ids} | output: {out.resolve()}", flush=True)

    t0 = time.perf_counter()
    for state_id in ids:
        if state_id in combined and not args.force:
            print(f"[{state_id:3d}/100] already complete; skipping", flush=True)
            continue
        rho = rho_for_state(state_id)
        s_a,s_b = theoretical_upper(rho)
        upper = min(s_a,s_b)
        rng2 = np.random.default_rng(2026 + state_id*1000 + 2)
        rng4 = np.random.default_rng(2026 + state_id*1000 + 4)
        started = time.perf_counter()

        r2, best_y2, logs2 = optimize_baseline(
            rho,state_id,2,rng2,random_starts=args.restarts2,
            polish=args.polish,maxiter=args.maxiter,
            ftol=args.ftol,gtol=args.gtol)
        r4, best_y4, logs4 = optimize_baseline(
            rho,state_id,4,rng4,warm_y2=best_y2,
            random_starts=args.restarts4,polish=args.polish,
            maxiter=args.maxiter,ftol=args.ftol,gtol=args.gtol)
        ep2,ep4 = r2["ep_direct"],r4["ep_direct"]
        if ep4 > ep2 + 5e-9:
            raise RuntimeError(f"FAIL dimensional embedding: state {state_id}: {ep4} > {ep2}")
        if ep4 > upper + 5e-9:
            raise RuntimeError(f"FAIL theoretical upper bound: state {state_id}: {ep4} > {upper}")
        for r in (r2,r4):
            if r["purification_hs2"] > 1e-15 or r["orthogonality_error"] > 1e-7:
                raise RuntimeError("Purification reconstruction failed: " + repr(r))
        row = dict(
            state_id=state_id, state_seed=STATE_SEED_BASE+state_id,
            rank=int(np.linalg.matrix_rank(rho,tol=1e-10)),
            purity=float(np.trace(rho@rho).real),
            S_A=s_a,S_B=s_b,upper_bound=upper,
            ep_direct_dim2=ep2, ep_direct_dim4=ep4,
            dim4_minus_dim2=ep4-ep2,
            bound_gap_dim4=upper-ep4,
            num_starts_dim2=r2["starts_attempted"],
            num_starts_dim4=r4["starts_attempted"],
            local_converged_starts_dim2=r2["starts_converged"],
            local_converged_starts_dim4=r4["starts_converged"],
            best_label_dim2=r2["best_start"],
            best_label_dim4=r4["best_start"],
            purification_hs2_dim2=r2["purification_hs2"],
            purification_hs2_dim4=r4["purification_hs2"],
            time_seconds=time.perf_counter()-started,
        )
        combined[state_id] = row
        for log in logs2+logs4:
            traces.append(log)
        sorted_rows = [combined[k] for k in sorted(combined)]
        atomic_save_csv(sorted_rows,fpcomb)
        atomic_save_csv(traces,fpruns)
        df = pd.DataFrame(sorted_rows)
        for d,fp in ((2,fp2),(4,fp4)):
            outdf = df[["state_id","state_seed","rank","purity",
                        "S_A","S_B","upper_bound",f"ep_direct_dim{d}"]].copy()
            outdf.rename(columns={f"ep_direct_dim{d}":"ep_direct"},inplace=True)
            atomic_save_csv(outdf.to_dict("records"),fp)
        print(f"[{state_id:3d}/100] d2={ep2:.10f} d4={ep4:.10f} "
              f"U={upper:.10f}  d4-d2={ep4-ep2:+.2e} "
              f"time={row['time_seconds']:.1f}s",flush=True)

    if not combined:
        return
    df = pd.DataFrame([combined[k] for k in sorted(combined)])
    summary = {
        "N_saved":int(len(df)),
        "dim4_le_dim2_count":int((df.ep_direct_dim4 <= df.ep_direct_dim2+5e-9).sum()),
        "dim4_le_theoretical_upper_count":int((df.ep_direct_dim4 <= df.upper_bound+5e-9).sum()),
        "max_positive_dim4_minus_dim2":float(df.dim4_minus_dim2.max()),
        "mean_dim4_minus_dim2":float(df.dim4_minus_dim2.mean()),
        "runtime_this_invocation_seconds":float(time.perf_counter()-t0),
        "WARNING":"Local multistart optimization is not a global-optimality proof."
    }
    (out/"random_1tn_direct_refined_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False), encoding="utf8")
    print("\nALL DONE")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print("Use random_1tn_direct_refined_dim4.csv column ep_direct for the new x-axis.")


if __name__ == "__main__":
    main()
