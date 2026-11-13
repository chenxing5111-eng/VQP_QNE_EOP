#!/usr/bin/env python3
"""Terhal-style *direct classical numerical reference* for the EoP of Werner states.

Extracted from the supplied two-stage VQC comparison script.  Only the
spectral purification, general ancillary-unitary optimization, and direct
parameter sweep are retained.  The numerical optimizer is not guaranteed to
find the globally optimal entanglement of purification; it reports the best
value found over the specified initializations.

Bell-diagonal Werner family:
    rho_W(e) = e|Psi_0><Psi_0| + (1-e)/3 sum_{i=1}^3 |Psi_i><Psi_i|

The minimized entropy is S(AA') in bits (base-2 logarithm).
For dim(A')=dim(B')=2 a full SU(4) acts on the purifier;
for dim(A')=dim(B')=4 a full SU(16) acts on the purifier.

Usage examples:
    python werner_eop_direct.py --dim 2 --e 0.25
    python werner_eop_direct.py --dim both --step 0.05 --output direct_results.csv

Dependencies: numpy, torch, pennylane (version supporting qml.SpecialUnitary).
"""

import argparse
import csv
from pathlib import Path

import numpy as onp
from pennylane import numpy as np
import pennylane as qml
import torch

torch.set_default_dtype(torch.float64)

# Direct optimization settings retained from the supplied code.
SEED_DIRECT = 2026
DIRECT_EPOCHS_DIM2 = 1200
DIRECT_LR_DIM2 = (0.06, 0.02, 0.005)
DIRECT_RANDOM_RESTARTS_DIM2 = 3
DIRECT_INIT_SCALE_DIM2 = 0.35

DIRECT_EPOCHS_DIM4 = 1600
DIRECT_LR_DIM4 = (0.025, 0.008, 0.002)
DIRECT_RANDOM_RESTARTS_DIM4 = 2
DIRECT_INIT_SCALE_DIM4 = 0.20

DIRECT_TOL = 1e-10
DIRECT_PATIENCE = 40
DIRECT_MIN_EPOCHS = 250
PRINT_EVERY_POINT = True

# bell_states
def bell_states():
    psi0 = np.array([1.0, 0.0, 0.0, 1.0], requires_grad=False) / np.sqrt(2.0)
    psi1 = np.array([1.0, 0.0, 0.0, -1.0], requires_grad=False) / np.sqrt(2.0)
    psi2 = np.array([0.0, 1.0, 1.0, 0.0], requires_grad=False) / np.sqrt(2.0)
    psi3 = np.array([0.0, 1.0, -1.0, 0.0], requires_grad=False) / np.sqrt(2.0)
    return psi0, psi1, psi2, psi3


# standard_purification_state
def standard_purification_state(e, aux_qubits_per_side):
    """
    Exact standard purification of the Werner state.

    Bell eigenvectors are used as the AB eigenbasis.  The four spectral labels
    i=0,1,2,3 are embedded in the first four computational basis states of
    A'B'.

    k=1:
        dim(A')=dim(B')=2, total purifier dimension = 4.

    k=2:
        dim(A')=dim(B')=4, total purifier dimension = 16.
        The standard purifier initially occupies only a 4-dimensional subspace;
        the subsequent full SU(16) auxiliary transformation can map that
        subspace to an arbitrary 4-dimensional isometric image.
    """
    k = int(aux_qubits_per_side)
    aux_dim = 2 ** (2 * k)

    q = (1.0 - float(e)) / 3.0
    lambdas = onp.array([float(e), q, q, q], dtype=float)

    # Use ordinary NumPy here because this is an exact fixed input state;
    # it is not an optimization variable.
    bells = [
        onp.asarray(v, dtype=complex)
        for v in bell_states()
    ]

    state = onp.zeros(4 * aux_dim, dtype=complex)
    for i, (lam, bell) in enumerate(zip(lambdas, bells)):
        aux_basis = onp.zeros(aux_dim, dtype=complex)
        aux_basis[i] = 1.0
        state += onp.sqrt(max(lam, 0.0)) * onp.kron(bell, aux_basis)

    norm = onp.linalg.norm(state)
    if norm <= 0:
        raise RuntimeError("Standard purification has zero norm.")
    state = state / norm
    return np.array(state, requires_grad=False)


# make_direct_model
def make_direct_model(aux_qubits_per_side):
    """
    Build the Terhal-style direct auxiliary-space optimization.

    IMPORTANT:
    qml.SpecialUnitary is NOT differentiable with PennyLane's Autograd
    interface because it relies on a matrix exponential.  Therefore the
    direct-reference part uses the Torch interface, while our original VQC
    part remains unchanged and continues to use PennyLane Autograd.

    k=1 -> n_aux=2 -> full SU(4), 15 real parameters.
    k=2 -> n_aux=4 -> full SU(16), 255 real parameters.

    For k=2, the 2002 paper directly optimized a 16x4 isometry.  A full
    SU(16) acting on the embedded four-dimensional support can realize an
    arbitrary 16x4 isometry via unitary extension.  This is therefore an
    equivalent search space, although the numerical parameterization differs.
    """
    if not hasattr(qml, "SpecialUnitary"):
        raise RuntimeError(
            "This code requires qml.SpecialUnitary. "
            "Please update PennyLane: pip install -U pennylane"
        )

    k = int(aux_qubits_per_side)
    if k not in (1, 2):
        raise ValueError("This script supports k=1 or k=2.")

    wire_A = 0
    wire_B = 1
    wires_Ap = list(range(2, 2 + k))
    wires_Bp = list(range(2 + k, 2 + 2 * k))
    aux_wires = wires_Ap + wires_Bp
    all_wires = list(range(2 + 2 * k))
    aa_wires = [wire_A] + wires_Ap

    n_aux = len(aux_wires)
    n_params = 4 ** n_aux - 1

    if k == 1:
        epochs = DIRECT_EPOCHS_DIM2
        lr = DIRECT_LR_DIM2
        random_restarts = DIRECT_RANDOM_RESTARTS_DIM2
        init_scale = DIRECT_INIT_SCALE_DIM2
    else:
        epochs = DIRECT_EPOCHS_DIM4
        lr = DIRECT_LR_DIM4
        random_restarts = DIRECT_RANDOM_RESTARTS_DIM4
        init_scale = DIRECT_INIT_SCALE_DIM4

    dev = qml.device("default.qubit", wires=len(all_wires))

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def entropy_qnode(params, initial_state):
        qml.StatePrep(initial_state, wires=all_wires)
        qml.SpecialUnitary(params, wires=aux_wires)
        return qml.vn_entropy(wires=aa_wires, log_base=2)

    return {
        "k": k,
        "dim": 2 ** k,
        "n_aux": n_aux,
        "n_params": n_params,
        "epochs": epochs,
        "lr": lr,
        "random_restarts": random_restarts,
        "init_scale": init_scale,
        "entropy_qnode": entropy_qnode,
    }


# optimize_direct_one_e
def optimize_direct_one_e(model, e, warm_params=None, seed_offset=0):
    """
    Direct minimization of S(AA') from the exact standard purification.

    The direct-reference optimization uses Torch Adam because
    qml.SpecialUnitary supports Torch/JAX/TensorFlow differentiation but
    does not support Autograd differentiation of its matrix exponential.

    Candidate starts:
      1) previous e-point optimum (warm start), if available;
      2) identity (all parameters zero);
      3) several random starts.

    The best converged result is retained.
    """
    # Exact standard purification is fixed, so it does not require gradients.
    initial_state_np = onp.asarray(
        standard_purification_state(e, model["k"]),
        dtype=onp.complex128
    )
    initial_state = torch.tensor(
        initial_state_np,
        dtype=torch.complex128,
        requires_grad=False
    )

    entropy_qnode = model["entropy_qnode"]

    starts = []

    if warm_params is not None:
        starts.append((
            "warm",
            onp.asarray(warm_params, dtype=onp.float64).copy()
        ))

    # Identity is important near the small-e region where the standard
    # purification can itself be optimal.
    starts.append((
        "identity",
        onp.zeros(model["n_params"], dtype=onp.float64)
    ))

    rng = onp.random.default_rng(SEED_DIRECT + seed_offset)
    for r in range(model["random_restarts"]):
        starts.append((
            f"random{r + 1}",
            rng.normal(
                0.0,
                model["init_scale"],
                size=model["n_params"]
            ).astype(onp.float64)
        ))

    global_best_ep = onp.inf
    global_best_params = None
    global_best_start = None
    global_best_epoch = 0

    for start_name, raw in starts:
        params = torch.tensor(
            raw,
            dtype=torch.float64,
            requires_grad=True
        )

        optimizer = torch.optim.Adam(
            [params],
            lr=float(model["lr"][0])
        )

        ep0 = entropy_qnode(params, initial_state)
        best_ep = float(ep0.detach().cpu().item())
        best_params = params.detach().cpu().numpy().copy()
        prev = best_ep
        stable = 0
        best_epoch = 0

        for epoch in range(1, model["epochs"] + 1):
            frac = epoch / model["epochs"]

            if abs(frac - 0.60) < 0.5 / model["epochs"]:
                for group in optimizer.param_groups:
                    group["lr"] = float(model["lr"][1])

            if abs(frac - 0.85) < 0.5 / model["epochs"]:
                for group in optimizer.param_groups:
                    group["lr"] = float(model["lr"][2])

            optimizer.zero_grad(set_to_none=True)

            loss = entropy_qnode(params, initial_state)

            # Protect against unexpected non-finite numerical values.
            if not torch.isfinite(loss):
                break

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ep_now_tensor = entropy_qnode(params, initial_state)
                ep_now = float(ep_now_tensor.detach().cpu().item())

            if ep_now < best_ep:
                best_ep = ep_now
                best_params = params.detach().cpu().numpy().copy()
                best_epoch = epoch

            change = abs(ep_now - prev)
            prev = ep_now
            stable = stable + 1 if change < DIRECT_TOL else 0

            if epoch >= DIRECT_MIN_EPOCHS and stable >= DIRECT_PATIENCE:
                break

        if best_ep < global_best_ep:
            global_best_ep = best_ep
            global_best_params = best_params.copy()
            global_best_start = start_name
            global_best_epoch = best_epoch

    if global_best_params is None:
        raise RuntimeError(
            f"Direct optimization failed for e={e:.6f}; "
            "all starts produced non-finite results."
        )

    return {
        "e": float(e),
        "ep": float(global_best_ep),
        "params": global_best_params,
        "best_start": global_best_start,
        "best_epoch": int(global_best_epoch),
    }


# sweep_direct_curve
def sweep_direct_curve(model, e_values, label="direct"):
    """
    Warm-start sweep centered at e=1/4, with extra random restarts at every e.
    """
    values = onp.asarray(e_values, dtype=float)
    values = onp.sort(values)

    center_index = int(onp.argmin(onp.abs(values - 0.25)))
    center_e = float(values[center_index])

    results = {}

    print("\n" + "=" * 78)
    print(
        f"Start {label}: Terhal-style direct optimization, "
        f"dim(A')=dim(B')={model['dim']}, "
        f"full SU({2 ** model['n_aux']}) with {model['n_params']} parameters"
    )
    print("=" * 78)

    r0 = optimize_direct_one_e(
        model, center_e,
        warm_params=None,
        seed_offset=int(round(center_e * 10000))
    )
    results[center_e] = r0
    if PRINT_EVERY_POINT:
        print(
            f"[{label}] e={center_e:7.4f}  Ep={r0['ep']:.9f}  "
            f"start={r0['best_start']}  epoch={r0['best_epoch']}"
        )

    prev = r0["params"]
    for e in values[center_index + 1:]:
        e = float(e)
        r = optimize_direct_one_e(
            model, e,
            warm_params=prev,
            seed_offset=int(round(e * 10000))
        )
        results[e] = r
        prev = r["params"]
        if PRINT_EVERY_POINT:
            print(
                f"[{label}] e={e:7.4f}  Ep={r['ep']:.9f}  "
                f"start={r['best_start']}  epoch={r['best_epoch']}"
            )

    prev = r0["params"]
    for e in values[:center_index][::-1]:
        e = float(e)
        r = optimize_direct_one_e(
            model, e,
            warm_params=prev,
            seed_offset=int(round(e * 10000))
        )
        results[e] = r
        prev = r["params"]
        if PRINT_EVERY_POINT:
            print(
                f"[{label}] e={e:7.4f}  Ep={r['ep']:.9f}  "
                f"start={r['best_start']}  epoch={r['best_epoch']}"
            )

    return [results[float(e)] for e in values]


def main():
    parser = argparse.ArgumentParser(
        description="Direct auxiliary-unitary numerical optimization of Werner-state EoP."
    )
    parser.add_argument(
        "--dim", choices=("2", "4", "both"), default="both",
        help="Local ancilla dimensions dim(A')=dim(B') (default: 2).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--e", type=float, default=None,
        help="Run one Werner mixing parameter e in [0,1].",
    )
    group.add_argument(
        "--step", type=float, default=None,
        help="Sweep e=0,step,...,1. Original step: 0.05.",
    )
    parser.add_argument(
        "--output", default="werner_eop_direct_results.csv",
        help="Output CSV for direct numerical results.",
    )
    args = parser.parse_args()

    if args.e is not None:
        if not 0 <= args.e <= 1:
            parser.error("--e must be between 0 and 1")
        values = onp.array([float(args.e)])
    else:
        step = 0.05 if args.step is None else args.step
        if not 0 < step <= 1:
            parser.error("--step must be in (0,1]")
        # Preserve the source script's e-grid construction.
        values = onp.round(onp.arange(0.0, 1.0 + 0.5 * step, step), 10)
        values = values[values <= 1.0]

    dims = (2,4) if args.dim == "both" else (int(args.dim),)
    rows = []
    for ancilla_dim in dims:
        k = 1 if ancilla_dim == 2 else 2
        model = make_direct_model(k)
        if len(values) == 1:
            e = float(values[0])
            results = [optimize_direct_one_e(
                model, e, seed_offset=int(round(e * 10000))
            )]
        else:
            results = sweep_direct_curve(
                model, values, label=f"Direct-dim{ancilla_dim}"
            )
        for result in results:
            rows.append({
                "e": result["e"],
                "ancilla_dimension_Aprime_Bprime": ancilla_dim,
                "eop_direct_bits": result["ep"],
                "best_start": result["best_start"],
                "best_epoch": result["best_epoch"],
            })
            if len(values) == 1:
                print(
                    f"dim(A')=dim(B')={ancilla_dim}, e={result['e']:.4f}, "
                    f"direct EoP numerical estimate={result['ep']:.9f} bits"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "e", "ancilla_dimension_Aprime_Bprime", "eop_direct_bits",
            "best_start", "best_epoch"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
