code/"""
Unknown-Werner-state variational purification + QNE estimation of E_p
=====================================================================

Purpose
-------
Treat the two-qubit Werner state as a BLACK-BOX quantum state:

    rho_W(e) = e |Psi_0><Psi_0|
             + (1-e)/3 * sum_{i=1}^3 |Psi_i><Psi_i|,

where the algorithm is NOT allowed to read rho_W(e), its eigenvalues, or its
matrix elements during training.

The workflow is:

Stage 1: Variational quantum purification (VQP)
    Learn a pure state |Psi(theta)>_{ABA'B'} whose AB marginal approximates the
    unknown target rho_AB.  The loss is the Hilbert-Schmidt distance

        D_HS^2(chi_AB, rho_AB)
          = Tr(chi_AB^2) + Tr(rho_AB^2) - 2 Tr(chi_AB rho_AB),

    but every term is evaluated by SWAP tests between state copies.  No target
    density matrix is used in the training loss.  We require

        D_HS^2 < 1e-7

    for several consecutive epochs before Stage 2 is allowed to start.

Stage 2: Purification-space optimization + Quantum Neural Entropy estimation
    Freeze theta*.  Apply a 4-layer hardware-efficient V_phi only on A'B' to explore
    purifications of the learned AB state.  For each current purification, QNE
    estimates S(rho_AA') without constructing rho_AA' or diagonalizing it.

    The QNE Hermitian operator is

        H(w,gamma) = sum_i f_w(i) U_gamma |i><i| U_gamma^dagger,

    where
      - U_gamma is a 4-layer Rot-CNOT basis circuit on AA',
      - f_w is a 2-10-1 classical neural network evaluated on the four bitstrings.

    With d=4, the objective is

        J(phi,gamma,w)
          = sum_i p_{phi,gamma}(i) f_w(i)
            - (1/4) sum_i exp(f_w(i)),

    where p_{phi,gamma}(i) are exact Born probabilities from qml.probs for the
    first noiseless validation version.  The second term is evaluated exactly
    over all four bitstrings (no classical Monte-Carlo noise).

    The QNE estimate of E_p is

        E_p^QNE = [ln(4) - 1 - max_{phi,gamma,w} J] / ln(2).

Validation only (NOT used in training)
--------------------------------------
1) The true Werner density matrix is used only AFTER optimization to compute
   theoretical E_p bounds:

       E_f,
       I(A:B)/2,
       S_R^(2)(A:B)/2,
       L = max{E_f, I/2, S_R^(2)/2},
       U = min{S(A), S(B)}.

2) qml.vn_entropy is evaluated only at the final QNE-optimized phi as a hidden
   simulator diagnostic.  It is never used by the QNE optimizer.

3) AB invariance under V_phi is checked numerically after Stage 2.

Outputs
-------
- CSV with E_p^QNE, all bounds, Stage-1 accuracy, diagnostics, runtimes.
- Stage-1 convergence CSV.
- Stage-2 convergence CSV.
- Main E_p-versus-bounds figure (PNG + PDF).
- Representative Stage-1 convergence figure.
- Representative Stage-2 convergence figure.

Dependencies
------------
    pip install -U pennylane numpy scipy matplotlib pandas torch

Notes
-----
- This implementation uses shots=None and qml.probs in Stage 2.  Therefore it
  validates the variational/QNE mathematics without finite-shot noise.
- The target Werner state is emulated as a black-box state-preparation oracle by
  preparing a hidden purification.  In an experiment, replace this oracle by the
  physical device/source that repeatedly prepares rho_AB.
"""

import os
import time
import copy
import warnings

import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


# ============================================================
# 0. Global configuration
# ============================================================

SEED_STAGE1 = 42
SEED_STAGE2 = 1024

# Werner grid
E_STEP = 0.05
E_VALUES = onp.round(onp.arange(0.0, 1.0 + 0.5 * E_STEP, E_STEP), 10)

# Output directory
OUT_DIR = "werner_unknown_vqp_qne_results_all_HEA"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_MAIN = os.path.join(OUT_DIR, "werner_unknown_vqp_qne_ep_bounds.csv")
CSV_STAGE1 = os.path.join(OUT_DIR, "stage1_hs_convergence.csv")
CSV_STAGE2 = os.path.join(OUT_DIR, "stage2_qne_convergence.csv")

FIG_MAIN_PNG = os.path.join(OUT_DIR, "werner_qne_ep_theoretical_bounds.png")
FIG_MAIN_PDF = os.path.join(OUT_DIR, "werner_qne_ep_theoretical_bounds.pdf")
FIG_S1_PNG = os.path.join(OUT_DIR, "stage1_hs_convergence.png")
FIG_S1_PDF = os.path.join(OUT_DIR, "stage1_hs_convergence.pdf")
FIG_S2_PNG = os.path.join(OUT_DIR, "stage2_qne_convergence.png")
FIG_S2_PDF = os.path.join(OUT_DIR, "stage2_qne_convergence.pdf")

# ---------------- Stage 1: unknown-state variational purification ----------------
# 4-qubit learned purifier: A,B,A',B'
S1_LAYERS = 4
S1_MAX_EPOCHS = 2200
S1_LR = (0.08, 0.03, 0.008)
S1_INIT_SCALE = 0.05

# User-requested strict target
HS_TARGET = 1e-6
HS_CONSECUTIVE = 10

# Only used if the warm/random first attempt fails to reach HS_TARGET.
S1_FALLBACK_RESTARTS = 2

# ---------------- Stage 2: purification search + QNE ----------------
# Stage 2: two-qubit Rot-CNOT HEAs, instead of SpecialUnitary SU(4).
VPHI_LAYERS = 4
VPHI_PARAM_SHAPE = (VPHI_LAYERS, 2, 3)  # 24 trainable parameters
GAMMA_LAYERS = 4
GAMMA_PARAM_SHAPE = (GAMMA_LAYERS, 2, 3)  # 24 trainable parameters

S2_MAX_EPOCHS = 1800
S2_MIN_EPOCHS = 300
S2_PATIENCE = 60
S2_TOL = 1e-9

# Alternating optimization learning rates.
S2_LR_NN = (0.030, 0.010, 0.003)
S2_LR_PHI = (0.018, 0.007, 0.002)
S2_LR_GAMMA = (0.020, 0.008, 0.002)

# At every e, use zero-angle plus random starts; if a warm start is available,
# include it as an additional candidate.
S2_RANDOM_RESTARTS = 1
S2_INIT_SCALE = 0.08

# Smoothly bound the NN spectral outputs for numerical stability at rank-deficient
# states.  +/-25 makes exp(-25) ~ 1.4e-11, negligible on this scale.
QNE_F_BOUND = 25.0

# Convergence logging
PRINT_STAGE1_EVERY = 100
PRINT_STAGE2_EVERY = 100

# Bound check tolerance: do not clip the result; report a violation if outside.
BOUND_TOL = 2e-3
NUMERICAL_EPS = 1e-14

# Representative e values used for convergence figures
REPRESENTATIVE_E = (0.00, 0.25, 0.50, 0.75, 1.00)


# ============================================================
# 1. Werner family and black-box target-preparation oracle
# ============================================================

def bell_states_numpy():
    """Standard Bell basis used in the previous Werner-state code."""
    psi0 = onp.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / onp.sqrt(2.0)
    psi1 = onp.array([1.0, 0.0, 0.0, -1.0], dtype=complex) / onp.sqrt(2.0)
    psi2 = onp.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / onp.sqrt(2.0)
    psi3 = onp.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / onp.sqrt(2.0)
    return psi0, psi1, psi2, psi3


def werner_state_for_validation_only(e):
    """
    True 4x4 Werner density matrix.

    IMPORTANT: this function is NEVER called by Stage-1 or Stage-2 training.
    It is only used after optimization for theoretical bounds/validation.
    """
    e = float(e)
    if not 0.0 <= e <= 1.0:
        raise ValueError("e must lie in [0,1].")

    psis = bell_states_numpy()
    projectors = [onp.outer(v, v.conj()) for v in psis]
    rho = e * projectors[0]
    rho += (1.0 - e) / 3.0 * (projectors[1] + projectors[2] + projectors[3])
    return rho


def target_oracle_purification(e):
    """
    Hidden purification used only to EMULATE repeated preparation of the unknown
    Werner state in a pure-state simulator.

    Wire ordering inside the oracle block is:
        [A, B, E0, E1].

    The algorithm never measures or accesses E0,E1 and never reads this vector.
    In real hardware, replace this entire function by the physical source that
    repeatedly prepares rho_AB.
    """
    e = float(e)
    q = (1.0 - e) / 3.0
    lambdas = onp.array([e, q, q, q], dtype=float)
    bells = bell_states_numpy()

    state = onp.zeros(16, dtype=complex)
    for i, (lam, bell) in enumerate(zip(lambdas, bells)):
        env = onp.zeros(4, dtype=complex)
        env[i] = 1.0
        state += onp.sqrt(max(lam, 0.0)) * onp.kron(bell, env)

    norm = onp.linalg.norm(state)
    if norm <= 0:
        raise RuntimeError("Target oracle purification has zero norm.")
    return state / norm


def prepare_target_oracle(e, wires):
    """Prepare one hidden purification copy of the unknown target rho_AB."""
    qml.StatePrep(target_oracle_purification(float(e)), wires=wires)


# ============================================================
# 2. Stage-1 VQP ansatz and SWAP-test Hilbert-Schmidt loss
# ============================================================

# Learned purification uses four logical wires [A,B,A',B'].
LEARNED_LOGICAL_WIRES = [0, 1, 2, 3]
SYSTEM_LOCAL = [0, 1]
AUX_LOCAL = [2, 3]
AA_LOCAL = [0, 2]


def stage1_ansatz(params, wires):
    """
    Four-qubit variational purification ansatz.

    Same structure as the previous Werner VQC work, but made deeper so the
    strict D_HS^2 < 1e-7 threshold is more robust.
    """
    wires = list(wires)
    n = len(wires)

    for layer in range(S1_LAYERS):
        for i, w in enumerate(wires):
            qml.Rot(
                params[layer, i, 0],
                params[layer, i, 1],
                params[layer, i, 2],
                wires=w,
            )

        # Circular entangling layer
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def swap_test(control_wire, system1_wires, system2_wires):
    """
    Standard SWAP test.  For two (possibly mixed) states rho and sigma on the
    selected system registers, <Z_control> = Tr(rho sigma).
    """
    qml.Hadamard(wires=control_wire)
    for w1, w2 in zip(system1_wires, system2_wires):
        qml.CSWAP(wires=[control_wire, w1, w2])
    qml.Hadamard(wires=control_wire)


# 1 ancilla + 4-qubit copy 1 + 4-qubit copy 2 = 9 qubits.
DEV_SWAP = qml.device("default.qubit", wires=9, shots=None)

CTRL = 0
COPY1 = [1, 2, 3, 4]       # [A,B,A',B'] for learned copy or [A,B,E0,E1]
COPY2 = [5, 6, 7, 8]
COPY1_AB = [1, 2]
COPY2_AB = [5, 6]


@qml.qnode(DEV_SWAP, interface="autograd", diff_method="backprop")
def learned_purity_swap_qnode(params):
    """Return Tr[chi_AB(theta)^2] using two learned purification copies."""
    stage1_ansatz(params, COPY1)
    stage1_ansatz(params, COPY2)
    swap_test(CTRL, COPY1_AB, COPY2_AB)
    return qml.expval(qml.PauliZ(CTRL))


@qml.qnode(DEV_SWAP, interface="autograd", diff_method="backprop")
def learned_target_overlap_swap_qnode(params, e):
    """Return Tr[chi_AB(theta) rho_target] by a SWAP test."""
    stage1_ansatz(params, COPY1)
    prepare_target_oracle(float(e), COPY2)
    swap_test(CTRL, COPY1_AB, COPY2_AB)
    return qml.expval(qml.PauliZ(CTRL))


@qml.qnode(DEV_SWAP, interface="autograd", diff_method=None)
def target_purity_swap_qnode(e):
    """Return Tr[rho_target^2] from two target copies; no density matrix used."""
    prepare_target_oracle(float(e), COPY1)
    prepare_target_oracle(float(e), COPY2)
    swap_test(CTRL, COPY1_AB, COPY2_AB)
    return qml.expval(qml.PauliZ(CTRL))


def stage1_train_cost(params, e):
    """
    Parameter-dependent part of D_HS^2:
        Tr(chi^2) - 2 Tr(chi rho).
    The constant Tr(rho^2) is omitted only from the optimizer objective.
    """
    purity_chi = learned_purity_swap_qnode(params)
    overlap = learned_target_overlap_swap_qnode(params, float(e))
    return purity_chi - 2.0 * overlap


def stage1_full_hs2(params, e, target_purity):
    """Full Hilbert-Schmidt squared distance, evaluated entirely by SWAP tests."""
    value = stage1_train_cost(params, e) + target_purity
    value = float(value)
    # Tiny negative values can appear from floating-point cancellation.
    return max(value, 0.0)


def _stage1_lr_for_epoch(epoch):
    frac = epoch / S1_MAX_EPOCHS
    if frac < 0.60:
        return S1_LR[0]
    if frac < 0.85:
        return S1_LR[1]
    return S1_LR[2]


def _run_stage1_attempt(e, init_params, attempt_name):
    """Run one Stage-1 optimization attempt."""
    target_purity = float(target_purity_swap_qnode(float(e)))

    params = np.array(onp.asarray(init_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=S1_LR[0])

    best_hs = onp.inf
    best_params = np.array(onp.asarray(params), requires_grad=False)
    best_epoch = 0
    consecutive = 0
    history = []

    prev_lr = S1_LR[0]

    for epoch in range(1, S1_MAX_EPOCHS + 1):
        lr = _stage1_lr_for_epoch(epoch)
        if lr != prev_lr:
            opt.stepsize = lr
            prev_lr = lr

        params, _ = opt.step_and_cost(
            lambda p: stage1_train_cost(p, float(e)),
            params,
        )

        hs_now = stage1_full_hs2(params, float(e), target_purity)
        history.append({
            "epoch": epoch,
            "hs2": hs_now,
        })

        if hs_now < best_hs:
            best_hs = hs_now
            best_params = np.array(onp.asarray(params), requires_grad=False)
            best_epoch = epoch

        consecutive = consecutive + 1 if hs_now < HS_TARGET else 0

        if epoch == 1 or epoch % PRINT_STAGE1_EVERY == 0 or consecutive == HS_CONSECUTIVE:
            print(
                f"    [S1:{attempt_name}] epoch={epoch:4d}  "
                f"D_HS^2={hs_now:.3e}  best={best_hs:.3e}"
            )

        if consecutive >= HS_CONSECUTIVE:
            break

    success = best_hs < HS_TARGET

    return {
        "success": success,
        "best_hs2": float(best_hs),
        "best_params": best_params,
        "best_epoch": int(best_epoch),
        "last_epoch": int(epoch),
        "attempt_name": attempt_name,
        "target_purity_swap": float(target_purity),
        "history": history,
    }


def train_stage1_one_e(e, warm_params=None, seed_offset=0):
    """
    Learn one purification of the unknown target.

    The function enforces D_HS^2 < 1e-7.  If the first warm/random attempt does
    not reach the threshold, extra random restarts are tried.  The code raises
    an error instead of silently entering Stage 2 with an inaccurate purifier.
    """
    t0 = time.time()
    e = float(e)

    attempts = []

    if warm_params is not None:
        attempts.append(("warm", onp.asarray(warm_params, dtype=float).copy()))
    else:
        rng = onp.random.default_rng(SEED_STAGE1 + seed_offset)
        raw = rng.normal(
            0.0,
            S1_INIT_SCALE,
            size=(S1_LAYERS, 4, 3),
        )
        attempts.append(("random0", raw))

    # First attempt
    name, init = attempts[0]
    best_result = _run_stage1_attempt(e, init, name)

    # Fallback restarts only if required.
    if not best_result["success"]:
        for r in range(S1_FALLBACK_RESTARTS):
            rng = onp.random.default_rng(SEED_STAGE1 + seed_offset + 1000 * (r + 1))
            init = rng.normal(
                0.0,
                S1_INIT_SCALE,
                size=(S1_LAYERS, 4, 3),
            )
            result = _run_stage1_attempt(e, init, f"fallback{r + 1}")
            if result["best_hs2"] < best_result["best_hs2"]:
                best_result = result
            if result["success"]:
                best_result = result
                break

    if not best_result["success"]:
        raise RuntimeError(
            f"Stage 1 failed at e={e:.4f}: best D_HS^2="
            f"{best_result['best_hs2']:.3e} >= {HS_TARGET:.1e}.\n"
            "Increase S1_MAX_EPOCHS / S1_LAYERS or add restarts before using "
            "the Stage-2 E_p result."
        )

    best_result["runtime_sec"] = time.time() - t0
    return best_result


# ============================================================
# 3. Stage-2 QNE model: two-qubit Rot-CNOT HEAs
# ============================================================

DEV_QNE = qml.device("default.qubit", wires=4, shots=None)

A_WIRE = 0
B_WIRE = 1
AP_WIRE = 2
BP_WIRE = 3
AUX_WIRES = [AP_WIRE, BP_WIRE]
AA_WIRES = [A_WIRE, AP_WIRE]
AB_WIRES = [A_WIRE, B_WIRE]
ALL_WIRES = [0, 1, 2, 3]



def hea_rot_cnot_ansatz(params, wires):
    """Hardware-efficient Rot + CNOT ansatz (two- or three-qubit).

    params: torch tensor of shape (layers, len(wires), 3).
    For two qubits, alternate the CNOT direction between layers.
    For three qubits, use a circular CNOT pattern in every layer.
    This is a gate-based ansatz, not the full SU(4)/SU(8) family.
    """
    wires = list(wires)
    n = len(wires)
    if n not in (2, 3):
        raise ValueError("Rot-CNOT ansatz supports only 2 or 3 wires.")

    for layer in range(params.shape[0]):
        for j, wire in enumerate(wires):
            qml.Rot(
                params[layer, j, 0],
                params[layer, j, 1],
                params[layer, j, 2],
                wires=wire,
            )
        if n == 2:
            if layer % 2 == 0:
                qml.CNOT(wires=[wires[0], wires[1]])
            else:
                qml.CNOT(wires=[wires[1], wires[0]])
        else:
            for j in range(n):
                qml.CNOT(wires=[wires[j], wires[(j + 1) % n]])


def v_phi_ansatz(phi):
    """Trainable two-ancilla-qubit HEA purification search (4 layers)."""
    hea_rot_cnot_ansatz(phi, AUX_WIRES)


def u_gamma_ansatz(gamma):
    """Trainable HEA measurement-basis unitary on AA'; used as adjoint."""
    hea_rot_cnot_ansatz(gamma, AA_WIRES)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def qne_probs_qnode(phi, gamma, frozen_theta):
    """
    Produce p_{phi,gamma}(i) on AA'.

    The sequence is:
        learned purification(theta*)
        -> V_phi on A'B'
        -> U_gamma^dagger on AA'
        -> computational-basis probabilities on AA'.
    """
    stage1_ansatz(frozen_theta, ALL_WIRES)

    # Explore candidate purifications via a Rot-CNOT HEA.
    v_phi_ansatz(phi)

    # QNE measurement/eigenbasis transformation U_gamma^dagger.
    qml.adjoint(u_gamma_ansatz)(gamma)

    return qml.probs(wires=AA_WIRES)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def exact_entropy_diagnostic_qnode(phi, frozen_theta):
    """
    Simulator-only diagnostic: exact S(AA') at the QNE-selected phi.
    NEVER used in Stage-2 optimization.
    """
    stage1_ansatz(frozen_theta, ALL_WIRES)
    v_phi_ansatz(phi)
    return qml.vn_entropy(wires=AA_WIRES, log_base=2)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def rho_ab_before_stage2_diagnostic_qnode(frozen_theta):
    """Simulator-only diagnostic; not used in optimization."""
    stage1_ansatz(frozen_theta, ALL_WIRES)
    return qml.density_matrix(wires=AB_WIRES)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def rho_ab_after_stage2_diagnostic_qnode(phi, frozen_theta):
    """Simulator-only diagnostic; not used in optimization."""
    stage1_ansatz(frozen_theta, ALL_WIRES)
    v_phi_ansatz(phi)
    return qml.density_matrix(wires=AB_WIRES)


class QNENet(torch.nn.Module):
    """Goldfeld-style 2-10-1 feed-forward network for the d=4 spectrum."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.fc2 = torch.nn.Linear(10, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


BITSTRINGS = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=torch.float64,
)


def bounded_f_values(net):
    """Return four smooth, numerically bounded spectral values f_w(i)."""
    raw = net(BITSTRINGS).reshape(-1)
    return QNE_F_BOUND * torch.tanh(raw / QNE_F_BOUND)


def qne_J_from_probs(probs, f_values):
    """
    J = sum_i p_i f_i - (1/4) sum_i exp(f_i).
    The second term is evaluated exactly over all d=4 bitstrings.
    """
    return torch.sum(probs * f_values) - torch.mean(torch.exp(f_values))


def qne_entropy_bits_from_J(J_value):
    """Convert the natural-log QNE variational objective to entropy in bits."""
    return (onp.log(4.0) - 1.0 - float(J_value)) / onp.log(2.0)


def _set_optimizer_lr(optimizer, lr_values):
    """Set learning rate(s) of an optimizer's parameter groups."""
    if not isinstance(lr_values, (list, tuple)):
        lr_values = [float(lr_values)] * len(optimizer.param_groups)
    for group, lr in zip(optimizer.param_groups, lr_values):
        group["lr"] = float(lr)


def _stage2_lr_tuple(epoch):
    frac = epoch / S2_MAX_EPOCHS
    if frac < 0.60:
        idx = 0
    elif frac < 0.85:
        idx = 1
    else:
        idx = 2
    return S2_LR_NN[idx], S2_LR_PHI[idx], S2_LR_GAMMA[idx]


def _new_qne_net(seed):
    torch.manual_seed(int(seed))
    net = QNENet().double()
    return net


def _run_stage2_start(frozen_theta_np, start, seed, start_name):
    """
    Run one alternating Stage-2 optimization start.

    Alternation per epoch:
      1) fix phi,gamma; update the NN once,
      2) fix the updated NN; update phi,gamma once.
    """
    frozen_theta = torch.tensor(
        onp.asarray(frozen_theta_np, dtype=onp.float64),
        dtype=torch.float64,
        requires_grad=False,
    )

    if start is None:
        rng = onp.random.default_rng(seed)
        phi0 = rng.normal(0.0, S2_INIT_SCALE, size=VPHI_PARAM_SHAPE)
        gamma0 = rng.normal(0.0, S2_INIT_SCALE, size=GAMMA_PARAM_SHAPE)
        net = _new_qne_net(seed + 17)
    else:
        phi0 = onp.asarray(start["phi"], dtype=onp.float64).copy()
        gamma0 = onp.asarray(start["gamma"], dtype=onp.float64).copy()
        net = _new_qne_net(seed + 17)
        if start.get("nn_state") is not None:
            net.load_state_dict(copy.deepcopy(start["nn_state"]))

    phi = torch.tensor(phi0, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(gamma0, dtype=torch.float64, requires_grad=True)

    nn_optimizer = torch.optim.Adam(net.parameters(), lr=S2_LR_NN[0])
    q_optimizer = torch.optim.Adam(
        [
            {"params": [phi], "lr": S2_LR_PHI[0]},
            {"params": [gamma], "lr": S2_LR_GAMMA[0]},
        ]
    )

    best_J = -onp.inf
    best_epoch = 0
    best_phi = None
    best_gamma = None
    best_nn_state = None

    prev_J = None
    stable = 0
    history = []

    current_lr_key = None

    for epoch in range(1, S2_MAX_EPOCHS + 1):
        lr_nn, lr_phi, lr_gamma = _stage2_lr_tuple(epoch)
        lr_key = (lr_nn, lr_phi, lr_gamma)
        if lr_key != current_lr_key:
            _set_optimizer_lr(nn_optimizer, lr_nn)
            _set_optimizer_lr(q_optimizer, [lr_phi, lr_gamma])
            current_lr_key = lr_key

        # ----------------------------------------------------
        # A. NN step: p is treated as fixed data.
        # ----------------------------------------------------
        nn_optimizer.zero_grad(set_to_none=True)

        probs_fixed = qne_probs_qnode(
            phi.detach(),
            gamma.detach(),
            frozen_theta,
        ).detach()

        f_nn = bounded_f_values(net)
        J_nn = qne_J_from_probs(probs_fixed, f_nn)
        loss_nn = -J_nn

        if not torch.isfinite(loss_nn):
            break

        loss_nn.backward()
        nn_optimizer.step()

        # ----------------------------------------------------
        # B. Quantum step: updated NN is treated as fixed.
        # ----------------------------------------------------
        q_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            f_fixed = bounded_f_values(net).detach()

        probs = qne_probs_qnode(phi, gamma, frozen_theta)
        J_quantum = qne_J_from_probs(probs, f_fixed)
        loss_quantum = -J_quantum

        if not torch.isfinite(loss_quantum):
            break

        # J_now corresponds to the CURRENT parameters before the quantum update;
        # this makes best-state bookkeeping exact.
        J_now = float(J_quantum.detach().cpu().item())
        ep_now = qne_entropy_bits_from_J(J_now)

        history.append({
            "epoch": epoch,
            "J": J_now,
            "Ep_QNE": ep_now,
        })

        if J_now > best_J:
            best_J = J_now
            best_epoch = epoch
            best_phi = phi.detach().cpu().numpy().copy()
            best_gamma = gamma.detach().cpu().numpy().copy()
            best_nn_state = copy.deepcopy(net.state_dict())

        if prev_J is not None:
            change = abs(J_now - prev_J)
            stable = stable + 1 if change < S2_TOL else 0
        prev_J = J_now

        if epoch == 1 or epoch % PRINT_STAGE2_EVERY == 0:
            print(
                f"    [S2:{start_name}] epoch={epoch:4d}  "
                f"J={J_now:+.10f}  Ep_QNE={ep_now:.9f}  "
                f"best={qne_entropy_bits_from_J(best_J):.9f}"
            )

        # Now update phi and gamma.
        loss_quantum.backward()
        q_optimizer.step()

        if epoch >= S2_MIN_EPOCHS and stable >= S2_PATIENCE:
            break

    # Evaluate one last state after the final quantum update.
    with torch.no_grad():
        probs_last = qne_probs_qnode(phi, gamma, frozen_theta)
        f_last = bounded_f_values(net)
        J_last = float(qne_J_from_probs(probs_last, f_last).cpu().item())

    if J_last > best_J:
        best_J = J_last
        best_epoch = int(epoch)
        best_phi = phi.detach().cpu().numpy().copy()
        best_gamma = gamma.detach().cpu().numpy().copy()
        best_nn_state = copy.deepcopy(net.state_dict())

    if best_phi is None:
        raise RuntimeError(f"Stage-2 start '{start_name}' failed to produce a finite objective.")

    return {
        "J_max": float(best_J),
        "Ep_QNE": float(qne_entropy_bits_from_J(best_J)),
        "phi": best_phi,
        "gamma": best_gamma,
        "nn_state": best_nn_state,
        "best_epoch": int(best_epoch),
        "last_epoch": int(epoch),
        "start_name": start_name,
        "history": history,
    }


def optimize_stage2_qne(frozen_theta_np, warm_start=None, seed_offset=0):
    """Optimize purification freedom phi and the QNE estimator gamma,w."""
    t0 = time.time()

    starts = []

    if warm_start is not None:
        starts.append(("warm", warm_start))

    # Zero-angle HEA start + fresh NN (not necessarily identity due to CNOTs).
    starts.append((
        "zero_angles",
        {
            "phi": onp.zeros(VPHI_PARAM_SHAPE, dtype=onp.float64),
            "gamma": onp.zeros(GAMMA_PARAM_SHAPE, dtype=onp.float64),
            "nn_state": None,
        },
    ))

    for r in range(S2_RANDOM_RESTARTS):
        starts.append((f"random{r + 1}", None))

    best = None

    for s_idx, (name, start) in enumerate(starts):
        result = _run_stage2_start(
            frozen_theta_np,
            start=start,
            seed=SEED_STAGE2 + seed_offset + 10000 * s_idx,
            start_name=name,
        )

        if best is None or result["J_max"] > best["J_max"]:
            best = result

    # Restore the best state for diagnostics.
    frozen_theta = torch.tensor(
        onp.asarray(frozen_theta_np, dtype=onp.float64),
        dtype=torch.float64,
        requires_grad=False,
    )
    best_phi_t = torch.tensor(best["phi"], dtype=torch.float64, requires_grad=False)

    exact_entropy = float(
        exact_entropy_diagnostic_qnode(best_phi_t, frozen_theta)
        .detach().cpu().item()
    )

    rho_before = (
        rho_ab_before_stage2_diagnostic_qnode(frozen_theta)
        .detach().cpu().numpy()
    )
    rho_after = (
        rho_ab_after_stage2_diagnostic_qnode(best_phi_t, frozen_theta)
        .detach().cpu().numpy()
    )
    delta = rho_after - rho_before
    ab_invariant_hs2 = float(onp.real(onp.trace(delta @ delta)))

    best["exact_entropy_at_qne_phi"] = exact_entropy
    best["qne_minus_exact_same_phi"] = float(best["Ep_QNE"] - exact_entropy)
    best["ab_invariant_hs2"] = ab_invariant_hs2
    best["runtime_sec"] = time.time() - t0

    # Warm-start payload for the next e point.
    best["warm_payload"] = {
        "phi": best["phi"].copy(),
        "gamma": best["gamma"].copy(),
        "nn_state": copy.deepcopy(best["nn_state"]),
    }

    return best


# ============================================================
# 4. Theoretical E_p bounds (validation only)
# ============================================================

def binary_entropy(x):
    x = float(onp.clip(x, 0.0, 1.0))
    if x <= NUMERICAL_EPS or 1.0 - x <= NUMERICAL_EPS:
        return 0.0
    return -(x * onp.log2(x) + (1.0 - x) * onp.log2(1.0 - x))


def eof_lower_bound(e):
    """Entanglement of formation for this Bell-diagonal Werner family."""
    e = float(e)
    concurrence = max(0.0, 2.0 * e - 1.0)
    if concurrence <= NUMERICAL_EPS:
        return 0.0
    x = (1.0 + onp.sqrt(max(0.0, 1.0 - concurrence ** 2))) / 2.0
    return binary_entropy(x)


def von_neumann_entropy_matrix(rho):
    """Von Neumann entropy in bits; used only for post-training validation."""
    rho = onp.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    vals = onp.linalg.eigvalsh(rho).real
    vals = onp.clip(vals, 0.0, None)
    total = vals.sum()
    if total <= NUMERICAL_EPS:
        return 0.0
    vals = vals / total
    vals = vals[vals > NUMERICAL_EPS]
    if len(vals) == 0:
        return 0.0
    return float(-onp.sum(vals * onp.log2(vals)))


def partial_trace_two_qubit(rho_ab, keep):
    rho = onp.asarray(rho_ab, dtype=complex).reshape(2, 2, 2, 2)
    if keep == "A":
        return onp.trace(rho, axis1=1, axis2=3)
    if keep == "B":
        return onp.trace(rho, axis1=0, axis2=2)
    raise ValueError("keep must be 'A' or 'B'.")


def matrix_sqrt_psd(rho):
    rho = onp.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = onp.linalg.eigh(rho)
    vals = onp.clip(vals.real, 0.0, None)
    return (vecs * onp.sqrt(vals)) @ vecs.conj().T


def renyi2_reflected_entropy(rho_ab):
    r"""
    S_R^(2)(A:B) = -log_2 Tr[(rho_AA*)^2]
    from the canonical purification |sqrt(rho_AB)>.
    """
    rho = onp.asarray(rho_ab, dtype=complex)
    sqrt_rho = matrix_sqrt_psd(rho)

    # Ordering A,B,A*,B*.
    psi = sqrt_rho.reshape(2, 2, 2, 2)

    # Trace over B and B*.
    rho_aa_star = onp.einsum(
        "abcd,ebfd->acef",
        psi,
        psi.conj(),
    ).reshape(4, 4)

    rho_aa_star = 0.5 * (rho_aa_star + rho_aa_star.conj().T)
    tr = float(onp.real(onp.trace(rho_aa_star)))
    if tr <= NUMERICAL_EPS:
        raise RuntimeError("Invalid canonical purification in reflected entropy.")
    rho_aa_star = rho_aa_star / tr

    purity = float(onp.real(onp.trace(rho_aa_star @ rho_aa_star)))
    purity = float(onp.clip(purity, NUMERICAL_EPS, 1.0))
    s_r2 = float(-onp.log2(purity))
    return s_r2, purity


def theoretical_bounds_werner(e):
    """
    External validation bounds for the true Werner state.

    These values NEVER enter Stage-1 or Stage-2 training.
    """
    rho_ab = werner_state_for_validation_only(float(e))
    rho_a = partial_trace_two_qubit(rho_ab, "A")
    rho_b = partial_trace_two_qubit(rho_ab, "B")

    s_a = von_neumann_entropy_matrix(rho_a)
    s_b = von_neumann_entropy_matrix(rho_b)
    s_ab = von_neumann_entropy_matrix(rho_ab)

    mutual_information = max(0.0, s_a + s_b - s_ab)
    i_half = 0.5 * mutual_information
    e_f = eof_lower_bound(float(e))

    s_r2, reflected_purity = renyi2_reflected_entropy(rho_ab)
    sr2_half = 0.5 * s_r2

    lower = max(e_f, i_half, sr2_half)
    upper = min(s_a, s_b)

    return {
        "S_A": float(s_a),
        "S_B": float(s_b),
        "S_AB": float(s_ab),
        "mutual_information": float(mutual_information),
        "I_half": float(i_half),
        "E_f": float(e_f),
        "S_R2": float(s_r2),
        "SR2_half": float(sr2_half),
        "reflected_purity": float(reflected_purity),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
    }


# ============================================================
# 5. Warm-start sweep centered at e = 1/4
# ============================================================

def run_one_e(e, warm_s1=None, warm_s2=None):
    e = float(e)
    seed_offset = int(round(e * 10000))

    print("\n" + "=" * 84)
    print(f"Werner target e = {e:.4f}  (treated as unknown black-box state)")
    print("=" * 84)

    print("[Stage 1] Variational purification from SWAP-test HS loss")
    s1 = train_stage1_one_e(
        e,
        warm_params=warm_s1,
        seed_offset=seed_offset,
    )

    print(
        f"  Stage 1 accepted: D_HS^2={s1['best_hs2']:.3e}  "
        f"(< {HS_TARGET:.1e}), best_epoch={s1['best_epoch']}, "
        f"start={s1['attempt_name']}"
    )

    print("[Stage 2] Auxiliary Rot-CNOT HEA search + HEA-based QNE")
    s2 = optimize_stage2_qne(
        frozen_theta_np=s1["best_params"],
        warm_start=warm_s2,
        seed_offset=seed_offset,
    )

    bounds = theoretical_bounds_werner(e)

    ep = float(s2["Ep_QNE"])
    lower = bounds["lower_bound"]
    upper = bounds["upper_bound"]
    in_bounds = bool(
        (ep >= lower - BOUND_TOL)
        and (ep <= upper + BOUND_TOL)
    )

    print(
        f"  QNE E_p              = {ep:.9f} ebits\n"
        f"  Exact S(AA') @ QNE phi [diagnostic only] = "
        f"{s2['exact_entropy_at_qne_phi']:.9f}\n"
        f"  QNE - exact(same phi)= {s2['qne_minus_exact_same_phi']:+.3e}\n"
        f"  Lower bound L         = {lower:.9f}\n"
        f"  Upper bound U         = {upper:.9f}\n"
        f"  In theoretical band?  = {in_bounds}\n"
        f"  AB invariance HS^2    = {s2['ab_invariant_hs2']:.3e}"
    )

    return {
        "e": e,
        "stage1": s1,
        "stage2": s2,
        "bounds": bounds,
        "in_bounds": in_bounds,
    }


def run_sweep():
    values = onp.sort(onp.asarray(E_VALUES, dtype=float))
    center_idx = int(onp.argmin(onp.abs(values - 0.25)))
    center_e = float(values[center_idx])

    results = {}

    # Center point
    r0 = run_one_e(center_e)
    results[center_e] = r0

    # Upward branch
    warm_s1 = r0["stage1"]["best_params"]
    warm_s2 = r0["stage2"]["warm_payload"]

    for e in values[center_idx + 1:]:
        e = float(e)
        r = run_one_e(e, warm_s1=warm_s1, warm_s2=warm_s2)
        results[e] = r
        warm_s1 = r["stage1"]["best_params"]
        warm_s2 = r["stage2"]["warm_payload"]

    # Downward branch
    warm_s1 = r0["stage1"]["best_params"]
    warm_s2 = r0["stage2"]["warm_payload"]

    for e in values[:center_idx][::-1]:
        e = float(e)
        r = run_one_e(e, warm_s1=warm_s1, warm_s2=warm_s2)
        results[e] = r
        warm_s1 = r["stage1"]["best_params"]
        warm_s2 = r["stage2"]["warm_payload"]

    return [results[float(e)] for e in values]


# ============================================================
# 6. Export tables
# ============================================================

def export_results(results):
    rows = []
    s1_rows = []
    s2_rows = []

    for r in results:
        e = r["e"]
        s1 = r["stage1"]
        s2 = r["stage2"]
        b = r["bounds"]

        ep = float(s2["Ep_QNE"])
        lower = float(b["lower_bound"])
        upper = float(b["upper_bound"])

        rows.append({
            "e": e,
            "Ep_QNE": ep,
            "E_f": b["E_f"],
            "I_half": b["I_half"],
            "S_R2": b["S_R2"],
            "SR2_half": b["SR2_half"],
            "lower_bound": lower,
            "upper_bound": upper,
            "Ep_minus_lower": ep - lower,
            "upper_minus_Ep": upper - ep,
            "in_bounds_with_tolerance": r["in_bounds"],
            "stage1_HS2": s1["best_hs2"],
            "stage1_best_epoch": s1["best_epoch"],
            "stage1_last_epoch": s1["last_epoch"],
            "stage1_start": s1["attempt_name"],
            "target_purity_swap": s1["target_purity_swap"],
            "stage1_runtime_sec": s1["runtime_sec"],
            "J_max": s2["J_max"],
            "stage2_best_epoch": s2["best_epoch"],
            "stage2_last_epoch": s2["last_epoch"],
            "stage2_start": s2["start_name"],
            "stage2_runtime_sec": s2["runtime_sec"],
            "exact_entropy_at_qne_phi": s2["exact_entropy_at_qne_phi"],
            "qne_minus_exact_same_phi": s2["qne_minus_exact_same_phi"],
            "ab_invariant_HS2": s2["ab_invariant_hs2"],
            "S_A": b["S_A"],
            "S_B": b["S_B"],
            "S_AB": b["S_AB"],
            "mutual_information": b["mutual_information"],
            "reflected_purity": b["reflected_purity"],
        })

        for item in s1["history"]:
            s1_rows.append({
                "e": e,
                "epoch": item["epoch"],
                "HS2": item["hs2"],
            })

        for item in s2["history"]:
            s2_rows.append({
                "e": e,
                "epoch": item["epoch"],
                "J": item["J"],
                "Ep_QNE": item["Ep_QNE"],
            })

    df = pd.DataFrame(rows).sort_values("e").reset_index(drop=True)
    df_s1 = pd.DataFrame(s1_rows)
    df_s2 = pd.DataFrame(s2_rows)

    df.to_csv(CSV_MAIN, index=False)
    df_s1.to_csv(CSV_STAGE1, index=False)
    df_s2.to_csv(CSV_STAGE2, index=False)

    return df, df_s1, df_s2


# ============================================================
# 7. Plotting
# ============================================================

def plot_main_bounds(df):
    e = df["e"].to_numpy(dtype=float)
    ep = df["Ep_QNE"].to_numpy(dtype=float)
    ef = df["E_f"].to_numpy(dtype=float)
    ih = df["I_half"].to_numpy(dtype=float)
    sr2h = df["SR2_half"].to_numpy(dtype=float)
    lower = df["lower_bound"].to_numpy(dtype=float)
    upper = df["upper_bound"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.fill_between(
        e,
        lower,
        upper,
        alpha=0.12,
        label=r"Theoretical interval $[L,U]$",
    )

    ax.plot(e, ef, linestyle=":", linewidth=1.1, label=r"$E_f$")
    ax.plot(e, ih, linestyle="-.", linewidth=1.1, label=r"$I(A:B)/2$")
    ax.plot(
        e,
        sr2h,
        linestyle="--",
        linewidth=1.2,
        label=r"$S_R^{(2)}(A:B)/2$",
    )
    ax.plot(
        e,
        lower,
        linewidth=1.5,
        label=r"$L=\max\{E_f,I/2,S_R^{(2)}/2\}$",
    )
    ax.plot(
        e,
        upper,
        linestyle=":",
        linewidth=1.5,
        label=r"$U=\min\{S_A,S_B\}$",
    )
    ax.plot(
        e,
        ep,
        marker="o",
        markersize=4.0,
        linewidth=1.8,
        label=r"VQP + QNE $E_p$",
    )

    ax.set_xlabel(r"Werner parameter $e$")
    ax.set_ylabel(r"$E_p$ (ebits)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, max(1.05, 1.05 * float(onp.nanmax(ep))))
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8.0, loc="best")

    fig.tight_layout()
    fig.savefig(FIG_MAIN_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_MAIN_PDF, bbox_inches="tight")
    plt.close(fig)


def _nearest_present_e(df, target):
    values = df["e"].drop_duplicates().to_numpy(dtype=float)
    return float(values[onp.argmin(onp.abs(values - target))])


def plot_stage1_convergence(df_s1):
    if len(df_s1) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for target_e in REPRESENTATIVE_E:
        e0 = _nearest_present_e(df_s1, target_e)
        sub = df_s1[onp.isclose(df_s1["e"], e0)].sort_values("epoch")
        if len(sub) == 0:
            continue
        ax.semilogy(
            sub["epoch"].to_numpy(dtype=int),
            onp.maximum(sub["HS2"].to_numpy(dtype=float), 1e-16),
            linewidth=1.3,
            label=fr"$e={e0:.2f}$",
        )

    ax.axhline(
        HS_TARGET,
        linestyle="--",
        linewidth=1.2,
        label=fr"target $10^{{{int(onp.log10(HS_TARGET))}}}$",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$D_{\mathrm{HS}}^2$")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8.0, loc="best")

    fig.tight_layout()
    fig.savefig(FIG_S1_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_S1_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_stage2_convergence(df_s2):
    if len(df_s2) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for target_e in REPRESENTATIVE_E:
        e0 = _nearest_present_e(df_s2, target_e)
        sub = df_s2[onp.isclose(df_s2["e"], e0)].sort_values("epoch")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["epoch"].to_numpy(dtype=int),
            sub["Ep_QNE"].to_numpy(dtype=float),
            linewidth=1.3,
            label=fr"$e={e0:.2f}$",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"QNE estimate of $E_p$ (ebits)")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8.0, loc="best")

    fig.tight_layout()
    fig.savefig(FIG_S2_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_S2_PDF, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Summary and entry point
# ============================================================

def print_summary(df):
    print("\n" + "#" * 84)
    print("FINAL SUMMARY")
    print("#" * 84)

    max_hs = float(df["stage1_HS2"].max())
    num_outside = int((~df["in_bounds_with_tolerance"].astype(bool)).sum())
    mean_qne_exact = float(onp.mean(onp.abs(df["qne_minus_exact_same_phi"].to_numpy(dtype=float))))

    print(f"Max Stage-1 D_HS^2       : {max_hs:.3e}")
    print(f"Required Stage-1 target  : < {HS_TARGET:.1e}")
    print(f"Points outside bounds    : {num_outside} / {len(df)}")
    print(f"Mean |QNE-exact| same phi: {mean_qne_exact:.3e} ebits")

    # Strong sanity checks for this Werner convention.
    idx025 = int(onp.argmin(onp.abs(df["e"].to_numpy(dtype=float) - 0.25)))
    idx1 = int(onp.argmin(onp.abs(df["e"].to_numpy(dtype=float) - 1.0)))

    row025 = df.iloc[idx025]
    row1 = df.iloc[idx1]

    print("\nSanity checks:")
    print(
        f"  e={row025['e']:.2f}: expected E_p=0, "
        f"QNE E_p={row025['Ep_QNE']:.9f}"
    )
    print(
        f"  e={row1['e']:.2f}: expected E_p=1, "
        f"QNE E_p={row1['Ep_QNE']:.9f}"
    )

    if num_outside > 0:
        print("\nWARNING: Some QNE E_p points lie outside the theoretical band.")
        print("Inspect Stage-2 convergence and QNE-exact diagnostics before using them.")

    print("\nFiles:")
    print(f"  Main CSV : {os.path.abspath(CSV_MAIN)}")
    print(f"  S1 CSV   : {os.path.abspath(CSV_STAGE1)}")
    print(f"  S2 CSV   : {os.path.abspath(CSV_STAGE2)}")
    print(f"  Main PNG : {os.path.abspath(FIG_MAIN_PNG)}")
    print(f"  Main PDF : {os.path.abspath(FIG_MAIN_PDF)}")
    print(f"  S1 PNG   : {os.path.abspath(FIG_S1_PNG)}")
    print(f"  S2 PNG   : {os.path.abspath(FIG_S2_PNG)}")


def run_main():
    t0 = time.time()

    print("\n" + "#" * 84)
    print("UNKNOWN WERNER STATE -> VQP -> AUXILIARY PURIFICATION SEARCH -> QNE -> E_p")
    print("#" * 84)
    print("Stage-1 target is accessed only through black-box state copies + SWAP tests.")
    print(f"Stage-1 acceptance criterion: D_HS^2 < {HS_TARGET:.1e}")
    print("Stage-2 entropy is estimated from qml.probs + 2-10-1 QNE network.")
    print("True Werner density matrices are used only AFTER training for bounds.")

    results = run_sweep()
    df, df_s1, df_s2 = export_results(results)

    plot_main_bounds(df)
    plot_stage1_convergence(df_s1)
    plot_stage2_convergence(df_s2)

    print_summary(df)
    print(f"\nTotal elapsed time: {(time.time() - t0) / 60.0:.2f} min")

    return df


if __name__ == "__main__":
    run_main()
