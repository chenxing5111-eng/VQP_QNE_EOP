"""
Random 1TN unknown-state variational purification + QNE estimation of E_p
4 auxiliary qubits, NO restarts, NO warm starts
=======================================================================

Purpose
-------
For each Haar-random 1TN pure state

    |psi>_ABC,  d_A=2, d_B=2, d_C=4,

construct the reduced two-qubit mixed state

    rho_AB = Tr_C |psi><psi|,

but treat rho_AB as an UNKNOWN black-box state during training.
The algorithm may only repeatedly prepare the target state; it does not read
rho_AB, its eigenvalues, or its matrix elements in Stage 1 or Stage 2.

The workflow is:

Stage 1: Variational quantum purification (VQP)
    Learn a six-qubit pure state |Phi(theta)>_{ABA'B'} whose AB marginal
    approximates the unknown target rho_AB.  A and B are one qubit each;
    A' and B' are two qubits each (four auxiliary qubits total).

    The squared Hilbert-Schmidt distance is

        D_HS^2(chi_AB, rho_AB)
          = Tr(chi_AB^2) + Tr(rho_AB^2) - 2 Tr(chi_AB rho_AB),

    and every term is evaluated using SWAP tests between state copies.
    No target density matrix is used by the Stage-1 optimizer.

    Required acceptance criterion:

        D_HS^2 < 1e-7

    for HS_CONSECUTIVE consecutive epochs.

    IMPORTANT: there are NO restarts.  Each 1TN target is trained exactly once
    from a deterministic state-specific random initialization.  If Stage 1 does
    not satisfy the threshold, that state is marked as failed and Stage 2 is
    skipped for that target.

Stage 2: Purification-space optimization + Quantum Neural Entropy estimation
    Freeze theta*.  Apply a 4-qubit hardware-efficient V_phi only on A'B' to
    explore purifications of the learned AB state.

    QNE estimates S(rho_AA') without constructing rho_AA' or diagonalizing it.
    Here AA' consists of three qubits, so d=8.  The QNE Hermitian ansatz is

        H(w,gamma) = sum_i f_w(i) U_gamma |i><i| U_gamma^dagger,

    where
      - U_gamma is a 6-layer Rot-CNOT circuit on AA',
      - f_w is a 3-15-1 classical neural network evaluated on all 8 bitstrings.

    The objective is

        J(phi,gamma,w)
          = sum_i p_{phi,gamma}(i) f_w(i)
            - (1/8) sum_i exp(f_w(i)).

    The QNE estimate is

        E_p^QNE = [ln(8) - 1 - max J] / ln(2).

    IMPORTANT: Stage 2 also uses exactly ONE state-specific random start.
    There are NO identity/random multiple starts, NO restarts, and NO warm start
    from neighboring 1TN samples.

Validation only (NOT used in training)
--------------------------------------
After optimization, the exact rho_AB is used only to compute:

    E_f,
    I(A:B)/2,
    S_R^(2)(A:B)/2,
    L = max{E_f, I/2, S_R^(2)/2},
    U = min{S(A), S(B)}.

qml.vn_entropy is evaluated only at the final QNE-selected phi as a simulator
 diagnostic.  It is never used by the QNE optimizer.

Random 1TN ensemble
-------------------
The state ensemble matches the previous 1TN experiments:

    d_A = 2, d_B = 2, d_C = 4,
    NUM_STATES = 100,
    STATE_SEED_BASE = 20260918.

Dependencies
------------
    pip install -U pennylane numpy scipy matplotlib pandas torch
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

# -------- Random 1TN ensemble --------
NUM_STATES = 100
STATE_SEED_BASE = 20260918

# -------- Optimization seeds --------
SEED_STAGE1 = 42
SEED_STAGE2 = 1024

# -------- Output --------
OUT_DIR = "2020random_1tn_unknown_vqp_qne_aux4_no_restart_results_all_HEA"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_MAIN = os.path.join(OUT_DIR, "random_1tn_vqp_qne_aux4_results.csv")
CSV_STAGE1 = os.path.join(OUT_DIR, "random_1tn_stage1_hs_convergence.csv")
CSV_STAGE2 = os.path.join(OUT_DIR, "random_1tn_stage2_qne_convergence.csv")
CSV_SUMMARY = os.path.join(OUT_DIR, "random_1tn_vqp_qne_aux4_summary.csv")

FIG_BOUNDS_PNG = os.path.join(OUT_DIR, "random_1tn_qne_theoretical_band.png")
FIG_BOUNDS_PDF = os.path.join(OUT_DIR, "random_1tn_qne_theoretical_band.pdf")
FIG_S1_PNG = os.path.join(OUT_DIR, "random_1tn_stage1_hs_convergence.png")
FIG_S1_PDF = os.path.join(OUT_DIR, "random_1tn_stage1_hs_convergence.pdf")
FIG_S2_PNG = os.path.join(OUT_DIR, "random_1tn_stage2_qne_convergence.png")
FIG_S2_PDF = os.path.join(OUT_DIR, "random_1tn_stage2_qne_convergence.pdf")

# -------- Stage 1: six-qubit learned purification --------
# A=1q, B=1q, A'=2q, B'=2q -> 6 qubits total.
N_AUX_PER_SIDE = 2
N_AUX_TOTAL = 4
N_LEARNED_QUBITS = 6

S1_LAYERS = 6
S1_MAX_EPOCHS = 4000
S1_LR = (0.06, 0.02, 0.005)
S1_INIT_SCALE = 0.05

HS_TARGET = 1e-7
HS_CONSECUTIVE = 10

# NO Stage-1 restarts.

# -------- Stage 2: four-qubit auxiliary V_phi + three-qubit QNE --------
# V_phi is a hardware-efficient unitary on A'B' rather than full SU(16).
VPHI_LAYERS = 12
VPHI_PARAM_SHAPE = (VPHI_LAYERS, N_AUX_TOTAL, 4)

# AA' has 3 qubits -> d=8. 6-layer Rot-CNOT HEA (54 parameters).
QNE_DIM = 8
GAMMA_LAYERS = 6
GAMMA_PARAM_SHAPE = (GAMMA_LAYERS, 3, 3)

S2_MAX_EPOCHS = 2400
S2_MIN_EPOCHS = 400
S2_PATIENCE = 80
S2_TOL = 1e-9

S2_LR_NN = (0.030, 0.010, 0.003)
S2_LR_PHI = (0.012, 0.004, 0.001)
S2_LR_GAMMA = (0.015, 0.006, 0.0015)
S2_INIT_SCALE = 0.05

# NO Stage-2 restarts and NO warm starts.

# Smooth numerical bound on the NN spectral outputs.
QNE_F_BOUND = 25.0

PRINT_STAGE1_EVERY = 100
PRINT_STAGE2_EVERY = 100

BOUND_TOL = 2e-3
NUMERICAL_EPS = 1e-14

# Representative random-state IDs for convergence plots.
REPRESENTATIVE_STATE_IDS = (1, 25, 50, 75, 100)


# ============================================================
# 1. Random 1TN state generation and black-box target oracle
# ============================================================

def generate_1tn_random_state(state_id):
    """
    Generate one Haar-random pure state |psi>_ABC with

        d_A=2, d_B=2, d_C=4.

    Returns
    -------
    psi_abc : complex ndarray, shape (16,)
        Pure 1TN state used as the hidden preparation oracle.
    rho_ab : complex ndarray, shape (4,4)
        Exact reduced target, used ONLY after training for validation/bounds.
    seed : int
        State seed.
    """
    state_id = int(state_id)
    seed = STATE_SEED_BASE + state_id
    rng = onp.random.default_rng(seed)

    z = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi_abc = z / onp.linalg.norm(z)

    # (AB,C) = (4,4)
    psi_matrix = psi_abc.reshape(4, 4)
    rho_ab = psi_matrix @ psi_matrix.conj().T
    rho_ab = 0.5 * (rho_ab + rho_ab.conj().T)
    rho_ab = rho_ab / onp.trace(rho_ab)

    return psi_abc, rho_ab, seed


def target_oracle_purification_1tn(psi_abc):
    """
    Embed |psi>_ABC into the six-qubit target-copy register.

    Target register ordering:
        [A, B, C0, C1, E0, E1].

    C already purifies rho_AB and has dimension 4 (two qubits).  Two additional
    environment qubits are appended in |00> only to make the target-copy register
    the same six-qubit size as the learned purification register.

        |target_oracle> = |psi>_ABC tensor |00>_E.
    """
    psi_abc = onp.asarray(psi_abc, dtype=complex).reshape(-1)
    if psi_abc.size != 16:
        raise ValueError("psi_abc must have 16 amplitudes for d_A=2,d_B=2,d_C=4.")

    zero_two_qubits = onp.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    state = onp.kron(psi_abc, zero_two_qubits)
    state = state / onp.linalg.norm(state)
    return state


def prepare_target_oracle(target_state_6q, wires):
    """Prepare one hidden six-qubit oracle copy.  rho_AB itself is never read."""
    qml.StatePrep(target_state_6q, wires=wires)


# ============================================================
# 2. Stage-1 VQP ansatz and SWAP-test Hilbert-Schmidt loss
# ============================================================

# Learned purification logical ordering:
# [A, B, A'_0, A'_1, B'_0, B'_1].
LEARNED_LOGICAL_WIRES = [0, 1, 2, 3, 4, 5]
SYSTEM_LOCAL = [0, 1]
AUX_LOCAL = [2, 3, 4, 5]
AA_LOCAL = [0, 2, 3]


def stage1_ansatz(params, wires):
    """Six-qubit variational purification ansatz: Rot + circular CNOT."""
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

        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def swap_test(control_wire, system1_wires, system2_wires):
    """
    Standard SWAP test on the selected subsystems.

    For reduced states rho and sigma on the selected registers,
        <Z_control> = Tr(rho sigma).
    """
    qml.Hadamard(wires=control_wire)
    for w1, w2 in zip(system1_wires, system2_wires):
        qml.CSWAP(wires=[control_wire, w1, w2])
    qml.Hadamard(wires=control_wire)


# 1 control + 6-qubit copy 1 + 6-qubit copy 2 = 13 qubits.
DEV_SWAP = qml.device("default.qubit", wires=13, shots=None)

CTRL = 0
COPY1 = [1, 2, 3, 4, 5, 6]
COPY2 = [7, 8, 9, 10, 11, 12]
COPY1_AB = [1, 2]
COPY2_AB = [7, 8]


@qml.qnode(DEV_SWAP, interface="autograd", diff_method="backprop")
def learned_purity_swap_qnode(params):
    """Return Tr[chi_AB(theta)^2] from two learned purification copies."""
    stage1_ansatz(params, COPY1)
    stage1_ansatz(params, COPY2)
    swap_test(CTRL, COPY1_AB, COPY2_AB)
    return qml.expval(qml.PauliZ(CTRL))


@qml.qnode(DEV_SWAP, interface="autograd", diff_method="backprop")
def learned_target_overlap_swap_qnode(params, target_state_6q):
    """Return Tr[chi_AB(theta) rho_AB] without reading rho_AB."""
    stage1_ansatz(params, COPY1)
    prepare_target_oracle(target_state_6q, COPY2)
    swap_test(CTRL, COPY1_AB, COPY2_AB)
    return qml.expval(qml.PauliZ(CTRL))


@qml.qnode(DEV_SWAP, interface="autograd", diff_method=None)
def target_purity_swap_qnode(target_state_6q):
    """Return Tr[rho_AB^2] using two black-box target copies."""
    prepare_target_oracle(target_state_6q, COPY1)
    prepare_target_oracle(target_state_6q, COPY2)
    swap_test(CTRL, COPY1_AB, COPY2_AB)
    return qml.expval(qml.PauliZ(CTRL))


def stage1_train_cost(params, target_state_6q):
    """
    Parameter-dependent part of D_HS^2:

        Tr(chi^2) - 2 Tr(chi rho).

    Tr(rho^2) is constant and omitted only from the optimizer objective.
    """
    purity_chi = learned_purity_swap_qnode(params)
    overlap = learned_target_overlap_swap_qnode(params, target_state_6q)
    return purity_chi - 2.0 * overlap


def stage1_full_hs2(params, target_state_6q, target_purity):
    """Full D_HS^2 evaluated entirely by SWAP tests."""
    value = stage1_train_cost(params, target_state_6q) + target_purity
    value = float(value)
    return max(value, 0.0)


def _stage1_lr_for_epoch(epoch):
    frac = epoch / S1_MAX_EPOCHS
    if frac < 0.60:
        return S1_LR[0]
    if frac < 0.85:
        return S1_LR[1]
    return S1_LR[2]


def train_stage1_one_state(target_state_6q, state_id):
    """
    Train Stage 1 exactly ONCE for one random 1TN target.

    No restart and no warm start are used.
    """
    t0 = time.time()
    state_id = int(state_id)

    target_purity = float(target_purity_swap_qnode(target_state_6q))

    rng = onp.random.default_rng(SEED_STAGE1 + 1000 * state_id)
    raw = rng.normal(
        0.0,
        S1_INIT_SCALE,
        size=(S1_LAYERS, N_LEARNED_QUBITS, 3),
    )

    params = np.array(raw, requires_grad=True)
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
            lambda p: stage1_train_cost(p, target_state_6q),
            params,
        )

        hs_now = stage1_full_hs2(params, target_state_6q, target_purity)
        history.append({"epoch": epoch, "hs2": hs_now})

        if hs_now < best_hs:
            best_hs = hs_now
            best_params = np.array(onp.asarray(params), requires_grad=False)
            best_epoch = epoch

        consecutive = consecutive + 1 if hs_now < HS_TARGET else 0

        if epoch == 1 or epoch % PRINT_STAGE1_EVERY == 0 or consecutive == HS_CONSECUTIVE:
            print(
                f"    [S1] epoch={epoch:4d}  "
                f"D_HS^2={hs_now:.3e}  best={best_hs:.3e}"
            )

        if consecutive >= HS_CONSECUTIVE:
            break

    success = bool(consecutive >= HS_CONSECUTIVE)

    return {
        "success": success,
        "best_hs2": float(best_hs),
        "best_params": best_params,
        "best_epoch": int(best_epoch),
        "last_epoch": int(epoch),
        "target_purity_swap": float(target_purity),
        "history": history,
        "runtime_sec": time.time() - t0,
    }


# ============================================================
# 3. Stage-2 QNE: 4-qubit Rot-CRY V_phi + 3-qubit Rot-CNOT QNE
# ============================================================

DEV_QNE = qml.device("default.qubit", wires=6, shots=None)

A_WIRE = 0
B_WIRE = 1
AP_WIRES = [2, 3]
BP_WIRES = [4, 5]
AUX_WIRES = AP_WIRES + BP_WIRES
AA_WIRES = [A_WIRE] + AP_WIRES
AB_WIRES = [A_WIRE, B_WIRE]
ALL_WIRES = [0, 1, 2, 3, 4, 5]


def v_phi_ansatz(phi):
    """Hardware-efficient 4-qubit purification-space unitary on A'B'."""
    n_aux = len(AUX_WIRES)

    for layer in range(VPHI_LAYERS):
        for i, w in enumerate(AUX_WIRES):
            qml.Rot(
                phi[layer, i, 0],
                phi[layer, i, 1],
                phi[layer, i, 2],
                wires=w,
            )

        for i in range(n_aux):
            qml.CRY(
                phi[layer, i, 3],
                wires=[AUX_WIRES[i], AUX_WIRES[(i + 1) % n_aux]],
            )



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


def u_gamma_ansatz(gamma):
    """Trainable HEA measurement-basis unitary on AA'; used as adjoint."""
    hea_rot_cnot_ansatz(gamma, AA_WIRES)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def qne_probs_qnode(phi, gamma, frozen_theta):
    """
    learned purification(theta*)
        -> V_phi on A'B'
        -> U_gamma^dagger on AA'
        -> qml.probs on AA'.
    """
    stage1_ansatz(frozen_theta, ALL_WIRES)
    v_phi_ansatz(phi)
    qml.adjoint(u_gamma_ansatz)(gamma)
    return qml.probs(wires=AA_WIRES)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def exact_entropy_diagnostic_qnode(phi, frozen_theta):
    """Simulator-only S(AA') diagnostic; never used by the optimizer."""
    stage1_ansatz(frozen_theta, ALL_WIRES)
    v_phi_ansatz(phi)
    return qml.vn_entropy(wires=AA_WIRES, log_base=2)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def rho_ab_before_stage2_diagnostic_qnode(frozen_theta):
    stage1_ansatz(frozen_theta, ALL_WIRES)
    return qml.density_matrix(wires=AB_WIRES)


@qml.qnode(DEV_QNE, interface="torch", diff_method="backprop")
def rho_ab_after_stage2_diagnostic_qnode(phi, frozen_theta):
    stage1_ansatz(frozen_theta, ALL_WIRES)
    v_phi_ansatz(phi)
    return qml.density_matrix(wires=AB_WIRES)


class QNENet(torch.nn.Module):
    """3-15-1 QNE network for the d=8 Hermitian spectrum."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 15)
        self.fc2 = torch.nn.Linear(15, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


BITSTRINGS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=torch.float64,
)


def bounded_f_values(net):
    raw = net(BITSTRINGS).reshape(-1)
    return QNE_F_BOUND * torch.tanh(raw / QNE_F_BOUND)


def qne_J_from_probs(probs, f_values):
    """J = sum p_i f_i - (1/8) sum exp(f_i)."""
    return torch.sum(probs * f_values) - torch.mean(torch.exp(f_values))


def qne_entropy_bits_from_J(J_value):
    return (onp.log(float(QNE_DIM)) - 1.0 - float(J_value)) / onp.log(2.0)


def _set_optimizer_lr(optimizer, lr_values):
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


def optimize_stage2_qne(frozen_theta_np, state_id):
    """
    Run Stage 2 exactly ONCE for one random 1TN target.

    No restart, no multiple starts, and no warm start are used.
    """
    t0 = time.time()
    state_id = int(state_id)
    seed = SEED_STAGE2 + 1000 * state_id

    frozen_theta = torch.tensor(
        onp.asarray(frozen_theta_np, dtype=onp.float64),
        dtype=torch.float64,
        requires_grad=False,
    )

    rng = onp.random.default_rng(seed)
    phi0 = rng.normal(0.0, S2_INIT_SCALE, size=VPHI_PARAM_SHAPE)
    gamma0 = rng.normal(0.0, S2_INIT_SCALE, size=GAMMA_PARAM_SHAPE)

    torch.manual_seed(seed + 17)
    net = QNENet().double()

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

        # A. NN update with quantum probabilities treated as fixed data.
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

        # B. Quantum update with the newly updated NN held fixed.
        q_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            f_fixed = bounded_f_values(net).detach()

        probs = qne_probs_qnode(phi, gamma, frozen_theta)
        J_quantum = qne_J_from_probs(probs, f_fixed)
        loss_quantum = -J_quantum

        if not torch.isfinite(loss_quantum):
            break

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
                f"    [S2] epoch={epoch:4d}  "
                f"J={J_now:+.10f}  Ep_QNE={ep_now:.9f}  "
                f"best={qne_entropy_bits_from_J(best_J):.9f}"
            )

        loss_quantum.backward()
        q_optimizer.step()

        if epoch >= S2_MIN_EPOCHS and stable >= S2_PATIENCE:
            break

    # Evaluate the final state after the final quantum update.
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
        raise RuntimeError(f"Stage 2 failed for state_id={state_id}: no finite QNE objective.")

    best_phi_t = torch.tensor(best_phi, dtype=torch.float64, requires_grad=False)

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

    ep_qne = float(qne_entropy_bits_from_J(best_J))

    return {
        "J_max": float(best_J),
        "Ep_QNE": ep_qne,
        "phi": best_phi,
        "gamma": best_gamma,
        "nn_state": best_nn_state,
        "best_epoch": int(best_epoch),
        "last_epoch": int(epoch),
        "history": history,
        "exact_entropy_at_qne_phi": exact_entropy,
        "qne_minus_exact_same_phi": float(ep_qne - exact_entropy),
        "ab_invariant_hs2": ab_invariant_hs2,
        "runtime_sec": time.time() - t0,
    }


# ============================================================
# 4. Generic two-qubit theoretical bounds (validation only)
# ============================================================

def binary_entropy(x):
    x = float(onp.clip(x, 0.0, 1.0))
    if x <= NUMERICAL_EPS or 1.0 - x <= NUMERICAL_EPS:
        return 0.0
    return -(x * onp.log2(x) + (1.0 - x) * onp.log2(1.0 - x))


def von_neumann_entropy_matrix(rho):
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


def purity(rho):
    rho = onp.asarray(rho, dtype=complex)
    return float(onp.real(onp.trace(rho @ rho)))


def concurrence_two_qubit(rho_ab):
    """Wootters concurrence for a general two-qubit mixed state."""
    rho = onp.asarray(rho_ab, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / onp.trace(rho)

    sigma_y = onp.array(
        [[0.0, -1j], [1j, 0.0]],
        dtype=complex,
    )
    yy = onp.kron(sigma_y, sigma_y)

    rho_tilde = yy @ rho.conj() @ yy
    R = rho @ rho_tilde

    eigvals = onp.linalg.eigvals(R)
    eigvals = onp.real_if_close(eigvals, tol=1000)
    eigvals = onp.real(eigvals)
    eigvals = onp.clip(eigvals, 0.0, None)

    roots = onp.sqrt(eigvals)
    roots = onp.sort(roots)[::-1]

    if len(roots) < 4:
        roots = onp.pad(roots, (0, 4 - len(roots)))

    C = max(0.0, float(roots[0] - roots[1] - roots[2] - roots[3]))
    return float(onp.clip(C, 0.0, 1.0))


def entanglement_of_formation_two_qubit(rho_ab):
    C = concurrence_two_qubit(rho_ab)
    if C <= NUMERICAL_EPS:
        return 0.0

    x = (1.0 + onp.sqrt(max(0.0, 1.0 - C * C))) / 2.0
    return binary_entropy(x)


def matrix_sqrt_psd(rho):
    rho = onp.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = onp.linalg.eigh(rho)
    vals = onp.clip(vals.real, 0.0, None)
    return (vecs * onp.sqrt(vals)) @ vecs.conj().T


def renyi2_reflected_entropy(rho_ab):
    r"""
    S_R^(2)(A:B) = -log_2 Tr[(rho_AA*)^2]
    using the canonical purification |sqrt(rho_AB)>.
    """
    rho = onp.asarray(rho_ab, dtype=complex)
    sqrt_rho = matrix_sqrt_psd(rho)

    # Tensor ordering A,B,A*,B*.
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
    p = float(onp.real(onp.trace(rho_aa_star @ rho_aa_star)))
    p = float(onp.clip(p, NUMERICAL_EPS, 1.0))

    s_r2 = float(-onp.log2(p))
    return s_r2, p


def theoretical_bounds_general_two_qubit(rho_ab):
    """
    Compute post-training theoretical E_p bounds for an arbitrary two-qubit state.
    These quantities NEVER enter Stage-1 or Stage-2 training.
    """
    rho_ab = onp.asarray(rho_ab, dtype=complex)
    rho_ab = 0.5 * (rho_ab + rho_ab.conj().T)
    rho_ab = rho_ab / onp.trace(rho_ab)

    rho_a = partial_trace_two_qubit(rho_ab, "A")
    rho_b = partial_trace_two_qubit(rho_ab, "B")

    s_a = von_neumann_entropy_matrix(rho_a)
    s_b = von_neumann_entropy_matrix(rho_b)
    s_ab = von_neumann_entropy_matrix(rho_ab)

    mutual_information = max(0.0, s_a + s_b - s_ab)
    i_half = 0.5 * mutual_information

    e_f = entanglement_of_formation_two_qubit(rho_ab)

    s_r2, reflected_purity = renyi2_reflected_entropy(rho_ab)
    sr2_half = 0.5 * s_r2

    lower_candidates = {
        "E_f": float(e_f),
        "I_half": float(i_half),
        "SR2_half": float(sr2_half),
    }
    strongest_lower = max(lower_candidates, key=lower_candidates.get)
    lower = lower_candidates[strongest_lower]
    upper = min(s_a, s_b)

    return {
        "S_A": float(s_a),
        "S_B": float(s_b),
        "S_AB": float(s_ab),
        "mutual_information": float(mutual_information),
        "I_half": float(i_half),
        "E_f": float(e_f),
        "concurrence": float(concurrence_two_qubit(rho_ab)),
        "S_R2": float(s_r2),
        "SR2_half": float(sr2_half),
        "reflected_purity": float(reflected_purity),
        "strongest_lower": strongest_lower,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
    }


# ============================================================
# 5. One 1TN state: generation -> Stage 1 -> Stage 2 -> bounds
# ============================================================

def run_one_state(state_id):
    state_id = int(state_id)
    t0 = time.time()

    psi_abc, rho_ab, state_seed = generate_1tn_random_state(state_id)
    target_state_6q = target_oracle_purification_1tn(psi_abc)

    eigvals = onp.linalg.eigvalsh(rho_ab).real
    rank = int(onp.sum(eigvals > 1e-10))
    rho_purity = purity(rho_ab)

    bounds = theoretical_bounds_general_two_qubit(rho_ab)

    print("\n" + "=" * 92)
    print(
        f"Random 1TN state {state_id:3d}/{NUM_STATES}  "
        f"seed={state_seed}  rank={rank}  purity={rho_purity:.6f}"
    )
    print("Target rho_AB is treated as an unknown black-box state during training.")
    print("=" * 92)

    print("[Stage 1] Single-start variational purification from SWAP-test HS loss")
    s1 = train_stage1_one_state(target_state_6q, state_id)

    print(
        f"  Stage 1 result: D_HS^2={s1['best_hs2']:.3e}, "
        f"target < {HS_TARGET:.1e}, best_epoch={s1['best_epoch']}, "
        f"success={s1['success']}"
    )

    # Strict rule: do not use an inaccurate Stage-1 purification for E_p.
    if not s1["success"]:
        print(
            "  Stage 1 did not satisfy the strict threshold. "
            "No restart is allowed, so Stage 2 is skipped for this state."
        )

        return {
            "state_id": state_id,
            "state_seed": state_seed,
            "rank": rank,
            "purity": rho_purity,
            "stage1": s1,
            "stage2": None,
            "bounds": bounds,
            "in_bounds": None,
            "elapsed_seconds": time.time() - t0,
        }

    print("[Stage 2] Single-start 4-qubit auxiliary search + 3-qubit QNE")
    s2 = optimize_stage2_qne(s1["best_params"], state_id)

    ep = float(s2["Ep_QNE"])
    lower = float(bounds["lower_bound"])
    upper = float(bounds["upper_bound"])
    in_bounds = bool(
        ep >= lower - BOUND_TOL
        and ep <= upper + BOUND_TOL
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
        "state_id": state_id,
        "state_seed": state_seed,
        "rank": rank,
        "purity": rho_purity,
        "stage1": s1,
        "stage2": s2,
        "bounds": bounds,
        "in_bounds": in_bounds,
        "elapsed_seconds": time.time() - t0,
    }


# ============================================================
# 6. Export helpers
# ============================================================

def _main_row_from_result(r):
    state_id = r["state_id"]
    s1 = r["stage1"]
    s2 = r["stage2"]
    b = r["bounds"]

    base = {
        "state_id": state_id,
        "state_seed": r["state_seed"],
        "rank": r["rank"],
        "purity": r["purity"],
        "aux_qubits_total": N_AUX_TOTAL,
        "A_prime_qubits": N_AUX_PER_SIDE,
        "B_prime_qubits": N_AUX_PER_SIDE,
        "QNE_dimension": QNE_DIM,
        "Vphi_layers": VPHI_LAYERS,
        "Gamma_layers": GAMMA_LAYERS,
        "Gamma_parameters": 54,
        "stage1_success": bool(s1["success"]),
        "stage1_HS2": s1["best_hs2"],
        "stage1_best_epoch": s1["best_epoch"],
        "stage1_last_epoch": s1["last_epoch"],
        "target_purity_swap": s1["target_purity_swap"],
        "stage1_runtime_sec": s1["runtime_sec"],
        "S_A": b["S_A"],
        "S_B": b["S_B"],
        "S_AB": b["S_AB"],
        "mutual_information": b["mutual_information"],
        "I_half": b["I_half"],
        "E_f": b["E_f"],
        "concurrence": b["concurrence"],
        "S_R2": b["S_R2"],
        "SR2_half": b["SR2_half"],
        "reflected_purity": b["reflected_purity"],
        "strongest_lower": b["strongest_lower"],
        "lower_bound": b["lower_bound"],
        "upper_bound": b["upper_bound"],
        "elapsed_seconds": r["elapsed_seconds"],
    }

    if s2 is None:
        base.update({
            "Ep_QNE": onp.nan,
            "J_max": onp.nan,
            "stage2_best_epoch": onp.nan,
            "stage2_last_epoch": onp.nan,
            "stage2_runtime_sec": onp.nan,
            "exact_entropy_at_qne_phi": onp.nan,
            "qne_minus_exact_same_phi": onp.nan,
            "ab_invariant_HS2": onp.nan,
            "Ep_minus_lower": onp.nan,
            "upper_minus_Ep": onp.nan,
            "within_bounds_with_tolerance": onp.nan,
        })
        return base

    ep = float(s2["Ep_QNE"])
    lower = float(b["lower_bound"])
    upper = float(b["upper_bound"])

    base.update({
        "Ep_QNE": ep,
        "J_max": s2["J_max"],
        "stage2_best_epoch": s2["best_epoch"],
        "stage2_last_epoch": s2["last_epoch"],
        "stage2_runtime_sec": s2["runtime_sec"],
        "exact_entropy_at_qne_phi": s2["exact_entropy_at_qne_phi"],
        "qne_minus_exact_same_phi": s2["qne_minus_exact_same_phi"],
        "ab_invariant_HS2": s2["ab_invariant_hs2"],
        "Ep_minus_lower": ep - lower,
        "upper_minus_Ep": upper - ep,
        "within_bounds_with_tolerance": bool(r["in_bounds"]),
    })

    return base


def export_results(results):
    rows = []
    s1_rows = []
    s2_rows = []

    for r in results:
        state_id = r["state_id"]
        rows.append(_main_row_from_result(r))

        for item in r["stage1"]["history"]:
            s1_rows.append({
                "state_id": state_id,
                "epoch": item["epoch"],
                "HS2": item["hs2"],
            })

        if r["stage2"] is not None:
            for item in r["stage2"]["history"]:
                s2_rows.append({
                    "state_id": state_id,
                    "epoch": item["epoch"],
                    "J": item["J"],
                    "Ep_QNE": item["Ep_QNE"],
                })

    df = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    df_s1 = pd.DataFrame(s1_rows)
    df_s2 = pd.DataFrame(s2_rows)

    df.to_csv(CSV_MAIN, index=False)
    df_s1.to_csv(CSV_STAGE1, index=False)
    df_s2.to_csv(CSV_STAGE2, index=False)

    successful = df[df["stage1_success"].astype(bool) & df["Ep_QNE"].notna()].copy()

    n_total = len(df)
    n_s1_success = int(df["stage1_success"].astype(bool).sum())
    n_stage2 = len(successful)

    if n_stage2 > 0:
        within_series = successful["within_bounds_with_tolerance"].astype(bool)
        n_outside = int((~within_series).sum())
        mean_abs_qne_exact = float(
            onp.nanmean(onp.abs(successful["qne_minus_exact_same_phi"].to_numpy(dtype=float)))
        )
        max_hs_success = float(successful["stage1_HS2"].max())
    else:
        n_outside = 0
        mean_abs_qne_exact = onp.nan
        max_hs_success = onp.nan

    summary = pd.DataFrame([
        {
            "NUM_STATES": n_total,
            "STATE_SEED_BASE": STATE_SEED_BASE,
            "stage1_success_count": n_s1_success,
            "stage1_failure_count": n_total - n_s1_success,
            "stage2_evaluated_count": n_stage2,
            "outside_bounds_count": n_outside,
            "mean_abs_QNE_minus_exact_same_phi": mean_abs_qne_exact,
            "max_stage1_HS2_among_successful": max_hs_success,
            "HS_TARGET": HS_TARGET,
            "aux_qubits_total": N_AUX_TOTAL,
            "QNE_dimension": QNE_DIM,
            "restarts": 0,
            "warm_start": False,
        }
    ])
    summary.to_csv(CSV_SUMMARY, index=False)

    return df, df_s1, df_s2, summary


# ============================================================
# 7. Plotting
# ============================================================

def plot_theoretical_band(df):
    plot_df = df[
        df["stage1_success"].astype(bool)
        & df["Ep_QNE"].notna()
    ].copy()

    if len(plot_df) == 0:
        return

    # Match the previous 1TN visualization: sort states by the estimated E_p.
    plot_df = plot_df.sort_values("Ep_QNE").reset_index(drop=True)

    x = onp.arange(1, len(plot_df) + 1)
    lower = plot_df["lower_bound"].to_numpy(dtype=float)
    upper = plot_df["upper_bound"].to_numpy(dtype=float)
    ep = plot_df["Ep_QNE"].to_numpy(dtype=float)
    within = plot_df["within_bounds_with_tolerance"].astype(bool).to_numpy()

    fig, ax = plt.subplots(figsize=(7.3, 4.5))

    ax.fill_between(
        x,
        lower,
        upper,
        alpha=0.22,
        label="Theoretical interval",
    )
    ax.plot(x, lower, linestyle="--", linewidth=1.0, label=r"Lower bound $L$")
    ax.plot(x, upper, linestyle=":", linewidth=1.0, label=r"Upper bound $U$")
    ax.plot(
        x,
        ep,
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        label=r"VQP + QNE $E_p$",
    )

    if onp.any(~within):
        ax.scatter(
            x[~within],
            ep[~within],
            marker="x",
            s=42,
            linewidths=1.3,
            label="Outside theoretical interval",
            zorder=5,
        )

    ax.set_xlabel(r"Random 1TN states sorted by $E_p^{\mathrm{QNE}}$")
    ax.set_ylabel(r"$E_p$ (ebits)")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8.0, loc="best")

    ymin = min(-0.02, float(onp.nanmin(lower)) - 0.03)
    ymax = max(1.05, float(onp.nanmax(onp.concatenate([upper, ep]))) * 1.05)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(FIG_BOUNDS_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_BOUNDS_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_stage1_convergence(df_s1):
    if len(df_s1) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for state_id in REPRESENTATIVE_STATE_IDS:
        sub = df_s1[df_s1["state_id"] == state_id].sort_values("epoch")
        if len(sub) == 0:
            continue

        ax.semilogy(
            sub["epoch"].to_numpy(dtype=int),
            onp.maximum(sub["HS2"].to_numpy(dtype=float), 1e-16),
            linewidth=1.3,
            label=f"State {state_id}",
        )

    ax.axhline(
        HS_TARGET,
        linestyle="--",
        linewidth=1.2,
        label=r"Stage-1 target $10^{-7}$",
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

    for state_id in REPRESENTATIVE_STATE_IDS:
        sub = df_s2[df_s2["state_id"] == state_id].sort_values("epoch")
        if len(sub) == 0:
            continue

        ax.plot(
            sub["epoch"].to_numpy(dtype=int),
            sub["Ep_QNE"].to_numpy(dtype=float),
            linewidth=1.3,
            label=f"State {state_id}",
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
# 8. Main experiment
# ============================================================

def run_main():
    t0 = time.time()

    print("\n" + "#" * 92)
    print("RANDOM 1TN UNKNOWN STATE -> SINGLE-START VQP -> SINGLE-START QNE -> E_p")
    print("#" * 92)
    print("1TN ensemble: d_A=2, d_B=2, d_C=4")
    print(f"Number of random states : {NUM_STATES}")
    print(f"STATE_SEED_BASE         : {STATE_SEED_BASE}")
    print("Purification auxiliaries: dim(A')=4, dim(B')=4 (4 auxiliary qubits total)")
    print("Stage 1                 : one random start, no restart, no warm start")
    print(f"Stage-1 acceptance      : D_HS^2 < {HS_TARGET:.1e} for {HS_CONSECUTIVE} epochs")
    print("Stage 2                 : one random start, no restart, no warm start")
    print("QNE                     : AA' has 3 qubits, d=8, network 3-15-1")
    print("True rho_AB is used only after training for bounds/validation.")
    print("#" * 92)

    results = []

    for state_id in range(1, NUM_STATES + 1):
        result = run_one_state(state_id)
        results.append(result)

        # Incremental main CSV save after every state, so long runs are not lost.
        partial_rows = [_main_row_from_result(r) for r in results]
        pd.DataFrame(partial_rows).sort_values("state_id").to_csv(CSV_MAIN, index=False)

        if result["stage2"] is None:
            print(
                f"[{state_id:3d}/{NUM_STATES}] STAGE1-FAIL  "
                f"HS1={result['stage1']['best_hs2']:.2e}  "
                f"time={result['elapsed_seconds']:.1f}s"
            )
        else:
            status = "BOUND-OK" if result["in_bounds"] else "BOUND-OUT"
            print(
                f"[{state_id:3d}/{NUM_STATES}] "
                f"Ep_QNE={result['stage2']['Ep_QNE']:.7f}  "
                f"L={result['bounds']['lower_bound']:.7f}  "
                f"U={result['bounds']['upper_bound']:.7f}  "
                f"{status}  "
                f"HS1={result['stage1']['best_hs2']:.2e}  "
                f"time={result['elapsed_seconds']:.1f}s"
            )

    df, df_s1, df_s2, summary = export_results(results)

    plot_theoretical_band(df)
    plot_stage1_convergence(df_s1)
    plot_stage2_convergence(df_s2)

    n_success = int(df["stage1_success"].astype(bool).sum())
    valid = df[df["Ep_QNE"].notna()].copy()
    n_outside = 0
    mean_abs_qne_exact = onp.nan

    if len(valid) > 0:
        n_outside = int((~valid["within_bounds_with_tolerance"].astype(bool)).sum())
        mean_abs_qne_exact = float(
            onp.nanmean(onp.abs(valid["qne_minus_exact_same_phi"].to_numpy(dtype=float)))
        )

    print("\n" + "#" * 92)
    print("FINAL SUMMARY")
    print("#" * 92)
    print(f"Total states                : {len(df)}")
    print(f"Stage-1 successes           : {n_success}")
    print(f"Stage-1 failures            : {len(df) - n_success}")
    print(f"Stage-2 evaluated states    : {len(valid)}")
    print(f"Points outside bounds       : {n_outside} / {len(valid)}")
    print(f"Mean |QNE-exact| same phi   : {mean_abs_qne_exact:.3e} ebits")
    print("Restarts                    : 0")
    print("Warm starts                 : False")

    print("\nOutput files:")
    print(f"  Main CSV    : {os.path.abspath(CSV_MAIN)}")
    print(f"  Summary CSV : {os.path.abspath(CSV_SUMMARY)}")
    print(f"  Stage1 CSV  : {os.path.abspath(CSV_STAGE1)}")
    print(f"  Stage2 CSV  : {os.path.abspath(CSV_STAGE2)}")
    print(f"  Bounds PNG  : {os.path.abspath(FIG_BOUNDS_PNG)}")
    print(f"  Bounds PDF  : {os.path.abspath(FIG_BOUNDS_PDF)}")
    print(f"  Stage1 PNG  : {os.path.abspath(FIG_S1_PNG)}")
    print(f"  Stage2 PNG  : {os.path.abspath(FIG_S2_PNG)}")
    print(f"Total elapsed time          : {(time.time() - t0) / 60.0:.2f} min")

    return df


if __name__ == "__main__":
    run_main()
