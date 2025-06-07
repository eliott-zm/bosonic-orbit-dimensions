import numpy as np
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Any


### Bosonic operators ###

def apply_a(i, state):
    n = list(state)
    if n[i-1] == 0:
        return []
    coeff = np.sqrt(n[i-1])
    new_n = n.copy()
    new_n[i-1] -= 1
    return [(coeff, tuple(new_n))]

def apply_adag(i, state):
    n = list(state)
    new_n = n.copy()
    new_n[i-1] += 1
    coeff = np.sqrt(new_n[i-1])
    return [(coeff, tuple(new_n))]

def e_kl_op(k, l):
    def op(state):
        term1 = []
        a_l_terms = apply_a(l, state)
        for coeff_a_l, s1 in a_l_terms:
            for coeff_adag_k, s2 in apply_adag(k, s1):
                term1.append((coeff_a_l * coeff_adag_k, s2))
        term2 = []
        a_k_terms = apply_a(k, state)
        for coeff_a_k, s1 in a_k_terms:
            for coeff_adag_l, s2 in apply_adag(l, s1):
                term2.append((coeff_a_k * coeff_adag_l, s2))
        combined = [(0.5 * coeff, s) for coeff, s in term1 + term2]
        return combined
    return op

def E_kl_op(k, l):
    def op(state):
        term1 = []
        a_l_terms = apply_a(l, state)
        for coeff_a_l, s1 in a_l_terms:
            for coeff_adag_k, s2 in apply_adag(k, s1):
                term1.append((coeff_a_l * coeff_adag_k, s2))
        term2 = []
        a_k_terms = apply_a(k, state)
        for coeff_a_k, s1 in a_k_terms:
            for coeff_adag_l, s2 in apply_adag(l, s1):
                term2.append((coeff_a_k * coeff_adag_l, s2))
        combined = []
        for coeff, s in term1:
            combined.append((0.5j * coeff, s))
        for coeff, s in term2:
            combined.append((-0.5j * coeff, s))
        return combined
    return op

def r_kl_op(k, l):
    def op(state):
        term1 = []
        for coeff1, s1 in apply_adag(k, state):
            for coeff2, s2 in apply_adag(l, s1):
                term1.append((coeff1 * coeff2, s2))
        term2 = []
        for coeff_l, s1 in apply_a(l, state):
            for coeff_k, s2 in apply_a(k, s1):
                term2.append((coeff_l * coeff_k, s2))
        combined = [(0.5 * coeff, s) for coeff, s in term1 + term2]
        return combined
    return op

def R_kl_op(k, l):
    def op(state):
        term1 = []
        for coeff1, s1 in apply_adag(k, state):
            for coeff2, s2 in apply_adag(l, s1):
                term1.append((coeff1 * coeff2, s2))
        term2 = []
        for coeff_l, s1 in apply_a(l, state):
            for coeff_k, s2 in apply_a(k, s1):
                term2.append((coeff_l * coeff_k, s2))
        combined = []
        for coeff, s in term1:
            combined.append((0.5j * coeff, s))
        for coeff, s in term2:
            combined.append((-0.5j * coeff, s))
        return combined
    return op

def N_k_op(k):
    def op(state):
        return [(state[k-1], state)]
    return op

def s_k_op(k):
    def op(state):
        term1 = []
        for coeff1, s1 in apply_adag(k, state):
            for coeff2, s2 in apply_adag(k, s1):
                term1.append((coeff1 * coeff2, s2))
        term2 = []
        for coeff1, s1 in apply_a(k, state):
            for coeff2, s2 in apply_a(k, s1):
                term2.append((coeff1 * coeff2, s2))
        combined = [(0.5 * coeff, s) for coeff, s in term1 + term2]
        return combined
    return op

def S_k_op(k):
    def op(state):
        term1 = []
        for coeff1, s1 in apply_adag(k, state):
            for coeff2, s2 in apply_adag(k, s1):
                term1.append((coeff1 * coeff2, s2))
        term2 = []
        for coeff1, s1 in apply_a(k, state):
            for coeff2, s2 in apply_a(k, s1):
                term2.append((coeff1 * coeff2, s2))
        combined = []
        for coeff, s in term1:
            combined.append((0.5j * coeff, s))
        for coeff, s in term2:
            combined.append((-0.5j * coeff, s))
        return combined
    return op

def q_k_op(k):
    def op(state):
        terms = []
        for coeff, s in apply_adag(k, state):
            terms.append((coeff / np.sqrt(2), s))
        for coeff, s in apply_a(k, state):
            terms.append((coeff / np.sqrt(2), s))
        return terms
    return op

def p_k_op(k):
    def op(state):
        terms = []
        for coeff, s in apply_adag(k, state):
            terms.append((1j * coeff / np.sqrt(2), s))
        for coeff, s in apply_a(k, state):
            terms.append((-1j * coeff / np.sqrt(2), s))
        return terms
    return op

def identity_op(state):
    return [(1.0, state)]

def generate_operators(m, algebra='GO'):
    operators = []
    if algebra == 'OLO':
        for k in range(1, m + 1):
            for l in range(k + 1, m + 1):
                operators.append((f'e_{k}{l}', e_kl_op(k, l)))
    elif algebra == 'PLO':
        for k in range(1, m + 1):
            for l in range(k + 1, m + 1):
                operators.append((f'e_{k}{l}', e_kl_op(k, l)))
                operators.append((f'E_{k}{l}', E_kl_op(k, l)))
        for k in range(1, m + 1):
            operators.append((f'N_{k}', N_k_op(k)))
    elif algebra == 'DPLO':
        for k in range(1, m + 1):
            for l in range(k + 1, m + 1):
                operators.append((f'e_{k}{l}', e_kl_op(k, l)))
                operators.append((f'E_{k}{l}', E_kl_op(k, l)))
        for k in range(1, m + 1):
            operators.append((f'N_{k}', N_k_op(k)))
            operators.append((f'q_{k}', q_k_op(k)))
            operators.append((f'p_{k}', p_k_op(k)))
        operators.append(('I', identity_op))
    elif algebra == 'ALO':
        for k in range(1, m + 1):
            for l in range(k + 1, m + 1):
                operators.append((f'e_{k}{l}', e_kl_op(k, l)))
                operators.append((f'E_{k}{l}', E_kl_op(k, l)))
                operators.append((f'r_{k}{l}', r_kl_op(k, l)))
                operators.append((f'R_{k}{l}', R_kl_op(k, l)))
        for k in range(1, m + 1):
            operators.append((f'N_{k}', N_k_op(k)))
            operators.append((f's_{k}', s_k_op(k)))
            operators.append((f'S_{k}', S_k_op(k)))
    elif algebra == 'WH':
        for k in range(1, m + 1):
            operators.append((f'q_{k}', q_k_op(k)))
            operators.append((f'p_{k}', p_k_op(k)))
        operators.append(('I', identity_op))
    elif algebra == 'GO':
        for k in range(1, m + 1):
            for l in range(k + 1, m + 1):
                operators.append((f'e_{k}{l}', e_kl_op(k, l)))
                operators.append((f'E_{k}{l}', E_kl_op(k, l)))
                operators.append((f'r_{k}{l}', r_kl_op(k, l)))
                operators.append((f'R_{k}{l}', R_kl_op(k, l)))
        for k in range(1, m + 1):
            operators.append((f'N_{k}', N_k_op(k)))
            operators.append((f's_{k}', s_k_op(k)))
            operators.append((f'S_{k}', S_k_op(k)))
            operators.append((f'q_{k}', q_k_op(k)))
            operators.append((f'p_{k}', p_k_op(k)))
        operators.append(('I', identity_op))
    else:
        raise ValueError(f"Unknown algebra: {algebra}")
    return operators

### ###


### Utils for mixed states ###

def compute_left_product(H_op, rho):
    """Compute H * rho."""
    H_rho = defaultdict(lambda: defaultdict(complex))
    for j in rho:
        for m in rho[j]:
            rho_jm = rho[j][m]
            # Apply H to ket j
            terms = H_op(j) if isinstance(j, tuple) else H_op((j,))
            for coeff, new_ket in terms:
                H_rho[new_ket][m] += coeff * rho_jm
    return H_rho

def compute_right_product(H_op, rho):
    """Compute rho * H."""
    rho_H = defaultdict(lambda: defaultdict(complex))
    # Collect all unique kets in rho's columns
    all_kets = set()
    for bra in rho:
        all_kets.update(rho[bra].keys())
    for ket in all_kets:
        # Apply H to ket
        terms = H_op(ket) if isinstance(ket, tuple) else H_op((ket,))
        for coeff, new_ket in terms:
            # Distribute to all bras that have this ket
            for bra in rho:
                if ket in rho[bra]:
                    rho_H[bra][new_ket] += rho[bra][ket] * coeff.conjugate()
    return rho_H

def compute_commutator(H_op, rho):
    """Compute [H, rho] = H*rho - rho*H."""
    H_rho = compute_left_product(H_op, rho)
    rho_H = compute_right_product(H_op, rho)
    commutator = defaultdict(lambda: defaultdict(complex))
    # Subtract rho_H from H_rho
    all_rows = set(H_rho.keys()).union(rho_H.keys())
    for row in all_rows:
        cols = set(H_rho[row].keys() if row in H_rho else []).union(
            rho_H[row].keys() if row in rho_H else [])
        for col in cols:
            h_val = H_rho.get(row, {}).get(col, 0j)
            r_val = rho_H.get(row, {}).get(col, 0j)
            commutator[row][col] = h_val - r_val
    return commutator

def compute_adjoint(matrix):
    """Compute the adjoint of a matrix."""
    adjoint = defaultdict(lambda: defaultdict(complex))
    for bra in matrix:
        for ket in matrix[bra]:
            adjoint[ket][bra] = matrix[bra][ket].conjugate()
    return adjoint

def compute_product(A, B):
    """Compute matrix product A * B."""
    product = defaultdict(lambda: defaultdict(complex))
    for a_bra in A:
        for a_ket in A[a_bra]:
            a_val = A[a_bra][a_ket]
            if a_ket not in B:
                continue
            for b_ket in B[a_ket]:
                product[a_bra][b_ket] += a_val * B[a_ket][b_ket]
    return product

def compute_trace(matrix):
    """Compute the trace of a matrix."""
    trace = 0j
    for diag in matrix:
        if diag in matrix[diag]:
            trace += matrix[diag][diag]
    return trace


def ket_to_dens(psi):
    """
    Convert a state vector (ket) to a density matrix (ket-bra outer product)
    
    Args:
        psi: Dictionary {(n1, n2,...): complex} representing Fock basis amplitudes
        
    Returns:
        Density matrix as nested defaultdict: rho[bra][ket] = <bra|rho|ket>
    """
    rho = defaultdict(lambda: defaultdict(complex))
    
    # Iterate over all basis states in the superposition
    for bra in psi:
        alpha_bra = psi[bra]
        for ket in psi:
            alpha_ket = psi[ket]

            rho[bra][ket] = alpha_bra.conjugate() * alpha_ket
            
    return rho

### ###




### Computation of the Gram matrices in the ket and density operator pictures (before taking real parts) ###


def compute_G_matrix(psi, algebra='GO'):
    if not psi:
        raise ValueError("State psi is empty.")
    first_state = next(iter(psi.keys()))
    m = len(first_state)
    for state in psi:
        if len(state) != m:
            raise ValueError("All states must have the same length.")
    operators = generate_operators(m, algebra)
    h_psi_list = []
    for _, op in operators:
        h_psi = defaultdict(complex)
        for basis_state, alpha in psi.items():
            for coeff, new_state in op(basis_state):
                h_psi[new_state] += alpha * coeff
        h_psi_list.append(dict(h_psi))
    b = len(operators)
    G = np.zeros((b, b), dtype=np.complex128)
    for i in range(b):
        for j in range(b):
            inner_product = 0j
            states_i = h_psi_list[i].keys()
            states_j = h_psi_list[j].keys()
            all_states = set(states_i).union(states_j)
            for s in all_states:
                ci = h_psi_list[i].get(s, 0j)
                cj = h_psi_list[j].get(s, 0j)
                inner_product += np.conj(ci) * cj
            G[i, j] = inner_product
    return G


def compute_Gdens_matrix(rho, algebra='GO'):
    if not rho:
        raise ValueError("Density matrix rho is empty.")
    
    # Determine the number of modes
    first_bra = next(iter(rho.keys()))
    m = len(first_bra)
    operators = generate_operators(m, algebra)
    b = len(operators)
    Gdens = np.zeros((b, b), dtype=np.complex128)
    
    # Precompute commutators and their adjoints
    commutators = []
    adjoints = []
    for _, H_op in operators:
        C = compute_commutator(H_op, rho)
        commutators.append(C)
        adjoints.append(compute_adjoint(C))
    
    # Compute all pairs
    for i in range(b):
        Ci_dag = adjoints[i]
        for j in range(b):
            Cj = commutators[j]
            product = compute_product(Ci_dag, Cj)
            trace = compute_trace(product)
            Gdens[i, j] = trace

    # Check that the obtained Gram marix Gdens is already real:
    if not np.allclose(Gdens.imag, 0):
        max_imag = np.max(np.abs(Gdens.imag))
        raise ValueError(f"Gdens matrix is not real. Max imaginary component of Gdens: {max_imag:.4e}")
    
    return Gdens

### ###


### Computation of the orbit dimensions in the ket and density operator pictures (via rank of real part of computed Gram matrices) ###

def compute_ket_orbit_dim(psi, algebra='GO'):
     G = compute_G_matrix(psi, algebra=algebra).real
     return int(np.linalg.matrix_rank( G ))

def compute_dens_orbit_dim(rho, algebra='GO'):
     Gdens = compute_Gdens_matrix(rho, algebra=algebra).real
     return int(np.linalg.matrix_rank( Gdens ))

### ###


