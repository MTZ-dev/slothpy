# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from numpy import zeros, arange, kron, eye
from slothpy.core._config import settings
from numpy import int64


def _kron_A_N(A, N):
    m, n = A.shape
    out = zeros((m, N, n, N), dtype=A.dtype)
    r = arange(N)
    out[:, r, :, r] = A
    out.shape = (m * N, n * N)
    return out

def _kron_mult(ops):
    if isinstance(ops[0], (int, int64)):
        result = eye(ops[0], dtype=settings.complex)
    else:
        result = ops[0]
    for op in ops[1:]:
        if isinstance(op, (int, int64)):
            result = _kron_A_N(result, op)
        else:
            result = kron(result, op)
    return result

def dipolar_interaction(i, j, cartesian_coords, momenta, dims, n):
    """
    Compute the dipolar interaction term between sites i and j.
    """
    H = np.zeros((np.prod(dims), np.prod(dims)), dtype=np.complex128)
    r_vec = np.array(cartesian_coords[i]) - np.array(cartesian_coords[j])
    r_ij = np.linalg.norm(r_vec)
    r_hat = r_vec / r_ij  # Normalized vector
    r3_inv = 1.0 / (r_ij ** 3)

    for l in range(3):
        for m in range(3):
            coeff = mu_0 / (4 * np.pi) * r3_inv * (np.eye(3)[l, m] - 3 * r_hat[l] * r_hat[m])
            if np.abs(coeff) < 1e-10:  # Skip terms with negligible coefficients
                continue
            mu1 = momenta[i][l]
            mu2 = momenta[j][m]
            ops = [mu1 if k == i else mu2 if k == j else None for k in range(n)]
            H_temp = kron_mult_with_identity_flag(ops, dims)
            H += coeff * H_temp

    return H

def hamiltonian_matrix(params):
    """
    Compute the full Hamiltonian matrix.
    """
    n = params['n']
    dims = params['dims']
    dim = np.prod(dims)
    H = np.zeros((dim, dim), dtype=np.complex128)
    single_site_ops = params['single_site_ops']
    two_site_ops = params['two_site_ops']
    cartesian_coords = params['cartesian_coords']
    momenta = params['momenta']

    # Single-site terms
    for site, (op, coeff) in single_site_ops.items():
        ops = [op if j == site else None for j in range(n)]  # None flags identity
        H_temp = kron_mult_with_identity_flag(ops, dims)
        H += coeff * H_temp

    # Two-site interaction terms
    for (site1, site2), (ops1, ops2, J) in two_site_ops.items():
        for l in range(3):
            for m in range(3):
                coeff = J[l, m]
                if np.abs(coeff) < 1e-10:  # Skip terms with negligible coefficients
                    continue
                op1 = ops1[l]
                op2 = ops2[m]
                ops = [op1 if j == site1 else op2 if j == site2 else None for j in range(n)]  # None flags identity
                H_temp = kron_mult_with_identity_flag(ops, dims)
                H += coeff * H_temp

    # Dipolar interaction terms
    for i in range(n):
        for j in range(i + 1, n):
            H += dipolar_interaction(i, j, cartesian_coords, momenta, dims, n)

    return H
