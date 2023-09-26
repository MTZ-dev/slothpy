from math import factorial
from numpy.linalg import eigh
from numpy import (
    ndarray,
    zeros,
    zeros_like,
    ascontiguousarray,
    diag,
    real,
    imag,
    trace,
    abs,
    sqrt,
    int32,
    int64,
    float64,
    complex128,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
    _get_soc_total_angular_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._math_expresions import (
    _hermitian_x_in_basis_of_hermitian_y,
    _decomposition_of_hermitian_matrix,
    _Wigner_3j,
)
from slothpy._magnetism._zeeman import _calculate_zeeman_matrix


def _set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(
    momenta_matrix, matrix
):
    """Convention:
    Jx[i,i+1] = real, negative
    Jy[i,i+1] = imag, positive
    J+/- = real (Condon_Shortley)
    Jz = real, diag"""

    # Transform momenta to "z" basis
    _, eigenvectors = eigh(momenta_matrix[2, :, :])
    for i in range(3):
        momenta_matrix[i, :, :] = (
            eigenvectors.conj().T @ momenta_matrix[i, :, :] @ eigenvectors
        )

    # Initialize phases of vectors with the first one = 1
    c = zeros(momenta_matrix.shape[1], dtype=complex128)
    c[0] = 1.0

    # Set Jx[i,i+1] to real negative and collect phases of vectors in c[:]
    for i in range(momenta_matrix[0, :, :].shape[0] - 1):
        if (
            real(momenta_matrix[0, i, i + 1]).any() > 1e-12
            or abs(imag(momenta_matrix[1, i, i + 1])).any() > 1e-12
        ):
            c[i + 1] = (
                momenta_matrix[0, i, i + 1].conj()
                / abs(momenta_matrix[0, i, i + 1])
                / c[i].conj()
            )
        else:
            c[i + 1] = 1.0

    for i in range(momenta_matrix[0].shape[0] - 1):
        if momenta_matrix[0, i, i + 1] * c[i].conj() * c[i + 1] > 0:
            c[i + 1] = -c[i + 1]

    matrix_out = zeros_like(matrix)

    for i in range(matrix_out.shape[0]):
        for j in range(matrix_out.shape[0]):
            matrix_out[i, j] = matrix[i, j] * c[i].conj() * c[j]

    return ascontiguousarray(matrix_out)


def _get_soc_matrix_in_z_pseudo_spin_basis(
    filename, group, start_state, stop_state, pseudo_kind, rotation=None
):
    if pseudo_kind == "magnetic":
        (
            momenta,
            soc_energies,
        ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
            filename, group, stop_state + 1, rotation
        )
    elif pseudo_kind == "total_angular":
        (
            momenta,
            soc_energies,
        ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
            filename, group, stop_state + 1, rotation
        )
    else:
        raise NotImplementedError(
            'The only options for pseudo-spin type are: "magnetic" and'
            ' "total_angular".'
        )

    momenta = momenta[:, start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    soc_matrix = diag(soc_energies).astype(complex128)
    soc_matrix = _hermitian_x_in_basis_of_hermitian_y(
        soc_matrix, momenta[2, :, :]
    )
    soc_matrix = _set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(
        momenta, soc_matrix
    )

    return soc_matrix


def _get_decomposition_in_z_pseudo_spin_basis(
    filename,
    group,
    matrix,
    pseudo_kind,
    start_state,
    stop_state,
    rotation=None,
    field=None,
    orientation=None,
):
    if matrix == "soc":
        soc_matrix = _get_soc_matrix_in_z_pseudo_spin_basis(
            filename, group, start_state, stop_state, pseudo_kind, rotation
        )
    elif (
        (matrix == "zeeman")
        and (field != None and field is not None)
        and (orientation != None and orientation is not None)
    ):
        soc_matrix = _get_zeeman_matrix_in_z_pseudo_spin_basis(
            filename,
            group,
            field,
            orientation,
            start_state,
            stop_state,
            pseudo_kind,
            rotation,
        )
    else:
        raise NotImplementedError(
            'The only options for matrix are: "soc" and "zeeman". For the'
            " second option one needs to provide magnetic field value and its"
            " orientation."
        )
    decopmosition = _decomposition_of_hermitian_matrix(soc_matrix)

    return decopmosition


def _get_zeeman_matrix_in_z_pseudo_spin_basis(
    filename,
    group,
    field,
    orientation,
    start_state,
    stop_state,
    pseudo_kind,
    rotation=None,
):
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, stop_state + 1, rotation
    )
    magnetic_momenta = magnetic_momenta[:, start_state:, start_state:]
    soc_energies = soc_energies[start_state:]
    zeeman_matrix = _calculate_zeeman_matrix(
        magnetic_momenta, soc_energies, field, orientation
    )

    if pseudo_kind == "total_angular":
        (
            magnetic_momenta,
            soc_energies,
        ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
            filename, group, stop_state + 1
        )
    elif pseudo_kind != "magnetic":
        raise NotImplementedError(
            'The only options for pseudo-spin type are: "magnetic" and'
            ' "total_angular".'
        )

    zeeman_matrix = _hermitian_x_in_basis_of_hermitian_y(
        zeeman_matrix, magnetic_momenta[2, :, :]
    )
    zeeman_matrix = (
        _set_condon_shortley_phases_for_matrix_in_z_pseudo_spin_basis(
            magnetic_momenta, zeeman_matrix
        )
    )

    return zeeman_matrix


def _ito_matrix(J, k, q):
    dim = int64(2 * J + 1)

    matrix = zeros((dim, dim), dtype=float64)

    for i in range(dim):
        mj1 = i - J
        for p in range(dim):
            mj2 = p - J
            matrix[i, p] = (-1) ** (J - mj1) * _Wigner_3j(
                J, k, J, -mj1, q, mj2
            )

    coeff = float64(1.0)

    for i in range(int(-k), int(k + 1)):
        coeff *= 2 * J + 1 + i

    coeff /= factorial(2 * k)
    coeff = sqrt(coeff)
    coeff *= ((-1) ** k) * factorial(k)

    N_k_k = ((-1) ** k) / (2 ** (k / 2))

    return matrix * N_k_k * coeff


def _calculate_b_k_q(matrix: ndarray, k: int32, q: int32):
    J = (matrix.shape[0] - 1) / 2

    matrix = ascontiguousarray(matrix)
    ITO_plus = _ito_matrix(J, k, q)
    ITO_plus = ascontiguousarray(ITO_plus).astype(complex128)
    ITO_minus = _ito_matrix(J, k, -q)
    ITO_minus = ascontiguousarray(ITO_minus).astype(complex128)

    numerator = trace(matrix @ ITO_minus)
    denominator = trace(ITO_plus @ ITO_minus)

    return numerator / denominator


def _ito_complex_decomp_matrix(
    matrix: ndarray, order: int, even_order: bool = False
):
    step = 1

    if even_order:
        step = 2

    result = []

    for k in range(0, order + 1, step):
        for q in range(-k, k + 1):
            B_k_q = _calculate_b_k_q(matrix, k, q)
            result.append([k, q, B_k_q])

    return result


def _matrix_from_ito_complex(J, coefficients):
    dim = int64(2 * J + 1)

    matrix = zeros((dim, dim), dtype=complex128)

    for i in coefficients:
        matrix += _ito_matrix(J, int(i[0].real), int(i[1].real)) * i[2]

    return matrix


def _ito_real_decomp_matrix(
    matrix: ndarray, order: int, even_order: bool = False
):
    step = 1

    if even_order:
        step = 2

    result = []

    J = (matrix.shape[0] - 1) / 2
    matrix = ascontiguousarray(matrix)

    for k in range(0, order + 1, step):
        for q in range(k, 0, -1):
            ITO_plus = _ito_matrix(J, k, q)
            ITO_plus = ascontiguousarray(ITO_plus).astype(complex128)
            ITO_minus = _ito_matrix(J, k, -q)
            ITO_minus = ascontiguousarray(ITO_minus).astype(complex128)
            B_k_q = (
                -1j
                * (
                    trace(matrix @ ITO_plus)
                    - ((-1) ** (-q)) * trace(matrix @ ITO_minus)
                )
                / trace(ITO_plus @ ITO_minus)
            )

            result.append([k, -q, B_k_q.real])

        B_k_q = _calculate_b_k_q(matrix, k, 0)
        result.append([k, 0, B_k_q.real])

        for q in range(1, k + 1):
            ITO_plus = _ito_matrix(J, k, q)
            ITO_plus = ascontiguousarray(ITO_plus).astype(complex128)
            ITO_minus = _ito_matrix(J, k, -q)
            ITO_minus = ascontiguousarray(ITO_minus).astype(complex128)
            B_k_q = (
                trace(matrix @ ITO_plus)
                + ((-1) ** (-q)) * trace(matrix @ ITO_minus)
            ) / trace(ITO_plus @ ITO_minus)

            result.append([k, q, B_k_q.real])

    return result


def _matrix_from_ito_real(J, coefficients):
    dim = int64(2 * J + 1)

    matrix = zeros((dim, dim), dtype=complex128)

    for i in coefficients:
        k = int64(i[0])
        q = int64(i[1])

        if q < 0:
            matrix += (
                1j
                * 0.5
                * (
                    ((-1) ** (-q + 1)) * _ito_matrix(J, k, -q)
                    + _ito_matrix(J, k, q)
                )
                * i[2]
            )
        if q > 0:
            matrix += (
                0.5
                * (_ito_matrix(J, k, -q) + ((-1) ** q) * _ito_matrix(J, k, q))
                * i[2]
            )
        if q == 0:
            matrix += _ito_matrix(J, k, q) * i[2]

    return matrix
