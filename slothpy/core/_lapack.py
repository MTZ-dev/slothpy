
from numpy import iscomplexobj, inf
# Local imports
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork



def eigh(a, lower=True, eigvals_only=False, overwrite_a=False,
         subset_by_index=None, subset_by_value=None, driver=None):

    _job = 'N' if eigvals_only else 'V'

    drv_str = [None, "ev", "evd", "evr", "evx"]
    if driver not in drv_str:
        raise ValueError('"{}" is unknown. Possible values are "None", "{}".'
                         ''.format(driver, '", "'.join(drv_str[1:])))

    overwrite_a = overwrite_a
    cplx = True if iscomplexobj(a) else False
    n = a.shape[0]
    drv_args = {'overwrite_a': overwrite_a}

    subset = (subset_by_index is not None) or (subset_by_value is not None)

    # Both subsets can't be given
    if subset_by_index and subset_by_value:
        raise ValueError('Either index or value subset can be requested.')

    # Check indices if given
    if subset_by_index:
        lo, hi = (int(x) for x in subset_by_index)
        if not (0 <= lo <= hi < n):
            raise ValueError('Requested eigenvalue indices are not valid. '
                             f'Valid range is [0, {n-1}] and start <= end, but '
                             f'start={lo}, end={hi} is given')
        # fortran is 1-indexed
        drv_args.update({'range': 'I', 'il': lo + 1, 'iu': hi + 1})

    if subset_by_value:
        lo, hi = subset_by_value
        if not (-inf <= lo < hi <= inf):
            raise ValueError('Requested eigenvalue bounds are not valid. '
                             'Valid range is (-inf, inf) and low < high, but '
                             f'low={lo}, high={hi} is given')

        drv_args.update({'range': 'V', 'vl': lo, 'vu': hi})

    # fix prefix for lapack routines
    pfx = 'he' if cplx else 'sy'

    # decide on the driver if not given
    # first early exit on incompatible choice
    if driver:
        if subset and (driver in ["ev", "evd"]):
            raise ValueError(f'"{driver}" cannot compute subsets of eigenvalues')

    # Default driver is evr
    else:
        driver = "evr"

    lwork_spec = {
                  'syevd': ['lwork', 'liwork'],
                  'syevr': ['lwork', 'liwork'],
                  'heevd': ['lwork', 'liwork', 'lrwork'],
                  'heevr': ['lwork', 'lrwork', 'liwork'],
                  }

    drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                    [a])
    clw_args = {'n': n, 'lower': lower}
    if driver == 'evd':
        clw_args.update({'compute_v': 0 if _job == "N" else 1})

    lw = _compute_lwork(drvlw, **clw_args)
    # Multiple lwork vars
    if isinstance(lw, tuple):
        lwork_args = dict(zip(lwork_spec[pfx+driver], lw))
    else:
        lwork_args = {'lwork': lw}

    drv_args.update({'lower': lower, 'compute_v': 0 if _job == "N" else 1})

    return drv, drv_args, lwork_args


    # # m is always the first extra argument
    # w = w[:other_args[0]] if subset else w
    # v = v[:, :other_args[0]] if (subset and not eigvals_only) else v

    # # Check if we had a  successful exit
    # if info == 0:
    #     if eigvals_only:
    #         return w
    #     else:
    #         return w, v
    # else:
    #     if info < -1:
    #         raise LinAlgError('Illegal value in argument {} of internal {}'
    #                           ''.format(-info, drv.typecode + pfx + driver))
    #     elif info > n:
    #         raise LinAlgError(f'The leading minor of order {info-n} of B is not '
    #                           'positive definite. The factorization of B '
    #                           'could not be completed and no eigenvalues '
    #                           'or eigenvectors were computed.')
    #     else:
    #         drv_err = {'ev': 'The algorithm failed to converge; {} '
    #                          'off-diagonal elements of an intermediate '
    #                          'tridiagonal form did not converge to zero.',
    #                    'evx': '{} eigenvectors failed to converge.',
    #                    'evd': 'The algorithm failed to compute an eigenvalue '
    #                           'while working on the submatrix lying in rows '
    #                           'and columns {0}/{1} through mod({0},{1}).',
    #                    'evr': 'Internal Error.'
    #                    }
    #         if driver in ['ev', 'gv']:
    #             msg = drv_err['ev'].format(info)
    #         elif driver in ['evx', 'gvx']:
    #             msg = drv_err['evx'].format(info)
    #         elif driver in ['evd', 'gvd']:
    #             if eigvals_only:
    #                 msg = drv_err['ev'].format(info)
    #             else:
    #                 msg = drv_err['evd'].format(info, n+1)
    #         else:
    #             msg = drv_err['evr']

    #         raise LinAlgError(msg)

    