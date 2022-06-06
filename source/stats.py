__author__ = "Denver Lloyd"
__copyright__ = "TBD"

import numpy as np


def get_stats(imgs, std=True):
    """
    get all component wise FPN and Temporal noise
    from a stack of images
    """

    var = {}
    var_temp = {}
    result = {}

    # average temporal variance
    cvar_temp = np.mean(np.var(np.mean(imgs, axis=1)))
    rvar_temp = np.mean(np.var(np.mean(imgs, axis=2)))
    tvar_temp = np.mean(np.var(imgs, axis=0))

    # average spatial variance
    cvar = np.var(np.mean(imgs, axis=1))
    rvar = np.var(np.mean(imgs, axis=2))
    tvar = np.var(np.mean(imgs, axis=0))

    L, M, N = imgs.shape

    # get exact temporal variance solution
    var_temp = exact_solution(col_var=cvar_temp,
                              row_var=rvar_temp,
                              tot_var=tvar_temp,
                              M=M,
                              N=N,
                              spatial=False)

    # get exact spatial variance solution
    var = exact_solution(col_var=cvar,
                         row_var=rvar,
                         tot_var=tvar,
                         M=M,
                         N=N)

    var = {**var, **var_temp}
    # if we want STDDEV values we update dict prior to
    # returning
    if std:
        for kk in var.keys():
            up_key = kk.replace('var', 'std')
            val = np.sqrt(var[kk])
            result[up_key] = val
    else:
        result = var.copy()

    return result


def exact_solution(col_var, row_var, tot_var, M, N, spatial=True):
    """
    solve for exact solution of component wise noise per
    EMVA 4.0 definition
    """

    var = {}

    col_var = ((M*N-M)/(M*N-M-N)*col_var -
               (N)/(M*N-M-N)*(tot_var-row_var))

    row_var = ((M*N-N)/(M*N-M-N)*row_var -
               (M)/(M*N-M-N)*(tot_var-col_var))

    pix_var = ((M*N)/(M*N-M-N)) * (tot_var - col_var - row_var)

    if spatial:
        key = ''
    else:
        key = '_temp'

    var = {f'total_var{key}': tot_var,
           f'pix_var{key}': pix_var,
           f'row_var{key}': row_var,
           f'col_var{key}': col_var}

    return var
