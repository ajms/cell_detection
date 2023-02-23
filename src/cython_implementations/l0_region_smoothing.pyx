# cython: infer_types=True
import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.image_loader import CellImage
from src.utils.storage import get_project_root

cimport cython


def neighbour(
    shape: tuple[int], index: tuple[int], flat: bool = False
) -> list[tuple[int]]:
    """Find neighbours of index in image, works for 1, 2 and 3d shapes

    Args:
        shape (tuple[int]): image shape
        index (tuple[int]): index of which to determine neighbours
        flat (bool, optional): return the flattened index, C-order. Defaults to False.

    Returns:
        list[tuple[int]]: list of neighbours
    """
    adder = [1, -1]

    if len(shape) == 3:
        i, j, k = index
        i_fixed = [
            (i, j + a, k + a)
            for a in adder
            if (0 <= j + a < shape[1] and 0 <= k + a < shape[2])
        ]
        j_fixed = [
            (i + a, j, k + a)
            for a in adder
            if (0 <= i + a < shape[0] and 0 <= k + a < shape[2])
        ]
        k_fixed = [
            (i + a, j + a, k)
            for a in adder
            if (0 <= i + a < shape[0] and 0 <= j + a < shape[1])
        ]
        neighbours = i_fixed + j_fixed + k_fixed
    elif len(shape) == 2:
        i, j = index
        i_fixed = [(i, j + a) for a in adder if (0 <= j + a < shape[1])]
        j_fixed = [(i + a, j) for a in adder if (0 <= i + a < shape[0])]
        neighbours = i_fixed + j_fixed
    elif len(shape) == 1:
        i = index
        neighbours = [(i - 1, i + 1)]

    if flat:
        return tuple(
            np.ravel_multi_index(
                tuple(zip(*neighbours)),
                dims=shape,
            )
        )

    else:
        return neighbours


def create_c_N(shape: tuple[int]) -> sp.lil_array:
    """Initialize sparse matrix of neighbour relations and dictionary of neighbours for each index.

    Args:
        shape (tuple[int]): image shape

    Returns:
        tuple[sp.coo_array, dict[int, tuple[int]]]:
            returns a MxM-matrix, where M is the product of the shape which is 1, if i and j are neighbours and
            a dictionary containing all neighbour indices for each index.

    """
    M = np.product(shape)
    zipped = [
        (
            i,
            neighbour(shape=shape, index=np.unravel_index(i, shape=shape), flat=True),
        )
        for i in np.arange(np.product(shape))
    ]

    col = [elt for elt_i in zipped for elt in elt_i[1]]
    row = [len(elt[1]) * [elt[0]] for elt in zipped]
    row = [elt for nested in row for elt in nested]
    assert len(col) == len(row), f"{len(col)=}, {len(row)=}"
    data = np.ones(len(row), dtype=np.intc)
    return sp.coo_array((data, (row, col)), shape=(M, M)).tolil()


def reconstruct_image(
    M: int,
    shape: tuple[int],
    G: sp.lil_array,
    Y: np.ndarray,
) -> np.ndarray:
    """Reconstruct the image from the groups from the l0-region-smoothing

    Args:
        M (int): length of flattened image
        shape (tuple[int]): shape of image
        N (dict[int, set]): dictionary of neighbours
        G (dict[int, list]): dictionary of groups
        Y (dict[int, np.float16]): dictionary of group intensities

    Returns:
        np.ndarray: reconstructed image
    """
    S = np.zeros((M,))

    for i in remaining_keys(G):
        for j in G.rows[i]:
            S[j] = Y[i]
    return S.reshape(shape)

v_keys = np.vectorize(lambda x: len(x))

def remaining_keys(G):
    g_len = v_keys(G.rows)
    assert g_len.sum() == len(G.rows), f"{g_len.sum()=}"
    n_keys = np.nonzero(g_len)[0]
    return n_keys

@cython.boundscheck(False)
@cython.wraparound(False)
def l0_region_smoothing(
    image,
    float lambda_,
    int K,
    float gamma,
    callback,
):
    """L0 region smoothing adapted from paper Fast and Effective L0 Gradient Minimization by Region Fusion by Nguyen et al.

    Args:
        image (np.ndarray): input image
        lambda_ (float, optional): level of sparseness. Defaults to 0.01.
        K (int, optional): number of iterations. Defaults to 20.
        gamma (float, optional): power for convergence criterion, if gamma = 1 then linear, else nonlinear. Defaults to 2.2.
        callback (None | Callable, optional): callback function for collecting metrics: iter, beta, n_keys, N, Y, G and w. Defaults to None.

    Returns:
        np.ndarray: return image with smoothened regions
    """
    cdef int M = np.product(image.shape, dtype=np.intc).item()
    Y = image.flatten()
    cdef double lhs, rhs
    cdef double[:] y_view = Y
    logging.info("Initialize G")
    G = sp.coo_array((np.ones(M), (np.arange(0, M, 1), np.arange(0, M, 1))), dtype=np.intc).tolil()
    w = np.ones(M, dtype=np.intc)
    cdef int[:] w_view = w
    logging.info("Initialize c")
    c = create_c_N(image.shape)
    cdef double beta
    cdef Py_ssize_t iter, i, j, k, l, jj, kk, ll
    cdef Py_ssize_t jj_max, kk_max, ll_max
    beta = 0.0
    iter = 0
    v_keys = np.vectorize(lambda x: len(x))

    logging.info("Start iterations")
    while beta < lambda_:
        n_keys = remaining_keys(G)
        if callback:
            callback(
                **{
                    "iter": iter,
                    "beta": beta,
                    "n_keys": n_keys,
                    "Y": Y,
                    "G": G,
                    "w": w,
                }
            )
        print(f"{iter=}, {beta=}, {len(n_keys)=}")
        for i in tqdm(n_keys):
            assert Y[i] == y_view[i]
            jj = 0
            jj_max = len(c.rows[i])
            while jj < jj_max:
                j = c.rows[i][jj]  # j in N[i]
                lhs = w_view[i] * w_view[j] * (y_view[i] - y_view[j]) ** 2
                rhs = beta * c[i,j] * (w_view[i] + w_view[j])
                if lhs <= rhs:
                    ll_max = len(G.rows[j])
                    for ll in range(ll_max):  # G[i] union G[j]
                        l = G.rows[j][ll]
                        G[i,l] = G.data[j][ll]
                    y_view[i] = (w_view[i] * y_view[i] + w_view[j] * y_view[j]) / (w_view[i] + w_view[j])
                    w_view[i] = w_view[i] + w_view[j]
                    c[i, j] = 0
                    jj_max -= 1
                    kk_max = len(c.rows[j])
                    for kk in range(kk_max):
                        k = c.rows[j][kk]  # k in N[j]
                        if k == i:
                            continue
                        if k in c.rows[i]:
                            c[i, k] += c[j, k]
                            c[k, i] += c[j, k]
                        else:
                            c[i, k] = c[j, k]
                            c[k, i] = c[j, k]
                        c[k, j] = 0
                    G.data[j] = []  # delete G[j]
                    G.rows[j] = []
                    c.data[j] = []  # delete N[j]
                    c.rows[j] = []
                    w_view[j] = 0  # delete w[j]
                jj += 1


        iter += 1
        beta = (iter / K) ** gamma * lambda_

    logging.info(f"Stopped at {beta=}")
    return reconstruct_image(M, image.shape, G, Y)
