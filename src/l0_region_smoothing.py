import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.image_loader import CellImage
from src.utils.storage import get_project_root


def neighbour(shape: tuple[int], index: tuple[int], flat: bool = False) -> slice:
    adder = [1, -1]

    match len(shape):
        case 3:
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
        case 2:
            i, j = index
            i_fixed = [(i, j + a) for a in adder if (0 <= j + a < shape[1])]
            j_fixed = [(i + a, j) for a in adder if (0 <= i + a < shape[0])]
            neighbours = i_fixed + j_fixed
        case 1:
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


def create_cij(shape: tuple[int]) -> tuple[sp.coo_array, dict[int, tuple[int]]]:
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
    data = np.ones(len(row), dtype=np.uint8)
    return sp.coo_array((data, (row, col)), shape=(M, M), dtype=np.uint32).tolil(), {
        i: set(nb) for i, nb in zipped
    }


def reconstruct_image(
    M: int,
    shape: tuple[int],
    N: dict[int, set],
    G: dict[int, list],
    Y: dict[int, np.float16],
) -> np.ndarray:
    S = np.zeros((M,))
    for i in N.keys():
        for j in G[i]:
            S[j] = Y[i]
    return S.reshape(shape)


def l0_region_smoothing(
    image: np.ndarray,
    lambda_: float = 0.01,
    K: int = 20,
    gamma: float = 2.2,
    callback: None | Callable = None,
):
    M = np.product(image.shape)
    Y = {i: Ii for i, Ii in enumerate(image.flatten())}
    G = {i: [i] for i in np.arange(M)}
    w = {i: 1 for i in np.arange(M)}
    c, N = create_cij(image.shape)
    beta = 0.0
    iter = 0
    i = 0
    while beta < lambda_:
        n_keys = list(N.keys())
        print(f"{iter=}, {beta=}, {len(n_keys)=}")
        for i in tqdm(n_keys):
            for j in N.get(i, []):
                lhs = w[i] * w[j] * (Y[i] - Y[j]) ** 2
                rhs = beta * c[i, j] * (w[i] + w[j])
                if lhs <= rhs:
                    G[i].append(j)
                    Y[i] = (w[i] * Y[i] + w[j] * Y[j]) / (w[i] + w[j])
                    w[i] = w[i] + w[j]
                    c[i, j] = 0
                    N[i] = N[i].difference({j})
                    for k in N[j].difference({i}):
                        if k in N[i]:
                            c[i, k] = c[i, k] + c[j, k]
                            c[k, i] = c[i, k] + c[j, k]
                        else:
                            N[i].add(k)
                            N[k].add(i)
                            c[i, k] = c[j, k]
                            c[k, i] = c[k, j]
                        N[k] = N[k].difference({j})
                        c[k, j] = 0
                    G.pop(j), N.pop(j), w.pop(j)
        if callback:
            callback(
                **{"iter": iter, "beta": beta, "n_keys": n_keys, "N": N, "Y": Y, "G": G}
            )
        iter += 1
        beta = (iter / K) ** gamma * lambda_

    logging.info(f"Stopped at {beta=} with {len(N.keys())=}")
    return reconstruct_image(M, image.shape, N, G, Y)


if __name__ == "__main__":
    # slc = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # arr = np.array([slc * 10, slc * 100, slc * 1_000, slc * 10_000])
    # neighbour_idx = neighbour(arr.shape, (2, 2, 2))
    # print(f"{neighbour_idx=}")
    # neighbour_idx2 = neighbour(arr.shape[:2], (2, 2))
    # print(f"{neighbour_idx2=}")
    # neightbour_flattend = np.ravel_multi_index(
    #     tuple(zip(*neighbour_idx)), dims=arr.shape
    # )
    # print(f"{neightbour_flattend=}")

    # c, N = create_cij((4, 4, 4))
    # print(f"{c=}, {N=}")

    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)

    imslice = ci.get_slice(
        x=356,
        equalize="local",
        lower_bound=0,
        unsharp_mask={"radius": 80, "amount": 2},
        regenerate=False,
    )

    imslice = imslice.astype(np.float16)
    # c, N = create_cij(imslice.shape)
    smooth = l0_region_smoothing(imslice)
    plt.imshow(smooth)

    plt.show()
