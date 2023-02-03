import logging

import numpy as np
from scipy import ndimage


def ol_loss(
    levelset: np.ndarray,
    image: np.ndarray,
    lambda1: float = 1,
    lambda2: float = 1,
    mu: float = 0,
    nu: float = 0,
    epsilon: float = 0.01,
):
    if np.max(image) > 1:
        image = image.astype(np.float64)
        image = image / np.max(image)

    m, n, o = image.shape
    levelset = np.reshape(levelset, (m, n, o))
    mu1 = np.mean(image[levelset > 0])
    mu2 = np.mean(image[levelset < 0])
    loss = 0
    dloss = np.zeros(image.shape)
    mom = 2
    levelset_basis = np.floor(levelset)
    t = levelset - levelset_basis
    p = (
        1
        / 6
        * np.array(
            [
                -(t**3) + 3 * t**2 - 3 * t + 1,
                3 * t**3 - 6 * t**2 + 4,
                -3 * t**3 + 3 * t**2 + 3 * t + 1,
                t**3,
            ]
        )
    )
    dp = (
        1
        / 6
        * np.array(
            [
                -3 * t**2 + 6 * t - 3,
                9 * t**2 - 12 * t,
                -9 * t**2 + 6 * t + 3,
                3 * t**2,
            ]
        )
    )
    for k in range(-1, 3, 1):
        loss_pos, loss_neg, loss_0 = (0, 0, 0)
        dloss_pos, dloss_neg, dloss_0 = (
            np.zeros(image.shape),
            np.zeros(image.shape),
            np.zeros(image.shape),
        )
        level_0 = np.abs(levelset_basis + k) < epsilon
        level_pos = levelset_basis + k > epsilon
        level_neg = levelset_basis + k < epsilon
        loss_0 = np.sum(level_0 * mu * p[k + 1])
        dloss_0 = level_0 * mu * dp[k + 1]
        loss_pos = np.sum(
            level_pos
            * (lambda1 * np.abs(image - mu1) ** mom * p[k + 1] + nu * p[k + 1])
        )
        dloss_pos = level_pos * (
            lambda1 * np.abs(image - mu1) ** mom * dp[k + 1] + nu * dp[k + 1]
        )
        loss_neg = np.sum(level_neg * (lambda2 * np.abs(image - mu2) ** mom * p[k + 1]))
        dloss_neg = level_neg * (lambda2 * np.abs(image - mu2) ** mom * dp[k + 1])
        loss += loss_pos + loss_neg + loss_0
        dloss += dloss_pos + dloss_neg + dloss_0

    logging.debug(f"{loss=}, {np.max(dloss)=}, {np.min(dloss)=}")
    return loss, dloss.flatten()


def signed_distance_map(binary_image: np.ndarray) -> np.ndarray:
    positive_image = ndimage.distance_transform_edt(binary_image)
    negative_image = ndimage.distance_transform_edt(np.abs(binary_image - 1))
    return (positive_image - negative_image) / np.max((positive_image, negative_image))
