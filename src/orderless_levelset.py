import numpy as np
from main import CellImage
from src.utils.storage import get_project_root
from skimage.segmentation import disk_level_set
from skimage import io
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage import filters
from scipy import ndimage
import logging

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)


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

    m, n = image.shape
    levelset = np.reshape(levelset, (m, n))
    mu1 = np.mean(image[levelset > 0])
    mu2 = np.mean(image[levelset < 0])
    loss = 0
    dloss = np.zeros(image.shape)
    mom = 2
    if True:
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
            loss_neg = np.sum(
                level_neg * (lambda2 * np.abs(image - mu2) ** mom * p[k + 1])
            )
            dloss_neg = level_neg * (lambda2 * np.abs(image - mu2) ** mom * dp[k + 1])
            loss += loss_pos + loss_neg + loss_0
            dloss += dloss_pos + dloss_neg + dloss_0
    else:
        for i in range(1, m * n):
            levelset_value = levelset.flatten()[i]
            image_value = image.flatten()[i]
            basis = np.floor(levelset_value)
            t = levelset_value - basis
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
            assert np.isnan(p).sum() == 0, print(p)
            assert np.isnan(dp).sum() == 0, print(dp)
            for k in range(-1, 3, 1):
                if basis + k == 0:
                    loss += mu * p[k + 1]
                    dloss[i] += mu * dp[k + 1]
                elif basis + k >= 0:
                    loss += (
                        lambda1 * np.abs(image_value - mu1) ** mom * p[k + 1]
                        + nu * p[k + 1]
                    )
                    dloss[i] += (
                        +lambda1 * np.abs(image_value - mu1) ** mom * dp[k + 1]
                        + nu * dp[k + 1]
                    )
                else:
                    loss += lambda2 * np.abs(image_value - mu2) ** mom * p[k + 1]
                    dloss[i] += lambda2 * np.abs(image_value - mu2) ** mom * dp[k + 1]
            assert np.isnan(dp).sum() == 0, print(i)
    logging.info(f"{loss=}, {np.max(dloss)=}, {np.min(dloss)=}")
    return loss, dloss.flatten()


def signed_distance_map(binary_image: np.ndarray) -> np.ndarray:
    positive_image = ndimage.distance_transform_edt(binary_image)
    negative_image = ndimage.distance_transform_edt(np.abs(binary_image - 1))
    return (positive_image - negative_image) / np.max((positive_image, negative_image))


if __name__ == "__main__":
    path_to_file = get_project_root() / "data/raw/cell-detection/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    imslice = disk_level_set(
        (100, 100), center=(30, 30), radius=20
    )  # ci.image[        355, :, :    ]  #
    imslice = filters.gaussian(imslice, sigma=1)
    plt.imshow(imslice)
    plt.show()
    levelset = signed_distance_map(
        disk_level_set(imslice.shape, center=(50, 50), radius=40)
    )

    res = minimize(
        fun=ol_loss,
        x0=levelset.flatten(),
        method="L-BFGS-B",
        jac=True,
        args=(
            imslice,
            1,  # lambda 1
            1,  # lambda 2
            0,  # mu
            0,  # nu
            0.1,  # epsilon
        ),
        options={"disp": True},
    )
    Z = res.x.reshape(imslice.shape)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(range(imslice.shape[1]), range(imslice.shape[0]))
    ha.plot_surface(X, Y, Z)

    plt.show()
