import numpy as np
from main import CellImage
from src.utils.storage import get_project_root
from skimage.segmentation import disk_level_set
from skimage import io
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage import filters
from scipy import ndimage
import logging
import hydra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.aim import experiment_context
from aim import Image

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)
CONFIG_PATH = str(get_project_root() / "conf")


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

    logging.info(f"{loss=}, {np.max(dloss)=}, {np.min(dloss)=}")
    return loss, dloss.flatten()


def signed_distance_map(binary_image: np.ndarray) -> np.ndarray:
    positive_image = ndimage.distance_transform_edt(binary_image)
    negative_image = ndimage.distance_transform_edt(np.abs(binary_image - 1))
    return (positive_image - negative_image) / np.max((positive_image, negative_image))


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    with experiment_context(cfg) as aim_run:
        path_to_file = (
            get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
        )
        ci = CellImage(path=path_to_file)
        imslice = ci.image[
            355, :, :
        ]  # disk_level_set((100, 100), center=(30, 30), radius=20)

        levelset = signed_distance_map(
            disk_level_set(imslice.shape, center=(50, 50), radius=40)
        )

        lambda2 = 2 - cfg.ol.lambda1

        imslice_smooth = filters.gaussian(imslice, sigma=cfg.ol.sigma)

        # TODO build callback for aim
        res = minimize(
            fun=ol_loss,
            x0=levelset.flatten(),
            method="L-BFGS-B",
            jac=True,
            args=(
                imslice_smooth,
                cfg.ol.lambda1,  # lambda 1
                lambda2,  # lambda 2
                cfg.ol.mu,  # mu
                cfg.ol.nu,  # nu
                cfg.ol.epsilon,  # epsilon
            ),
            options={"disp": True},
        )

        logging.info("Preparing plots")
        # levelset function 3d
        Z = filters.gaussian(
            res.x.reshape(imslice_smooth.shape), sigma=cfg.ol.lvlset_sigma
        )
        X, Y = np.meshgrid(
            range(imslice_smooth.shape[1]),
            range(imslice_smooth.shape[0]),
        )

        # segmentation
        img_segmentation = res.x.reshape(imslice_smooth.shape)
        img_segmentation[img_segmentation > 0] = 1
        img_segmentation[img_segmentation <= 0] = 0

        # create figures for tracking
        fig1 = plt.figure()
        ha = fig1.add_subplot(projection="3d")
        fig2 = plt.figure()
        hb = fig2.add_subplot()
        fig3 = plt.figure()
        hc = fig3.add_subplot()
        fig4 = plt.figure()
        hd = fig4.add_subplot()

        ha.plot_surface(X, Y, Z, cmap="viridis")
        hb.imshow(imslice_smooth)
        hc.imshow(img_segmentation, cmap="viridis")
        hd.imshow(imslice)

        # track stats
        logging.info("Tracking results in aim...")
        aim_run.track(np.max(img_segmentation), "max", context={"hparam": True})
        aim_run.track(np.max(img_segmentation), "min", context={"hparam": True})
        aim_run.track(
            np.count_nonzero(img_segmentation), "num_zeros", context={"hparam": True}
        )

        # track images
        logging.info("Tracking images in aim...")
        aim_run.track(Image(fig1), name="levelset", context={"hparam": True})
        aim_run.track(Image(fig2), name="smoothened image", context={"hparam": True})
        aim_run.track(Image(fig3), name="segmentation", context={"hparam": True})
        aim_run.track(Image(fig4), name="original image", context={"hparam": True})

        logging.info("Finished tracking.")


if __name__ == "__main__":
    main()
