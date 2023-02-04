import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.optimize import minimize
from skimage import filters, io
from skimage.segmentation import disk_level_set

from image_loader import CellImage
from src.orderless_levelset import ol_loss, signed_distance_map
from src.utils.aim import ScipyCallback, experiment_context
from src.utils.storage import get_project_root
from src.visualization import plot_2d, plot_3d

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)
CONFIG_PATH = str(get_project_root() / "conf")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    with experiment_context(cfg) as aim_run:
        path_to_file = get_project_root() / "data" / cfg.image.path
        ci = CellImage(path=path_to_file)

        ci.read_image(
            equalize=cfg.image.equalize,
            lower_bound=cfg.image.lower_bound,
            unsharp_mask=cfg.image.unsharp_mask,
            regenerate=cfg.image.regenerate,
        )
        image = ci.image[
            cfg.image.slice.x[0] : cfg.image.slice.x[1],
            cfg.image.slice.y[0] : cfg.image.slice.y[1],
            cfg.image.slice.z[0] : cfg.image.slice.z[1],
        ]

        del ci

        levelset_center = image.shape[0] // 2

        levelset = signed_distance_map(
            disk_level_set(image.shape, center=(levelset_center, 100, 100), radius=40)
        )

        image = filters.gaussian(image, sigma=cfg.preprocessing.sigma)
        lambda2 = 2 - cfg.ol.lambda1

        logging.info("Initial plots")

        # levelset function 3d
        logging.debug(
            f"{levelset[levelset_center,:,:].shape=}, {levelset[levelset_center,:,:].max()=}, {levelset[levelset_center,:,:].min()=}"
        )
        aim_run.track(
            {
                "initial levelset": plot_3d(levelset[levelset_center, :, :]),
                "smoothened image": plot_2d(image[levelset_center, :, :]),
            },
            context={"context": "final", "x": levelset_center},
        )
        plt.close()

        logging.info("Finished initial tracking.")

        # initialise callback for aim tracking of optimisation
        scallback = ScipyCallback(
            aim_run=aim_run,
            cfg=cfg,
            fun=ol_loss,
            image=image,
            lambda2=lambda2,
            x=(
                levelset_center // 2,
                levelset_center,
                levelset_center + levelset_center // 2,
            ),
        )

        res = minimize(
            fun=ol_loss,
            x0=levelset.flatten(),
            method="L-BFGS-B",
            jac=True,
            args=(
                image,
                cfg.ol.lambda1,  # lambda 1
                lambda2,  # lambda 2
                cfg.ol.mu,  # mu
                cfg.ol.nu,  # nu
                cfg.ol.epsilon,  # epsilon
            ),
            options={"disp": True},
            callback=scallback.scipy_optimize_callback,
        )

        logging.info("Preparing plots")

        # segmentation
        img_segmentation = res.x.reshape(image.shape)
        img_segmentation[img_segmentation > 0] = 1
        img_segmentation[img_segmentation <= 0] = 0
        io.imsave(Path.cwd() / "segmentation.tif", img_segmentation)

        # track stats
        logging.info("Tracking results in aim...")
        non_zero = np.count_nonzero(img_segmentation)
        aim_run.track(
            {
                "min": np.min(img_segmentation),
                "max": np.max(img_segmentation),
                "num_zero": non_zero,
                "num_zero_prc": non_zero / np.product(img_segmentation.shape),
            },
            context={"context": "final"},
        )

        # track images
        logging.info("Tracking images in aim...")
        aim_run.track(
            {
                "final levelset": plot_3d(levelset[levelset_center, :, :]),
                "segmentation": plot_2d(img_segmentation[levelset_center, :, :]),
            },
            context={"context": "final"},
        )
        plt.close()

        logging.info("Finished tracking.")


if __name__ == "__main__":
    main()
