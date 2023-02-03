import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from aim import Image
from omegaconf import DictConfig
from scipy.optimize import minimize
from skimage import filters, io
from skimage.segmentation import disk_level_set

from image_loader import CellImage
from src.orderless_levelset import ol_loss, signed_distance_map
from src.utils.aim import ScipyCallback, experiment_context
from src.utils.storage import get_project_root

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
        image = ci.image[300:400, :, :]

        del ci

        levelset = signed_distance_map(
            disk_level_set(image.shape, center=(50, 50, 50), radius=40)
        )

        image = filters.gaussian(image, sigma=cfg.preprocessing.sigma)
        lambda2 = 2 - cfg.ol.lambda1

        # initialise callback for aim tracking of optimisation
        scallback = ScipyCallback(
            aim_run=aim_run, cfg=cfg, fun=ol_loss, image=None, lambda2=lambda2
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
        # levelset function 3d
        Z = filters.gaussian(
            res.x.reshape(image.shape)[0, :, :], sigma=cfg.postprocessing.sigma
        )
        X, Y = np.meshgrid(
            range(image.shape[1]),
            range(image.shape[2]),
        )

        # segmentation
        img_segmentation = res.x.reshape(image.shape)
        img_segmentation[img_segmentation > 0] = 1
        img_segmentation[img_segmentation <= 0] = 0
        io.imsave(Path.cwd() / "segmentation.tif", img_segmentation)

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
        hb.imshow(image[0, :, :])
        hc.imshow(img_segmentation[0, :, :], cmap="viridis")
        hd.imshow(image[0, :, :])

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
                "final levelset": Image(fig1),
                "smoothened image": Image(fig2),
                "segmentation": Image(fig3),
                "original image": Image(fig4),
            },
            context={"context": "final"},
        )
        plt.close()

        logging.info("Finished tracking.")


if __name__ == "__main__":
    main()
