import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.optimize import minimize
from skimage import filters, io
from skimage.segmentation import checkerboard_level_set

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
            l0_smoothing=cfg.image.l0_smoothing,
        )
        image = ci.image[
            cfg.image.slice.x[0] : cfg.image.slice.x[1],
            cfg.image.slice.y[0] : cfg.image.slice.y[1],
            cfg.image.slice.z[0] : cfg.image.slice.z[1],
        ]

        del ci

        levelset_center = np.array(image.shape) // 2
        levelset_radius = int(levelset_center.min() * 0.8)
        logging.info(f"{levelset_center=}, {levelset_radius=}")

        levelset = signed_distance_map(
            checkerboard_level_set(
                image_shape=image.shape,
                square_size=10,
            )
        )

        image = filters.gaussian(image, sigma=cfg.preprocessing.sigma)
        lambda2 = 2 - cfg.ol.lambda1

        logging.info("Initial plots")

        # levelset function 3d
        logging.info(
            f"{levelset[levelset_center[0],:,:].shape=}, {levelset[levelset_center[0],:,:].max()=}, {levelset[levelset_center[0],:,:].min()=}"
        )
        aim_run.track(
            {
                "initial levelset": plot_3d(levelset[levelset_center[0], :, :]),
                "smoothened image": plot_2d(image[levelset_center[0], :, :]),
            },
            context={"context": "initial", "x": f"{levelset_center[0]}"},
        )

        logging.info("Finished initial tracking.")

        # initialise callback for aim tracking of optimisation
        scallback = ScipyCallback(
            aim_run=aim_run,
            cfg=cfg,
            fun=ol_loss,
            image=image,
            lambda2=lambda2,
            x=(
                2 * levelset_center[0] // 3,
                4 * levelset_center[0] // 3,
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
        levelset = res.x.reshape(image.shape)
        del res
        logging.info(f"{np.count_nonzero(np.abs(levelset) < cfg.ol.epsilon)=}")
        io.imsave(Path.cwd() / "levelset.tif", levelset)

        # segmentation
        img_segmentation = np.zeros(levelset.shape)
        img_segmentation[levelset > 0] = 1
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
                "final levelset": plot_3d(levelset[levelset_center[0], :, :]),
                "segmentation": plot_2d(img_segmentation[levelset_center[0], :, :]),
            },
            context={"context": "final"},
        )

        logging.info("Finished tracking.")


if __name__ == "__main__":
    main()
