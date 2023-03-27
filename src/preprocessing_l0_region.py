import logging

import aim
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from skimage import io

from src.cython_implementations.l0_region_smoothing import l0_region_smoothing
from src.image_loader import CellImage
from src.utils.aim import L0Callback, experiment_context
from src.utils.storage import get_project_root
from src.visualization import plot_2d

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)
CONFIG_PATH = str(get_project_root() / "conf")


@hydra.main(config_path=CONFIG_PATH, config_name="preprocessing", version_base="1.3")
def main(cfg: DictConfig):
    with experiment_context(cfg) as aim_run:
        path_to_file = get_project_root() / "data" / cfg.image.path
        ci = CellImage(path=path_to_file, cfg=cfg)

        # uncomment for slice instead of 3d image
        # image = ci.get_slice(
        #     x=cfg.image.slice.x[0],
        #     equalize=cfg.image.equalize,
        #     lower_bound=cfg.image.lower_bound,
        #     unsharp_mask=cfg.image.unsharp_mask,
        #     regenerate=cfg.image.regenerate,
        # )
        ci.read_image()

        image_center = np.array(ci.image.shape) // 2

        logging.info("Initial tracking")
        aim_run.track(
            {
                "image": plot_2d(ci.image[image_center[0]]),
                "histogram": aim.Distribution(ci.image.flatten()),
            },
            context={"context": "initial"},
        )

        lcallback = L0Callback(aim_run=aim_run, cfg=cfg, shape=ci.image.shape)
        logging.info("Starting region smoothing")
        smooth = l0_region_smoothing(
            ci.image, **cfg.image.l0_region, callback=lcallback.l0_callback
        )

        unsharp_mask = ci.equalize_local(
            smooth, unsharp_mask={"radius": 80, "amount": 2}
        )
        aim_run.track(
            {
                "L0": plot_2d(smooth[:, image_center[1]]),
                "L0 + unsharp": plot_2d(unsharp_mask[image_center[0]]),
            },
            context={"context": "final"},
        )
        plt.close()

        io.imsave("smooth_image.tif", smooth)

        # ci.show_3d()


if __name__ == "__main__":
    main()
