import logging

import cv2 as cv
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.image_loader import CellImage
from src.utils.aim import experiment_context
from src.utils.storage import get_project_root
from src.visualization import plot_2d

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)
CONFIG_PATH = str(get_project_root() / "conf")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    with experiment_context(cfg) as aim_run:
        path_to_file = get_project_root() / "data" / cfg.image.path
        ci = CellImage(path=path_to_file)

        imslice = ci.get_slice(
            x=cfg.image.slice.x[0],
            equalize=cfg.image.equalize,
            lower_bound=cfg.image.lower_bound,
            unsharp_mask=cfg.image.unsharp_mask,
            regenerate=cfg.image.regenerate,
            return_path=False,
        )

        imslice_cv = (imslice * 255).astype(np.uint8)

        imslice_colour = cv.cvtColor(imslice_cv, cv.COLOR_GRAY2RGB)
        aim_run.track(
            {
                "image": plot_2d(imslice),
            },
        )
        logging.info(f"{type(imslice_colour)=}, {imslice_colour.shape=}")
        assert imslice_colour.shape[2] == 3, print(f"{imslice_colour.shape=}")
        l0 = cv.ximgproc.l0Smooth(imslice_colour)
        ad = cv.ximgproc.anisotropicDiffusion(
            imslice_colour, alpha=0.1, K=25.5, niters=10
        )
        aim_run.track(
            {
                "AD": plot_2d(ad),
                "L0": plot_2d(l0),
            },
        )
        plt.close()


if __name__ == "__main__":
    main()
