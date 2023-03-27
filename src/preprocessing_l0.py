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
        ci = CellImage(path=path_to_file, cfg=cfg)
        ci.read_image()

        image_center = np.array(ci.image.shape) // 2

        ci.image = (ci.image * 255).astype(np.uint8)

        aim_run.track(
            {
                "image": plot_2d(ci.image[:, image_center[1], :]),
            },
        )
        logging.info(f"{type(ci.image)=}, {ci.image.shape=}")
        l0 = np.zeros(list(ci.image.shape))
        for idx, slice in enumerate(ci.image):
            logging.info(f"[{idx}] {slice.shape=}")
            l0[idx] = cv.ximgproc.l0Smooth(slice)

        aim_run.track(
            {
                "L0": plot_2d(l0[:, image_center[1], :]),
            },
        )
        plt.close()


if __name__ == "__main__":
    main()
