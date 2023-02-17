import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.image_loader import CellImage
from src.l0_region_smoothing import l0_region_smoothing
from src.utils.aim import L0Callback, experiment_context
from src.utils.storage import get_project_root
from src.visualization import plot_2d, plot_histogram

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)
CONFIG_PATH = str(get_project_root() / "conf")


@hydra.main(config_path=CONFIG_PATH, config_name="preprocessing", version_base="1.3")
def main(cfg: DictConfig):
    with experiment_context(cfg) as aim_run:
        path_to_file = get_project_root() / "data" / cfg.image.path
        ci = CellImage(path=path_to_file)

        imslice = (
            ci.get_slice(
                x=cfg.image.slice.x[0],
                equalize=cfg.image.equalize,
                lower_bound=cfg.image.lower_bound,
                unsharp_mask=cfg.image.unsharp_mask,
                regenerate=cfg.image.regenerate,
            )
            / 65536
        ).astype(np.float16)

        aim_run.track(
            {
                "image": plot_2d(imslice),
                "histogram": plot_histogram(imslice),
            },
            context={"context": "initial"},
        )

        lcallback = L0Callback(aim_run=aim_run, cfg=cfg, shape=imslice.shape)
        smooth = l0_region_smoothing(
            imslice, **cfg.image.l0_region, callback=lcallback.l0_callback
        )

        aim_run.track(
            {
                "L0": plot_2d(smooth),
            },
            context={"context": "final"},
        )
        plt.close()


if __name__ == "__main__":
    main()
