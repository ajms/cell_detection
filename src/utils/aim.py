import getpass
import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import aim
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.utils.storage import get_project_root


@contextmanager
def experiment_context(cfg: DictConfig):
    repo = get_project_root() / "data/cell-detection/aim"
    aim_run = aim.Run(
        repo=str(repo),
        experiment=cfg.experiment,
    )

    cwd = Path.cwd()

    logging.info("Experiment metadata")
    logging.info(f"aim: {repo=}")
    logging.info(f"{aim_run.hash=}")
    logging.info(f"{cwd=}")
    logging.info(f"{OmegaConf.to_yaml(cfg)=}")

    # Log various param
    aim_run["metadata"] = {
        "user": getpass.getuser(),
        "cwd": str(cwd.relative_to(get_project_root())),
        "git": {"sha": _git_sha(), "branch": _git_branch()},
    }
    logging.info("Adding hydra data...")
    _hydra_cfg(aim_run, cfg)

    logging.info("Serve aim for tracking")
    try:
        yield aim_run
    except Exception as e:
        logging.info(f"Error '{e}' occured")
    finally:
        aim_run.close()


def _hydra_cfg(aim_run, cfg) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            aim_run[k] = v
        else:
            aim_run["metadata"][k] = v


def _git_sha() -> str:
    cmd = "git rev-parse HEAD"
    sha = subprocess.check_output(cmd.split(" ")).decode("utf-8").strip()
    return sha


def _git_branch() -> str:
    cmd = "git rev-parse --abbrev-ref HEAD"
    branch = subprocess.check_output(cmd.split(" ")).decode("utf-8")
    return branch


class ScipyCallback:
    def __init__(
        self,
        aim_run: aim.Run,
        cfg: OmegaConf,
        fun: Callable,
        image: np.ndarray,
        **kwargs,
    ):
        self.aim_run = aim_run
        self.cfg = cfg
        self.step = 0
        self.image = image
        self.fun = fun
        self.kwargs = kwargs
        self.X, self.Y = np.meshgrid(
            range(self.image.shape[1]),
            range(self.image.shape[0]),
        )

    def scipy_optimize_callback(self, xk: np.ndarray):
        self.step += 1
        loss, dloss = self.fun(xk, self.image, **self.cfg.ol, **self.kwargs)

        fig = plt.figure()
        ha = fig.add_subplot(projection="3d")
        ha.plot_surface(self.X, self.Y, xk.reshape(self.image.shape), cmap="viridis")

        self.aim_run.track(
            {
                "loss": loss,
                "max_dloss": np.max(dloss),
                "min_dloss": np.min(dloss),
                "levelset": aim.Image(fig),
            },
            step=self.step,
            context={"context": "step"},
        )
        plt.close()