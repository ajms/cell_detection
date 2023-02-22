import getpass
import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import aim
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.cython_implementations.l0_region_smoothing import reconstruct_image
from src.utils.storage import get_project_root
from src.visualization import plot_2d, plot_3d, plot_quantiles


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


class L0Callback:
    def __init__(
        self,
        aim_run: aim.Run,
        cfg: OmegaConf,
        shape: tuple[int],
    ):
        self.aim_run = aim_run
        self.cfg = cfg
        self.shape = shape
        self.M = np.product(shape, dtype=np.int32).item()

    def l0_callback(
        self,
        iter: int | None = None,
        beta: float | None = None,
        num_keys: int | None = None,
        N: None | dict[int, set] = None,
        G: None | dict[int, list] = None,
        Y: None | dict[int, np.float16] = None,
        w: None | dict[int, int] = None,
    ):
        logging.info(
            f"In the callback: {self.M=}, {self.shape=}, {iter=}, {beta=}, {num_keys=}"
        )
        image = reconstruct_image(self.M, self.shape, G, Y)
        v_len = np.vectorize(lambda x: len(x) if x else 0)
        g_len = v_len(G.data)
        max_G = g_len.max()
        P = np.count_nonzero(g_len)
        logging.info(f"{max_G=}, {P=}")
        logging.debug(f"{image.shape}")
        self.aim_run.track(
            {
                "beta": beta,
                "n_keys": num_keys,
                "P": P,
                "number of groups": G.rows.shape[0],
                "biggest group": max_G,
                "max weight": w.max(),
                "image": plot_2d(
                    image if len(self.shape) == 2 else image[self.shape[0] // 2]
                ),
                "histogram": aim.Distribution(image.flatten()),
                "quantiles": plot_quantiles(image=image),
                "N memory": sys.getsizeof(N),
                "G memory": sys.getsizeof(G),
                "Y memory": sys.getsizeof(Y),
                "w memory": sys.getsizeof(w),
            },
            step=iter,
            context={"context": "step"},
        )
        logging.info("Tracking complete")
        plt.close()


class ScipyCallback:
    def __init__(
        self,
        aim_run: aim.Run,
        cfg: OmegaConf,
        fun: Callable,
        image: None | np.ndarray,
        x: None | tuple[int],
        **kwargs,
    ):
        self.aim_run = aim_run
        self.cfg = cfg
        self.step = 0
        self.image = image
        self.fun = fun
        self.x = x
        self.kwargs = kwargs

        self.X, self.Y = np.meshgrid(
            range(self.image.shape[2]),
            range(self.image.shape[1]),
        )

    def scipy_optimize_callback(self, xk: np.ndarray):
        self.step += 1
        loss, dloss = self.fun(xk, self.image, **self.cfg.ol, **self.kwargs)
        if self.step % 7 == 0:
            if self.x:
                surfaces = {
                    f"levelset at {xl}": plot_3d(
                        levelset=xk.reshape(self.image.shape)[xl, :, :],
                        X=self.X,
                        Y=self.Y,
                    )
                    for xl in self.x
                }
            else:
                surfaces = (
                    {
                        "levelset": plot_3d(
                            xk.reshape(self.image.shape), X=self.X, Y=self.Y
                        ),
                    },
                )

            self.aim_run.track(
                value=surfaces,
                step=self.step,
                context={"context": "step"},
            )

        self.aim_run.track(
            {
                "loss": loss,
                "max_dloss": np.max(dloss),
                "min_dloss": np.min(dloss),
            },
            step=self.step,
            context={"context": "step"},
        )

        plt.close()
