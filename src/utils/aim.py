from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager
import aim
from pathlib import Path
from src.utils.storage import get_project_root
import logging
import getpass
import subprocess


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
