import aim
import matplotlib.pyplot as plt
import numpy as np

from src.image_loader import CellImage
from src.utils.storage import get_project_root
from src.visualization import plot_2d


def edge_preserving_smoothing():
    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    repo = get_project_root() / "data/cell-detection/aim"
    imslice = ci.get_slice(
        x=356,
        equalize=None,
        regenerate=False,
    )
    experiment = "edge preserving smoothing"
    lower_bound = 0
    unsharp_mask = {"radius": 80, "amount": 2}
    aim_run = aim.Run(
        repo=str(repo),
        experiment=experiment,
    )

    aim_run["metadata"] = {"type": experiment}
    lambda_ = 0.01

    for kappa in [1.5, 2.5, 3]:
        l0_smoothing = {"lambda_": lambda_, "kappa": kappa}

        imslice = ci.get_slice(
            x=356,
            equalize="local",
            lower_bound=lower_bound,
            unsharp_mask=unsharp_mask,
            l0_smoothing=l0_smoothing,
            regenerate=False,
        )
        aim_run.track(
            {"L0": plot_2d(imslice)},
            context={
                "lower_bound": lower_bound,
                "l0_smoothing": l0_smoothing,
                "unsharp_mask": unsharp_mask,
            },
        )
        aim_run.close()


def plot_quantiles(mx=2e-3, step=1e-5):
    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    imslice = ci.get_slice(
        x=356,
        equalize=None,
        regenerate=False,
    )

    plt.plot(
        np.arange(0, mx, step),
        [np.quantile(imslice, q) for q in np.arange(0, mx, step)],
    )
    plt.show()


if __name__ == "__main__":
    edge_preserving_smoothing()
