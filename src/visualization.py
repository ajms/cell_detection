import aim
import matplotlib.pyplot as plt
import numpy as np


def plot_2d(image: np.ndarray) -> aim.Image:
    fig = plt.figure()
    subplot = fig.add_subplot()
    subplot.imshow(image, cmap="viridis")
    aim_image = aim.Image(fig)
    plt.close()
    return aim_image


def plot_3d(
    levelset: np.ndarray, X: None | np.ndarray = None, Y: None | np.ndarray = None
) -> aim.Image:
    fig = plt.figure()
    subplot = fig.add_subplot(projection="3d")
    if X is None or Y is None:
        X, Y = np.meshgrid(
            range(levelset.shape[1]),
            range(levelset.shape[0]),
        )

    subplot.plot_surface(
        X,
        Y,
        levelset,
        cmap="viridis",
    )
    aim_image = aim.Image(fig)
    plt.close()
    return aim_image
