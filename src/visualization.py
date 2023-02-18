import aim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d(image: np.ndarray) -> aim.Image:
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(image)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    # subplot = fig.add_subplot()
    # subplot.imshow(image, cmap="viridis")
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
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


def plot_histogram(image: np.ndarray, bins: int = 256):
    # create the histogram
    counts, bins = np.histogram(image, bins=bins, range=(0, 1))
    # configure and draw the histogram figure
    fig = plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.stairs(counts, bins)  # <- or here
    aim_image = aim.Image(fig)
    plt.close()
    return aim_image


def plot_quantiles(image: np.ndarray):
    xs = np.arange(0, 1, 0.01)
    ys = [np.quantile(image, x) for x in xs]
    fig = plt.figure()
    plt.title("Grayscale quantiles")
    plt.xlabel("grayscale value")
    plt.ylabel("mass")
    plt.plot(xs, ys)
    aim_figure = aim.Figure(fig)
    plt.close()
    return aim_figure
