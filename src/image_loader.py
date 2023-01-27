import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import napari
import numpy as np
from skimage import exposure, filters, io, morphology

from src.utils.storage import get_project_root

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)


@dataclass
class CellImage:
    path: Path
    hf: h5py.File = field(init=False)
    dset: h5py.Dataset = field(init=False)
    image: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)

    def __post_init__(self):
        self.hf = h5py.File(self.path, "r")
        self.dset = self.hf[list(self.hf.keys())[-1]]
        logging.debug(f"{list(self.hf.keys())=}")

    def read_image(self):
        self.image = self.dset[0, :, 890:, 950:1500, 0]
        logging.info(f"{self.image.shape=}")
        logging.info(f"{self.image.dtype=}")
        logging.info(f"{self.image.min()=}, {self.image.max()=}")

    def equalize_histogram(self):
        path_to_equalized = self.path.parent / f"{self.path.stem}_eq.tif"
        if path_to_equalized.exists():
            logging.info(f"Reading image from harddisk: {path_to_equalized}")
            self.image = io.imread(path_to_equalized)
        else:
            logging.info("equalizing histogram")
            self.read_image()
            self.image = exposure.equalize_hist(self.image)
            logging.info(f"Writing image to harddisk: {path_to_equalized}")
            io.imsave(path_to_equalized, self.image)

    def edge_detection(self, show: bool = False):
        logging.info("detecting edges")
        self.edges = filters.sobel(self.image)
        logging.info("edge detection finished")
        if show:
            viewer = napari.view_image(
                self.image, blending="additive", colormap="green", name="image"
            )
            viewer.add_image(
                self.edges, blending="additive", colormap="magenta", name="edges"
            )
            napari.run()

    def remove_holes(self, width: int = 20):
        logging.info("removing holes")
        image = morphology.remove_small_holes(self.edges, area_threshold=width**3)
        logging.info("holes removed")
        self.show_3d(image)

    def get_slice(
        self,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
        equalize: None | str = "global",
        regenerate: bool = False,
        show: bool = False,
    ) -> np.ndarray:
        path_to_slice = self.path.parent / f"{self.path.stem}_eq_{x}_{y}_{z}.tif"
        if path_to_slice.exists() and not regenerate:
            logging.info(f"Reading slice from harddisk: {path_to_slice}")
            imslice = io.imread(path_to_slice)
        else:
            if equalize == "global":
                self.equalize_histogram()
            else:
                self.read_image()
            if x:
                imslice = self.image[x, :, :]
            elif y:
                imslice = self.image[:, y, :]
            elif z:
                imslice = self.image[:, :, z]

            if show:
                io.imshow(imslice)
                plt.show()
            if equalize == "local":
                logging.info("equalizing locally")
                imslice[imslice < 3e4] = 3e4
                b = np.max(imslice)
                a = np.min(imslice)
                imslice = (imslice - a) / (b - a)
                assert np.round(np.max(imslice), 0) == 1, np.max(imslice)
                assert np.round(np.min(imslice), 0) == 0, np.min(imslice)
            logging.info(f"Writing slice to harddisk: {path_to_slice}")
            io.imsave(path_to_slice, imslice)
        return imslice

    def show_3d(self, image: np.ndarray | None = None):
        if image is None:
            image = self.image
        _ = napari.view_image(image, contrast_limits=[image.min(), image.max()])
        napari.run()


if __name__ == "__main__":
    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    imslice = ci.get_slice(x=356, equalize="local", regenerate=False)
    plt.imshow(imslice)
    plt.show()

    ci.show_3d()
