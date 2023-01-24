import h5py
import napari
import logging
from src.utils.storage import get_project_root
import numpy as np
from skimage import io, exposure, filters, morphology
from skimage.segmentation import chan_vese
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path

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
        logging.info(f"{list(self.hf.keys())=}")
        self.equalize_histogram()

    def read_image(self):
        self.image = self.dset[0, :, 890:, 950:1500, 0]
        logging.info(f"{self.image.shape=}")
        logging.info(f"{self.image.dtype=}")
        logging.info(f"{self.image.min()=}, {self.image.max()=}")

    def equalize_histogram(self):
        path_to_equalized = self.path.parent / f"{self.path.stem}_eq.tif"
        if path_to_equalized.exists():
            self.image = io.imread(path_to_equalized)
        else:
            logging.info("equalizing histogram")
            self.read_image()
            self.image = exposure.equalize_hist(self.image)
            logging.info("equalizing finished")
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
        x=int | None,
        y=int | None,
        z=int | None,
        show: bool = False,
    ) -> np.ndarray:
        if x:
            imslice = self.dset[0, x, :, :, 0]
        elif y:
            imslice = self.dset[0, :, y, :, 0]
        elif z:
            imslice = self.dset[0, :, :, z, 0]
        if show:
            io.imshow(imslice)
            plt.show()
        return imslice

    def show_3d(self, image: np.ndarray | None = None):
        if image is None:
            image = self.image
        _ = napari.view_image(image, contrast_limits=[image.min(), image.max()])
        napari.run()


if __name__ == "__main__":
    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    imslice = ci.get_slice(x=335)
    plt.imshow(imslice)
    plt.show()

    ci.show_3d()
