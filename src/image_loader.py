import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
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

    def read_image(
        self,
        regenerate: bool = False,
        equalize: None | str = None,
        q_lower_bound: float = 0.0,
        unsharp_mask: dict = {"radius": 0, "amount": 0},
        l0_smoothing: None | dict = None,
    ):
        if not l0_smoothing:
            l0_smoothing = {}
        fname = "_".join(
            map(
                str,
                [equalize, q_lower_bound] + list(unsharp_mask) + list(l0_smoothing),
            )
        )
        path_to_img = self.path.parent / f"{self.path.stem}_{fname}.tif"
        if path_to_img.exists() and not regenerate:
            logging.info(f"Reading image from harddisk: {path_to_img}")
            self.image = io.imread(path_to_img)
        else:
            self.image = self.dset[0, :, 890:, 950:1500, 0]
            logging.info(f"{self.image.shape=}")
            logging.info(f"{self.image.dtype=}")
            logging.info(f"{self.image.min()=}, {self.image.max()=}")
            if q_lower_bound:
                self.image = self.lower_bound(image=self.image, q=q_lower_bound)
            if equalize == "global":
                self.image = self.equalize_histogram(self.image)
            elif equalize == "local":
                self.image = self.equalize_local(self.image, unsharp_mask=unsharp_mask)
            else:
                self.image = self._normalize(image=self.image)
            if l0_smoothing:
                self.image = (self.image * 255).astype(np.uint8)
                l0 = np.zeros(list(self.image.shape))
                for idx, slice in enumerate(self.image):
                    l0[idx] = cv2.ximgproc.l0Smooth(src=slice, **l0_smoothing)
                assert isinstance(l0, np.ndarray), type(l0)
                self.image = self._normalize(l0)
                del l0

            logging.info(f"Writing slice to harddisk: {path_to_img}")
            io.imsave(path_to_img, self.image)

    def get_slice(
        self,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
        equalize: None | str = "global",
        q_lower_bound: float = 0.0,
        unsharp_mask: dict = {"radius": 0, "amount": 0},
        l0_smoothing: None | dict = None,
        regenerate: bool = False,
    ) -> np.ndarray | Path:
        if not l0_smoothing:
            l0_smoothing = {}
        fname = "_".join(
            map(
                str,
                [x, y, z, equalize, q_lower_bound]
                + list(unsharp_mask.values())
                + list(l0_smoothing.values()),
            )
        )
        path_to_slice = self.path.parent / f"{self.path.stem}_{fname}.tif"
        if path_to_slice.exists() and not regenerate:
            logging.info(f"Reading slice from harddisk: {path_to_slice}")
            imslice = io.imread(path_to_slice)
        else:
            if not hasattr(self, "image"):
                self.read_image()
            if x:
                imslice = self.image[x, :, :]
            elif y:
                imslice = self.image[:, y, :]
            elif z:
                imslice = self.image[:, :, z]

            if q_lower_bound:
                imslice = self.lower_bound(image=imslice, q=q_lower_bound)

            if equalize == "local":
                imslice = self.equalize_local(imslice, unsharp_mask=unsharp_mask)
            elif equalize == "global":
                imslice = self.equalize_histogram(imslice)
            else:
                imslice = self._normalize(image=imslice)

            if l0_smoothing:
                imslice = (imslice * 255).astype(np.uint8)
                imslice = cv2.ximgproc.l0Smooth(src=imslice, **l0_smoothing)
                imslice = self._normalize(imslice)

            logging.info(f"Writing slice to harddisk: {path_to_slice}")
            io.imsave(path_to_slice, imslice)
        return imslice

    def equalize_histogram(self, image: np.ndarray):
        logging.info("equalizing histogram")
        return exposure.equalize_hist(image)

    def lower_bound(self, image: np.ndarray, q: float) -> np.ndarray:
        q_percentile = np.quantile(image, q)
        image[image < q_percentile] = q_percentile
        return image

    def equalize_local(self, image: np.ndarray, unsharp_mask: dict) -> np.ndarray:
        logging.info(f"equalizing locally: {np.min(image)=},{np.max(image)=}")
        image = self._normalize(image)
        image = filters.unsharp_mask(image, **unsharp_mask)
        return image

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

    def show(self, image: np.ndarray):
        plt.imshow(image)
        plt.show()

    def show_3d(self, image: np.ndarray | None = None):
        if image is None:
            image = self.image
        _ = napari.view_image(image, contrast_limits=[image.min(), image.max()])
        napari.run()

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        logging.info(f"Normalizing {image.shape=}")
        b = np.max(image)
        a = np.min(image)
        image = (image - a) / (b - a)
        return image


if __name__ == "__main__":
    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    repo = get_project_root() / "data/cell-detection/aim"
    # imslice = ci.get_slice(
    #     x=356,
    #     equalize=None,
    #     q_lower_bound=0.01,
    #     regenerate=False,
    # )
    # ci.show(imslice)
    ci.read_image(
        equalize=None,
        q_lower_bound=0.01,
        regenerate=False,
    )
    img = ci.image[360:370, 50:-50, 50:-50]
    # img = io.imread(
    #     "/home/albert/repos/cell_detection/data/cell-detection/exp/2023-02-23_12:23:19/smooth_image.tif"
    # )
    ci.show_3d(img)
