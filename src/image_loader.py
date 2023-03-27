import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import napari
import numpy as np
from omegaconf import DictConfig, OmegaConf
from skimage import exposure, filters, io, morphology

from src.utils.storage import get_project_root

logging.basicConfig(
    format="%(levelname)s [%(asctime)s]: %(message)s", level=logging.INFO
)


@dataclass
class CellImage:
    path: Path
    cfg: DictConfig
    hf: h5py.File = field(init=False)
    dset: h5py.Dataset = field(init=False)
    image: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.path.suffix == ".tif":
            self.image = io.imread(self.path)
        else:
            self.hf = h5py.File(self.path, "r")
            self.dset = self.hf[list(self.hf.keys())[-1]]
            logging.debug(f"{list(self.hf.keys())=}")

    def read_image(self):
        fname = "_".join(
            map(
                str,
                [self.cfg.image.equalize, self.cfg.image.q_lower_bound]
                + list(self.cfg.image.unsharp_mask),
            )
        )
        path_to_img = self.path.parent / f"{self.path.stem}_{fname}.tif"
        if path_to_img.exists() and not self.cfg.image.regenerate:
            logging.info(f"Reading image from harddisk: {path_to_img}")
            self.image = io.imread(path_to_img)
        else:
            logging.info(f"Load image from h5: {path_to_img}")
            if self.path.suffix != ".tif":
                # self.image = self.dset[0, :, 890:, 950:1500, 0]
                self.image = self.dset[
                    0,
                    self.cfg.image.slice.x[0] : self.cfg.image.slice.x[1],
                    self.cfg.image.slice.y[0] : self.cfg.image.slice.y[1],
                    self.cfg.image.slice.z[0] : self.cfg.image.slice.z[1],
                    0,
                ]
                logging.info(f"{self.image.shape=}")
                logging.info(f"{self.image.dtype=}")
                logging.info(f"{self.image.min()=}, {self.image.max()=}")
            if self.cfg.image.q_lower_bound:
                self.image = self._set_bounds(
                    image=self.image, lower_q=self.cfg.image.q_lower_bound
                )
            if self.cfg.image.equalize == "global":
                self.image = self.equalize_histogram(self.image)
            elif self.cfg.image.equalize == "local":
                self.image = self.equalize_local(
                    self.image, unsharp_mask=self.cfg.image.unsharp_mask
                )
            else:
                self.image = self._normalize(image=self.image)
            # if self.cfg.image.l0_smoothing:
            #     self.image = (self.image * 255).astype(np.uint8)
            #     l0 = np.zeros(list(self.image.shape))
            #     for idx, slice in enumerate(self.image):
            #         l0[idx] = cv2.ximgproc.l0Smooth(
            #             src=slice, **self.cfg.image.l0_smoothing
            #         )
            #     assert isinstance(l0, np.ndarray), type(l0)
            #     self.image = self._normalize(l0)
            #     del l0

            logging.info(f"Writing image to harddisk: {path_to_img}")
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
                imslice = self._set_bounds(image=imslice, lower_q=q_lower_bound)

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

    def _set_bounds(
        self, image: np.ndarray, lower_q: float, upper_q: float = 1.0
    ) -> np.ndarray:
        q_percentile = np.quantile(image, lower_q)
        _q_percentile = np.quantile(image, upper_q)
        logging.info(
            f"lower_q_percentile: {q_percentile}, upper_q_percentile: {_q_percentile}"
        )
        image[image < q_percentile] = q_percentile
        image[image > _q_percentile] = _q_percentile
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
        a = image.min()
        b = image.max()
        image_normalized = ((image - a) / (b - a)).astype(np.single)
        return image_normalized


if __name__ == "__main__":
    # path_to_file = (
    #     get_project_root()
    #     / "data/cell-detection/exp/2023-03-05_19:26:45/image_step_7.tif"
    # )
    # path_to_file = (
    #     get_project_root()
    #     / "data/cell-detection/exp/2023-03-05_16:44:19/image_step_5.tif"
    # )
    # path_to_file = (
    #     get_project_root()
    #     / "data/cell-detection/exp/2023-02-26_10:22:51/image_step_9.tif"
    # )
    cfg = OmegaConf.load("conf/preprocessing.yaml")
    cfg.image.slice.x = [0, 20]
    cfg.image.slice.y = [1000, 1500]
    ci = CellImage(path=get_project_root() / f"data/{cfg.image.path}", cfg=cfg)
    ci.read_image()
    # imslice = ci.get_slice(
    #     x=356,
    #     equalize=None,
    #     q_lower_bound=0.01,
    #     regenerate=False,
    # )
    # ci.show(imslice)
    # ci.read_image(
    #     equalize=None,
    #     q_lower_bound=0.01,
    #     regenerate=False,
    # )

    # img = io.imread(
    #     "/home/albert/repos/cell_detection/data/cell-detection/exp/2023-03-05_19:26:45/image_step_7.tif"
    # )
    ci.show_3d()
