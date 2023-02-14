import logging
from dataclasses import dataclass, field
from pathlib import Path

import aim
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
        lower_bound: float = 0,
        unsharp_mask: dict = {"radius": 0, "amount": 0},
    ):
        path_to_img = (
            self.path.parent
            / f"{self.path.stem}_eq_{equalize}_lower_{lower_bound}_unsharp_{unsharp_mask['radius']}_{unsharp_mask['amount']}.tif"
        )
        if path_to_img.exists() and not regenerate:
            logging.info(f"Reading slice from harddisk: {path_to_img}")
            self.image = io.imread(path_to_img)
        else:
            self.image = self.dset[0, :, 890:, 950:1500, 0]
            logging.info(f"{self.image.shape=}")
            logging.info(f"{self.image.dtype=}")
            logging.info(f"{self.image.min()=}, {self.image.max()=}")
            if equalize == "global":
                self.image = self.equalize_histogram(self.image)
            if equalize == "local":
                self.image = self.equalize_local(
                    self.image, lower_bound=lower_bound, unsharp_mask=unsharp_mask
                )
            logging.info(f"Writing slice to harddisk: {path_to_img}")
            io.imsave(path_to_img, self.image)

    def equalize_histogram(self, image: np.ndarray):
        logging.info("equalizing histogram")
        return exposure.equalize_hist(self.image)

    def equalize_local(
        self, image: np.ndarray, lower_bound: float, unsharp_mask: dict
    ) -> np.ndarray:
        logging.info(f"equalizing locally: {np.min(self.image)=},{np.max(self.image)=}")
        if lower_bound >= 1:
            image[image < lower_bound] = lower_bound
            logging.info(f"{np.average(image)=} before unsharp mask")
        b = np.max(image)
        a = np.min(image)
        image = (image - a) / (b - a)
        assert np.round(np.max(image), 0) == 1, np.max(image)
        assert np.round(np.min(image), 0) == 0, np.min(image)
        image = filters.unsharp_mask(image, **unsharp_mask)
        if lower_bound < 1:
            logging.info(f"{np.average(image)=} after unsharp mask")
            image[image < lower_bound] = lower_bound
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

    def get_slice(
        self,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
        equalize: None | str = "global",
        lower_bound: float = 3e-4,
        unsharp_mask: dict = {"radius": 0, "amount": 0},
        regenerate: bool = False,
        show: bool = False,
        return_path: bool = False,
    ) -> np.ndarray | Path:
        path_to_slice = (
            self.path.parent
            / f"{self.path.stem}_eq_{equalize}_{x}_{y}_{z}_lower_{lower_bound}_unsharp_{unsharp_mask['radius']}_{unsharp_mask['amount']}.tif"
        )
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

            if equalize == "local":
                imslice = self.equalize_local(
                    imslice, lower_bound=lower_bound, unsharp_mask=unsharp_mask
                )
            elif equalize == "global":
                imslice = self.equalize_histogram(imslice)

            logging.info(f"Writing slice to harddisk: {path_to_slice}")
            io.imsave(path_to_slice, imslice)

            if show:
                io.imshow(imslice)
                plt.show()
        if return_path:
            return path_to_slice
        return imslice

    def show_3d(self, image: np.ndarray | None = None):
        if image is None:
            image = self.image
        _ = napari.view_image(image, contrast_limits=[image.min(), image.max()])
        napari.run()


if __name__ == "__main__":
    path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
    ci = CellImage(path=path_to_file)
    repo = get_project_root() / "data/cell-detection/aim"
    experiment = "unsharp mask"
    lower_bound = 0
    # amount = 1
    radius = 80

    aim_run = aim.Run(
        repo=str(repo),
        experiment=experiment,
    )
    aim_run["metadata"] = {"type": "unsharp_mask"}
    for amount in range(2, 10, 1):
        imslice = ci.get_slice(
            x=356,
            equalize="local",
            lower_bound=lower_bound,
            unsharp_mask={"radius": radius, "amount": amount},
            regenerate=True,
        )
        fig = plt.figure()
        ha = fig.add_subplot()
        ha.imshow(imslice)
        aim_run.track(
            aim.Image(fig),
            "unsharp mask",
            step=radius,
            context={
                "lower_bound": lower_bound,
                "amount": amount,
                "radius": radius,
            },
        )
    aim_run.close()
    # ci.show_3d()
