import numpy as np
from skimage import io

from src.image_loader import CellImage
from src.utils.storage import get_project_root

if __name__ == "__main__":
    img = io.imread(
        get_project_root()
        / "data/cell-detection/exp/2023-03-05_14:54:59/segmentation.tif"
    )
    print(
        f"{np.count_nonzero(img)=}, {np.count_nonzero(img == 0)=}, {np.product(img.shape)=}"
    )
    path_to_file = (
        get_project_root()
        / "data/cell-detection/exp/2023-02-26_10:22:51/smooth_image.tif"
    )
    ci = CellImage(path=path_to_file)
    ci.show_3d(img)
