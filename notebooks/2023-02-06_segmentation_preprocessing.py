# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.restoration import (
    denoise_bilateral,
    denoise_tv_chambolle,
    denoise_wavelet,
    estimate_sigma,
)

from src.image_loader import CellImage
from src.utils.storage import get_project_root

# %%

rel_path = "data/cell-detection/exp/2023-02-05_12:46:19/1/levelset.tif"
levelset = io.imread(get_project_root() / rel_path)
# %%
rel_path = "data/cell-detection/exp/2023-01-27_13:48:42/0/segmentation.tif"  # "data/cell-detection/exp/2023-01-27_09:23:40/segmentation.tif"
segmentation = io.imread(get_project_root() / rel_path)
path_to_file = get_project_root() / "data/cell-detection/raw/cropped_first_third.h5"
ci = CellImage(path=path_to_file)

# %%
n = levelset.shape[0] // 2
segmentation = np.zeros(levelset.shape)
segmentation[levelset >= 0] = 1
# %%
imslice1 = ci.get_slice(x=355, equalize=None, regenerate=False)
imslice2 = ci.get_slice(
    x=355,
    equalize="local",
    lower_bound=0,
    unsharp_mask={"radius": 60, "amount": 2},
    regenerate=False,
)

# %%

plt.rcParams["figure.figsize"] = [20, 20]
fig = plt.figure()
sp0 = fig.add_subplot(241)
sp1 = fig.add_subplot(242)
sp2 = fig.add_subplot(243)
sp3 = fig.add_subplot(244)
sp4 = fig.add_subplot(245)
sp5 = fig.add_subplot(246)
sp6 = fig.add_subplot(247)
sp7 = fig.add_subplot(248)
sp0.imshow(imslice1)
sp1.imshow(imslice2)
sp2.imshow(denoise_tv_chambolle(imslice1, weight=0.5, channel_axis=None))

sp3.imshow(denoise_wavelet(imslice1, channel_axis=None, rescale_sigma=True))
sp4.imshow(denoise_wavelet(imslice2, channel_axis=None, rescale_sigma=True))
sp5.imshow(denoise_tv_chambolle(imslice2, weight=0.5, channel_axis=None))
sp6.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=None,
    )
)
sp7.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.05,
        sigma_spatial=20,
        channel_axis=None,
    )
)
# %%
sigma_est = estimate_sigma(imslice2, channel_axis=-1, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f"Estimated Gaussian noise standard deviation = {sigma_est}")
# %%
fig = plt.figure()
sp0 = fig.add_subplot(121)
sp1 = fig.add_subplot(122)
sp0.imshow(imslice2)
sp1.imshow(
    denoise_wavelet(
        imslice2, sigma=0.1, mode="hard", channel_axis=None, rescale_sigma=True
    )
)
# %%
fig = plt.figure()
sp0 = fig.add_subplot(131)
sp1 = fig.add_subplot(132)
sp2 = fig.add_subplot(133)
sp0.imshow(imslice2)
sp1.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=None,
    )
)
sp2.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.1,
        sigma_spatial=15,
        channel_axis=None,
    )
)
# %%
fig = plt.figure()
sp0 = fig.add_subplot(131)
sp1 = fig.add_subplot(132)
sp2 = fig.add_subplot(133)
sp0.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.05,
        sigma_spatial=30,
        channel_axis=None,
    )
)
sp1.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.1,
        sigma_spatial=20,
        channel_axis=None,
    )
)
sp2.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.2,
        sigma_spatial=15,
        channel_axis=None,
    )
)

# %%
plt.imshow(
    denoise_bilateral(
        imslice2,
        sigma_color=0.1,
        sigma_spatial=50,
        channel_axis=None,
    )
)
# %%
