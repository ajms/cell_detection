#!/usr/bin/env python
# coding: utf-8

# In[96]:


import re
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, io
from skimage.color import label2rgb
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)

from src.utils.storage import get_project_root

# In[3]:


rel_path = "data/cell-detection/exp/2023-02-05_12:46:19/1/levelset.tif"
levelset = io.imread(get_project_root() / rel_path)


# In[4]:


print(np.count_nonzero(np.abs(levelset) < 1e-2))
for sigma in range(0, 0, 5):
    print(f"{sigma=}")
    print(
        np.count_nonzero(np.abs(filters.gaussian(levelset, sigma=sigma)) < 1e-2)
        / np.product(levelset.shape)
    )


# In[121]:


def create_boundary(k, image):
    boundary = np.zeros(image.shape)
    boundary[np.abs(image) < k] = 1
    return boundary


# In[143]:


n = levelset.shape[0] // 2
segmentation = np.zeros(levelset.shape)
segmentation[levelset >= 0] = 1
smooth_dict = {
    str(sigma): filters.gaussian(levelset, sigma=sigma)[n] for sigma in range(1, 11, 2)
}
boundary_dict = {
    f"{i}_{sigma}": create_boundary(i, v)
    for sigma, v in smooth_dict.items()
    for i in range(1, 6, 1)
}


# In[145]:


plt.rcParams["figure.figsize"] = [20, 20]
labs = False
fig = plt.figure(constrained_layout=True)
nrows = 5
ncols = ceil(len(boundary_dict) / nrows)
for idx, (name, seg) in enumerate(boundary_dict.items()):
    h = fig.add_subplot(nrows, ncols, idx + 1)
    if labs:
        label_image = label(seg, connectivity=1, background=1)
        image_label_overlay = label2rgb(label_image, image=seg, bg_label=0)
    else:
        image_label_overlay = seg
    h.imshow(image_label_overlay)
    h.set_title(name)


# In[ ]:


# In[172]:


segmentation_dict = {
    "segmentation_onb": np.minimum(segmentation[n], 1 - boundary_dict["3_5"]),
    "segmentation_ab": np.maximum(segmentation[n], boundary_dict["3_5"]),
    "segmentation_ob": np.minimum(segmentation[n], boundary_dict["5_5"]),
    "segmentation_clever": np.maximum(
        np.minimum(segmentation[n], boundary_dict["5_1"]), boundary_dict["5_5"]
    ),
    "segmentation": segmentation[n],
}
footprint = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
erosion_dict = {f"er_{k}": binary_erosion(v) for k, v in segmentation_dict.items()}
erosion2_dict = {f"2{k}": binary_erosion(v) for k, v in erosion_dict.items()}
erosionk_dict = {
    f"ker_{k}": binary_erosion(v, footprint=footprint) for k, v in erosion_dict.items()
}
dilation_dict = {f"di_{k}": binary_dilation(v) for k, v in segmentation_dict.items()}
dilationk_dict = {
    f"kdi_{k}": binary_dilation(v, footprint=footprint)
    for k, v in segmentation_dict.items()
}
dilation2_dict = {f"2{k}": binary_dilation(v) for k, v in dilation_dict.items()}
opening_dict = {f"op_{k}": binary_opening(v) for k, v in segmentation_dict.items()}
closing_dict = {f"cl_{k}": binary_closing(v) for k, v in segmentation_dict.items()}


# In[186]:


fig = plt.figure()
h0 = fig.add_subplot(241)
h1 = fig.add_subplot(242)
h2 = fig.add_subplot(243)
h3 = fig.add_subplot(244)
h4 = fig.add_subplot(245)
h5 = fig.add_subplot(246)
h6 = fig.add_subplot(247)
h7 = fig.add_subplot(248)

h0.imshow(segmentation_dict["segmentation_ob"])
h1.imshow(dilation_dict["di_segmentation_ob"])
h2.imshow(label(dilationk_dict["kdi_segmentation_ob"], background=1), cmap="Dark2")
h3.imshow(label(dilation2_dict["2di_segmentation_ob"], background=1), cmap="Dark2")
repeated_subtraction = np.minimum(
    dilationk_dict["kdi_segmentation_ob"], boundary_dict["5_1"]
)
dilated_3 = binary_dilation(dilation2_dict["2di_segmentation_ob"])
dilated_4 = binary_dilation(dilated_3)
h4.imshow(repeated_subtraction)
h5.imshow(label(dilated_3, background=1), cmap="Dark2")
h6.imshow(label(dilated_4, background=1), cmap="Dark2")
h7.imshow(segmentation[n])


# In[179]:


plt.rcParams["figure.figsize"] = [20, 10]

segments = (
    segmentation_dict | erosion_dict | erosion2_dict | erosionk_dict | dilationk_dict
)
segments = {
    k: v
    for k, v in segments.items()
    if re.match(r"(?:\w*_|)segmentation_(?:ub|ab)", k) is None
}

fig = plt.figure(constrained_layout=True)
nrows = 2
ncols = ceil(len(segments) / nrows)
for idx, (name, seg) in enumerate(segments.items()):
    h = fig.add_subplot(nrows, ncols, idx + 1)
    if re.match(r"(?:\w*_|)segmentation_ob", name):
        seg = 1 - seg
    label_image = label(seg, connectivity=1, background=0)
    image_label_overlay = label2rgb(label_image, image=segmentation[n], bg_label=0)
    h.imshow(image_label_overlay)
    h.set_title(name)


# In[73]:


nrows


# In[74]:


ncols


# In[ ]:
