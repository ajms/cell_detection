# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from src.image_loader import CellImage
from src.utils.storage import get_project_root

# %%
cfg = OmegaConf.load(get_project_root() / "conf/preprocessing.yaml")
cfg.image.q_lower_bound = 0
cfg.image.q_upper_bound = 1
ci = CellImage(cfg=cfg, path=get_project_root() / "data" / cfg.image.path)
img = ci.get_slice(10, equalize=None, q_lower_bound=0, regenerate=True)
img = (img * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# %%
plt.imshow(img)


# %%
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


# %%
model_type = "vit_b"
sam_checkpoint = str(
    get_project_root() / "data/cell-detection/models/sam_vit_b_01ec64.pth"
)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)
# %%
masks = mask_generator.generate(img)  # Requires open-cv to run post-processing)


# %%

plt.figure(figsize=(20, 20))
plt.imshow(img)
show_anns(masks)
plt.axis("off")
plt.show()


# %%
