# cell_detection
This repo is for developing a method of segmenting and classifying cells from microscopy images.
The method is work in progress.

## requirements
python 3.10, poetry > 3.1

## getting started
Install dependencies by
```bash
poetry install
```

`orderless_levelsets.py` implements an orderless image loss function for segmentation
`segmentation.py` runs the orderless segmentation method
`image_loader.py` contains the class for reading and preprocessing the images
