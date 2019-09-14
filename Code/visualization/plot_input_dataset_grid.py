import os
import shutil
from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

GRID_H = 2
GRID_W = 5

def main():

    dataset_folder_path = Path("../../Generated/preprocessing/flattened_images/")
    output_filepath = Path("../../Generated/visualization/original_dataset.png")

    shutil.rmtree(output_filepath.parent, ignore_errors=True)
    os.makedirs(output_filepath.parent, exist_ok=True)

    filepaths_list = list(dataset_folder_path.iterdir())
    selected_filepaths = random.sample(filepaths_list, GRID_H * GRID_W)

    images_list = [cv2.imread(str(p)) for p in selected_filepaths]

    shapes_list = [img.shape[:2] for img in images_list]
    max_h, max_w = np.max(shapes_list, axis=0)

    resized_images_list = []
    for img in images_list:

        h, w = img.shape[:2]
        dh1 = int((max_h - h) / 2)
        dh2 = max_h - h - dh1
        dw1 = int((max_w - w) / 2)
        dw2 = max_w - w - dw1

        resized_img = np.pad(img, [(dh1, dh2), (dw1, dw2), (0, 0)], mode="constant", constant_values=255)

        resized_images_list.append(resized_img)

    fh, fw = GRID_H * max_h, GRID_W * max_w
    full_img = np.zeros((fh, fw, 3), np.uint8)

    for i, img in enumerate(resized_images_list):

        x = (i % GRID_W) * max_w
        y = (i // GRID_W) * max_h

        full_img[y:y+max_h, x:x+max_w] = img

    cv2.imwrite(str(output_filepath), full_img)


if __name__ == "__main__":
    main()
