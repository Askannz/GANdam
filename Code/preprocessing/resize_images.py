# Resizes all images to a target step.
# /!\ First, you need to manually sort the images you want to keep in the scraped dataset,
# and put them in a folder at Generated/preprocessing/manually_selected/.

import os
import shutil
from pathlib import Path
import numpy as np
import cv2

TARGET_SHAPE = (128, 128)

def main():

    input_folder_path = Path("../../Generated/preprocessing/manually_selected/")
    output_folder_path = Path("../../Generated/preprocessing/resized_images/")

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path, exist_ok=True)

    th, tw = TARGET_SHAPE

    for img_path in input_folder_path.iterdir():

        img_name = img_path.name

        img = cv2.imread(str(img_path))
        assert img.ndim == 3
        h, w, _ = img.shape

        dw, dh = w - tw, h - th

        if dw <= 0 and dh <= 0:
            f = tw / w if dw > dh else th / h
        elif dw >= 0 and dh >= 0:
            f = tw / w if dw > dh else th / h
        else:
            f = tw / w if dw > 0 else th / h

        scaled = cv2.resize(img, None, fx=f, fy=f)

        sh, sw, _ = scaled.shape

        pw_l, pw_r = int((tw - sw)/2), tw - sw - int((tw - sw)/2)
        ph_t, ph_b = int((th - sh)/2), th - sh - int((th - sh)/2)

        resized = np.pad(scaled, [(ph_t, ph_b), (pw_l, pw_r), (0, 0)], mode="constant", constant_values=255)

        rh, rw, _ = resized.shape
        assert rh == th and rw == tw

        resized_filepath = output_folder_path / (img_name + ".png")
        cv2.imwrite(str(resized_filepath), resized)


if __name__ == "__main__":
    main()
