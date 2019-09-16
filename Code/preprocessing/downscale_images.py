# Resizes all images to a target step.
# /!\ First, you need to manually sort the images you want to keep in the scraped dataset,
# and put them in a folder at Generated/preprocessing/manually_selected/.

import os
import shutil
from pathlib import Path
import cv2

TARGET_SHAPE = (64, 64)

def main():

    input_folder_path = Path("../../Generated/preprocessing/augmented_images/")
    output_folder_path = Path("../../Generated/preprocessing/downscaled_images/")

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path, exist_ok=True)

    for img_path in input_folder_path.iterdir():

        img_name = img_path.name

        img = cv2.imread(str(img_path))
        assert img.ndim == 3

        resized = cv2.resize(img, TARGET_SHAPE)

        resized_filepath = output_folder_path / (img_name + ".png")
        cv2.imwrite(str(resized_filepath), resized)


if __name__ == "__main__":
    main()
