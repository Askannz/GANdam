# Resizes all images to a target step.
# /!\ First, you need to manually sort the images you want to keep in the scraped dataset,
# and put them in a folder at Generated/preprocessing/manually_selected/.

import os
import shutil
from pathlib import Path
import cv2


def main():

    input_folder_path = Path("../../Generated/preprocessing/manually_sorted_flip/")
    output_folder_path = Path("../../Generated/preprocessing/flip_corrected/")

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path, exist_ok=True)

    left_images_folder_path = input_folder_path / "left"
    right_images_folder_path = input_folder_path / "right"

    for img_path in left_images_folder_path.iterdir():
        shutil.copy(img_path, output_folder_path)

    for img_path in right_images_folder_path.iterdir():

        img_name = img_path.name

        img = cv2.imread(str(img_path))
        assert img.ndim == 3

        img_flipped = cv2.flip(img, 1)

        flipped_filepath = output_folder_path / (img_name + ".png")
        cv2.imwrite(str(flipped_filepath), img_flipped)


if __name__ == "__main__":
    main()
