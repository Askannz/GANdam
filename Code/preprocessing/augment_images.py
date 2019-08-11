# Doubles the dataset size by flipping all images along the horizontal axis.

import os
import shutil
from pathlib import Path
import numpy as np
import cv2


def main():

    input_folder_path = Path("../../Generated/preprocessing/resized_images/")
    output_folder_path = Path("../../Generated/preprocessing/augmented_images/")

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path, exist_ok=True)

    for img_path in input_folder_path.iterdir():

        img_name = img_path.name

        img = cv2.imread(str(img_path))
        flipped_img = np.flip(img, axis=1)

        normal_filepath = output_folder_path / (img_name + ".png")
        flipped_filepath = output_folder_path / (img_name + "_f.png")
        cv2.imwrite(str(normal_filepath), img)
        cv2.imwrite(str(flipped_filepath), flipped_img)


if __name__ == "__main__":
    main()
 
