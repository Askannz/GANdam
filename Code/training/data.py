from pathlib import Path
import cv2
import numpy as np
from shared_constants import IMG_SHAPE


def load_data():

    data_folder_path = Path("../../Generated/preprocessing/resized_images/")

    h, w = IMG_SHAPE

    filepaths_list = list(data_folder_path.iterdir())
    n = len(filepaths_list)

    images_array = np.zeros((n, h, w, 3), np.uint8)

    for i, filepath in enumerate(filepaths_list):
        img = cv2.imread(str(filepath))
        images_array[i, :, :, :] = img

    # Rescaling values between -1 and 1
    images_array_norm = images_array / 127.5 - 1

    return images_array_norm
