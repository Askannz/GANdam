from pathlib import Path
import cv2
import numpy as np
from shared_constants import DOWNSCALED_IMG_SHAPE, ORIGINAL_IMG_SHAPE


def load_gan_data():
    data_folder_path = Path("../../Generated/preprocessing/downscaled_images/")
    return _load_images(data_folder_path, DOWNSCALED_IMG_SHAPE)

def load_upscaler_data():

    samples_folder_path = Path("../../Generated/preprocessing/downscaled_images/")
    samples_data = _load_images(samples_folder_path, DOWNSCALED_IMG_SHAPE)

    labels_folder_path = Path("../../Generated/preprocessing/augmented_images/")
    labels_data = _load_images(labels_folder_path, ORIGINAL_IMG_SHAPE)

    return samples_data, labels_data


def _load_images(data_folder_path, img_shape):

    h, w = img_shape

    filepaths_list = list(sorted(data_folder_path.iterdir(), key=lambda p: p.name))
    n = len(filepaths_list)

    images_array = np.zeros((n, h, w, 3), np.uint8)

    for i, filepath in enumerate(filepaths_list):
        img = cv2.imread(str(filepath))
        images_array[i, :, :, :] = img

    # Rescaling values between -1 and 1
    images_array_norm = images_array / 127.5 - 1

    return images_array_norm
