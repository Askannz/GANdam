import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from keras.models import load_model

LATENT_DIM = 300
NB_SAMPLES = 32

def main():

    generator_model_path = Path("../../Generated/training/gan_models/final/generator.h5")
    upscaler_model_path = Path("../../Generated/training/upscaler_models/final/upscaler.h5")
    samples_path = Path("../../Generated/testing/samples/")

    shutil.rmtree(samples_path, ignore_errors=True)
    os.makedirs(samples_path)

    generator = load_model(str(generator_model_path))
    upscaler = load_model(str(upscaler_model_path))

    noise = np.random.normal(0, 1, (NB_SAMPLES, LATENT_DIM))
    img_array_float = generator.predict(noise)
    img_array_upscaled_float = upscaler.predict(img_array_float)
    img_array = np.clip((img_array_upscaled_float + 1) * 127.5, 0, 255).astype(np.uint8)

    for i, img in enumerate(img_array):
        img_path = samples_path / ("%d.png" % i)
        cv2.imwrite(str(img_path), img)


def _sample_generated_images(generator, epoch):

    samples_path = Path("../../Generated/testing/samples/")

    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    img_float = generator.predict(noise)[0]
    img = np.clip((img_float + 1) * 127.5, 0, 255).astype(np.uint8)

    img_path = samples_path / ("%d.png" % epoch)
    cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    main()
