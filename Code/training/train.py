import sys
import os
import shutil
import signal
from pathlib import Path
import numpy as np
import cv2
import keras
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from models import make_generator, make_discriminator
from shared_constants import LATENT_DIM
from data import load_data


# This is a workaround to stop Keras from complaining about training a model with frozen weights.
# See https://github.com/keras-team/keras/issues/8585
def _check_trainable_weights_consistency(self):
    return
keras.Model._check_trainable_weights_consistency = _check_trainable_weights_consistency


BATCH_SIZE = 32
LR_DISCRIMINATOR = 0.00003
LR_COMBINED = 0.0001
DISCRIMINATOR_LABELS_NOISE = 0.10

SAMPLE_FREQ = 10
CHECKPOINT_FREQ = 2000

def main():

    handler = _SignalHandler()
    signal.signal(signal.SIGTERM, handler.handler)
    signal.signal(signal.SIGINT, handler.handler)

    generator, discriminator, combined = _make_trainable_models()
    data = load_data()
    dataset_size = data.shape[0]

    os.makedirs("../../Generated/training/models/", exist_ok=True)
    os.makedirs("../../Generated/training/training_samples/", exist_ok=True)

    epoch = 0
    while not handler.stop:  # Stop on kill signal (Ctrl+C)

        #
        # Discriminator

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        gen_imgs_batch = generator.predict(noise)

        random_indices = np.random.randint(0, dataset_size, BATCH_SIZE)
        true_imgs_batch = data[random_indices, :, :, :]

        true_ground_truth = 1.0 - np.random.rand(BATCH_SIZE) * DISCRIMINATOR_LABELS_NOISE
        fake_ground_truth = np.random.rand(BATCH_SIZE) * DISCRIMINATOR_LABELS_NOISE

        discr_loss_true, discr_acc_true = discriminator.train_on_batch(true_imgs_batch, true_ground_truth)
        discr_loss_fake, discr_acc_fake = discriminator.train_on_batch(gen_imgs_batch, fake_ground_truth)

        #
        # Combined

        true_ground_truth = np.ones(BATCH_SIZE, np.float)
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        comb_loss = combined.train_on_batch(noise, true_ground_truth)

        #
        # Progression
    
        print("(%d) Discriminator : %f / %f / %.2f %% / %.2f %% || Combined : %f"
              % (epoch, discr_loss_true, discr_loss_fake, 100 * discr_acc_true, 100 * discr_acc_fake, comb_loss))

        if epoch % SAMPLE_FREQ == 0:
            _sample_generated_images(generator, epoch)

        if epoch != 0 and epoch % CHECKPOINT_FREQ == 0:
            _save_models(generator, discriminator, "checkpoint-%d" % epoch)

        epoch += 1

    #
    # Saving final models

    _save_models(generator, discriminator, "final")


def _make_trainable_models():

    optimizer_discriminator = Adam(LR_DISCRIMINATOR, 0.9)
    optimizer_combined = Adam(LR_COMBINED, 0.9)

    generator = make_generator()
    discriminator = make_discriminator()

    #
    # Trainable discriminator
    discriminator.compile(loss="binary_crossentropy",
                          optimizer=optimizer_discriminator, metrics=["accuracy"])

    #
    # Trainable stacked generator+discriminator
    gen_input = Input(shape=(LATENT_DIM,))
    gen_img_output = generator(gen_input)

    discriminator.trainable = False
    discr_output = discriminator(gen_img_output)

    combined = Model(gen_input, discr_output)
    combined.compile(loss="binary_crossentropy", optimizer=optimizer_combined)

    return generator, discriminator, combined

def _sample_generated_images(generator, epoch):

    samples_path = Path("../../Generated/training/training_samples/")

    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    img_float = generator.predict(noise)[0]
    img = np.clip((img_float + 1) * 127.5, 0, 255).astype(np.uint8)

    img_path = samples_path / ("%d.png" % epoch)
    cv2.imwrite(str(img_path), img)

def _save_models(generator, discriminator, name):

    saved_folder_path = Path("../../Generated/training/models/%s/" % name)
    os.makedirs(saved_folder_path, exist_ok=True)

    print("saving models to %s" % str(saved_folder_path))
    generator.save(str(saved_folder_path / "generator.h5"))
    discriminator.save(str(saved_folder_path / "discriminator.h5"))

class _SignalHandler:
    def __init__(self):
        self.stop = False
    def handler(self, signum, frame):
        self.stop = True

if __name__ == "__main__":
    main()