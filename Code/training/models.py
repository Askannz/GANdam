from keras.layers import Input, Reshape, Dense, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from shared_constants import IMG_SHAPE, LATENT_DIM


BN_ENABLED = True
BN_MOMENTUM = 0.3

def make_generator():

    input_tensor = Input(shape=(LATENT_DIM,), dtype="float32")
    dense1 = Dense(4096, activation="relu")(input_tensor)
    dense2 = Dense(2 * 2 * 2048, activation="relu")(dense1)
    reshape = Reshape((2, 2, 2048))(dense2)  # 2x2

    conv1 = Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same", activation="relu")(reshape)  # 4x4
    batch1 = BatchNormalization(momentum=BN_MOMENTUM)(conv1)

    conv2 = Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", activation="relu")(batch1)  # 8x8
    batch2 = BatchNormalization(momentum=BN_MOMENTUM)(conv2)

    conv3 = Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", activation="relu")(batch2) # 16x16
    batch3 = BatchNormalization(momentum=BN_MOMENTUM)(conv3)

    conv4 = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu")(batch3) # 32x32
    batch4 = BatchNormalization(momentum=BN_MOMENTUM)(conv4)

    conv5 = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(batch4) # 64x64
    batch5 = BatchNormalization(momentum=BN_MOMENTUM)(conv5)

    conv6 = Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(batch5) # 128x128
    batch6 = BatchNormalization(momentum=BN_MOMENTUM)(conv6)

    output_tensor = Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="tanh")(batch6) # 256x256

    return Model(inputs=input_tensor, outputs=output_tensor)


def make_discriminator():

    h, w = IMG_SHAPE

    input_tensor = Input(shape=(h, w, 3), dtype="float32")

    conv1 = Conv2D(64, kernel_size=3, strides=2, padding="same")(input_tensor)  # 128x128
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    batch_norm1 = BatchNormalization(momentum=BN_MOMENTUM)(activ1)
    dropout1 = Dropout(0.25)(batch_norm1)

    conv2 = Conv2D(128, kernel_size=3, strides=2, padding="same")(dropout1)  # 64x64
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    batch_norm2 = BatchNormalization(momentum=BN_MOMENTUM)(activ2)
    dropout2 = Dropout(0.25)(batch_norm2)

    conv3 = Conv2D(256, kernel_size=3, strides=2, padding="same")(dropout2)  # 32x32
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    batch_norm3 = BatchNormalization(momentum=BN_MOMENTUM)(activ3)
    dropout3 = Dropout(0.25)(batch_norm3)

    conv4 = Conv2D(512, kernel_size=3, strides=2, padding="same")(dropout3)  # 16x16
    activ4 = LeakyReLU(alpha=0.2)(conv4)
    batch_norm4 = BatchNormalization(momentum=BN_MOMENTUM)(activ4)
    dropout4 = Dropout(0.25)(batch_norm4)

    conv5 = Conv2D(1024, kernel_size=3, strides=2, padding="same")(dropout4)  # 8x8
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    dropout5 = Dropout(0.25)(activ5)

    conv6 = Conv2D(2048, kernel_size=3, strides=2, padding="same")(dropout5)  # 4x4
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    dropout6 = Dropout(0.25)(activ6)

    flattened = Flatten()(dropout6)
    dense1 = Dense(1024, activation="relu")(flattened)
    output_tensor = Dense(1, activation="sigmoid")(dense1)

    return Model(inputs=input_tensor, outputs=output_tensor)
