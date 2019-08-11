from keras.layers import Input, Reshape, Dense, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from shared_constants import IMG_SHAPE, LATENT_DIM


BN_ENABLED = True
BN_MOMENTUM = 0.3

def make_generator():

    input_tensor = Input(shape=(LATENT_DIM,), dtype="float32")
    dense = Dense(4 * 4 * 2048, activation="relu")(input_tensor)
    reshape = Reshape((4, 4, 2048))(dense)  # 4x4

    conv2048 = Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same", activation="relu")(reshape)  # 8x8
    batch_norm2048 = BatchNormalization(momentum=BN_MOMENTUM)(conv2048)

    conv1024 = Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same", activation="relu")(batch_norm2048)  # 16x16
    batch_norm1024 = BatchNormalization(momentum=BN_MOMENTUM)(conv1024)

    conv512 = Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", activation="relu")(batch_norm1024) # 32x32
    batch_norm512 = BatchNormalization(momentum=BN_MOMENTUM)(conv512)

    conv256 = Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", activation="relu")(batch_norm512) # 64x64
    batch_norm256 = BatchNormalization(momentum=BN_MOMENTUM)(conv256)

    output_tensor = Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="tanh")(batch_norm256) # 128x128

    return Model(inputs=input_tensor, outputs=output_tensor)


def make_discriminator():

    h, w = IMG_SHAPE

    input_tensor = Input(shape=(h, w, 3), dtype="float32")

    conv1 = Conv2D(256, kernel_size=3, strides=2, padding="same")(input_tensor)  # 64x64
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    batch_norm1 = BatchNormalization(momentum=BN_MOMENTUM)(activ1)
    dropout1 = Dropout(0.25)(batch_norm1)

    conv2 = Conv2D(512, kernel_size=3, strides=2, padding="same")(dropout1)  # 32x32
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    batch_norm2 = BatchNormalization(momentum=BN_MOMENTUM)(activ2)
    dropout2 = Dropout(0.25)(batch_norm2)

    conv3 = Conv2D(512, kernel_size=3, strides=2, padding="same")(dropout2)  # 16x16
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    batch_norm3 = BatchNormalization(momentum=BN_MOMENTUM)(activ3)
    dropout3 = Dropout(0.25)(batch_norm3)

    conv4 = Conv2D(1024, kernel_size=3, strides=2, padding="same")(dropout3)  # 8x8
    activ4 = LeakyReLU(alpha=0.2)(conv4)
    dropout4 = Dropout(0.25)(activ4)

    flattened = Flatten()(dropout4)
    output_tensor = Dense(1, activation="sigmoid")(flattened)

    return Model(inputs=input_tensor, outputs=output_tensor)
