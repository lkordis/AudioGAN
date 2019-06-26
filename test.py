from model.io import upsample_wav
from wgan import WGAN
from keras.utils import plot_model
from keras.models import load_model


if __name__ == '__main__':
    model = WGAN()
    model.load_weights("./4_blocks_generator.h5", "./4_blocks_discriminator.h5")
    model.save_model("4_blocks")
