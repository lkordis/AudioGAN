from __future__ import print_function, division

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam

from model.DeConv1D import Conv1DTranspose
from model.dataset import DataSet
from model.io import load_h5
from segan_test import segan_hr


class WGAN():
    def __init__(self, upsample_scale = 2):
        self.audio_shape = 8192
        self.shape_hr = (8192, 1)
        self.shape_lr = (8192, 1)
        self.lr = 1e-4
        self.gen_lr = 1e-4

        # Following parameter and optimizer set as recommended in paper
        self.n_discriminator = 5
        self.clip_value = 0.01
        self.gan_loss = self.wasserstein_loss
        self.dis_loss = self.wasserstein_loss

        optimizer = Adam(lr=0.0001)

        # Now we initialize the generator and discriminator.
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.compile_generator(self.generator)
        self.compile_discriminator(self.discriminator)

        self.discriminator.trainable = False
        # The generator takes noise as input and generated imgs
        audio = Input(shape=self.shape_lr)
        gen_audio = self.generator(audio)

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(gen_audio)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(audio, valid)
        self.compile_combined(self.combined)
        self.discriminator.trainable = True


    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=['mse', self.PSNR]
        )

    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.dis_loss,
            optimizer=Adam(self.lr, 0.9),
            metrics=['accuracy']
        )

    def compile_combined(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=self.wasserstein_loss,
            optimizer=Adam(self.lr, 0.9)
        )

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def PSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    def SubpixelConv1D(self, scale=2):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space

        :param scale: upsampling scale compared to input_shape. Default=2
        :return:
        """

        def subpixel(I):
            X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
            X = tf.batch_to_space_nd(X, [scale], [[0, 0]])  # (1, r*w, b)
            X = tf.transpose(X, [2, 1, 0])
            return X

        def subpixel_shape(input_shape):
            # (None, a * r, 1)
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    1]
            output_shape = tuple(dims)
            return output_shape

        return Lambda(subpixel, output_shape=subpixel_shape)

    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}_generator.h5".format(filepath))
        self.discriminator.save_weights("{}_discriminator.h5".format(filepath))

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)

    def save_model(self, filepath):
        self.generator.save("{}_generator_model.h5".format(filepath))
        self.discriminator.save("{}_discriminator_model.h5".format(filepath))

    def load_model(self, generator=None, discriminator=None, **kwargs):
        if generator:
            self.generator = load_model(generator, **kwargs)
        if discriminator:
            self.discriminator = load_model(discriminator, **kwargs)

    def build_generator(self):
        lr_audio = Input(shape=self.shape_lr)
        x = Conv1D(16, 3, strides=1, padding="same")(lr_audio)

        #n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        #n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
        n_filters = [128, 256]
        n_filtersizes = [65, 33]

        for nf, fs in zip(n_filters, n_filtersizes):
            x = Conv1D(nf, fs, strides=1, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = Conv1D(n_filters[-1], 9)(x)

        for nf, fs in zip(reversed(n_filters), reversed(n_filtersizes)):
            x = UpSampling1D(2)(x)
            x = Conv1D(nf, fs, strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.5)(x)

        x = Activation("tanh")(x)
        hr_output = Conv1DTranspose(1,9)(x)

        model = Model(inputs=lr_audio, outputs=hr_output)
        model.summary()

        return model


    def build_discriminator(self):
        filters = 64

        def conv1d_block(input, filters, strides=1, bn=True):
            d = Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        audio = Input(shape=self.shape_hr)
        x = conv1d_block(audio, filters, bn=False)
        x = conv1d_block(x, filters, strides=2)
        x = conv1d_block(x, filters * 2)
        x = conv1d_block(x, filters * 2, strides=2)
        x = conv1d_block(x, filters * 4, strides=2)
        x = conv1d_block(x, filters * 8, strides=2)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create model and compile
        model = Model(inputs=audio, outputs=x)
        model.summary()

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train_, Y_train_ = load_h5('./data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5')
        X_val_, Y_val_ = load_h5('./data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5')

        train_data = DataSet(X_train_, Y_train_)
        val_data = DataSet(X_val_, Y_val_)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            audios_lr, audios_hr = train_data.next_batch(batch_size)
            audios_lr = segan_hr(audios_lr)
            # print("Batch size: ", audios_hr.shape, audios_lr.shape)

            for _ in range(self.n_discriminator):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                # idx = np.random.randint(0, X_train.shape[0], batch_size)

                # Generate a batch of new images
                gen_audios = self.generator.predict(audios_lr)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(audios_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_audios, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(audios_lr, valid)
            self.discriminator.trainable = True
            # Plot the progress
            print("[{} D loss:{}] [G loss: {}]".format(epoch, d_loss, g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_weights("./")
                print(epoch)


if __name__ == '__main__':
    wgan = WGAN()
    # wgan.load_weights("./4_blocks_generator.h5", "./4_blocks_discriminator.h5")
    wgan.train(epochs=400, batch_size=128, sample_interval=10)
    wgan.save_weights("./")
