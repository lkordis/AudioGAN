#! /usr/bin/python
import os
import sys
import pickle
import datetime

import numpy as np

# Import keras + tensorflow without the "Using XX Backend" message
from model.dataset import DataSet
from model.io import load_h5

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Add, Conv2DTranspose
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv1D, Dense
from keras.layers import UpSampling2D, Lambda
from keras.optimizers import Adam
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

sys.stderr = stderr

class SRGAN():
    """
    Implementation of SRGAN as described in the paper:
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802
    """

    def __init__(self,
                 height_lr=24, width_lr=24, channels=1,
                 upscaling_factor=4,
                 gen_lr=1e-4, dis_lr=1e-4,
                 # VGG scaled with 1/12.75 as in paper
                 loss_weights=[1e-3, 0.006],
                 training_mode=True
                 ):
        """
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """
        # Following parameter and optimizer set as recommended in paper
        self.n_discriminator = 5
        self.clip_value = 0.01

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError('Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.audio_shape = (8192, 1)

        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        # Scaling of losses
        self.loss_weights = loss_weights

        # Gan setup settings
        self.gan_loss = self.wasserstein_loss
        self.dis_loss = self.wasserstein_loss

        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        # If training, build rest of GAN network
        if training_mode:
            self.discriminator = self.build_discriminator()
            self.compile_discriminator(self.discriminator)
            self.srgan = self.build_srgan()
            self.compile_srgan(self.srgan)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}_generator_{}X.h5".format(filepath, self.upscaling_factor))
        self.discriminator.save_weights("{}_discriminator_{}X.h5".format(filepath, self.upscaling_factor))

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)

    def Conv1DTranspose(self, input_tensor, filters = 64, kernel_size = 3, strides=2, padding='same'):
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    def SubpixelConv1D(self, name, scale=2):
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
            #(None, a * r, 1)
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    1]
            output_shape = tuple(dims)
            return output_shape

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)

    def build_generator(self, residual_blocks=16):
        """
        Build the generator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """

        n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

        # for nf, fs in zip(n_filters, n_filtersizes):
        #     model.add(Conv1D(filters=nf, kernel_size=fs,
        #                        activation=None))
        #     model.add(LeakyReLU(alpha=0.2))
        #
        # model.add(Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1],
        #                    activation=None))
        # model.add(Dropout(0.5))
        # # x = BatchNormalization(mode=2)(x)
        # model.add(LeakyReLU(alpha=0.2))
        #
        # # upsampling layers
        # for nf, fs in zip(reversed(n_filters), reversed(n_filtersizes)):
        #     model.add(Conv1D(filters=2 * nf, kernel_size=fs, activation=None))
        #     model.add(Dropout(0.5))
        #     model.add(LeakyReLU(alpha=0.2))
        #     #print(model.layers[-1].get_output_at(0))
        #     #model.add(SubPixel(model.layers[-1], r=2))

        def residual_block(input, filter=64, kernel=3):
            x = Conv1D(filter, kernel_size=kernel, strides=1, padding='same')(input)
            x = BatchNormalization(momentum=0.8)(x)
            x = PReLU(shared_axes=[1, 2])(x)
            return x

        def upsample(x, filter, kernel):
            x = Conv1D(filter, kernel_size=kernel, strides=1, padding='same')(x)
            # x = self.SubpixelConv1D('upSampleSubPixel', 2)(x)
            # x = self.Conv1DTranspose(x)
            x = PReLU(shared_axes=[1, 2])(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=self.audio_shape)

        # Residual blocks
        r = residual_block(lr_input)
        # for _ in range(residual_blocks - 1):
        #     r = residual_block(r)
        for nf, fs in zip(n_filters, n_filtersizes):
            r = residual_block(r, nf, fs)

        # Bottleneck layer
        x = Conv1D(n_filters[-1], kernel_size=n_filtersizes[-1], strides=1, padding='same', input_shape=self.audio_shape)(r)
        x = PReLU(shared_axes=[1, 2])(x)

        for nf, fs in zip(reversed(n_filters), reversed(n_filtersizes)):
            x = upsample(x, nf, fs)

        # Generate high resolution output
        # tanh activation, see:
        # https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
        hr_output = Conv1D(
            self.channels,
            kernel_size=9,
            strides=1,
            padding='same',
            activation='tanh'
        )(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        model.summary()

        return model

    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """
        print("Here")
        def Conv1D_block(input, filters, strides=1, bn=True):
            d = Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.audio_shape)
        x = Conv1D_block(img, filters, bn=False)
        x = Conv1D_block(x, filters, strides=2)
        x = Conv1D_block(x, filters * 2)
        x = Conv1D_block(x, filters * 2, strides=2)
        x = Conv1D_block(x, filters * 4)
        x = Conv1D_block(x, filters * 4, strides=2)
        x = Conv1D_block(x, filters * 8)
        x = Conv1D_block(x, filters * 8, strides=2)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x)
        model.summary()
        return model

    def build_srgan(self):
        """Create the combined SRGAN network"""

        # Input LR images
        img_lr = Input(self.audio_shape)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)


        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator HR images are OK
        generated_check = self.discriminator(generated_hr)

        # Create sensible names for outputs in logs
        generated_features = Lambda(lambda x: x, name='Content')(generated_hr)
        generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)

        # Create model and compile
        # Using binary_crossentropy with reversed label, to get proper loss, see:
        # https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
        model = Model(inputs=img_lr, outputs=[generated_check, generated_hr])
        return model

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
            optimizer=Adam(self.dis_lr, 0.9),
            metrics=['accuracy']
        )

    def compile_srgan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=[self.dis_loss, self.gan_loss],
            loss_weights=self.loss_weights,
            optimizer=Adam(self.gen_lr, 0.9)
        )

    def PSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


    def train_srgan(self,
                    epochs, batch_size,
                    dataname,
                    datapath_validation=None,
                    steps_per_validation=1000,
                    workers=4, max_queue_size=10,
                    first_epoch=0,
                    print_frequency=1,
                    crops_per_image=2,
                    log_weight_frequency=None,
                    log_weight_path='./data/weights/',
                    log_tensorboard_path='./data/logs/',
                    log_tensorboard_name='AudioGAN',
                    log_tensorboard_update_freq=10000,
                    log_test_frequency=500,
                    log_test_path="./images/samples/",
                    ):
        """Train the SRGAN network
        :param int epochs: how many epochs to train the network for
        :param str dataname: name to use for storing model weights etc.
        :param int print_frequency: how often (in epochs) to print progress to terminal. Warning: will run validation inference!
        :param int log_weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int log_weight_path: where should network weights be saved
        :param int log_test_frequency: how often (in epochs) should testing & validation be performed
        :param str log_test_path: where should test results be saved
        :param str log_tensorboard_path: where should tensorflow logs be sent
        :param str log_tensorboard_name: what folder should tf logs be saved under
        """
        # Load the dataset
        X_train_, Y_train_ = load_h5('./data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5')
        X_val_, Y_val_ = load_h5('./data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5')

        # Create train data loader
        loader = DataSet(X_train_, Y_train_)

        # Validation data loader
        if datapath_validation is not None:
            validation_loader = DataSet(X_val_, Y_val_)

        # Callback: tensorboard
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, log_tensorboard_name),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=False,
                write_grads=False,
                update_freq=log_tensorboard_update_freq
            )
            tensorboard.set_model(self.srgan)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: format input value
        def named_logs(model, logs):
            """Transform train_on_batch return value to dict expected by on_batch_end callback"""
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        # Adversarial ground truths
        valid = -np.ones((batch_size,512, 1))
        fake = np.ones((batch_size, 512, 1))

        # Each epoch == "update iteration" as defined in the paper
        print_losses = {"G": [], "D": []}
        start_epoch = datetime.datetime.now()

        # Loop through epochs / iterations
        for epoch in range(first_epoch, int(epochs) + first_epoch):

            # Start epoch time
            if epoch % (print_frequency + 1) == 0:
                start_epoch = datetime.datetime.now()

                # Train discriminator
            audios_lr, audios_hr = loader.next_batch(batch_size)
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

            g_loss = self.srgan.train_on_batch(audios_lr, valid)

            # Callbacks
            logs = named_logs(self.srgan, g_loss)
            tensorboard.on_epoch_end(epoch, logs)

            # Save losses
            print_losses['G'].append(g_loss)
            print_losses['D'].append(d_loss)

            # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['G']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                print("\nEpoch {}/{} | Time: {}s\n>> Generator/GAN: {}\n>> Discriminator: {}".format(
                    epoch, epochs + first_epoch,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.srgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.discriminator.metrics_names, d_avg_loss)])
                ))
                print_losses = {"G": [], "D": []}

                # Run validation inference if specified
                if datapath_validation:
                    validation_losses = self.generator.evaluate_generator(
                        validation_loader,
                        steps=steps_per_validation,
                        use_multiprocessing=workers > 1,
                        workers=workers
                    )
                    print(">> Validation Losses: {}".format(
                        ", ".join(
                            ["{}={:.4f}".format(k, v) for k, v in zip(self.generator.metrics_names, validation_losses)])
                    ))

                    # If test images are supplied, run model on them and save to log_test_path
            # if datapath_test and epoch % log_test_frequency == 0:
            #     plot_test_images(self, loader, datapath_test, log_test_path, epoch)

            # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:
                # Save the network weights
                self.save_weights(os.path.join(log_weight_path, dataname))


# Run the SRGAN network
if __name__ == '__main__':
    # Instantiate the SRGAN object
    print(">> Creating the SRGAN network")
    gan = SRGAN(gen_lr=1e-5)

    # Load previous imagenet weights
    #print(">> Loading old weights")
    #gan.load_weights('../data/weights/imagenet_generator.h5', '../data/weights/imagenet_discriminator.h5')

    # Train the SRGAN
    gan.train_srgan(500, 32, dataname="AudioGAN")