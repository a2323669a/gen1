import tensorflow as tf
tf.enable_eager_execution()

import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense, Conv2D,Conv2DTranspose
from typing import Tuple
import numpy as np
import os
import time
import matplotlib.pyplot as plt
class GAN:
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.noise = tf.random.normal(shape=(16, self.input_dim))

        self.prepare_dataset()
        self.build_model()
        self.create_checkpoint()
        self.load()

    def build_model(self):
        self.generator = self.generator_model(self.input_dim)
        self.discriminator = self.discriminator_model()

        self.d_optimizer = tf.train.AdamOptimizer(0.0001)
        self.g_optimizer = tf.train.AdamOptimizer(0.0001)

    def load(self):
        if tf.train.latest_checkpoint(self.checkpoint_dir):
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def prepare_dataset(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train / 225.
        x_test = x_test / 225.
        self.x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32')

        self.batch_size = 32
        self.batch_count = self.x_train.shape[0] // self.batch_size
        self.buffer_size = self.x_train.shape[0]
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(self.buffer_size).batch(
            self.batch_size,
            drop_remainder=True)

    def create_checkpoint(self):
        self.epoch = tf.Variable(0,trainable=False)
        self.checkpoint_dir = './mnist'
        self.checkpoint = tf.train.Checkpoint(g_optimizer=self.g_optimizer,
                                              d_optimizer=self.d_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator,
                                              epoch = self.epoch)

    def generate_and_save_images(self,model, epoch, test_input):
      predictions = model(test_input, training=False)

      fig = plt.figure(figsize=(4,4))

      for i in range(predictions.shape[0]):
          plt.subplot(4, 4, i+1)
          plt.imshow(predictions[i, :, :, 0] * 255., cmap='gray')
          plt.axis('off')

      plt.savefig('./result/mnist/epoch{:04d}.png'.format(epoch))
      plt.show()

    def discriminator_model(self) -> keras.Sequential:
        model = keras.Sequential()

        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', name='d_conv_1', input_shape=(28,28,1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('tanh'))

        model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', name='d_conv_2'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('tanh'))

        model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', name='d_conv_3'))
        assert model.output_shape[1:3] == (4,4)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('tanh'))

        model.add(keras.layers.Flatten())

        model.add(Dense(1, name='discriminator'))

        return model

    def generator_model(self,input_dim) -> keras.Sequential:
        model = keras.Sequential()

        model.add(Dense(4*4*128, input_shape=(input_dim,), name='dense_1'))
        model.add(keras.layers.PReLU())
        assert model.output_shape[1:] == (4*4*128,)

        model.add(keras.layers.Reshape(target_shape=(4,4,128)))

        model.add(keras.layers.Conv2DTranspose(128,(4,4), strides=(2,2),padding='same', output_padding=1, name='g_convt_1'))
        assert model.output_shape[1:] == (7,7,128)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('tanh'))

        model.add(keras.layers.Conv2DTranspose(64,(4,4), strides=(2,2), padding='same', name='g_convt_2'))
        assert model.output_shape[1:] == (14,14,64)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('tanh'))

        model.add(keras.layers.Conv2DTranspose(1,(4,4), strides=(2,2), padding='same', name='generator', activation='tanh'))

        assert model.output_shape[1:] == (28,28,1)

        return model

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def train(self, epochs):
        initial_epoch = int(self.epoch.numpy())

        for epoch in range(initial_epoch, epochs):
            start = time.time()

            for i,img_batch  in enumerate(self.train_dataset):
                noise_batch = tf.random.normal(shape=(self.batch_size, self.input_dim))
                print("{}/{}".format(i, self.batch_count))

                fake_batch = self.generator(noise_batch, training=True)
                for di in range(5):
                    with tf.GradientTape() as d_tape:
                        real = self.discriminator(img_batch, training=True)
                        fake = self.discriminator(fake_batch, training=True)

                        d_loss = self.discriminator_loss(real, fake)

                    # update
                    d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
                    print("d_loss:{}".format(d_loss))

                for gi in range(1):
                    with tf.GradientTape() as g_tape:
                        fake_batch = self.generator(noise_batch, training=True)
                        fake = self.discriminator(fake_batch, training=True)
                        g_loss = self.generator_loss(fake)

                    # update
                    g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
                    self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
                    print("g_loss:{}".format(g_loss))

                print("\n--------------------------------------------------------------------------------")
            self.generate_and_save_images(self.generator,epoch + 1, self.noise)
            self.epoch = epoch + 1

            self.checkpoint.save(file_prefix = os.path.join(self.checkpoint_dir, "{}".format(epoch+1)))

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

if __name__ == '__main__':
    gan = GAN(input_dim=62)
    gan.train(epochs = 20)