def train(self, epochs):
    initial_epoch = int(self.epoch.numpy())

    for epoch in range(initial_epoch, epochs):
        start = time.time()

        for i, img_batch in enumerate(self.train_dataset):
            noise_batch = tf.random.normal(shape=(self.batch_size, self.input_dim))
            with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                fake_batch = self.generator(noise_batch, training=True)

                real = self.discriminator(img_batch, training=True)
                fake = self.discriminator(fake_batch, training=True)

                d_loss = self.discriminator_loss(real, fake)
                g_loss = self.generator_loss(fake)

            # update
            d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)

            self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
            self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

            print("{}/{} d_loss:{}, g_loss:{}".format(i, self.batch_count, d_loss, g_loss))

        self.generate_and_save_images(self.generator, epoch + 1, self.noise)
        self.epoch = epoch + 1

        self.checkpoint.save(file_prefix=os.path.join(self.checkpoint_dir, "{}".format(epoch + 1)))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))