import os
import numpy as np

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from face_model import FaceModel

n_epochs = 1500
batch_size = 2056

full_data_file = "../Data/Cleaned/Full.csv"
full_checkpoint_dir = "../Data/Results/Checkpoints/PC100/Full/"

training_data_files = ["../Data/Cleaned/"+str(x)+"_train.csv" for x in range(10)]
training_checkpoint_dirs = ["../Data/Results/Checkpoints/Gauss/"+str(x) for x in range(10)]

def main():
    #train(full_data_file, full_checkpoint_dir)
    #return
    for data_file, checkpoint_dir in zip(training_data_files, training_checkpoint_dirs):
        train(data_file, checkpoint_dir)

def load_training_data(data_file):
    data = pd.read_csv(data_file)

    pcs = data.loc[:, "PC 0":"PC 99"].values.astype(np.float32)
    scale_vector = np.std(pcs, axis=0)
    sexes = np.expand_dims(data['Sex'].values.astype(np.float32), axis=1)
    ages = np.expand_dims(data['Age'].values.astype(np.float32), axis=1)
    synds = np.expand_dims(data['Syndrome'].values.astype(np.int32), axis=1)

    data = tf.data.Dataset.from_tensor_slices((pcs, ages, sexes, synds))
    data = data.shuffle(ages.shape[0], reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

    return data, scale_vector


def train(data_file, checkpoint_dir):
    data, scale_vector = load_training_data(data_file)
    model = FaceModel(scale_vector, checkpoint_dir)
    avg_loss = tf.keras.metrics.Mean(name='neg_log_prob', dtype=tf.float32)
    model.manager.save()

    @tf.function
    def train_step(pc, age, sex, synd):
        with tf.GradientTape() as tape:
            neg_log_prob = -model.log_prob(pc, age, sex, synd)

        grads = tape.gradient(neg_log_prob, model.bijector.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.bijector.trainable_variables))
        avg_loss.update_state(neg_log_prob)

    for epoch in range(n_epochs):
        for pc, age, sex, synd in data:
            train_step(pc, age, sex, synd)
            if tf.equal(model.optimizer.iterations % 100, 0):
                print( "Step {} Loss {:.6f}".format(
                        model.optimizer.iterations.numpy(), avg_loss.result()))
                avg_loss.reset_states()

        if epoch % 100 == 0:
            print("EPOCH: ", epoch)
            model.manager.save()

main()
