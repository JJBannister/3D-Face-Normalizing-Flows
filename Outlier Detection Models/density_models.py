import os
import pickle as pk
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import roc_auc_score, accuracy_score
tfd = tfp.distributions
tfb = tfp.bijectors

from bijectors import ConditionalTranslation, ConditionalCayleyRotation, ConditionalAffineCoupling, RandomRotation, CayleyRotation, ConditionalScaleDiag

n_epochs = 5
batch_size = 128
lr = 1e-3

training_data_files = ["../Data/FB/Cleaned_1k/"+str(x)+"_train.pkl" for x in range(5)]
testing_data_files = ["../Data/FB/Cleaned_1k/"+str(x)+"_test.pkl" for x in range(5)]
checkpoint_parent_dir = "D:/OOD/Checkpoints/Density/1k/"


def main():
    density_type = "NonGaussian"
    test_auc_results = []
    checkpoint_dirs = [checkpoint_parent_dir + density_type + "/" + str(x) for x in range(5)]

    for train_data_file, test_data_file, checkpoint_dir in zip(training_data_files, testing_data_files, checkpoint_dirs):
        train_data, mean, std = load_data(train_data_file, train=True)
        test_data, _, _ = load_data(test_data_file, train=False)

        model = DensityFlow(train_data, mean, std, checkpoint_dir, n_epochs, type=density_type)

        test_scores = []
        test_ood = []
        for x, age, sex, ood in test_data:
            test_ood.append(ood.numpy()[0])
            test_scores.append(-model.log_prob(tf.expand_dims(x, axis=0),
                                                         tf.expand_dims(age, axis=0),
                                                         tf.expand_dims(sex, axis=0)).numpy()[0])

        test_auc = roc_auc_score(test_ood, test_scores)
        test_auc_results.append(test_auc)
        break


    print("Density: ", density_type)
    print("AUC mean ", np.mean(test_auc_results))
    print("AUC std ", np.std(test_auc_results))



def load_data(data_file, train=True):
    data = pk.load(open(data_file, "rb"))
    data.columns = data.columns.astype(str)

    if train:
        data = data[data["OOD"] == 0] # control only

    points = data.loc[:, '0':].values.astype(np.float32)
    sexes = np.expand_dims(data['Sex'].values.astype(np.float32), axis=1)
    ages = np.expand_dims(data['Age'].values.astype(np.float32), axis=1)
    oods = np.expand_dims(data['OOD'].values.astype(np.int32), axis=1)

    mean = np.mean(points, axis=0)
    std = np.mean(points, axis=0)

    return tf.data.Dataset.from_tensor_slices((points, ages, sexes, oods)), mean, std


class DensityFlow(tf.Module):
    def __init__(self, train_data, mean, std, checkpoint_dir, n_epochs, type="gaussian"):
        self.input_dim = mean.size
        self.checkpoint_dir = checkpoint_dir

        def _init_once(x, name, dtype):
            return tf.Variable(x, name=name, trainable=False, dtype=dtype)

        bijector_chain = []
        bijector_chain.append(tfb.Shift(_init_once(-mean, "MeanSubtract", tf.float32)))
        bijector_chain.append(tfb.ScaleMatvecDiag(_init_once(1./std, "Normalize", tf.float32)))

        if type == "Gaussian":
            bijector_chain.append(ConditionalTranslation(self.input_dim))
            bijector_chain.append(ConditionalCayleyRotation(self.input_dim))
            bijector_chain.append(ConditionalScaleDiag(self.input_dim))

        elif type == "IGaussian":
            bijector_chain.append(ConditionalTranslation(self.input_dim))
            bijector_chain.append(ConditionalScaleDiag(self.input_dim))

        elif type == "NonGaussian":
            bijector_chain.append(ConditionalTranslation(self.input_dim))
            bijector_chain.append(ConditionalScaleDiag(self.input_dim))
            for _ in range(3):
                bijector_chain.append(ConditionalAffineCoupling(self.input_dim))
                bijector_chain.append(
                    tfb.Permute(permutation=_init_once(
                        np.linspace(self.input_dim - 1, 0, self.input_dim).astype('int32'),
                        'permutation', tf.int32)))
            bijector_chain.append(ConditionalAffineCoupling(self.input_dim))

        self.bijector = tfb.Chain(list(reversed(bijector_chain)))
        self.optimizer = tf.optimizers.Nadam(learning_rate=lr, clipnorm=0.5)
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros([self.input_dim]),
            scale_diag=tf.ones([self.input_dim]))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=1)

        # Train
        n_train = train_data.cardinality().numpy()
        train_data = train_data.shuffle(n_train, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
        avg_loss = tf.keras.metrics.Mean(name='log_prob', dtype=tf.float32)
        min_loss = 9999999

        @tf.function
        def train_step(x, age, sex):
            with tf.GradientTape() as tape:
                loss = -self.log_prob(x, age, sex)

            grads = tape.gradient(loss, self.bijector.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.bijector.trainable_variables))
            avg_loss.update_state(loss)

        self.load_checkpoint()
        for epoch in range(n_epochs):
            for x, age, sex, ood in train_data:
                train_step(x, age, sex)

            if avg_loss.result().numpy() < min_loss:
                print("***** Saving checkpoint ***** loss=", avg_loss.result().numpy())

                min_loss = avg_loss.result().numpy()
                self.manager.save()

            print("Epoch {} Avg Loss {:.6f}".format(epoch, avg_loss.result()))
            avg_loss.reset_states()

        # Reset to most recent save after training
        self.load_checkpoint()

    def log_prob(self, x, age, sex):
        z = self.bijector.inverse(x, ConditionalLayer={"age": age, "sex": sex})
        p_z = self.base_dist.log_prob(z)
        log_det_j = self.bijector.forward_log_det_jacobian(
            z, 1, ConditionalLayer={"age": age, "sex": sex})
        return p_z - log_det_j

    def load_checkpoint(self):
        if self.manager.latest_checkpoint:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            # print(tf.train.list_variables(self.manager.latest_checkpoint))
            # print(self.bijector.variables)
            # status.expect_partial()


if __name__ == '__main__':
    main()
