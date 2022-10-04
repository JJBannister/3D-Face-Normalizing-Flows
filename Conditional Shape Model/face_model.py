import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from affine_coupling import AffineCoupling
from conditional_affine import ConditionalAffine
from rotation import RandomRotation, LearnableRotation

input_shape = 100
n_syndromes = 48
n_coupling_blocks = 3
lr = 1e-3

class FaceModel(tf.Module):
    def __init__(self, scale_vector, checkpoint_dir):
        bijector_chain = []

        bijector_chain.append(
            ConditionalAffine(input_shape, n_syndromes))

        bijector_chain.append(
            LearnableRotation(input_shape))

        def _init_once(x, name, dtype):
            return tf.Variable(x, name=name, trainable=False, dtype=dtype)

        for i in range(n_coupling_blocks):
            bijector_chain.append(AffineCoupling(input_shape, name="AC1_"+str(i)))

            bijector_chain.append(
                tfb.Permute(permutation=_init_once(
                    np.linspace(input_shape-1, 0, input_shape).astype('int32'),
                    'permutation', tf.int32)))

            bijector_chain.append(AffineCoupling(input_shape, name="AC2_"+str(i)))
            bijector_chain.append(RandomRotation(input_shape))

        bijector_chain.append(tfb.ScaleMatvecDiag(_init_once(scale_vector, "scale_vector", tf.float32)))

        self.bijector = tfb.Chain(list(reversed(bijector_chain)))

        self.base_dist = tfd.MultivariateNormalDiag(
                        loc=tf.zeros([input_shape]),
                        scale_diag = tf.ones([input_shape]))

        self.optimizer = tf.optimizers.Nadam(learning_rate=lr)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=3)

        if self.manager.latest_checkpoint:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            #print(tf.train.list_variables(self.manager.latest_checkpoint))
            #print(self.bijector.variables)
            #status.expect_partial()

    def log_prob(self, pc, age, sex, synd):
        z = self.bijector.inverse(pc, ConditionalAffine={"age": age, "sex": sex, "syndrome": synd})
        p_z = self.base_dist.log_prob(z)
        log_det_j = self.bijector.forward_log_det_jacobian(
            z, 1, ConditionalAffine={"age": age, "sex": sex, "syndrome": synd})
        return p_z - log_det_j

