import tensorflow as tf
import numpy as np
from scipy.stats import special_ortho_group
import tensorflow_probability as tfp
tfb = tfp.bijectors

class RandomRotation(tfb.Bijector):
    def __init__(
            self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="random_rotation"):
        super(RandomRotation, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        self.rand_matrix = tf.Variable(
            special_ortho_group.rvs(input_shape),
            trainable=False,
            dtype=tf.float32)

    def _forward(self, x):
        return tf.matmul(x, self.rand_matrix)

    def _inverse(self, y):
        return tf.matmul(y, tf.transpose(self.rand_matrix))

    def _forward_log_det_jacobian(self, x):
        return tf.reduce_sum(x, axis=-1) * 0


class LearnableRotation(tfb.Bijector):
    def __init__(
            self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="learnable_rotation"):
        super(LearnableRotation, self).__init__(
            validate_args=validate_args,
            dtype=tf.float32,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        n = input_shape
        self.matrix_vector = tf.Variable(
            np.random.randn(np.int32(n * (n + 1) / 2)),
            dtype=tf.float32, trainable=True,
            name='rotation_vector')

        self.I = tf.constant(tf.linalg.eye(n))

    def _construct_ortho(self):
        A = tfp.math.fill_triangular(self.matrix_vector)
        W = A - tf.transpose(A)
        return tf.linalg.lstsq((self.I + W), (self.I - W))

    def _forward(self, x, **kwargs):
        return tf.matmul(x, self._construct_ortho())

    def _inverse(self, y, **kwargs):
        return tf.matmul(y, tf.transpose(self._construct_ortho()))

    def _forward_log_det_jacobian(self, x, **kwargs):
        return tf.reduce_sum(x, axis=-1) * 0.