import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import special_ortho_group
from tensorflow.keras.layers import Layer, Dense, Lambda, Input, Concatenate
from tensorflow.keras import Model, regularizers
tfb = tfp.bijectors

class CayleyRotation(tfb.Bijector):
    def __init__(
            self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="CayleyRotation"):
        super(CayleyRotation, self).__init__(
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




class ConditionalCayleyRotation(tfb.Bijector):
    def __init__(self,
            input_shape,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="ConditionalLayer"):
        super().__init__(
            dtype=tf.float32,
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        self.age_scale = 40

        # Inputs
        sex = Input(shape=(1,), dtype=tf.float32)
        age = Input(shape=(1,), dtype=tf.float32)

        # Age and Sex
        x = Concatenate(axis=-1)([age, sex])
        x = Dense(5, activation='elu')(x)
        x = Dense(5, activation='elu')(x)
        age_sex = Dense(5, activation='elu')(x)

        matrix_vector = Dense(np.int32(input_shape * (input_shape + 1) / 2),
                              activation='linear')(age_sex)

        self.matrix_vector = Model([age, sex], matrix_vector, name=self.name + "/rotation_matrix_vec", trainable=True)
        self.I = tf.constant(tf.linalg.eye(input_shape))

    def _construct_ortho(self, age, sex):
        A_vec = self.matrix_vector([age, sex])
        A = tfp.math.fill_triangular(A_vec)
        W = A - tf.transpose(A, perm=[0,2,1])
        return tf.linalg.lstsq((self.I + W), (self.I - W))

    def _forward(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        return tf.matmul(x, self._construct_ortho(age, sex))

    def _inverse(self, y, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        return tf.matmul(y, tf.transpose(self._construct_ortho(age, sex), perm=[0,2,1]))

    def _forward_log_det_jacobian(self, x, **kwargs):
        return tf.constant(0., dtype=x.dtype)


class ConditionalTranslation(tfb.Bijector):
    def __init__(self,
                 input_shape,
                 forward_min_event_ndims=1,
                 validate_args: bool = False,
                 name="ConditionalLayer"):
        super().__init__(
            dtype=tf.float32,
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        self.age_scale = 40

        # Inputs
        sex = Input(shape=(1,), dtype=tf.float32)
        age = Input(shape=(1,), dtype=tf.float32)

        # Age and Sex
        x = Concatenate(axis=-1)([age, sex])
        x = Dense(5, activation='elu')(x)
        x = Dense(5, activation='elu')(x)
        age_sex = Dense(5, activation='elu')(x)

        # Translation
        t = Dense(input_shape, activation='linear',
                  kernel_initializer='zeros',
                  bias_initializer='zeros')(age_sex)
        self.t = Model([age, sex], t, name=self.name + "/t", trainable=True)

    def _forward(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        t = self.t([age, sex])
        return x + t

    def _inverse(self, y, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        t = self.t([age, sex])
        return y - t

    def _forward_log_det_jacobian(self, x, **kwargs):
        return tf.constant(0., dtype=x.dtype)



class ConditionalScaleDiag(tfb.Bijector):
    def __init__(self,
                 input_shape,
                 forward_min_event_ndims=1,
                 validate_args: bool = False,
                 name="ConditionalLayer"):
        super().__init__(
            dtype=tf.float32,
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        self.age_scale = 40

        # Inputs
        sex = Input(shape=(1,), dtype=tf.float32)
        age = Input(shape=(1,), dtype=tf.float32)

        # Age and Sex
        x = Concatenate(axis=-1)([age, sex])
        x = Dense(5, activation='elu')(x)
        x = Dense(5, activation='elu')(x)
        age_sex = Dense(5, activation='elu')(x)

        # Translation
        log_s = Dense(input_shape, activation='linear',
                  kernel_initializer='zeros',
                  bias_initializer='zeros')(age_sex)
        self.log_s = Model([age, sex], log_s, name=self.name + "/log_s", trainable=True)

    def _forward(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        log_s = self.log_s([age, sex])
        return x * tf.exp(log_s)

    def _inverse(self, y, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        log_s = self.log_s([age, sex])
        return y * tf.exp(-log_s)

    def _forward_log_det_jacobian(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']
        log_s = self.log_s([age, sex])
        return tf.reduce_sum(log_s, axis=-1)


class ConditionalAffineCoupling(tfb.Bijector):
    def __init__(self,
                 input_shape,
                 forward_min_event_ndims=1,
                 validate_args: bool = False,
                 name="ConditionalLayer"):
        super().__init__(
            dtype=tf.float32,
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)
        self.age_scale = 40

        self.input_shape = input_shape
        self.a_dim = input_shape // 2
        self.b_dim = input_shape - self.a_dim
        self.hidden_dim = 32
        volume_preserve=True

        # Inputs
        sex = Input(shape=(1,), dtype=tf.float32)
        age = Input(shape=(1,), dtype=tf.float32)
        x_b = Input(shape=(self.a_dim,), dtype=tf.float32)

        # Mixing
        x = Concatenate(axis=-1)([x_b, age, sex])
        x = Dense(self.hidden_dim, activation='elu')(x)
        x = Dense(self.hidden_dim, activation='elu')(x)

        t = Dense(self.a_dim, activation='linear',
                  kernel_initializer='zeros',
                  bias_initializer='zeros')(x)

        self.t = Model([x_b, age, sex], t, name=self.name + "/t", trainable=True)

        log_s = Dense(self.b_dim, activation='tanh',
                  kernel_initializer='zeros',
                  bias_initializer='zeros')(x)

        def mean_subtract(x):
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            return tf.subtract(x, mean)

        log_s_vp = Lambda(mean_subtract)(log_s)

        if volume_preserve:
            self.log_s = Model([x_b, age, sex], log_s_vp, name="/log_s", trainable=True)
        else:
            self.log_s = Model([x_b, age, sex], log_s, name="/log_s", trainable=True)


    def _forward(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']

        x_a, x_b = (x[:,:self.a_dim], x[:,self.a_dim:])
        y_b = x_b

        log_s = self.log_s([x_b, age, sex])
        s = tf.exp(log_s)
        t = self.t([x_b, age, sex])

        y_a = s * x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']

        y_a, y_b = (y[:,:self.a_dim], y[:,self.a_dim:])
        x_b = y_b

        log_s = self.log_s([y_b, age, sex])
        s = tf.exp(-log_s)
        t = self.t([y_b, age, sex])

        x_a = (y_a - t) * s
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']

        x_a, x_b = (x[:,:self.a_dim], x[:,self.a_dim:])
        log_s = self.log_s([x_b, age, sex])

        return tf.reduce_sum(log_s, axis=-1)
