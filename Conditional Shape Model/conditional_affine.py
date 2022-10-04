import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
import tensorflow_probability as tfp
tfb = tfp.bijectors


class ConditionalAffine(tfb.Bijector):
    def __init__(self,
            input_shape,
            n_syndromes,
            forward_min_event_ndims=1,
            validate_args: bool = False,
            name="ConditionalAffine"):
        super().__init__(
            dtype=tf.float32,
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        self.n_syndromes = n_syndromes
        self.age_scale = 60

        # Inputs
        sex = Input(shape=(1,), dtype=tf.float32)
        age = Input(shape=(1,), dtype=tf.float32)
        synd = Input(shape=(n_syndromes,), dtype=tf.float32)

        # Age and Sex
        x = Concatenate(axis=-1)([age, sex])
        x = Dense(10, activation='elu')(x)
        x = Dense(10, activation='elu')(x)
        age_sex = Dense(10, activation='elu')(x)

        # Age Sex and Syndrome
        x = Concatenate(axis=-1)([age_sex, synd])
        x = Dense(100, activation='elu')(x)
        age_sex_synd = Dense(100, activation='elu')(x)

        # Translate Network
        t = Dense(input_shape,
                  activation='linear',
                  kernel_initializer='zeros',
                  bias_initializer='zeros'
                  )(age_sex_synd)

        self.t = Model([age, sex, synd], t, name=self.name + "/t", trainable=True)

        # Scale Network
        log_s = Dense(input_shape,
                      activation='linear',
                      kernel_initializer='zeros',
                      bias_initializer='zeros'
                      )(age_sex)

        self.log_s = Model([age, sex, synd], log_s, name=self.name + "/log_s", trainable=True)


    def _forward(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']

        syndrome = kwargs['syndrome']
        syndrome_t = tf.transpose(tf.one_hot(syndrome, self.n_syndromes, axis=0))
        syndrome_t = tf.squeeze(syndrome_t, axis=0)

        t = self.t([age, sex, syndrome_t])
        log_s = self.log_s([age, sex, syndrome_t])

        s = tf.exp(log_s)
        return x*s+t


    def _inverse(self, y, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']

        syndrome = kwargs['syndrome']
        syndrome_t = tf.transpose(tf.one_hot(syndrome, self.n_syndromes, axis=0))
        syndrome_t = tf.squeeze(syndrome_t, axis=0)

        t = self.t([age, sex, syndrome_t])
        log_s = self.log_s([age, sex, syndrome_t])

        s = tf.exp(log_s)
        return (y-t)/s

    def _forward_log_det_jacobian(self, x, **kwargs):
        age = tf.math.divide(kwargs['age'], self.age_scale)
        sex = kwargs['sex']

        syndrome = kwargs['syndrome']
        syndrome_t = tf.transpose(tf.one_hot(syndrome, self.n_syndromes, axis=0))
        syndrome_t = tf.squeeze(syndrome_t, axis=0)

        log_s = self.log_s([age, sex, syndrome_t])
        return tf.reduce_sum(log_s, axis=-1)


