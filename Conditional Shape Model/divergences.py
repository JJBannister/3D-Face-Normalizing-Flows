import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from face_model import FaceModel, input_shape, n_syndromes

full_data_file = "../Data/Cleaned/Full.csv"
full_checkpoint_dir = "../Data/Results/Checkpoints/Full/"
synd_codes = pd.read_csv("../Data/Cleaned/synd_codes.csv", index_col=0, squeeze=True, header=None).to_dict()
sex_codes = pd.read_csv("../Data/Cleaned/sex_codes.csv", index_col=0, squeeze=True, header=None).to_dict()



def main():
    model = FaceModel(np.zeros(input_shape), full_checkpoint_dir)
    #d_kl_synd_figure(model)
    #d_kl_age_figure(model)
    #d_kl_sex_figure(model)
    entropy_age_sex_figure(model)
    #total_entropy(model)

def total_entropy(model):
    n_batches = 100
    batch_size = 10000

    def rand_age():
        return tf.convert_to_tensor(
            np.expand_dims(
                np.random.uniform(1, 80, batch_size),
                axis=1),
            dtype=tf.float32)

    def rand_sex():
        return tf.convert_to_tensor(
            np.expand_dims(
                np.random.binomial(1, 0.5, batch_size),
                axis=1),
            dtype=tf.float32)

    def rand_synd():
        return tf.convert_to_tensor(
            np.full((batch_size, 1),
                    np.random.randint(0, n_syndromes, (batch_size, 1))))

    def conditions(age_match=True, sex_match=False, synd_match=False):
        age_1 = rand_age()
        sex_1 = rand_sex()
        synd_1 = rand_synd()

        if age_match:
            age_2 = age_1
        else:
            age_2 = rand_age()

        if sex_match:
            sex_2 = sex_1
        else:
            sex_2 = rand_sex()

        if synd_match:
            synd_2 = synd_1
        else:
            synd_2 = rand_synd()

        y1 = {"age": age_1, "sex": sex_1, "syndrome": synd_1}
        y2 = {"age": age_2, "sex": sex_2, "syndrome": synd_2}

        return y1, y2

    h = 0
    for _ in range(n_batches):
        y1, y2 = conditions()

        # sample with y1
        z = model.base_dist.sample(batch_size)
        x = model.bijector.forward(z, ConditionalAffine=y1)

        # prob with y2
        z2 = model.bijector.inverse(x, ConditionalAffine=y2)
        log_p = model.base_dist.log_prob(z2) - \
                model.bijector.forward_log_det_jacobian(z2, 1, ConditionalAffine=y2)
        h = h + tf.reduce_mean(log_p)

    entropy =  -float(h) / float(n_batches)
    print("Total Entropy: ", entropy)


def entropy_age_sex_figure(model):
    synd = 45
    m=1
    f=0
    ages = list(range(80))
    h_male = np.zeros(len(ages))
    h_female = np.zeros(len(ages))

    for i in range(len(ages)):
        h_male[i] = entropy(model, ages[i], m, synd)
        h_female[i] = entropy(model, ages[i], f, synd)

    plt.figure(figsize=(20,8))
    plt.plot(ages, h_male, 'r', label='Male')
    plt.plot(ages, h_female, 'b', label='Female')
    plt.xlabel("Age (years)", fontsize=45)
    plt.ylabel("Differential Entropy", fontsize=45)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=40)

    plt.tight_layout()
    plt.savefig("../Data/Results/Entropy/entropy.pdf", bbox_inches='tight')


def d_kl_age_figure(model):
    synd = 45
    m=1
    f=0
    ages = list(range(80))
    age_dkl_male = np.zeros(len(ages)-1)
    age_dkl_female = np.zeros(len(ages)-1)

    for i in range(len(ages)-1):
        print("age: ", i)
        age_dkl_male[i] = dkl_age(model, ages[i], ages[i+1], m, synd)
        age_dkl_female[i] = dkl_age(model, ages[i], ages[i+1], f, synd)

    plt.figure(figsize=(20,8))
    plt.plot(ages[:-1], age_dkl_male, 'r', label='Male')
    plt.plot(ages[:-1], age_dkl_female, 'b', label='Female')
    plt.xlabel("Age (years)", fontsize=45)
    plt.ylabel("KL Divergence", fontsize=45)
    plt.legend(fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    #plt.show()
    plt.tight_layout()
    plt.savefig("../Data/Results/Entropy/dkl_age.pdf", bbox_inches='tight')


def d_kl_sex_figure(model):
    synd = 45
    m=1
    f=0

    ages = list(range(80))
    sex_dkl_mf = np.zeros(len(ages))
    sex_dkl_fm = np.zeros(len(ages))

    for i in range(len(ages)):
        print("sex: ", i)
        sex_dkl_mf[i] = dkl_sex(model, ages[i], m, f, synd)
        sex_dkl_fm[i] = dkl_sex(model, ages[i], f, m, synd)

    sex_dkl = (sex_dkl_fm + sex_dkl_mf)/2.0

    plt.figure(figsize=(20,8))
    plt.plot(ages, sex_dkl_mf, 'r',  label="$D_{KL}(Male || Female)$")
    plt.plot(ages, sex_dkl_fm, 'b', label="$D_{KL}(Female || Male)$")
    plt.xlabel("Age (years)", fontsize=45)
    plt.ylabel("KL Divergence", fontsize=45)
    plt.legend(fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.tight_layout()
    plt.savefig("../Data/Results/Entropy/dkl_sex.pdf", bbox_inches='tight')


def d_kl_synd_figure(model):
    kl_matrix = kl_synd_matrix(model)
    kl_df = pd.DataFrame(kl_matrix,
                           index=list(synd_codes.values()),
                           columns=list(synd_codes.values()))

    sns.heatmap(kl_df, xticklabels=True, yticklabels=True, annot=False, linewidths=0.01, linecolor='snow',
                vmin=0, vmax=70)

    plt.ylabel("$Synd_i$", fontsize=30)
    plt.xlabel("$Synd_j$", fontsize=30)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplots_adjust(bottom=0.3)
    plt.subplots_adjust(left=0.2)
    plt.show()


def kl_synd_matrix(model):
    x = np.zeros(shape=(n_syndromes, n_syndromes))
    #return x
    for i in list(range(n_syndromes)):
        print("synd: ", i)
        for j in list(range(n_syndromes)):
            x[i,j] = dkl_synd(model, i, j)
    return x


def dkl_synd(model, synd_i, synd_j):
    n_batches = 5
    batch_size = 10000

    dkl = 0
    for _ in range(n_batches):
        # conditions
        age_t = tf.convert_to_tensor(
            np.expand_dims(
                np.random.uniform(1,80, batch_size),
                axis=1),
            dtype=tf.float32)

        sex_t = tf.convert_to_tensor(
            np.expand_dims(
                np.random.binomial(1, 0.5, batch_size),
                axis=1),
            dtype=tf.float32)

        synd_i_t = tf.convert_to_tensor(np.full((batch_size, 1), synd_i), dtype=tf.int32)
        synd_j_t = tf.convert_to_tensor(np.full((batch_size, 1), synd_j), dtype=tf.int32)

        y_i = {"age": age_t, "sex": sex_t, "syndrome": synd_i_t}
        y_j = {"age": age_t, "sex": sex_t, "syndrome": synd_j_t}

        #dkl
        z = model.base_dist.sample(batch_size)
        x = model.bijector.forward(z, ConditionalAffine=y_i)
        z_j = model.bijector.inverse(x, ConditionalAffine=y_j)

        log_p_i = model.base_dist.log_prob(z) - \
                  model.bijector.forward_log_det_jacobian(z, 1, ConditionalAffine=y_i)

        log_p_j = model.base_dist.log_prob(z_j) - \
                  model.bijector.forward_log_det_jacobian(z_j, 1, ConditionalAffine=y_j)

        dkl = dkl + tf.reduce_mean(log_p_i - log_p_j)

    return float(dkl) / float(n_batches)


def dkl_age(model, age_i, age_j, sex_ij, synd_ij):
    n_batches = 5
    batch_size = 10000

    dkl = 0
    for _ in range(n_batches):
        # conditions
        age_i_t = tf.convert_to_tensor(
            np.full((batch_size, 1), age_i),
            dtype=tf.float32)

        age_j_t = tf.convert_to_tensor(
            np.full((batch_size, 1), age_j),
            dtype=tf.float32)

        sex_t = tf.convert_to_tensor(
            np.full((batch_size, 1), sex_ij),
            dtype=tf.float32)

        synd_t = tf.convert_to_tensor(np.full((batch_size, 1), synd_ij), dtype=tf.int32)

        y_i = {"age": age_i_t, "sex": sex_t, "syndrome": synd_t}
        y_j = {"age": age_j_t, "sex": sex_t, "syndrome": synd_t}

        #dkl
        z = model.base_dist.sample(batch_size)
        x = model.bijector.forward(z, ConditionalAffine=y_i)
        z_j = model.bijector.inverse(x, ConditionalAffine=y_j)

        log_p_i = model.base_dist.log_prob(z) - \
                  model.bijector.forward_log_det_jacobian(z, 1, ConditionalAffine=y_i)

        log_p_j = model.base_dist.log_prob(z_j) - \
                  model.bijector.forward_log_det_jacobian(z_j, 1, ConditionalAffine=y_j)

        dkl = dkl + tf.reduce_mean(log_p_i - log_p_j)

    return float(dkl) / float(n_batches)


def dkl_sex(model, age_ij, sex_i, sex_j, synd_ij):
    n_batches = 5
    batch_size = 10000

    dkl = 0
    for _ in range(n_batches):
        # conditions
        age_t = tf.convert_to_tensor(
            np.full((batch_size, 1), age_ij),
            dtype=tf.float32)

        sex_i_t = tf.convert_to_tensor(
            np.full((batch_size, 1), sex_i),
            dtype=tf.float32)

        sex_j_t = tf.convert_to_tensor(
            np.full((batch_size, 1), sex_j),
            dtype=tf.float32)

        synd_t = tf.convert_to_tensor(np.full((batch_size, 1), synd_ij), dtype=tf.int32)

        y_i = {"age": age_t, "sex": sex_i_t, "syndrome": synd_t}
        y_j = {"age": age_t, "sex": sex_j_t, "syndrome": synd_t}

        #dkl
        z = model.base_dist.sample(batch_size)
        x = model.bijector.forward(z, ConditionalAffine=y_i)
        z_j = model.bijector.inverse(x, ConditionalAffine=y_j)

        log_p_i = model.base_dist.log_prob(z) - \
                  model.bijector.forward_log_det_jacobian(z, 1, ConditionalAffine=y_i)

        log_p_j = model.base_dist.log_prob(z_j) - \
                  model.bijector.forward_log_det_jacobian(z_j, 1, ConditionalAffine=y_j)

        dkl = dkl + tf.reduce_mean(log_p_i - log_p_j)

    return float(dkl) / float(n_batches)

def entropy(model, age, sex, synd):
    n_batches = 5
    batch_size = 10000

    h = 0
    for _ in range(n_batches):
        # conditions
        age_t = tf.convert_to_tensor(
            np.full((batch_size, 1), age),
            dtype=tf.float32)

        sex_t = tf.convert_to_tensor(
            np.full((batch_size, 1), sex),
            dtype=tf.float32)

        synd_t = tf.convert_to_tensor(np.full((batch_size, 1), synd), dtype=tf.int32)

        y = {"age": age_t, "sex": sex_t, "syndrome": synd_t}

        #entropy
        z = model.base_dist.sample(batch_size)
        log_p = model.base_dist.log_prob(z) - \
                  model.bijector.forward_log_det_jacobian(z, 1, ConditionalAffine=y)
        h = h + tf.reduce_mean(log_p)

    return -float(h) / float(n_batches)

if __name__ == "__main__":
    main()
