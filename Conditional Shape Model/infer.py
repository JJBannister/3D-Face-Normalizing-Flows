import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from face_model import FaceModel, input_shape, n_syndromes
batch_size = 10000

synd_codes = pd.read_csv("../Data/Cleaned2/synd_codes.csv", index_col=0, squeeze=True, header=None).to_dict()
sex_codes = pd.read_csv("../Data/Cleaned2/sex_codes.csv", index_col=0, squeeze=True, header=None).to_dict()

full_data_file = "../Data/Cleaned/Full.csv"
full_checkpoint_dir = "../Data/Results/Checkpoints/Full"

testing_data_files = ["../Data/Cleaned2/"+str(x)+"_test.csv" for x in range(10)]
training_checkpoint_dirs = ["../Data/Results/Checkpoints2/PC100/"+str(x) for x in range(10)]


def main():
    synd_confusion_figures()
    #age_prediction_figures()
    #sex_prediction_figures()

# SEX
########################################################################################################################

def sex_prediction_figures():
    sns.set_context("paper", rc={"font.size": 30, "axes.titlesize": 30, "axes.labelsize": 45})
    def _mle_sex(results):
        ml_sex = results.apply(lambda row: int(row[1] > row[0]), axis=1)
        def _correct_sex_posterior(row):
            true_sex = int(row["True Sex"])
            p_true = np.exp(row[true_sex])
            norm = np.exp(row[0]) + np.exp(row[1])
            return p_true / norm
        p_correct = results.apply(_correct_sex_posterior, axis=1)

        z = results[["Syndrome", "Age (years)", "True Sex"]]
        z["MAP Sex"] = ml_sex
        z["Correct Sex Posterior Probability"] = p_correct
        z["Correct"] = p_correct > 0.5
        z = z.replace({"Syndrome": synd_codes,
                       "True Sex": sex_codes,
                       "MAP Sex": sex_codes})
        return z

    # Full Data
    full_data = load_validation_data(full_data_file)
    full_model = FaceModel(np.zeros(input_shape), full_checkpoint_dir)
    full_results = sex_likelihoods(full_data, full_model)
    full_results = _mle_sex(full_results)

    # CV
    if True:
        cv_results = []
        for data_file, checkpoint_dir in zip(testing_data_files, training_checkpoint_dirs):
            data = load_validation_data(data_file)
            model = FaceModel(np.zeros(input_shape), checkpoint_dir)
            cv_results.append(sex_likelihoods(data, model))
        cv_results = pd.concat(cv_results)
        cv_results = _mle_sex(cv_results)

    # Figs
    mle_sex_figure(full_results, "Training Data Sex Prediction")
    mle_sex_figure(cv_results, "CV Data Sex Prediction")

def mle_sex_figure(results, title):
    unaffected_results = results.loc[results["Syndrome"] == "Unaffected"]
    p = sns.regplot(x="Age (years)", y="Correct Sex Posterior Probability", logistic=True, data=unaffected_results)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim(0,80)
    plt.show()

    synd_results = results.loc[results["Syndrome"] != "Unaffected"]
    #p = sns.lmplot(x="Age (years)", y="Correct Sex Posterior Probability", hue="Syndrome", fit_reg=False, data=synd_results)
    #plt.xticks(fontsize=30)
    #plt.yticks(fontsize=30)
    #plt.show()

    print("Unaffected Sex Accuracy:", unaffected_results["Correct"].values.sum() / len(unaffected_results.index))
    print("Syndromic Sex Accuracy:", synd_results["Correct"].values.sum() / len(synd_results.index))


def sex_likelihoods(data, model):
    sexes = [0,1]
    col_names = ["Syndrome", "Age (years)", "True Sex"] + sexes
    results = pd.DataFrame(columns=col_names)

    @tf.function
    def log_prob(pc, conditions):
        return model.log_prob(pc, bijector_kwargs=conditions)

    for pc, true_age, true_sex, true_synd in data:
        x = np.zeros(shape=(true_age.shape[0], len(sexes)))

        for i in range(len(sexes)):
            sex_test_tf = tf.convert_to_tensor(np.ones(shape=(true_age.shape[0], 1)) * sexes[i], dtype=tf.float32)
            log_probs = model.log_prob(pc, true_age, sex_test_tf, true_synd).numpy()
            x[:, i] = log_probs.T

        y = np.concatenate(
            (true_synd.numpy().reshape([-1, 1]),
             true_age.numpy().reshape([-1, 1]),
             true_sex.numpy().reshape([-1, 1]),
             x),
            axis=1)

        batch_results = pd.DataFrame(y, columns=col_names)
        results = results.append(batch_results)
    return results


# AGE
########################################################################################################################

def age_prediction_figures():
    sns.set_context("paper", rc={"font.size": 30, "axes.titlesize": 30, "axes.labelsize": 45})

    def _mle_age(results):
        probs = results.drop(["Syndrome", "True Age"], axis=1)
        ml_age = probs.idxmax(axis=1)

        ages = np.asarray([int(y) for y in probs.columns])

        def _mean(row):
            x = row.to_numpy()
            x = np.exp(x) / np.sum(np.exp(x))
            return np.average(ages, weights=x)

        def _std(row):
            x = row.to_numpy()
            x = np.exp(x) / np.sum(np.exp(x))
            mean = np.average(ages, weights=x)
            return np.sqrt(np.average((ages - mean) ** 2, weights=x))

        std = probs.apply(_std, axis=1)
        mean = probs.apply(_mean, axis=1)

        z = results[["Syndrome", "True Age"]]
        z["MAP Age"] = ml_age
        z["Absolute Age Error"] = (z["True Age"] - z["MAP Age"]).abs()
        z["Posterior STD"] = std
        z = z.replace({"Syndrome": synd_codes})
        return z

    # Full Data
    full_data = load_validation_data(full_data_file)
    full_model = FaceModel(np.zeros(input_shape), full_checkpoint_dir)
    full_results = age_likelihoods(full_data, full_model)
    full_results = _mle_age(full_results)

    # CV
    if True:
        cv_results = []
        for data_file, checkpoint_dir in zip(testing_data_files, training_checkpoint_dirs):
            data = load_validation_data(data_file)
            model = FaceModel(np.zeros(input_shape), checkpoint_dir)
            cv_results.append(age_likelihoods(data, model))
        cv_results = pd.concat(cv_results)
        cv_results = _mle_age(cv_results)

    # Figs
    print(full_results)
    #mle_age_figure(full_results, "Training Data Age Prediction")
    mle_age_figure(cv_results, "Cross Validated Age Prediction")

    #age_error_figure(full_results, "Training Data Age Uncertainty")
    #age_error_figure(cv_results, "Cross Validated Age Uncertainty")


def age_error_figure(results, title):
    x_line = np.linspace(0, 80, 100)

    print(results["Posterior STD"].mean())

    unaffected_results = results.loc[results["Syndrome"] == "Unaffected"]
    sns.lmplot(x="Absolute Age Error", y="Posterior STD", data=unaffected_results)
    plt.plot(x_line, x_line, label="$y=x$", color='r')
    plt.legend(fontsize=25)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    plt.show()

    return

    synd_results = results.loc[results["Syndrome"] != "Unaffected"]
    sns.lmplot(x="Absolute Age Error", y="Posterior STD", hue="Syndrome", data=synd_results, fit_reg=False)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.plot(x_line, x_line, label="$y=x$", color='r')
    plt.legend(fontsize=25)
    plt.show()



def mle_age_figure(results, title):
    x_line = np.linspace(0, 80, 100)

    unaffected_results = results.loc[results["Syndrome"] == "Unaffected"]
    sns.lmplot(x="True Age", y="MAP Age", data=unaffected_results)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.plot(x_line, x_line, label="$y=x$", color='r')
    plt.legend(fontsize=25)
    plt.show()

    synd_results = results.loc[results["Syndrome"] != "Unaffected"]
    #sns.lmplot(x="True Age", y="MAP Age", hue="Syndrome", data=synd_results, fit_reg=False)
    #plt.plot(x_line, x_line, label="$y=x$", color='r')
    ##plt.title(title)
    #plt.xticks(fontsize=30)
    #plt.yticks(fontsize=30)
    #plt.show()

    print("Unaffected Age MAE:", unaffected_results["Absolute Age Error"].mean())
    print("Syndromic Age MAE:", synd_results["Absolute Age Error"].mean())


def age_likelihoods(data, model):
    ages = [x for x in range(81)]
    col_names = ["Syndrome", "True Age"] + ages
    results = pd.DataFrame(columns=col_names)

    @tf.function
    def log_prob(pc, conditions):
        return model.log_prob(pc, bijector_kwargs=conditions)

    for pc, true_age, true_sex, true_synd in data:
        x = np.zeros(shape=(true_age.shape[0], len(ages)))

        for i in range(len(ages)):
            age_test_tf = tf.convert_to_tensor(np.ones(shape=(true_age.shape[0], 1))*ages[i], dtype=tf.float32)
            log_probs = model.log_prob(pc, age_test_tf, true_sex, true_synd).numpy()
            x[:, i] = log_probs.T

        y = np.concatenate(
            (true_synd.numpy().reshape([-1, 1]),
             true_age.numpy().reshape([-1, 1]),
             x),
            axis=1)

        batch_results = pd.DataFrame(y, columns=col_names)
        results = results.append(batch_results)
    return results


# SYNDROME
########################################################################################################################

def synd_confusion_figures():
    # Full Data
    #full_data = load_validation_data(full_data_file)
    #full_model = FaceModel(np.zeros(input_shape), full_checkpoint_dir)
    #full_results = syndrome_likelihoods(full_data, full_model)

    # CV
    if True:
        cv_results = []
        for data_file, checkpoint_dir in zip(testing_data_files, training_checkpoint_dirs):
            data = load_validation_data(data_file)
            model = FaceModel(np.zeros(input_shape), checkpoint_dir)
            cv_results.append(syndrome_likelihoods(data, model))
        cv_results = pd.concat(cv_results)

    #MLP
    #mlp_figure(full_results, "Training Data Median Log Probabilities")
    #mlp_figure(cv_results, "Cross Validated Median Log Probabilities")

    #MLE
    #mle_figure(full_results, "Training Data Syndrome Confusion Matrix")
    mle_figure(cv_results, "Cross Validated Syndrome Confusion Matrix")


def mle_figure(results, title):
    true_synds = results["Correct Syndrome"].astype(np.int32)
    predicted_synds = results.drop(labels=["Correct Syndrome"], axis=1).idxmax(axis=1)
    top1 = pd.DataFrame({"True": true_synds, "Predicted": predicted_synds})

    heat_matrix = np.zeros(shape=[n_syndromes, n_syndromes])
    for _, row in top1.iterrows():
        heat_matrix[int(row["True"]), int(row["Predicted"])] += 1

    # Accuracy
    n_total = len(top1.index)

    def compare(x):
        return int(x["True"]) == int(x["Predicted"])
    correct_top_1 = top1[top1.apply(compare, axis=1)]
    n_correct = len(correct_top_1.index)

    print("###")
    print("Overall Accuracy: ", n_correct/n_total)
    print("###")

    # Sensitivity
    sens_matrix = heat_matrix / heat_matrix.sum(axis=1)[:,np.newaxis]
    print()
    print("###")
    synd_avg = 0.0
    for i in range(n_syndromes):
        print(list(synd_codes.values())[i], " sensitivity & ", "%.2f" % round(sens_matrix[i,i], 2), " & ")
        synd_avg += sens_matrix[i,i]
    print("Syndrome avg sensitivity: ", synd_avg/float(n_syndromes))

    heat_df = pd.DataFrame(sens_matrix,
                           index=list(synd_codes.values()),
                           columns=list(synd_codes.values()))

    # Fig
    sns.heatmap(heat_df, xticklabels=True, yticklabels=True, annot=False, linewidths=0.01, linecolor='snow',
                vmin=0, vmax=1)
    #plt.title(title)
    plt.subplots_adjust(bottom=0.3)
    plt.subplots_adjust(left=0.2)
    plt.ylabel("Diagnosed Syndrome", fontsize=30)
    plt.xlabel("Predicted Syndrome", fontsize=30)
    plt.show()

    most_confused = heat_df.apply(lambda row: row.nlargest(1).idxmin(),axis=1)
    confuse_rate = heat_df.apply(lambda row: row.nlargest(1).values[-1],axis=1)
    print(most_confused)
    print(confuse_rate)


def mlp_figure(results, title):
    median_log_prob = results.groupby("Correct Syndrome").median().values
    heat_df = pd.DataFrame(median_log_prob,
                           index=list(synd_codes.values()),
                           columns=list(synd_codes.values()))

    sns.heatmap(heat_df, xticklabels=True, yticklabels=True, annot=False, linewidths=0.01, linecolor='snow',
                vmin=-550, vmax=-400)

    #plt.title(title)
    plt.subplots_adjust(bottom=0.3)
    plt.subplots_adjust(left=0.2)
    plt.ylabel("Diagnosed Syndrome", fontsize=30)
    plt.xlabel("Hypothesized Syndrome", fontsize=30)
    plt.show()


def syndrome_likelihoods(data, model):
    col_names = ["Correct Syndrome"]+[x for x in range(n_syndromes)]
    results = pd.DataFrame(columns=col_names)
    z_results = pd.DataFrame(columns=[str(x) for x in range(input_shape)])

    @tf.function
    def log_prob(pc, conditions):
        return model.log_prob(pc, bijector_kwargs=conditions)

    for pc, age, sex, synd in data:
        x = np.zeros(shape=(age.shape[0], n_syndromes))
        true_synds = synd.numpy().reshape([-1,1])

        for synd_test in range(n_syndromes):
            synd_test_tf = tf.convert_to_tensor(np.ones(shape=(age.shape[0], 1))*synd_test, dtype=tf.int32)
            log_probs = model.log_prob(pc, age, sex, synd_test_tf).numpy()
            x[:, synd_test] = log_probs.T

        z = model.bijector.inverse(pc, ConditionalAffine={"age": age, "sex": sex, "syndrome": synd}).numpy()

        y = np.concatenate(
            (true_synds, x),
            axis=1)

        z = pd.DataFrame(z, columns=[str(x) for x in range(input_shape)])
        z_results = z_results.append(z)

        batch_results = pd.DataFrame(y, columns=col_names)
        results = results.append(batch_results)

    #results.to_csv("../Data/Results/Inference/"+checkpoint_dir.split("/")[-1]+".csv")
    #z_results.to_csv("../Data/Results/Inference/"+checkpoint_dir.split("/")[-1]+"_latent_scores.csv")
    #print("STD")
    #print(np.std(z_results.to_numpy(), axis=1))

    return results


########################################################################################################################

def load_validation_data(data_file, no_synd=False):
    data = pd.read_csv(data_file)

    if no_synd:
        data = data[data['Syndrome'] == 'Unaffected']

    pcs = data.loc[:, "PC 0":"PC 99"].values.astype(np.float32)
    sexes = np.expand_dims(data['Sex'].values.astype(np.float32), axis=1)
    ages = np.expand_dims(data['Age'].values.astype(np.float32), axis=1)
    synds = np.expand_dims(data['Syndrome'].values.astype(np.int32), axis=1)

    data = tf.data.Dataset.from_tensor_slices((pcs, ages, sexes, synds)).batch(batch_size, drop_remainder=False)
    return data

main()

