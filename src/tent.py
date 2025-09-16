import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ER import ClassBalancedExperienceReplayMemory, ExperienceReplayMemory

import random
from sklearn.utils import shuffle

def evaluate(X, Y, model):
    test_cat_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    logits = model(X, training=False)
    test_cat_acc_metric.update_state(Y, logits)
    result = test_cat_acc_metric.result()
    test_cat_acc_metric.reset_state()

    return result

def crossEntropyLoss(y_pred, y_target):
    return -1 * tf.reduce_mean(tf.nn.softmax(y_target) * tf.nn.log_softmax(y_pred), axis=-1)

def entropyLoss_logit(y_pred):
    return -1 * tf.reduce_mean(tf.nn.softmax(y_pred) * tf.nn.log_softmax(y_pred), axis=-1)

def get_random_subset_array(X, Y, one_hot_encoder, subset_size=4):
    Y_ = one_hot_encoder.inverse_transform(Y)
    classes = np.unique(Y_, axis=0)

    subset_X = list()
    subset_Y = list()

    for c in classes:
        indices = (Y_ == c).nonzero()[0]
        label = c[0]
        random_indices = np.random.choice(indices, subset_size, replace=False)

        X_c = X[random_indices]
        Y_c = Y_[random_indices]
        Y_c = one_hot_encoder.transform(Y_c)

        subset_X.extend(X_c)
        subset_Y.extend(Y_c)

    subset_X = np.asarray(subset_X)
    subset_Y = np.asarray(subset_Y)

    return subset_X, subset_Y

def get_random_subset(X, Y, one_hot_encoder, subset_size=4):
    Y_ = one_hot_encoder.inverse_transform(Y)
    classes = np.unique(Y_, axis=0)

    subset = dict()

    for c in classes:
        indices = (Y_ == c).nonzero()[0]
        label = c[0]
        ssize = subset_size
        random_indices = np.random.choice(indices, ssize, replace=False)
        X_c = X[random_indices]
        subset[label] = X_c

    return subset


def run_TENT(train_time_model, data, model_online_student):
    ## FOR VERGARA DRIFT DATASET
    ###################### Online retraining with TENT

    # Hyperparameters
    THR = 0.98
    #THR = 0.99
    online_batch_size = 2
    mem_size = 4
    lr_online = 1e-4
    #lr_online = 1e-5

    print("Number of samples in an online batch is: ", online_batch_size)
    print("Replay memory size is: ", mem_size)
    print("Online update learning rate: ", lr_online)

    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)
    tf.keras.utils.set_random_seed(0)

    ### Get dataset
    test_batches = data["test_batches"]
    one_hot_encoder = data["one_hot_encoder"]

    ### Optimizer set_up
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_online)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_online)

    model_online_student.compile(optimizer=optimizer, loss=cce_loss,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    for layer in model_online_student.layers:
        layer.trainable = True

    ### Replay memory

    ### Replay memory
    C_replay_memory = ClassBalancedExperienceReplayMemory(mem_size)
    random_subset = get_random_subset(data["X_train"], data["Y_train"], one_hot_encoder, subset_size=8)
    C_replay_memory.initMemory(random_subset)

    ### For metrics
    plot_list_moving_acc = list()
    batch_accuracies = list()

    print("Evaluate test-time training dataset\n")
    mv_acc = 0
    for k, v in test_batches.items():

        print("##################################################################")
        print("Test dataset shape of gas_models:", v[0].shape)

        print("Evaluate test dataset #", k)

        moving_accuracy_static_list = list()
        moving_accuracy_online_list = list()

        X_retrain = v[0]
        Y_retrain = v[1]

        X_retrain, Y_retrain = shuffle(X_retrain, Y_retrain, random_state=0)

        num_of_splits = int(X_retrain.shape[0] / online_batch_size)
        batches_X = np.array_split(X_retrain, num_of_splits)
        batches_Y = np.array_split(Y_retrain, num_of_splits)

        count = 0
        for batch_X, batch_Y in zip(batches_X, batches_Y):

            # labels_logits = model_online_student(batch_X, training=False)
            # labels = tf.nn.softmax(labels_logits)
            # labels = labels.numpy()
            # labels_thr = labels
            # X_retrain_thr = batch_X
            # hard_labels = np.zeros((labels_thr.shape[0], labels_thr.shape[1]))
            # argmx = np.argmax(labels_thr, axis=1)
            # for ps in range(0, labels_thr.shape[0]):
            #     hard_labels[ps, argmx[ps]] = 1
            #
            # for i in range(0, X_retrain_thr.shape[0]):
            #     sampleX = np.expand_dims(X_retrain_thr[i], axis=0)
            #     sampleY = np.expand_dims(hard_labels[i], axis=0)
            #     hard_label_inverse = one_hot_encoder.inverse_transform(sampleY)[0, 0]
            #     C_replay_memory.addSample(sampleX, hard_label_inverse)
            #
            # replay_memory_X, _ = C_replay_memory.getMemoryAsArray()
            # online_training_batch_X = replay_memory_X

            with tf.GradientTape() as tape:
                logits_student = model_online_student(batch_X, training=True)
                et_loss = tf.reduce_mean(entropyLoss_logit(logits_student), axis=0)
                loss_value = et_loss

            grads = tape.gradient(loss_value, model_online_student.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_online_student.trainable_weights))


            moving_acc_online = evaluate(batch_X, batch_Y, model_online_student)
            moving_acc_static = evaluate(batch_X, batch_Y, train_time_model)

            moving_accuracy_online_list.append(moving_acc_online)
            moving_accuracy_static_list.append(moving_acc_static)


        print("======================================================")

        MV_acc_static = float(np.mean(np.asarray(moving_accuracy_static_list)))
        MV_acc_online = float(np.mean(np.asarray(moving_accuracy_online_list)))

        mv_acc += MV_acc_online

        batch_accuracies.append(MV_acc_online)

        print("Online moving accuracy:", MV_acc_online)
        print("Static moving accuracy:", MV_acc_static)

        print("======================================================")

        plot_list_moving_acc.append(MV_acc_online)

    print("Final acc:", (mv_acc / 9))

    print("Online accuracies:")
    for i in batch_accuracies:
        v = i * 100.0
        value = f"{v:.2f}"
        print(value, end=", ")
    print("\n")

    print("Finished TENT learning approach")
