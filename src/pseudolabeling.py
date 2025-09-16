import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ER import ClassBalancedExperienceReplayMemory

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

def run_pseudolabeling(train_time_model, data, model_online_student, path):
    ## FOR VERGARA DRIFT DATASET
    ###################### Online retraining

    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)
    tf.keras.utils.set_random_seed(0)

    online_batch_size = 2
    THR = 0.98

    ### Get dataset
    test_batches = data["test_batches"]
    one_hot_encoder = data["one_hot_encoder"]

    ### Optimizer set_up
    #cce_loss = tf.keras.losses.CategoricalCrossentropy()
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #lr_online = 1e-5
    lr_online = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_online)

    model_online_student.compile(optimizer=optimizer, loss=cce_loss,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    for layer in model_online_student.layers:
        layer.trainable = True

    ### Replay memory
    mem_size = 4
    C_replay_memory = ClassBalancedExperienceReplayMemory(mem_size)
    random_subset = get_random_subset(data["X_train"], data["Y_train"], one_hot_encoder, subset_size=mem_size)
    C_replay_memory.initMemory(random_subset)

    ### For metrics
    plot_list_moving_acc = list()

    #updates = list()
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

        num_of_pseudo_learning_samples = 0

        # shuffle data inside batches
        X_retrain, Y_retrain = shuffle(X_retrain, Y_retrain, random_state=0)

        num_of_splits = int(X_retrain.shape[0] / online_batch_size)
        batches_X = np.array_split(X_retrain, num_of_splits)
        batches_Y = np.array_split(Y_retrain, num_of_splits)

        count = 0
        for batch_X, batch_Y in zip(batches_X, batches_Y):
            labels = model_online_student(batch_X, training=False)
            labels = tf.nn.softmax(labels)
            labels = labels.numpy()
            thr_indices = np.where(np.max(labels, axis=1)[..., None] > THR)[0]
            thr_not_indices = np.where(np.max(labels, axis=1)[..., None] <= THR)[0]

            labels_thr = labels[thr_indices]

            X_retrain_thr = batch_X[thr_indices]

            num_of_pseudo_learning_samples += X_retrain_thr.shape[0]

            hard_labels = np.zeros((labels_thr.shape[0], labels_thr.shape[1]))
            argmx = np.argmax(labels_thr, axis=1)
            for ps in range(0, labels_thr.shape[0]):
                hard_labels[ps, argmx[ps]] = 1

            for i in range(0, X_retrain_thr.shape[0]):
                sampleX = np.expand_dims(X_retrain_thr[i], axis=0)
                sampleY = np.expand_dims(hard_labels[i], axis=0)
                hard_label_inverse = one_hot_encoder.inverse_transform(sampleY)[0, 0]
                C_replay_memory.addSample(sampleX, hard_label_inverse)

            online_training_batch_X, online_training_batch_Y = C_replay_memory.getMemoryAsArray()
            #online_training_batch_X = np.concatenate((subset_X, online_training_batch_X), axis=0)
            online_training_batch_Y = one_hot_encoder.transform(online_training_batch_Y)

            if True:
                with tf.GradientTape() as tape:
                    logits_student = tf.cast(model_online_student(online_training_batch_X, training=True), tf.float32)
                    hard_labels = tf.cast(online_training_batch_Y, tf.float32)
                    et_loss = cce_loss(hard_labels, logits_student)
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

        print("Online moving accuracy: ", MV_acc_online)
        print("Static moving accuracy: ", MV_acc_static)
        print("Num of pseudolabel learning samples: ", num_of_pseudo_learning_samples)

        print("======================================================")
        plot_list_moving_acc.append(MV_acc_online)

    print("Final acc:", (mv_acc/9))

    print("Stop running pseudolabeling.")
