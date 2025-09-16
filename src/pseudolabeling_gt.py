import numpy as np
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

def decision(probability):
    #np.random.seed(0)
    #random.seed(0)
    return random.random() < probability

def run_pseudolabeling_with_gt(train_time_model, data, online_model, path):
    ## FOR VERGARA DRIFT DATASET
    ###################### Online retraining with pseudolabels (mixed with active labels)

    # Hyperparameters
    #THR = 0.98
    #THR = 0.98
    THR = 0.98
    #THR = 0.5
    online_batch_size = 2
    #gt_prob = 0.2
    gt_prob = 0.2
    #mem_size = 4
    mem_size = 4
    lr_online = 1e-4
    #lr_online = 1e-5

    print("Confidence threshold is: ", THR)
    print("Number of samples in an online batch is: ", online_batch_size)
    print("Replay memory size is: ", mem_size)
    print("Assigned probability is: ", gt_prob)
    print("Online update learning rate: ", lr_online)


    # Set random seeds
    np.random.seed(0)
    tf.random.set_seed(0)
    #random.seed(0)
    tf.keras.utils.set_random_seed(0)

    # Get dataset test batches
    test_batches = data["test_batches"]
    one_hot_encoder = data["one_hot_encoder"]

    # Set-up optimizer. Softmax is not part of the trained layers, so form_logits=True
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_online)
    online_model.compile(optimizer=optimizer, loss=cce_loss,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Set all layers trainable
    for layer in online_model.layers:
        layer.trainable = True

    ### Initialize replay memory
    C_replay_memory = ClassBalancedExperienceReplayMemory(mem_size)
    random_subset = get_random_subset(data["X_train"], data["Y_train"], one_hot_encoder, subset_size=mem_size)
    C_replay_memory.initMemory(random_subset)

    ### For metrics
    plot_list_moving_acc = list()

    #updates = list()
    print("Evaluate test-time training dataset\n")
    batch_accuracies = []
    batch_gt_sample_nums = []
    batch_ps_sample_nums = []
    batch_unused_samples_nums = []
    baseline_accuracies = []
    mv_acc = 0
    count = 0
    for k, v in test_batches.items():

        print("##################################################################")
        print("Test dataset shape of gas_models:", v[0].shape)
        print("Evaluate test dataset #", k)

        moving_accuracy_static_list = list()
        moving_accuracy_online_list = list()
        moving_accuracy_online_gt = list()

        X_retrain = v[0]
        Y_retrain = v[1]

        num_of_active_learning_samples = 0
        num_of_pseudo_learning_samples = 0

        num_of_samples_in_test = X_retrain.shape[0]

        # shuffle data inside batches
        X_retrain, Y_retrain = shuffle(X_retrain, Y_retrain, random_state=0)

        num_of_splits = int(X_retrain.shape[0] / online_batch_size)
        batches_X = np.array_split(X_retrain, num_of_splits)
        batches_Y = np.array_split(Y_retrain, num_of_splits)

        for batch_X, batch_Y in zip(batches_X, batches_Y):
            #print(batch_X)

            # Variable declarations
            X_not_retrain = None

            # Inference current batch
            labels = online_model(batch_X, training=False)
            labels = tf.nn.softmax(labels)
            labels = labels.numpy()

            # Get predictions that are above and below the threshold
            thr_indices = np.where(np.max(labels, axis=1)[..., None] > THR)[0]
            thr_not_indices = np.where(np.max(labels, axis=1)[..., None] <= THR)[0]

            labels_above_thr = labels[thr_indices]

            # Get those input samples where the confidence was above the threshold
            X_retrain_thr = batch_X[thr_indices]

            num_pseudo_labels = X_retrain_thr.shape[0]
            num_active_learning = 0

            #If probability of assigning ground truth label is ...
            if decision(gt_prob):
                # Get those samples (X and Y) where the confidence was below threshold
                X_not_retrain = batch_X[thr_not_indices]
                labels_not_thr = batch_Y[thr_not_indices]

                num_active_learning = X_not_retrain.shape[0]

                # If there are more than 0, add them to the samples used for retraining
                if X_not_retrain.shape[0] > 0:
                  X_retrain_thr = np.concatenate((X_retrain_thr, X_not_retrain), axis=0)
                  labels_above_thr = np.concatenate((labels_above_thr, labels_not_thr), axis=0)

            #     ##for ablation study:
            #     labels_above_thr = labels_not_thr
            #     X_retrain_thr = X_not_retrain
            #     num_pseudo_labels = 0
            # else: #this should only be uncommented if above 3 lines are uncommented
            #     # Evaluate on the same batch
            #     moving_acc_online = evaluate(batch_X, batch_Y, online_model)
            #     moving_acc_static = evaluate(batch_X, batch_Y, train_time_model)
            #
            #     moving_accuracy_online_list.append(moving_acc_online)
            #     moving_accuracy_static_list.append(moving_acc_static)
            #
            #     count = count + 1
            #     continue

            if X_retrain_thr.shape[0] < 1:
                # Evaluate on the same batch
                moving_acc_online = evaluate(batch_X, batch_Y, online_model)
                moving_acc_static = evaluate(batch_X, batch_Y, train_time_model)

                moving_accuracy_online_list.append(moving_acc_online)
                moving_accuracy_static_list.append(moving_acc_static)

                count = count + 1
                continue

            # Number of samples used for pseudolabel training
            num_of_pseudo_learning_samples += num_pseudo_labels

            # Increment the variable that tracks the number of gt labeled samples
            #num_of_active_learning_samples += X_not_retrain.shape[0]
            num_of_active_learning_samples += num_active_learning

            # Get hard labels, for example [0.2, 0.8] -> [0, 1]
            hard_labels = np.zeros((labels_above_thr.shape[0], labels_above_thr.shape[1]))
            argmx = np.argmax(labels_above_thr, axis=1)
            for ps in range(0, labels_above_thr.shape[0]):
                hard_labels[ps, argmx[ps]] = 1

            # Add samples to replay memory
            for i in range(0, X_retrain_thr.shape[0]):
                sampleX = np.expand_dims(X_retrain_thr[i], axis=0)
                sampleY = np.expand_dims(hard_labels[i], axis=0)
                hard_label_inverse = one_hot_encoder.inverse_transform(sampleY)[0, 0]
                C_replay_memory.addSample(sampleX, hard_label_inverse)

            # Get replay memory content
            online_training_batch_X, online_training_batch_Y = C_replay_memory.getMemoryAsArray()
            online_training_batch_Y = one_hot_encoder.transform(online_training_batch_Y)

            # Don't use replay memory
            #online_training_batch_X = sampleX
            #online_training_batch_Y = sampleY

            #if count == 10:
            #    print("Num of samples (memory) used in the update: ", online_training_batch_X.shape)

            # Shuffle the minibatch
            online_training_batch_X, online_training_batch_Y = shuffle(online_training_batch_X, online_training_batch_Y, random_state=0)

            if True:
                with tf.GradientTape() as tape:
                    # Inference again, this time the update batch
                    logits_student = tf.cast(online_model(online_training_batch_X, training=True), tf.float32)
                    logits_student = tf.nn.softmax(logits_student)
                    # Get the hard labels
                    hard_labels = tf.cast(online_training_batch_Y, tf.float32)

                    # Compute the CCE loss between the hard labels and the logits
                    et_loss = cce_loss(hard_labels, logits_student)
                    #cce_source = cce_loss(Y_train, logits_gt)
                    #et_loss = tf.reduce_mean(crossEntropyLoss(logits_student, hard_labels), axis=0)
                    loss_value = et_loss

                # Update the online model
                grads = tape.gradient(loss_value, online_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, online_model.trainable_weights))

            # Evaluate on the same batch
            moving_acc_online = evaluate(batch_X, batch_Y, online_model)
            moving_acc_static = evaluate(batch_X, batch_Y, train_time_model)

            # Evaluate only on the ground truth samples
            if X_not_retrain is not None:
                moving_acc_ = evaluate(X_not_retrain, labels_not_thr, online_model)
                moving_accuracy_online_gt.append(moving_acc_)

            moving_accuracy_online_list.append(moving_acc_online)
            moving_accuracy_static_list.append(moving_acc_static)

            count = count + 1

        print("======================================================")

        MV_acc_static = float(np.mean(np.asarray(moving_accuracy_static_list)))
        MV_acc_online = float(np.mean(np.asarray(moving_accuracy_online_list)))
        MV_acc_gt = float(np.mean(np.asarray(moving_accuracy_online_gt)))

        mv_acc += MV_acc_online

        samples_not_used = num_of_samples_in_test - num_of_pseudo_learning_samples - num_of_active_learning_samples

        batch_accuracies.append(MV_acc_online)
        baseline_accuracies.append(MV_acc_static)
        batch_gt_sample_nums.append(num_of_active_learning_samples)
        batch_ps_sample_nums.append(num_of_pseudo_learning_samples)
        batch_unused_samples_nums.append(samples_not_used)

        print("Online moving accuracy: ", MV_acc_online)
        print("Static moving accuracy: ", MV_acc_static)
        print("Active learning samples accuracy: ", MV_acc_gt)
        print("Num of active learning samples: ", num_of_active_learning_samples)
        print("Num of pseudolabel learning samples: ", num_of_pseudo_learning_samples)
        print("Num of samples not used for updating: ", samples_not_used)

        print("======================================================")
        plot_list_moving_acc.append(MV_acc_online)

    print("Online accuracies:")
    for i in batch_accuracies:
        v = i * 100.0
        value = f"{v:.2f}"
        print(value, end=", ")
    print("\n")

    print("Number of ground truth samples used:")
    for i in batch_gt_sample_nums:
        v = i
        value = str(v)
        print(value, end=", ")
    print("\n")

    print("Number of pseudo-label samples used:")
    for i in batch_ps_sample_nums:
        v = i
        value = str(v)
        print(value, end=", ")
    print("\n")


    print("Samples not used in update:")
    for i in batch_unused_samples_nums:
        v = i
        value = str(v)
        print(value, end=", ")
    print("\n")


    print("Baseline accuracy:")
    for i in baseline_accuracies:
        v = i * 100.0
        value = f"{v:.2f}"
        print(value, end=", ")
    print("\n")

    print("Baseline acc: ", np.asarray(baseline_accuracies).mean())
    print("Final acc:", (mv_acc/9))

    print("Stop running pseudolabeling.")
