import numpy as np
import tensorflow as tf
import os
from datetime import datetime

from vergara_drift_batches_dataset import get_drift_channels
from models.mlp import define_mlp
from teacher_student import run_pseudolabeling_TA
from pseudolabeling import run_pseudolabeling
from pseudolabeling_gt import run_pseudolabeling_with_gt
from tent import run_TENT
import random

model_func = define_mlp

def run_train(model, data):

    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)
    tf.keras.utils.set_random_seed(0)

    epochs = 100
    batch_size = 8

    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    X_train = data["X_train"]
    Y_train = data["Y_train"]

    # # Create a new folder to save the plot
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # save_dir = f""
    # os.makedirs(save_dir, exist_ok=True)

    test_batches = data["test_batches"]

    training_result = model.fit(X_train,
                                Y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.1,
                                #sample_weight=data["weights"],
                                callbacks=[tensorboard_callback, earlystop_callback],
                                shuffle=True)


    one_hot_encoder = data["one_hot_encoder"]

    # Uncomment if want to save the weights
    #model.save_weights("./saved_weights_n128.weights.h5", overwrite=True)

    #################
    # Uncomment if want to save learning curve
    # Plot the learning curve
    # plt.figure(figsize=(10, 6))
    # plt.plot(training_result.history['loss'], label='Training Loss')
    # plt.plot(training_result.history['accuracy'], label='Training Accuracy')
    # plt.title('Learning Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.grid(True)
    #
    # # Save the plot
    # plot_path = os.path.join(save_dir, 'learning_curve.png')
    # plt.savefig(plot_path)
    # plt.close()

    ##################
    print("Class accuracies: ")
    for c in range(1, 7):
        y_test_inv = one_hot_encoder.inverse_transform(Y_train)
        indices = (y_test_inv == c).nonzero()[0]
        if len(indices) > 0:
            x_test_ = X_train[indices]
            y_test_ = Y_train[indices]
            test_eval = model.evaluate(x_test_, y_test_, verbose=0)
            print('Accuracy on train dataset class {} with samples {} is {:.4%}'.format(c, x_test_.shape[0], test_eval[1]))
        else:
            print('Class {} is not present'.format(c))

    print("Train dataset shape of gas_models: ", X_train.shape)
    print("Number of epochs: ", epochs)
    print("Batch size: ", batch_size)

    running_acc = 0
    for k,v in test_batches.items():
        x_test = v[0]

        y_test = v[1]
        test_eval = model.evaluate(x_test, y_test, verbose=0)

        print('Accuracy on test dataset # {} is {:.4%}'.format(k, test_eval[1]))
        print("Test dataset shape:", x_test.shape, y_test.shape)
        running_acc += test_eval[1]

        print("Class accuracies: ")
        for c in range(1, 7):
            y_test_inv = one_hot_encoder.inverse_transform(y_test)
            indices = (y_test_inv == c).nonzero()[0]
            if len(indices) > 0:
                x_test_ = x_test[indices]
                y_test_ = y_test[indices]
                test_eval = model.evaluate(x_test_, y_test_, verbose=0)
                print('Accuracy on test dataset class {} with sample #{} is {:.4%}'.format(c, x_test_.shape[0], test_eval[1]))
            else:
                print('Class {} is not present'.format(c))

    print("Final accuracy is: ", running_acc/9)

    return model

def student_teacher_ps(path):
    dataset = get_drift_channels()

    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)
    tf.keras.utils.set_random_seed(0)

    model1 = model_func(dataset["shape"], dataset["num_of_classes"])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model1.compile(optimizer=opt, loss=cce,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


    model_online_student = model_func(dataset["shape"], dataset["num_of_classes"])
    model_online_teacher = model_func(dataset["shape"], dataset["num_of_classes"])

    #trained_model = run_train(model1, dataset)   # uncomment this and comment next line to train from scratch
    trained_model = model1
    trained_model.load_weights('./saved_weights_n128.weights.h5')

    model_online_student.set_weights(trained_model.get_weights())
    model_online_teacher.set_weights(trained_model.get_weights())

    for layer in model1.layers:
        layer.trainable = False

    acc = run_pseudolabeling_TA(trained_model, dataset, model_online_student, model_online_teacher, path)


def pseudolabeling(path):
    dataset = get_drift_channels()

    model1 = model_func(dataset["shape"], dataset["num_of_classes"])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model1.compile(optimizer=opt, loss=cce,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


    model_online_student = model_func(dataset["shape"], dataset["num_of_classes"])

    #trained_model = run_train(model1, dataset)  # uncomment this and comment next line to train from scratch
    trained_model = model1
    trained_model.load_weights('./saved_weights_n128.weights.h5')

    model_online_student.set_weights(trained_model.get_weights())

    for layer in model1.layers:
        layer.trainable = False

    acc = run_pseudolabeling(trained_model, dataset, model_online_student, path)

def pseudolabeling_gt(path):
    print("Start running pseudolabeling experiments with ground truth labels")

    dataset = get_drift_channels()

    print("Dataset shape: ", dataset["shape"], dataset["num_of_classes"])

    model1 = model_func(dataset["shape"], dataset["num_of_classes"])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model1.compile(optimizer=opt, loss=cce,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


    model_online_student = model_func(dataset["shape"], dataset["num_of_classes"])

    #trained_model = run_train(model1, dataset) # uncomment this and comment next line to train from scratch
    trained_model = model1
    trained_model.load_weights('./saved_weights_n128.weights.h5')

    model_online_student.set_weights(trained_model.get_weights())

    for layer in model1.layers:
        layer.trainable = False

    acc = run_pseudolabeling_with_gt(trained_model, dataset, model_online_student, path)


def tent(path):
    dataset = get_drift_channels()

    model1 = model_func(dataset["shape"], dataset["num_of_classes"])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model1.compile(optimizer=opt, loss=cce,
                       metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model_online_student = model_func(dataset["shape"], dataset["num_of_classes"])

    #trained_model = run_train(model1, dataset) #train from scratch or load pre-trained model
    trained_model = model1
    trained_model.load_weights('./saved_weights_n128.weights.h5')

    model_online_student.set_weights(trained_model.get_weights())

    for layer in model1.layers:
        layer.trainable = False

    acc = run_TENT(trained_model, dataset, model_online_student)

if __name__ == '__main__':

    print("Start test-time adaptation program.")

    #time = datetime.today().strftime('%Y_%m_%d_%H%M%S')
    #path_to_save = "" # Change this
    #folder = path_to_save + time + "/"
    #if not os.path.exists(folder):
    #    os.makedirs(folder)

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)
    tf.keras.utils.set_random_seed(0)

    # Uncomment the one you would like to run
    #tent(folder)
    #student_teacher_ps(folder)
    pseudolabeling(folder)

    #pseudolabeling_gt(folder)
