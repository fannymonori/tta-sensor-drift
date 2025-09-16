import numpy as np
import glob
import sys

from sklearn.preprocessing import OneHotEncoder

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

path_drift_full = "" #Replace it with path to downloaded sensor drift dataset

num_of_batches = 10
num_of_sensors = 16

num_of_classes = 6

def read_batches_drift(path):
    files = glob.glob(path + "/*.dat")

    data_in_batches = dict()
    for b in range(0, num_of_batches):
        data_in_batches[b] = (list(), list())

    for file in files:
        batch_num = int(file.strip().split("/")[-1][5:-4]) - 1
        with open(file, 'r') as file_content:
            for line in file_content:
                splitted = line.rstrip().split(' ')
                class_label = int(splitted[0].strip().split(";")[0])
                data_in_batches[batch_num][1].append(class_label)

                sensor = 0
                feature = 0
                sample = np.empty((8, 16))
                for s in splitted[1:]:
                    value = float(s.split(':')[1].strip())
                    sample[feature, sensor] = value

                    feature = feature + 1

                    if (feature % 8 == 0) and (feature > 0):
                        feature = 0
                        sensor = sensor + 1

                data_in_batches[batch_num][0].append(sample)

    for k,v in data_in_batches.items():
        tmp = (np.asarray(v[0]), np.asarray(v[1]))
        data_in_batches[k] = tmp

    return data_in_batches


def read_batches_drift_2(path):
    files = glob.glob(path + "/*.dat")

    data_in_batches = dict()
    for b in range(0, num_of_batches):
        data_in_batches[b] = (list(), list())

    for file in files:
        batch_num = int(file.strip().split("/")[-1][5:-4]) - 1
        with open(file, 'r') as file_content:
            for line in file_content:
                splitted = line.rstrip().split(' ')
                class_label = int(splitted[0].strip().split(";")[0])
                data_in_batches[batch_num][1].append(class_label)

                sensor = 0
                feature = 0
                sample = list()
                for s in splitted[1:]:
                    value = float(s.split(':')[1].strip())
                    sample.append(value)

                sample = np.asarray(sample)

                data_in_batches[batch_num][0].append(sample)

    for k,v in data_in_batches.items():
        tmp = (np.asarray(v[0]), np.asarray(v[1]))
        data_in_batches[k] = tmp

    return data_in_batches

class PreProcesser:

    def __init__(self):
        self.scaler = None
        self.one_hot_encoder = None
        self.mean = None
        self.std = None

        self.standardizer = StandardScaler()

        self.hasMean = False

    def preProcess(self, X, Y):
        X_ = X

        if self.one_hot_encoder:
            Y_ = self.one_hot_encoder.transform(Y)
        else:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False)
            self.one_hot_encoder.fit(Y)
            Y_ = self.one_hot_encoder.transform(Y)


        if not self.hasMean:
            print("calc mean")
            scaler = StandardScaler()
            scaler.fit(X_)

            rbscaler = RobustScaler()
            rbscaler.fit(X_)

            self.mean = scaler.mean_
            self.std = np.sqrt(scaler.var_) + 0.0001

            self.hasMean = True

            X_ = rbscaler.transform(X_)
            self.scaler = rbscaler

        return X_, Y_

    def getOneHotEncoder(self):
        return self.one_hot_encoder

    def getStatistics(self):
        return self.mean, self.std

    def getScaler(self):
        return self.scaler



def get_random_subset(X, Y, one_hot_encoder, subset_size=4):
    Y_ = one_hot_encoder.inverse_transform(Y)
    classes = np.unique(Y_, axis=0)

    subset = dict()

    for c in classes:
        indices = (Y_ == c).nonzero()[0]
        label = c[0]

        random_indices = np.random.choice(indices, subset_size, replace=False)

        X_c = X[random_indices]
        Y_c = Y_[random_indices]

        subset[label] = (X_c, Y_c)

    return subset


def get_drift_channels():
    print("Start processing vergara drift dataset.")

    # samples
    # 8 - num of features
    # 16 - num of sensors

    # full_ds_size = 13910
    # test_ds_size = full_ds_size - train_ds_size

    # self.classes = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toulene"]

    num_of_features = 8
    min_clip = -100
    max_clip = 100

    data_in_batches = read_batches_drift(path_drift_full)
    preprocesser = PreProcesser()

    train_X = list()
    train_Y = list()
    test_data = dict()
    test_data_unprocessed = dict()
    test_data_orig = dict()

    for k, v in data_in_batches.items():
        if k < 1:
            train_X.extend(v[0])
            train_Y.extend(v[1])
        else:
            test_data[k] = v

    train_X = np.asarray(train_X)
    train_X_orig = train_X.copy()

    train_Y = np.asarray(train_Y)

    # Flatten data for pre-processing
    shape = train_X.shape
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    train_Y = np.expand_dims(train_Y, axis=1)

    # Uncomment to calculate weights in training
    #_weights = sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced', y=train_Y)
    _weights = None

    train_X_unprocessed = train_X.copy()

    # Pre-process training data to get scaling statistics
    train_X, train_Y = preprocesser.preProcess(train_X, train_Y)
    scaler = preprocesser.getScaler()
    mean, std = preprocesser.getStatistics()

    # Get one-hot-encoder
    one_hot_encoder = preprocesser.getOneHotEncoder()

    # Clip training data for large values
    train_X = np.clip(train_X, min_clip, max_clip)
    train_X = np.reshape(train_X, (shape[0], shape[1], shape[2]))

    new_test_data = {}
    # Pre-process all test data
    for k, v in test_data.items():
        x = v[0]
        y = np.expand_dims(v[1], axis=1)

        shape = x.shape
        x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

        # Pre-process test data
        test_data_unprocessed[k] = (x, y)
        x, y = preprocesser.preProcess(x, y)
        x = scaler.transform(x)
        x = np.clip(x, min_clip, max_clip)
        x = np.reshape(x, (shape[0], shape[1], shape[2]))

        new_test_data[k] = (x, y)

        print("Dataset shape for test batch #:", k)
        print(x.shape)


    sample_shape = [num_of_features, num_of_sensors]

    # Pre-shuffle training dataset
    train_X, train_Y = shuffle(train_X, train_Y, random_state=0)

    dataset = {
        "X_train" : train_X,
        "Y_train" : train_Y,
        "X_train_unprocessed": train_X_unprocessed,
        "X_train_orig": train_X_orig,
        "test_batches": new_test_data,
        "test_batches_unprocessed": test_data_unprocessed,
        "test_batches_orig" : test_data_orig,
        "shape" : sample_shape,
        "one_hot_encoder" : one_hot_encoder,
        "num_of_sensors" : num_of_sensors,
        "num_of_features" : num_of_features,
        "num_of_classes": num_of_classes,
        "mean" : mean,
        "std" : std,
        "weights" : _weights,
        "scaler": scaler
    }

    return dataset

