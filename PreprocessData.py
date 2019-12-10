import ember
import random
import numpy as np
from sklearn import preprocessing
import sys


"""
Removes all of the unlabeled samples from the data set (target value of -1)
"""
def remove_unlabeled(X_data, y_data):

    removal_index = []
    for i in range(len(y_data)):
        if y_data[i] < 0:
            removal_index.append(i)

    X_trim = np.delete(X_data, removal_index, axis=0)
    y_trim = np.delete(y_data, removal_index, axis=0)

    return X_trim, y_trim


train_size = int(sys.argv[1])
test_size = int(sys.argv[2])

# Takes the json files from the ember2018 data and creates vectorized features
# that can be manipulated for machine learning.
X_train, y_train, X_test, y_test = ember.read_vectorized_features("data/ember2018/")

# Remove the unlabeled data before sampling.
X_train, y_train = remove_unlabeled(X_train, y_train)
X_test, y_test = remove_unlabeled(X_test, y_test)

# Create random sample of the indices of our training and test data which can
# be used to create a random sample of the data itself.
train_sampler = random.sample(range(len(y_train)), train_size)
test_sampler  = random.sample(range(len(y_test)), test_size)

# Apply the sample to training data
X_train = X_train[train_sampler]
y_train = y_train[train_sampler]

# Apply the sample to test data
X_test = X_test[test_sampler]
y_test = y_test[test_sampler]

# Use Min/Max normalization to normalize our feature vectors. This is necessary
# for PCA and helps training in general.
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test  = min_max_scaler.fit_transform(X_test)

# Saving the processed data in csv format to the data directory
np.savetxt("data/X_train.csv", X_train, delimiter=',')
np.savetxt("data/y_train.csv", y_train, delimiter=',')
np.savetxt("data/X_test.csv", X_test, delimiter=',')
np.savetxt("data/y_test.csv", y_test, delimiter=',')
