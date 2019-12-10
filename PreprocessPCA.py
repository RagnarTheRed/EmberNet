from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle as pk

'''
Performs Principal Component Analysis on our preprocessed files. Writes the
results to new csv files.

'''


# Importing our processed data from the data directory as pandas dataframes
X_train = pd.read_csv("data/X_train.csv", header=None)
y_train = pd.read_csv("data/y_train.csv", header=None)
X_test  = pd.read_csv("data/X_test.csv", header=None)
y_test  = pd.read_csv("data/y_test.csv", header=None)

# Instantiate our PCA object. We are using an explained variance level of 0.99,
# meaning that we will keep the n most explanatory PCA variables that together
# explain 99% of the variability in the data.
pca = PCA(0.99)
pca.fit(X_train)


print("After applying PCA, 99% of variance can be explained by " + str(pca.n_components_) + " components.")
print("Using PCA to transform the training and testing data and saving as csv.")

# Applying PCA to our training and test sets
X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)

# Saving csv's of the PCA transformed data to the data directory
np.savetxt("data/X_train_pca.csv", X_train_pca, delimiter=',')
np.savetxt("data/X_test_pca.csv", X_test_pca, delimiter=',')

# Saving the PCA object in order to use it for classification later.
pk.dump(pca, open("data/pca.pkl", "wb"))
