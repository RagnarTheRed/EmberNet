import torch
import ember
import numpy as np
from sklearn import preprocessing
import pickle as pk
import sys
from ModelTrainer import MLP4

'''
Performs classification on raw PE files, and outputs the decision to the console.
'''

pe_file_path = sys.argv[1]

# Load the previously trained EmberNet from the save file
EmberNet = MLP4(input_dim=1205)
EmberNet.load_state_dict(torch.load("data/ModelParams.txt"))
EmberNet.eval()

# Open the PE file and extract the vectorized features
file_data = open(pe_file_path, "rb").read()
extractor = ember.features.PEFeatureExtractor(feature_version=2)
features = np.array(extractor.feature_vector(file_data), dtype=np.float32).reshape(1, -1)

# Normalize the feature vector
min_max_scaler = preprocessing.MinMaxScaler()
norm_features = min_max_scaler.fit_transform(features)

# Transform with PCA
pca_reload = pk.load(open("data/pca.pkl", "rb"))
pca_features = pca_reload.transform(norm_features)
# Cast as a tensor object
cleaned_features = torch.Tensor(pca_features)

# Classify and label!
class_prob = EmberNet(cleaned_features)
label = np.where(class_prob.detach().numpy() < 0.5, 0, 1)
if label < 1:
    print("This file is likely to be safe!")
else:
    print("WARNING: This file may be unsafe! It is recommended that this file is deleted.")
