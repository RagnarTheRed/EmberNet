import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import random
import sys

'''
    This file is used to train and save the EmberNet model. Before using this
    file to train the model, use PreprocessData.py and PreprocessPCA.py to 
    create the processed data that is required for maximum performance.
    
    After this model is trained and saved to data/ModelParams.txt, ClassifyPE.py
    can be used to classify new Microsoft PE files.
    
    *** some notes ***
    
    - The model architecture is configured in the shape that I had the most success
      with. Changing the architecture can be done by changing the values of the
      hid_dim vector parameter in the MLP4 class, however if you do so results 
      may very drastically.
      
    - If there is too much activity on the console for your liking, change the 
      value of the modulos in the training loop that you choose to use.
      

    @author Jack Ryan
'''


"""

Defines the architecture of our network as a 2-layer MLP, using sigmoid
as the activation function for both layers. There is a single output node
that is probabilistic.

"""
class MLP4(nn.Module):
    def __init__(self, input_dim, hid_dim=[350, 50, 300], num_classes=1):
        super(MLP4, self).__init__()
        # These objects represent the layers of the network
        self.hiddenLayer1 = nn.Linear(input_dim, hid_dim[0])
        self.hiddenLayer2 = nn.Linear(hid_dim[0], hid_dim[1])
        self.hiddenLayer3 = nn.Linear(hid_dim[1], hid_dim[2])
        self.outputLayer = nn.Linear(hid_dim[2], num_classes)

        # These objects are used to perform batch normalization
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim[0])
        self.bn2 = nn.BatchNorm1d(hid_dim[1])


    def forward(self, x):
        x = self.hiddenLayer1(self.bn0(x))
        x = torch.tanh(x)
        x = self.hiddenLayer2(self.bn1(x))
        x = torch.tanh(x)
        x = self.hiddenLayer3(self.bn2(x))
        x = torch.tanh(x)
        x = self.outputLayer(x)
        x = torch.sigmoid(x)
        return x


"""
Method used to initialize the weights of our network
"""
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1)


def report_confusion_matrix(num_classes, target, pred):
    conf_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(target.long(), pred.long()):
        conf_matrix[t, p] += 1

    true_pos = conf_matrix.diag()

    for c in range(num_classes):
        idx = torch.ones(num_classes).bool()
        idx[c] = 0

        true_neg = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        false_pos = conf_matrix[idx, c].sum()
        false_neg = conf_matrix[c, idx].sum()

        print('Class {}\nTrue Pos: {}, True Neg.: {}, False Pos: {}, False Neg {}'.format(
            c, true_pos[c], true_neg, false_pos, false_neg))



def main():

    # File locations for our data
    train_features = sys.argv[1]
    train_labels   = sys.argv[2]
    test_features  = sys.argv[3]
    test_labels    = sys.argv[4]

    # Training method to be used.
    method = str(sys.argv[5])

    # Number of epochs to train our model. 5000+ recommended for best results.
    epochs = int(sys.argv[6])

    # Size of mini-batches if batch training method is chosen.
    batch_size = None
    if method == "batch":
        batch_size = int(sys.argv[7])


    """
    Setup:
        Loading data files given by the command line.
        Initializing our neural network model. 
    """

    # First, load in the csv files that have been prepared in pre-processing.
    # I personally used the PCA transformed feature data.
    X_train = pd.read_csv(train_features, header=None)
    y_train = pd.read_csv(train_labels, header=None)
    X_test  = pd.read_csv(test_features, header=None)
    y_test  = pd.read_csv(test_labels, header=None)

    # Convert the data into Tensors that can be used by PyTorch.
    trainX = torch.Tensor(X_train.values)
    trainY = torch.Tensor(y_train.values)
    testX  = torch.Tensor(X_test.values)
    testY  = torch.Tensor(y_test.values)

    # Initialize our network with an input layer to match our given data.
    EmberNet = MLP4(input_dim=X_train.shape[1])
    weights_init(EmberNet)

    # We will use Mean Square Error as our loss function and regular Stochastic
    # Gradient Descent to optimize.
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(EmberNet.parameters(), lr=0.005, momentum=0.90)
    # Learning rate ans momentum are fixed at these values since they are
    # good values for most problems, which applies here.


    """
    Training:
        We will train the model using one of three methods
        according to the choice given from the command line.  
    """

    # This method trains the model one sample at a time for every sample in the
    # training set at each epoch. This is a very inefficient method but usually
    # produces the most generalizable model.
    if method == "sample":
        steps = trainX.size(0)
        for i in range(epochs):
            for j in range(steps):

                x_var = Variable(trainX[j], requires_grad=False)
                y_var = Variable(trainY[j], requires_grad=False)

                optimizer.zero_grad()

                y_hat = EmberNet(x_var)

                loss = loss_func.forward(y_hat, y_var)
                loss.backward()

                optimizer.step()

            if i % 5 == 0:
                print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))


    # This method computes the gradient across all training samples as a single
    # batch at each epoch. It is the most efficient method but usually results
    # in the model being over-fit compared to other methods.
    elif method == "full":
        for i in range(epochs):

            optimizer.zero_grad()

            y_hat = EmberNet(trainX)
            loss = loss_func(y_hat, trainY)
            loss.backward()

            optimizer.step()

            if i % 5 == 0:
                print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))


    # This method trains the model in batches of a specified size at each epoch.
    # The batches are created at each epoch by randomly sampling from the
    # training data. This is the most balanced method between efficiency and
    # producing a generalizable model.
    elif method == "batch":

        for i in range(epochs):

            batch_sampler = random.sample(range(trainX.size(0)), batch_size)
            batchX = torch.Tensor(trainX.numpy()[batch_sampler])
            batchY = torch.Tensor(trainY.numpy()[batch_sampler])


            optimizer.zero_grad()
            y_hat = EmberNet(batchX)

            loss = loss_func(y_hat, batchY)
            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))

            if i % 10000 == 0:

                test_prob = EmberNet(testX)
                test_class = torch.Tensor(
                    np.where(test_prob.detach().numpy() < 0.5, 0, 1))
                test_correct = torch.sum((testY == test_class).long())
                test_acc = (test_correct * 100.0 / testY.shape[0])
                print("#################################################################")
                print("Test Accuracy: ", test_acc.data.numpy(), "%")
                report_confusion_matrix(2, testY, test_class)
                print("#################################################################")


    EmberNet.eval()
    # Our output node is probabilistic, so we must use this probability to
    # predict a class for each sample in order to get our complete results.
    test_prob = EmberNet(testX)
    train_prob = EmberNet(trainX)


    # A step function is used to determine the cutoff between classes. The
    # logical choice was 0.50 but a higher or lower threshold could be used
    # depending on what kind of classification tolerance is warranted.
    test_class = torch.Tensor(np.where(test_prob.detach().numpy() < 0.5, 0, 1))
    train_class = torch.Tensor(np.where(train_prob.detach().numpy() < 0.5, 0, 1))

    # Calculate and report a basic accuracy score.
    test_correct = torch.sum((testY == test_class).long())
    test_acc = (test_correct * 100.0 / testY.shape[0])

    train_correct = torch.sum((trainY == train_class).long())
    train_acc = (train_correct * 100.0 / trainY.shape[0])

    print("Train Accuracy: ",  train_acc.data.numpy(), "%")
    print("Test Accuracy: ", test_acc.data.numpy(), "%")

    print("Confusion Matrix of Test Results: ")
    report_confusion_matrix(2, testY, test_class)
    print()
    print("Confusion Matrix of Training Results: ")
    report_confusion_matrix(2, trainY, train_class)

    torch.save(EmberNet.state_dict(), "data/ModelParams.txt")



if __name__ == '__main__':
    main()