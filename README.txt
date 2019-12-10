

This code is part ofa semester project for CSE 4471 in the Fall semester of 2019.
We are the Antivirus group, consisting of

@author Jack Ryan


## Using EmberNet ##

This folder contains all of the files used to create and use EmberNet, a neural network that uses the
Endgame Malware BEnchmark for Research (Ember for short) data set to classify Microsoft PE files as
either safe or potentially malicious.


** Dependencies **

In order to use the EmberNet, you will python version >= 3.6 and will need to install a few libraries including:
    - Ember (instructions for installation at https://github.com/endgameinc/ember)
    - PyTorch
    - LIEF (dependency of Ember)(Version 0.9.0 only!)
    - LightGBM (dependency of Ember)
    - The usual suspects of machine learning (skLearn, numpy, pandas, etc. should be included in python3)


** The Data **

I have not provided the raw Ember data in this code base because it is quite large. I will instead provide the csv file
pre-computed by PreprocessData.py and PreprocessPCA.py so that you can bypass this tedious step of downloading,
converting, sampling and pre-processing the data.

    !! IMPORTANT: The pre-processed data files are too large for GitHub and I will have to email !!
    !! them to you if you wish to use them. Reach me at ryan.1369@osu.edu !!

If you are planning on creating the data set from scratch, you will also need to download the Ember2018
data set (available at https://github.com/endgameinc/ember) and add the folder labled "ember2018" to the data
directory.


** Training and Using the Model **

If you do not wish to download the entire 1.1 million sample Ember2018 data set, you must skip to step 2 (and can
optionally skip all the way to step 3).

Once all the dependencies are downloaded (and optionally the Ember dataset as well) and you have navigated to this
directory in your CLI there are 4 steps to using this model that must be carried out in order if you are starting
from scratch:

 1.) Pre-processing the data:

        to run: python3 PreprocessData.py train_size test_size

        train_size: number of samples to be in training data set
        test_size:  number of samples to be in validation/test data set

        This file will create the vectorized features from the ember2018 data set, sample
        this data, normalize the feature vectors and then output the data to the data directory
        in a csv format.

        !! This step is only necessary when starting from the raw Ember data !!

 2.) Principal Component Analysis:

        to run: python3 PreprocessPCA.py

        No arguments required. This file takes the output from step 1 and applies Principal Component Analysis
        to the data in order to obtain feature vectors that are both more separable with less dimensionality.
        The transformed and reduced feature vectors will also be output to the data directory in csv format.

 3.) Training the Model

        to run: python3 ModelTrainer.py <X Train> <y Train> <X Test> <y Test> <method>
                                        <epoch count> <batch size(used if method is "batch)">

            the first four arguments are the file locations of your train and test data
            method: the method to train the model with (sample, full or batch)
            epoch count: number of epochs used to train the model
            batch size: size of mini batches, only used if method is batch

        to run exactly as I have, enter:
        python3 ModelTrainer.py data/X_train_pca.csv data/y_train.csv data/X_test_pca.csv data/y_test.csv "batch" 1000000 128

        This is the meat of the software where we train the neural network using our processed data. While the model is
        training, the training error will be printed to the console every 100 epochs and the test accuracy plus confusion
        matrix will be printed to the console every 10,000 epochs. When the model is finished training, it will be saved
        to data/ModelParams.txt

        The model architecture I finalized on is a 4-layer deep multi-layer perceptron with layers of size
        1205|350|50|300|Output(1). This type of architecture that "squeezes" the data into a few nodes in
        the middle and then re-expands into more nodes is sometimes used for file compression which may
        be in some way related to why this architecture was the most useful for classifying files, but I am
        unsure to be honest.

        To get our final results I used mini-batch gradient descent (method = "batch") with a batch size of
        128 over 1,000,000 epochs. I also used a learning rate of 0.005 and a momentum factor of 0.90.

 4.) Classification:

     to run: python3 ClassifyPE.py /path/to/PE_File

     Only command line argument required is the path to the executable file to be classified.

     This is file will quickly load the model that has been trained, read in the PE file to be classified,
     convert it into a feature vector and then feed this vector to EmberNet, which will classify the file as
     either safe or malicious and then print the result to the console.


## Results ##

Using the 1205|350|50|300|Output(1) architecture, with mini-batch gradient descent, a batch size of 128, a learning
rate of 0.005 and a momentum factor of 0.9 and training with these parameters over one million epochs we created a model
that was more effective than just a few days ago when we presented to the class. Our updated statistics are:

    - Training Accuracy: 99.45%
    - Testing Accuracy:  81.35%
    - False Positive Ratio: 3.00%

While our model did not generalize perfectly, we think these are great results overall. A testing accuracy of 81.35%
may not be industry standard but it is a good indication we are headed in the right direction. Because of hardware
limitations, we could only use roughly 1% of all the available data from Ember. We believe that if we used a more
significant portion of the available data we would see proportional increases in testing performance. We would also
have liked to try a deeper network, something like ResNet, but hardware limitations made this impossible as well.

We are very happy with our training accuracy because this shows that at the very least, this model is extremely reliable
at classifying files that are the same or similar to files it has already seen. This means that if we were to update the
training data with newly discovered malware samples periodically, we would be able to reliably detect all malware known
to our data set. This would be like a form of machine learning driven Signature Detection that would be harder for bad
actors to avoid without significantly altering the contents of their malware files. At the very least it would perform
much faster than a traditional signature search and be tougher to bypass.

Additionally our false positive ratio is very low. This is important in malware detection because most files we download
are safe and we do not want to be calling a large portion of these safe files unsafe. In general, this model is a bit
biased towards classifying files as non-malicious, which is not a bad thing! To make up for this bias we could report the
probability that a file is malicious to the user instead of a binary classification. This would allow the user to make
a more educated and personalized decision about removing the file or not. This would be very easy to do because the output
node of our model is a probabilistic one (fed with a sigmoid), and we could just report the un-rounded value of this node
after feeding it a sample.



## Citations ##

H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

@ARTICLE{2018arXiv180404637A,
  author = {{Anderson}, H.~S. and {Roth}, P.},
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.04637},
  primaryClass = "cs.CR",
  keywords = {Computer Science - Cryptography and Security},
  year = 2018,
  month = apr,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
