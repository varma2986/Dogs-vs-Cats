import logging
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
import math




def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W1a = parameters['W1a']
    W2a = parameters['W2a']
    W3a = parameters['W3a']
    W4a = parameters['W4a']
   
    #CONV2D*2:stride of 1, padding 'SAME"
    Z1a = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1a = tf.nn.relu(Z1a)
    Z1 = tf.nn.conv2d(A1a, W1a, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    
    # MAXPOOL: window 2x2 stride 2, padding 'VALID'
    P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')


    #CONV2D*2:stride of 1, padding 'SAME"
    Z2a = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2a = tf.nn.relu(Z2a)
    Z2 = tf.nn.conv2d(A2a, W2a, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    
    # MAXPOOL: window 2x2 stride 2, padding 'VALID'
    P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')


    #CONV2D*3:stride of 1, padding 'SAME"
    Z3a = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    A3a = tf.nn.relu(Z3a)
    Z3b = tf.nn.conv2d(A3a, W3a, strides=[1, 1, 1, 1], padding='SAME')
    A3b = tf.nn.relu(Z3b)
    Z3 = tf.nn.conv2d(A3b, W3a, strides=[1, 1, 1, 1], padding='SAME')
    A3 = tf.nn.relu(Z3)
    
    # MAXPOOL: window 2x2 stride 2, padding 'VALID'
    P3 = tf.nn.max_pool(A3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')


    #CONV2D*3:stride of 1, padding 'SAME"
    Z4a = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    A4a = tf.nn.relu(Z4a)
    Z4b = tf.nn.conv2d(A4a, W4a, strides=[1, 1, 1, 1], padding='SAME')
    A4b = tf.nn.relu(Z4b)
    Z4 = tf.nn.conv2d(A4b, W4a, strides=[1, 1, 1, 1], padding='SAME')
    A4 = tf.nn.relu(Z4)
    
    # MAXPOOL: window 2x2 stride 2, padding 'VALID'
    P4 = tf.nn.max_pool(A4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')


    #CONV2D*3:stride of 1, padding 'SAME"
    Z5a = tf.nn.conv2d(P4, W5, strides=[1, 1, 1, 1], padding='SAME')
    A5a = tf.nn.relu(Z5a)
    Z5b = tf.nn.conv2d(A5a, W5, strides=[1, 1, 1, 1], padding='SAME')
    A5b = tf.nn.relu(Z5b)
    Z5 = tf.nn.conv2d(A5b, W5, strides=[1, 1, 1, 1], padding='SAME')
    A5 = tf.nn.relu(Z5)
    
    # MAXPOOL: window 2x2 stride 2, padding 'VALID'
    P5 = tf.nn.max_pool(A5, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')

    # FLATTEN
    P = tf.contrib.layers.flatten(P5)
    FC2 = tf.contrib.layers.fully_connected(P, 4096, activation_fn=None)
    FC3 = tf.contrib.layers.fully_connected(FC2, 4096, activation_fn=None)
    FC4 = tf.contrib.layers.fully_connected(FC3, 2, activation_fn=None)

    ####VGG16 ends here

    return FC4

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : []
                        W2 : []
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W1a = tf.get_variable("W1a", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2a = tf.get_variable("W2a", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3a = tf.get_variable("W3a", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4a = tf.get_variable("W4a", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W5 = tf.get_variable("W5", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W1a": W1a,
                  "W2a": W2a,
                  "W3a": W3a,
                  "W4a": W4a,
                  "W5": W5}
    
    return parameters

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    
    return X, Y

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    return cost

def create_cnn_vgg16_model(n_H0,n_W0, n_C0, n_y):
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

    return optimizer,cost,X,Y,Z
