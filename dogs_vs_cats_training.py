import logging
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
import math
from models.cnn_2conv import create_cnn_2conv_model
from models.cnn_2conv_fc2 import create_cnn_2conv_fc2_model
from models.cnn_3conv_fc1 import create_cnn_3conv_fc1_model
from models.cnn_3conv_fc2 import create_cnn_3conv_fc2_model
from models.cnn_3conv_fc3_new import create_cnn_3conv_fc3_model
from models.cnn_3conv_fc2_sigmoid import create_cnn_3conv_fc2_sigmoid_model
from models.cnn_3conv_fc1_l2reg import create_cnn_3conv_fc1_l2reg_model
from models.cnn_vgg16 import create_cnn_vgg16_model
from random import shuffle

np.random.seed(1337)
"""
64 * 64 for VGG16
224 * 224 for GoogleNet
64 * 64 for initial testing
"""
WIDTH = 224
HEIGHT = 224


def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def convert_image_to_data(image, WIDTH, HEIGHT):
    image_resized = Image.open(image).resize((WIDTH, HEIGHT))
    image_array = np.array(image_resized)
    #image_array = np.array(image_resized).T
    return image_array

def process_train_test_data(X_train, X_test, Y_train, Y_test):
    
    #Normalize the pixel data
    X_train = X_train/255
    X_test = X_test/255
    
    #Reshape Y_train and Y_test so as to avoid empty second dimenstion
    Y_train = (Y_train.reshape((1,Y_train.shape[0]))).T
    Y_test = (Y_test.reshape((1,Y_test.shape[0]))).T
    
    #Convert Y_train and Y_test to one_hot mode for easy softmax classification
    Y_train = convert_to_one_hot(Y_train, 2).T
    Y_test = convert_to_one_hot(Y_test, 2).T

    return X_train, X_test, Y_train, Y_test

def shuffle_in_unison(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(len(a))
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def create_train_test_data(WIDTH, HEIGHT):

    #1. Pick images with cat.1* and dog.1*
    #cat1_files = glob.glob("data/train/cat.1*")
    #dog1_files = glob.glob("data/train/dog.1*")
    
    # Restrict cat and dog files here for testing
    #cat1_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat1_files]
    #dog1_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog1_files]

    #2. Pick images with cat.2* and dog.2*
    cat2_files = glob.glob("data/train/cat.2*")
    dog2_files = glob.glob("data/train/dog.2*")
    
    # Restrict cat and dog files here for testing
    cat2_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat2_files]
    dog2_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog2_files]

    #3. Pick images with cat.3* and dog.3*
    #cat3_files = glob.glob("data/train/cat.3*")
    #dog3_files = glob.glob("data/train/dog.3*")
    
    # Restrict cat and dog files here for testing
    #cat3_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat3_files]
    #dog3_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog3_files]
   
    #4. Pick images with cat.4* and dog.4*
    #cat4_files = glob.glob("data/train/cat.4*")
    #dog4_files = glob.glob("data/train/dog.4*")
    
    # Restrict cat and dog files here for testing
    #cat4_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat4_files]
    #dog4_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog4_files]
   

    #5. Pick images with cat.5* and dog.5*
    #cat5_files = glob.glob("data/train/cat.5*")
    #dog5_files = glob.glob("data/train/dog.5*")
    
    # Restrict cat and dog files here for testing
    #cat5_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat5_files]
    #dog5_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog5_files]
   

    #6. Pick images with cat.6* and dog.6*
    #cat6_files = glob.glob("data/train/cat.6*")
    #dog6_files = glob.glob("data/train/dog.6*")
    
    # Restrict cat and dog files here for testing
    #cat6_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat6_files]
    #dog6_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog6_files]
   

    #7. Pick images with cat.7* and dog.7*
    #cat7_files = glob.glob("data/train/cat.7*")
    #dog7_files = glob.glob("data/train/dog.7*")
    
    # Restrict cat and dog files here for testing
    #cat7_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat7_files]
    #dog7_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog7_files]
   

    #8. Pick images with cat.8* and dog.8*
    #cat8_files = glob.glob("data/train/cat.8*")
    #dog8_files = glob.glob("data/train/dog.8*")
    
    # Restrict cat and dog files here for testing
    #cat8_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat8_files]
    #dog8_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog8_files]
   

    #9. Pick images with cat.9* and dog.9*
    #cat9_files = glob.glob("data/train/cat.9*")
    #dog9_files = glob.glob("data/train/dog.9*")
    
    # Restrict cat and dog files here for testing
    #cat9_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat9_files]
    #dog9_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog9_files]
   
    cat_list = cat2_list
    dog_list = dog2_list

    #cat_list = cat1_list + cat2_list + cat3_list + cat4_list + cat5_list + cat6_list + cat7_list + cat8_list + cat9_list
    #dog_list = dog1_list + dog2_list + dog3_list + dog4_list + dog5_list + dog6_list + dog7_list + dog8_list + dog9_list
    
    y_cat = np.zeros(len(cat_list),dtype=int)
    y_dog = np.ones(len(dog_list),dtype=int)

    X = np.concatenate([cat_list, dog_list])
    Y = np.concatenate([y_cat, y_dog])

    X,Y = shuffle_in_unison(X,Y)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1, stratify=Y)
    
    return X_train, X_test, Y_train, Y_test




def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


### Model starts here
if __name__ == "__main__":
    logger = create_logger()

    logger.info("Create train and test dataset.")
    X_train, X_test, Y_train, Y_test = create_train_test_data(WIDTH, HEIGHT)
    X_train, X_test, Y_train, Y_test = process_train_test_data(X_train, X_test, Y_train, Y_test)

    conv_layers = {}
    logger.info("Shape for X_train " + str(X_train.shape))
    logger.info("Shape for X_test " + str(X_test.shape))
    logger.info("Shape for Y_train " + str(Y_train.shape))
    logger.info("Shape for Y_test " + str(Y_test.shape))
    logger.info("Number of training examples m= " + str(X_train.shape[0]))

   
    #Develop the model
    #ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    minibatch_size = 64
    num_epochs = 10
    

    #optimizer,cost,X,Y,Z = create_cnn_2conv_model(n_H0,n_W0,n_C0,n_y)
    #optimizer,cost,X,Y,Z = create_cnn_2conv_fc2_model(n_H0,n_W0,n_C0,n_y)
    #optimizer,cost,X,Y,Z = create_cnn_3conv_fc1_model(n_H0,n_W0,n_C0,n_y)
    #optimizer,cost,X,Y,Z = create_cnn_3conv_fc2_model(n_H0,n_W0,n_C0,n_y)
    #optimizer,cost,X,Y,Z = create_cnn_3conv_fc3_model(n_H0,n_W0,n_C0,n_y)
    optimizer,cost,X,Y,Z = create_cnn_vgg16_model(n_H0,n_W0,n_C0,n_y)
    #optimizer,cost,X,Y,Z = create_cnn_3conv_fc2_sigmoid_model(n_H0,n_W0,n_C0,n_y)
    #optimizer,cost,X,Y,Z = create_cnn_3conv_fc1_l2reg_model(n_H0,n_W0,n_C0,n_y)

    init = tf.global_variables_initializer()


   # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            #print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        #plt.plot(np.squeeze(costs))
        #plt.ylabel('cost')
        #plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)




    #logger.info("Create the model.")
    # model = modified_vgg_16l(WIDTH, HEIGHT)
    #model = modified_googlenet(WIDTH, HEIGHT)

    #logger.info("Train the model.")
    # flag=False if no GoogleNet
    #trained_model = train_model(model, X_train, Y_train, flag=True)

    #logger.info("Evaluate the model.")
    #accuracy_score = evaluate_model(trained_model, X_test, Y_test)
    #logger.info("The accurarcy is " + str(accuracy_score))

    #logger.info("Save model")
    #trained_model.save("dogs_vs_cats_model_VGG-like_convnet.h5")
    #trained_model.save("dogs_vs_cats_model_googlenet.h5")
