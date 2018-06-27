from helpers import *
from CNN import *
from PIL import Image


import numpy as np
np.random.seed(10) # Set the seed for reproductibility

PATCH_SIZE = 16 #Size of the patch
WINDOW_SIZE = 64 #Size of the context
ORIGINAL_SIZE = 400 #Size of the training images
BATCH_SIZE = 16 #Size of the batch
num_epoch = 40 #Number of epochs
learning_rate = 1e-4 #Initial learning rate
seed = 10 #Seed for reproductibility
pad_size = int((WINDOW_SIZE-PATCH_SIZE)/2) #Padding at each border


#Instantiate the CNN
New_CNN = CNN(num_epoch,learning_rate, PATCH_SIZE, WINDOW_SIZE, ORIGINAL_SIZE,  BATCH_SIZE)

print("")
save_stuff = ""
while not (save_stuff is "1" or save_stuff is "2" or save_stuff is "3" or save_stuff is "4"):
    print("Choose one : ")
    print("Predict the Kaggle result [1]")
    print("Train the Kaggle Model, VGG16 + classifier [2]")
    print("Train VGG16 from scratch [3]")
    print("Train the custom CNN from scratch [4]")
    save_stuff = input("Choose one of [1,2,3,4]")

if save_stuff == "1":

    #Load the model
    New_CNN.load_best_model('CNN_I_FineTuned.h5')

    #Predict the test set and create the submission file
    gen = Generator(PATCH_SIZE, WINDOW_SIZE, 608)
    make_predictions(New_CNN,gen,PATCH_SIZE,WINDOW_SIZE,pad_size)
elif save_stuff == "2":
    X_train, Y_train, X_val, Y_val = load_all_augmented_images()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    New_CNN.create_model_vgg()
    # First train the classifier
    New_CNN.train_model(X_train,Y_train,X_val,Y_val, 'history_no_fine_tune.csv')
    # Save the model
    New_CNN.save_model('NoFineTune.h5')
    #Load the model to be sure that we update the good one
    New_CNN.load_best_model('NoFineTune.h5')
    # Defreeze the last 3 layers of the VGG16
    New_CNN.update_model()
    #Small learning rates for training VGG and retrain last layers
    New_CNN.update_learning_rate(1e-6)
    New_CNN.train_model(X_train,Y_train,X_val,Y_val, 'history_fine_tune.csv')
    #Save the model
    New_CNN.save_model('CNN_I_FineTuned.h5')
    #Predict the test set and create the submission file
    gen = Generator(PATCH_SIZE, WINDOW_SIZE, 608)
    make_predictions(New_CNN,gen,PATCH_SIZE,WINDOW_SIZE,pad_size)

elif save_stuff == "3":
    X_train, Y_train, X_val, Y_val = load_all_augmented_images()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    New_CNN.create_model_vgg_from_scratch()
    # First train the classifier
    New_CNN.train_model(X_train,Y_train,X_val,Y_val, 'history_vgg_scratch.csv')
    # Save the model
    New_CNN.save_model('VGG_from_scratch.h5')
    #Load the model to be sure that we update the good one
    New_CNN.load_best_model('VGG_from_scratch.h5')
    #Predict the test set and create the submission file
    gen = Generator(PATCH_SIZE, WINDOW_SIZE, 608)
    make_predictions(New_CNN,gen,PATCH_SIZE,WINDOW_SIZE,pad_size)

elif save_stuff == "4":
    number_of_iter_training = 2500
    #Instantiate the CNN
    New_CNN = CNN(num_epoch,learning_rate, PATCH_SIZE, WINDOW_SIZE, ORIGINAL_SIZE,  BATCH_SIZE, number_of_iter_training)

    X_train, Y_train, X_val, Y_val = load_all_augmented_images()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    New_CNN.get_Model_from_scratch()
    # First train the classifier
    New_CNN.train_model(X_train,Y_train,X_val,Y_val, 'history_model_scratch.csv')
    # Save the model
    New_CNN.save_model('Model_from_scratch.h5')
    #Load the model to be sure that we update the good one
    New_CNN.load_best_model('Model_from_scratch.h5')
    #Predict the test set and create the submission file
    gen = Generator(PATCH_SIZE, WINDOW_SIZE, 608)
    make_predictions(New_CNN,gen,PATCH_SIZE,WINDOW_SIZE,pad_size)
