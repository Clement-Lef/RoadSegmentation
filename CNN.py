from helpers import *
from Generator_class import *

import tensorflow as tf

import keras
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import initializers
from keras.constraints import max_norm
from keras.models import load_model, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras import backend as K

import numpy as np

def mean_f_score(actual,predicted):
    """Function used to compute the f1-score at each iteration"""
    actual = tf.cast(actual, "int32")
    predicted = tf.cast(tf.round(predicted), "int32")

    TP = tf.count_nonzero(predicted * actual) #True positive
    TN = tf.count_nonzero((predicted - 1) * (actual - 1)) #True Negative
    FP = tf.count_nonzero(predicted * (actual - 1)) #False Positive
    FN = tf.count_nonzero((predicted - 1) * actual) #False Negative
    preci = tf.reduce_sum(TP / (TP + FP)) #Precision
    recall = tf.reduce_sum(TP / (TP + FN)) #Recall
    f1 = 2 * preci * recall / (preci + recall) #F1 Score
    return f1

class CNN:
    """Class which contains a Convolutional Neural Network and the
    training method"""

    def __init__(self,num_epoch, learning_rate, PATCH_SIZE, WINDOW_SIZE, ORIGINAL_SIZE, BATCH_SIZE, num_iter_training = 5000):
        """Initialize all the necessary variable to train the CNN """
        self.PATCH_SIZE = PATCH_SIZE #Size of the patch (16*16)
        self.WINDOW_SIZE = WINDOW_SIZE #Total size of the window (64*64)
        self.ORIGINAL_SIZE = ORIGINAL_SIZE #Size of the input image (400*400)
        self.BATCH_SIZE = BATCH_SIZE #Size of each batch
        self.PATCH_PER_IMAGE = int((ORIGINAL_SIZE*ORIGINAL_SIZE)/(PATCH_SIZE*PATCH_SIZE)) #Number of patch per image
        self.pad_size = int((WINDOW_SIZE-PATCH_SIZE)/2) #Padding at each side of the image
        self.model = None #Model used
        self.num_epoch = num_epoch #Number of epoch
        self.learning_rate = learning_rate #Learning rate
        self.conv_base = None #VGG model
        self.num_iter_training = num_iter_training #Number of iterations per epoch


    def get_Model_from_scratch(self, image_heigth = 64,
              image_width = 64,
              image_depth = 3,
              kernel_size = 5,
              pool_size = 2,
              conv_depth_1 = 64,
              conv_depth_2 = 128,
              conv_depth_3 = 256,
              conv_depth_4 = 512,
              initial_bias = 0.001,
              drop_prob_1 = 0.25,
              drop_prob_2 = 0.5,
              drop_prob_3 = 0.5,
              drop_prob_4 = 0.5,
              drop_prob_hidden = 0.3,
              hidden_size = 256,
              regularization_factor = 1e-5,
              max_weight_norm = 4.0):
        """ Build the CNN modle in keras and returns it
        """

        # Input layer: takes a window from a sattelite image. Image depth is 3 (RGB)
        input_ = Input(shape=(image_heigth, image_width, image_depth))

        # Model is organized in 4 block, each containing 2 convolution layers. Each
        # convolution is followed by a BatchNormalization to reduce internal
        # covariate shift
        # Following the 2 convolutions, a max pooling is performed, submitted to
        # dropout

        # First block
        conv_11 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(input_)
        batch_N_11 = BatchNormalization()(conv_11)
        conv_12 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(batch_N_11)
        batch_N_12 = BatchNormalization()(conv_12)
        pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_N_12)
        drop_1 = Dropout(drop_prob_1)(pool_1)

        # Second block
        conv_21 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(drop_1)
        batch_N_21 = BatchNormalization()(conv_21)
        conv_22 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(batch_N_21)
        batch_N_22 = BatchNormalization()(conv_22)
        pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_N_22)
        drop_2 = Dropout(drop_prob_2)(pool_2)

        # Third block
        conv_31 = Convolution2D(conv_depth_3, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(drop_2)
        batch_N_31 = BatchNormalization()(conv_31)
        conv_32 = Convolution2D(conv_depth_3, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(batch_N_31)
        batch_N_32 = BatchNormalization()(conv_32)
        pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_N_32)
        drop_3 = Dropout(drop_prob_3)(pool_3)

        # Fourth block
        conv_41 = Convolution2D(conv_depth_4, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(drop_3)
        batch_N_41 = BatchNormalization()(conv_41)
        conv_42 = Convolution2D(conv_depth_4, (kernel_size, kernel_size),
                               bias_initializer = initializers.Constant(value=initial_bias),
                               padding='same', activation='relu',
                               kernel_constraint=max_norm(max_weight_norm))(batch_N_41)
        batch_N_42 = BatchNormalization()(conv_42)
        pool_4 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_N_42)
        drop_4 = Dropout(drop_prob_4)(pool_4)

        # Flatten the last activation cuboïde to get the flat representation of the features
        flat = Flatten()(drop_4)
        # Fully connected layer between the features to feed the classifier
        hidden = Dense(hidden_size, activation='relu', kernel_regularizer=l2(regularization_factor))(flat)
        drop_5 = Dropout(drop_prob_hidden)(hidden)

        # The last classifier, using softmax to get the probabilities of been "road" or not "road"
        out = Dense(2, activation='softmax')(drop_5)

        # get the model
        self.model = Model(inputs=input_, outputs=out)


    def create_model_vgg(self):
        """Download the ImageNet VGG16 model without the classifier layer
        and add our own classifier on top"""

        # Download VGG16 without the classifier layer
        self.conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(self.WINDOW_SIZE, self.WINDOW_SIZE, 3))


        self.model = Sequential()
        self.model.add(self.conv_base)
        self.model.add(layers.Flatten()) #Flatten the output of VGG16
        # Create new features from the last layer of VGG16
        self.model.add(layers.Dense(256, activation='relu',kernel_regularizer=l2(1e-4)))
        self.model.add(layers.Dropout(0.5)) #Dropout to avoid overfitting
        self.model.add(layers.Dense(2, activation='softmax')) #Classifier in two classes

        self.conv_base.trainable = False #Freeze the VGG16 model to keep the
        #ImageNet weights

    def create_model_vgg_from_scratch(self):
        """Download the ImageNet VGG16 model without the classifier layer
        and add our own classifier on top"""

        # Download VGG16 without the classifier layer
        self.conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(self.WINDOW_SIZE, self.WINDOW_SIZE, 3))


        self.model = Sequential()
        self.model.add(self.conv_base)
        self.model.add(layers.Flatten()) #Flatten the output of VGG16
        # Create new features from the last layer of VGG16
        self.model.add(layers.Dense(256, activation='relu',kernel_regularizer=l2(1e-4)))
        self.model.add(layers.Dropout(0.5)) #Dropout to avoid overfitting
        self.model.add(layers.Dense(2, activation='softmax')) #Classifier in two classes

        self.conv_base.trainable = True #Freeze the VGG16 model to keep the
        #ImageNet weights

    def update_model(self):
        """Function used to update the state of the last 3 layers of VGG16
        to unfreeze it and make it trainable for fine tuning"""
        self.model.trainable = True

        set_trainable = False
        for layer in self.model.layers:
            if layer.name == 'block5_conv1': #If first layer of last conv block
                set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False


    def train_model(self, X_train,Y_train,X_val,Y_val,history_name = 'history.csv'):
        """Function used to compile and train the CNN
        input :
                X_train : Training dataset
                Y_train : Training groundtruth
                X_val : Validation dataset
                Y_val : Validation groundtruth
                history_name : Name of the csv file which save the evolution of
                our training"""

        #Instantiate a Keras data generator which gives us a new batch at each step
        gen = Generator(self.PATCH_SIZE, self.WINDOW_SIZE, self.ORIGINAL_SIZE,  self.BATCH_SIZE)

        #Create the training dataset generator
        train_generator = gen.custom_generator_random(X_train,Y_train)
        #Create the validation dataset generator
        validation_generator = gen.custom_generator_random(X_val,Y_val)

        #Compile the model, we use the Adam Optimizer provided by Keras,
        #the loss is categorical_crossentropy and the metrics is a f1-score
        self.model.compile(optimizer=optimizers.Adam(self.learning_rate),
                        loss='categorical_crossentropy',
                        metrics=[mean_f_score])

        #### Definition of callbacks functions which are used after each epochs

        #Divide the learning rate by 5 if the validation mean_f_score did not improve
        #after 3 epochs
        reduce_lr = ReduceLROnPlateau(monitor='val_mean_f_score', factor=0.2,
                                  patience=3,verbose = 1,mode='max', min_lr=0)
        #Stop the training if the validation mean_f_score did not improve after
        #10 epochs
        early_stop = EarlyStopping(monitor='val_mean_f_score',
        min_delta=0.0001, patience=10, verbose=1, mode='max')

        #Save the best model after each epoch by comparing the validation
        #mean_f_score
        model_checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_mean_f_score', verbose=0, save_best_only=True,
         save_weights_only=False, mode='max', period=1)

        #Save the evolution of the training at each epoch (learning rate, losses, metrics)
        csv_logger = CSVLogger(history_name, separator=',', append=False)


        # Train the CNN, we put a try/except to stop the training by doing a CTRL+C
        try :
            history = self.model.fit_generator(
                train_generator, #Training generator which provides the training batches
                steps_per_epoch=self.num_iter_training, #Arbitrary number of batch per epoch
                epochs=self.num_epoch, #Number of epochs
                validation_data=validation_generator, #Provide the validation batches
                validation_steps=2000, #Arbitrary number of validation step
                callbacks=[reduce_lr, model_checkpoint, csv_logger, early_stop]) #Callbacks functions after each epoch
        except KeyboardInterrupt:
            pass

        # We load the best saved with the callbacks
        self.model = load_model('checkpoint.h5',custom_objects={'mean_f_score': mean_f_score})

    def update_learning_rate(self, lr):
        """Function used to manually update the learning rate of the training"""
        self.learning_rate = lr

    def save_model(self, name = 'model.h5'):
        """Function used to save the model to a file"""
        self.model.save(name)

    def load_best_model(self, name = 'model.h5'):
        """Function used to save load a model saved"""
        self.model = load_model(name,custom_objects={'mean_f_score': mean_f_score})


    def predict(self,img):
        """Function used to predict the label of a patch"""
        prediction = self.model.predict(img) # Predict the patch
        prediction = np.argmax(prediction,1) # Take the highest probability to define
        # if a patch is a road [1,0] or background [0,1]
        return prediction
