import numpy as np
from helpers import *

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

class Generator:
    """Class used to create a generator which will give us batches to feed
    the CNN"""
    def __init__(self,PATCH_SIZE, WINDOW_SIZE, ORIGINAL_SIZE, BATCH_SIZE = 16):
        """Initialization of the class"""
        self.PATCH_SIZE = PATCH_SIZE #Size of the patch (16*16)
        self.WINDOW_SIZE = WINDOW_SIZE #Size of the context window (64*64)
        self.CONTEXT_SIZE = int(WINDOW_SIZE/2 - PATCH_SIZE/2) #Size of the padding
        self.ORIGINAL_SIZE = ORIGINAL_SIZE #Original size of the image (400*400)
        self.BATCH_SIZE = BATCH_SIZE #Batch Size
        self.PATCH_PER_IMAGE = int((ORIGINAL_SIZE*ORIGINAL_SIZE)/(PATCH_SIZE*PATCH_SIZE)) #Number of patch per images


    def get_window(self,images_mirrored, patch_number):
        """Function used for the prediction to get a context window for a patch"""

        #Get the image number
        img_number = int(patch_number/self.PATCH_PER_IMAGE)
        #Get the relative patch number for the image studied
        relative_patch_number = int(patch_number-img_number*self.PATCH_PER_IMAGE)
        # Compute the position of the patch
        patch_per_row = int(self.ORIGINAL_SIZE/self.PATCH_SIZE)
        patch_row = int(relative_patch_number/patch_per_row)
        patch_column = int(relative_patch_number - patch_row*patch_per_row)
        # Compute the center of the patch
        row_center = int(patch_row*self.PATCH_SIZE + self.WINDOW_SIZE/2)
        column_center = int(patch_column*self.PATCH_SIZE  + self.WINDOW_SIZE/2)

        #Compute the full window
        window_image = images_mirrored[img_number][int(row_center-self.WINDOW_SIZE/2):int(row_center+self.WINDOW_SIZE/2),
                                     int(column_center-self.WINDOW_SIZE/2):int(column_center+self.WINDOW_SIZE/2),:]

        return window_image


    def custom_generator_random(self, X,Y):
        """Create a batch of window from the training input X/Y :
        X : Training Image, Y : Validation Image"""
        while True:
            #Initialize the batch
            batch_X = np.zeros((self.BATCH_SIZE, self.WINDOW_SIZE, self.WINDOW_SIZE,3))
            batch_label = np.zeros((self.BATCH_SIZE,1))

            #Fill the batch with the window
            for i in range(batch_X.shape[0]):
                #Compute a random number chosing a random image
                index = np.random.choice(X.shape[0])
                #Compute a window and the label that goes with it
                window, label = self.get_random_window_and_label(X[index],
                                                               Y[index])
                #Fill the batches
                batch_X[i] = window
                batch_label[i] = label
            # Return the batch and the label by converting the label 0 or 1 to
            # [1,0] (road) or [0,1] background
            yield batch_X, np_utils.to_categorical(batch_label, 2)

    def patch_to_label(self,patch, label_threshold = 0.25):
        """ Assign a label to a patch
            Returns: label 1 if the proportion of elements in the patch is superior
            to foreground_threshold, 0 otherwise
        """
        df = np.mean(patch)
        if df >= label_threshold:
            return 1
        else:
            return 0

    def mirror_padding(self,image, heigth, width):
        """ Pads the image using symmetric boundary conditions
            heigth : number of elements to pad on the right and bottom edges
            width  : number of elements to pad on the top and left edges
            Returns: Padded image
        """

        newShape = list(image.shape)
        newShape[0] += 2*heigth
        newShape[1] += 2*width

        newImage = np.empty(newShape)

        #Pad the groundtruth
        if len(image.shape)<3:
            newImage = np.pad(image[:,:], (heigth, width), 'reflect')
        else: #Pad the satellite image with RGB channel
            newImage = np.asarray([np.pad(image[:,:,i], (heigth, width), 'reflect')\
                                  for i in range(image.shape[2])])
            newImage = np.moveaxis(newImage, 0, -1)

        return newImage




    def get_random_window_and_label(self,image, gt_image,
                                   road_classification_threshold = 0.25):
        """ Get a random patch from image, and its label from the groundtruth image
            gt_image.
            A patch is defined by a subpart of the image of size patch_heigth x patch_width
            The label is the groundtruth value for a patch
            If context values are non 0, the patch is extended to the size
            patch_heigth + 2 *context_heigth   x   patch_width + 2 * context_width
            This is done in order to get information about the surronding of the patch.
            However, the label is still computed only on the smaller patch, so that the
            additional information does not impact the label.
        """

        padImage = self.mirror_padding(image, self.CONTEXT_SIZE, self.CONTEXT_SIZE)

        patch_corner_h = np.random.randint(self.CONTEXT_SIZE, image.shape[0] + self.CONTEXT_SIZE - self.PATCH_SIZE)
        patch_corner_w = np.random.randint(self.CONTEXT_SIZE, image.shape[1] + self.CONTEXT_SIZE - self.PATCH_SIZE)

        window = padImage[patch_corner_h-self.CONTEXT_SIZE: patch_corner_h + self.PATCH_SIZE + self.CONTEXT_SIZE,
                         patch_corner_w-self.CONTEXT_SIZE : patch_corner_w + self.PATCH_SIZE  + self.CONTEXT_SIZE ]

        gt_patch = gt_image[patch_corner_h - self.CONTEXT_SIZE : patch_corner_h + self.PATCH_SIZE - self.CONTEXT_SIZE,
                                        patch_corner_w - self.CONTEXT_SIZE: patch_corner_w + self.PATCH_SIZE - self.CONTEXT_SIZE]


        label = self.patch_to_label(gt_patch,road_classification_threshold)
        return window, label
