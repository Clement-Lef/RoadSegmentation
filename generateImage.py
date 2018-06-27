import numpy as np
np.random.seed(10) #Set the seed

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

from helpers import load_all_images, split_data

# Chose the number of augmented images to create
# (There are 100 images in the provided set)
NumberOfAugmentedImages = 1500
# Proportion of images to keep for validation purposes (from the dataset without the test set)
training_ratio = 0.9


training_dir = 'training'

def generateAugmentedImage(image_path, groundtruth_path,
                           aug_image_path, aug_groundtruth_path,
                           val_image_path, val_groundtruth_path,
                           howMany):
    """ Generate new images using rotation, shear, zoom, etc transformation
        Load the images from the disk (image_path) and writes the augmented images
        to write path
    """

    # Check if the directories exists, if not: create them
    if not os.path.exists(aug_image_path):
        os.makedirs(aug_image_path)
    if not os.path.exists(aug_groundtruth_path):
        os.makedirs(aug_groundtruth_path)
    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    if not os.path.exists(val_groundtruth_path):
        os.makedirs(val_groundtruth_path)

    tr_images, gt_images = load_all_images(100,'training/')
    tr_images = np.array(tr_images)
    gt_images = np.array(gt_images)

    Im_tr, GT_tr, Im_val, GT_val = split_data(tr_images, gt_images, training_ratio,10)

    # Create 2 generator with same seed for training and groundtruth transformation
    data_gen_args = dict(featurewise_center=False,
                         samplewise_center=False,
                         featurewise_std_normalization=False,
                         samplewise_std_normalization=False,
                         zca_whitening=False,
                         rotation_range=90,
                         width_shift_range=0.,
                         height_shift_range=0.,
                         shear_range=0.3,
                         zoom_range=0.,
                         fill_mode='reflect',
                         horizontal_flip=True,
                         vertical_flip=True)


    image_gen = ImageDataGenerator(**data_gen_args)
    gt_gen = ImageDataGenerator(**data_gen_args)


    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = np.random.randint(2**32 - 1)

     # a hack to get the transformed images without given a vector of labels
    Y_trash = np.ones(Im_tr.shape[0])


    iter_tr = image_gen.flow(Im_tr,
                             Y_trash,
                             batch_size = 1,
                             shuffle = None,
                             seed=seed,
                             save_to_dir=aug_image_path,
                             save_prefix='aug', save_format='png')

    iter_gt = gt_gen.flow(GT_tr[...,np.newaxis],
                          Y_trash,
                          batch_size = 1,
                          shuffle = None,
                          seed=seed,
                          save_to_dir=aug_groundtruth_path,
                          save_prefix='aug', save_format='png')


    # Save the augmented images
    for i in range(howMany):
        if i % 100 == 0:
            print("Saving image " + str(i) + "...")
        iter_tr.next()
        iter_gt.next()

    # Save the validation images
    for i in range(Im_val.shape[0]):
        Image.fromarray(((255*Im_val[i]).astype(np.uint8))).save(os.path.normpath(val_image_path) + '/val_Sat_Image_' + str(i) + '.png')
        Image.fromarray(((255*GT_val[i]).astype(np.uint8))).save(os.path.normpath(val_groundtruth_path) + '/val_Sat_Image_' + str(i) + '.png')

generateAugmentedImage(training_dir+'/images',
                       training_dir+'/groundtruth',
                       'training_augmented/images_augmented',
                       'groundtruth_augmented/groundtruth_augmented',
                       'images_validation/images_validation',
                       'groundtruth_validation/groundtruth_validation',
                       NumberOfAugmentedImages)
