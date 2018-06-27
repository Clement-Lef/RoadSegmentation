import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from mask_to_submission import *

test_dir = 'test_set_images'

def load_image(infilename):
    """Function to load an image"""
    data = mpimg.imread(infilename)
    return data

def load_all_images(number_images, root_dir):
    # Loaded a set of images
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(number_images, len(files)) # Load maximum 100 images
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    return imgs,gt_imgs


def load_all_augmented_images(number_images = 1500):
    """Load all the augmented images """

    #Load the satellite augmented training images
    image_dir = 'training_augmented/images_augmented/'
    files = os.listdir(image_dir) #Get the filenames of the images
    n = min(number_images, len(files)) # Load maximum 1500 images
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    #Load the groundtruth augmented training
    gt_dir = 'groundtruth_augmented/groundtruth_augmented/'
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    #Load the validation satellite images
    image_dir = 'images_validation/images_validation/'
    files = os.listdir(image_dir) #Get the filenames of the images
    n = min(number_images, len(files)) # Load maximum 1500 images
    print("Loading " + str(n) + " images")
    imgs_val = [load_image(image_dir + files[i]) for i in range(n)]

    #Load the validation groundtruth images
    gt_dir = 'groundtruth_validation/groundtruth_validation/'
    print("Loading " + str(n) + " images") #Get the filenames of the images
    gt_imgs_val = [load_image(gt_dir + files[i]) for i in range(n)]

    return imgs,gt_imgs, imgs_val, gt_imgs_val


def mirror_image(image, pad_size):
    """Mirror Pad one image"""
    return np.pad(image,((pad_size,pad_size),(pad_size,pad_size),(0,0)),mode = 'reflect')

def mirror_all_images(image_list, pad_size):
    """Mirror Pad all the images"""
    mirrored_images = [mirror_image(image_list[i],pad_size) for i in range(len(image_list))]
    return mirrored_images




def img_float_to_uint8(img):
    """Convert float numpy array to uint8 for saveing """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def label_to_img(imgwidth, imgheight, w, h, labels):
    """Convert the patch label to an image patch"""
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[i:i+w, j:j+h] = labels[idx]
            idx = idx + 1
    return im

def split_data(x, y, ratio, seed=None):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # Compute a permutation of the dataset
    indices = np.random.permutation(len(y))
    training_size = int(round(len(y)*ratio))
    training_index = indices[: training_size]
    test_index = indices[training_size:]

    x_training = x[training_index]
    y_training = y[training_index]
    x_test = x[test_index]
    y_test = y[test_index]
    return x_training, y_training, x_test, y_test


def predict_image(img,CNN, patch_size, window_size, pad_size, generator):
    """Function used to compute the prediction of an image"""

    height = img.shape[0]
    width = img.shape[1]
    number_of_patches = (height*width)/(patch_size*patch_size)

    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    #Mirror pad the image to use our context window
    img = mirror_all_images(img,pad_size)
    img = np.array(img)

    #Compute all the window to predict
    img_patches = np.array([generator.get_window(img,i) for i in range(int(number_of_patches))])
    #Predict the windows
    prediction = CNN.predict(img_patches)
    #Convert the prediction to an image
    img_prediction = label_to_img(height,width,patch_size,patch_size,prediction)
    #Convert the image to uint8
    pimg = img_float_to_uint8(img_prediction)
    return pimg

def create_submission(submission_filename, prediction_filenames):
    """Create the csv submission file"""
    masks_to_submission(submission_filename, *prediction_filenames)

def get_predictions_filenames(test_folder):
    """Get the filename of the predictions"""
    prediction_filenames = []
    for i in range(1,51):
        prediction_filenames.append(test_folder + '/test_' + str(i) + '/prediction.png')
    return prediction_filenames

def make_predictions(CNN, generator, patch_size, window_size, pad_size):
    """Function used to make all the predictions"""
    prediction_folder = 'predictions'

    #Create the predictions folder
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)
    #Do all the predictions
    print("")
    for i in range(1,51):

        print('Save prediction : ' + str(i))
        test_data_filename = test_dir + '/test_' + str(i) +'/'
        #Load test image
        img = load_image(test_data_filename + 'test_' + str(i) +'.png')
        #Predict the label
        pimg = predict_image(img,CNN, patch_size, window_size, pad_size, generator)
        if not os.path.exists('predictions/test_'+str(i)):
            os.mkdir('predictions/test_' + str(i))
        save_folder = 'predictions/test_' + str(i)
        #Save the image
        Image.fromarray(pimg).save(save_folder + "/prediction.png")

    ### Create the submission csv file
    submission_filename = 'submission_CNN.csv'
    prediction_filenames = get_predictions_filenames(prediction_folder);
    create_submission(submission_filename, prediction_filenames)
    return 0
