import keras

from keras import models
from keras.models import load_model

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append('../')

from CNN import mean_f_score

from keras.applications import VGG16
from keras import backend as K
from keras.applications import VGG16


# This is a script to vizalize the intermediate activation and the kernels of a
# convolutional network.
# It is higly inspired from
# http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb




# The size of the window used in the network
window_size = 64

cnn_is_imageNet = False

# set if not using pre-trained ImageNet
model_name = 'Model_from_scratch.h5'

# Load the model, if it is imageNet, load from Keras
if cnn_is_imageNet:
    model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(window_size, window_size, 3))
else:
    model = load_model(model_name,
                       custom_objects={'mean_f_score': mean_f_score})

# Print the model architecture
model.summary()

# path to the test image
img_path = 'visualization/test_1.png'
# Load the test image and convert it to numpy array
img = image.load_img(img_path, target_size=(600, 600))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# normalize the image to [0,1]
img_tensor /= 255.

# Take the first possible window in the image (top left corner)
img_tensor = img_tensor[:,:window_size,:window_size,:]

# remove the unecessary dimension (list of 1 image -> 1 image)
plt.imshow(img_tensor[0])
plt.savefig('visualization/test_window.svg')


# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# These are the names of the layers, so we can have them as part of our plot
layer_names = []
for layer in model.layers[0:]:
    layer_names.append(layer.name)
print('Name of the layers in the model:')
print(layer_names)

images_per_row = 8

# A loop to save the activation maps
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    n_features = layer_activation.shape[-1]

    # Feature map shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    fig, ax = plt.subplots(images_per_row,n_cols) # (images_per_row,n_cols)
    fig.set_size_inches(10, 10.5)
    for col in range(n_cols): # n_cols
        for row in range(images_per_row): # images_per_row
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            ax[row,col].imshow(channel_image)
            ax[row,col].axis('off')

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig('visualization/Activation_' + layer_name + '.eps')
    plt.savefig('visualization/Activation_' + layer_name + '.svg')




def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=window_size):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.

    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input, K.learning_phase()], [loss, grads])
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data, 1])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


def plot_activation(nb_row, nb_col, layer_idx, save_path = None):
    """ Saves the images of the of the kernels activation on the test window
    """
    print('layer_idx:' + str(layer_idx))
    print('nb_row: ' + str(nb_row) + 'nb_col' + str(nb_col))
    fig, ax = plt.subplots(nb_row,nb_col)
    for i in range(nb_row):
        for j in range(nb_col):
            ax[i,j].imshow(generate_pattern(layer_names[layer_idx], i*nb_col +j))
            ax[i,j].axis('off')
    if save_path is not None:
        fig.set_size_inches(10.5, 10.5)
        gs1 = gridspec.GridSpec(nb_row, nb_col)
        gs1.tight_layout(fig)
        plt.savefig(save_path + '.eps')
        plt.savefig(save_path + '.svg')


# Adapt layer number (3rd argument) to get the convolutional layers
plot_activation(8,8,0,'visualization/conv_bloc_1_layer_1') # 8, 8
plot_activation(8,8,2,'visualization/conv_bloc_1_layer_2') # 8, 8
# plt.show()
