# Project 2 : Road segmentation
## Team : All Roads Lead 2 Segmentation
## Members : Vincent Pollet, Aymeric Galan, Clément Lefebvre



### Setup used for training :

The models were trained using the Elastic Cloud Computing Amazon Web Service
using one GPU :
  - OS : Ubuntu 16.04.2 LTS
  - GPU : Tesla K80 12Go RAM

  Libraries used :
    - Python 3.6.1
    - Tensorflow 1.1.0 with GPU support
    - Keras 2.0.2
    - Numpy 1.12.1
    - Pillow 4.1.1
    - matplotlib 2.0.2


  Training time on GPU :
    - Model I (VGG pre-trained) : ~12h
    - Model V (VGG random weights) : ~2h30
    - Model S (Custom CNN) : ~2h

  Prediction time :
    - GPU Tesla K80 : 4min
    - GPU 660M : 8min

### How to run the code ?

1. If the training and test_set_images folder are not in the project folder :
Put the training folder 'training' in the root of the project directory (with the other scripts)
and the test set folder 'test_set_images' also in the root of the project directory. If the script does not
find the images, you can change the path to the test set on top of the helpers.py script and the path to the
training set on top of the generateImage.py script.

2. Download the trained model 'CNN_I_FineTuned.h5' : https://drive.switch.ch/index.php/s/ltMiAyLvIdgrhCh
   Put the model with the other script in the project root folder.

3. If you want to train the model run the generateImage.py script
   to generate 1500 augmented images (Not necessary for only prediction)

    Command : KERAS_BACKEND=tensorflow python generateImage.py

4. Run the run.py script

    Command : KERAS_BACKEND=tensorflow python run.py

    1. Choice 1 : Use the already trained model "FineTuned.h5" to predict the
    label of the images in "test_set_images". It will output our Kaggle result as submission_CNN.csv
    The prediction was tested on a Nvidia 660M GPU in ~8min. It can take a much higher
    time on CPU.

    2. Choice 2 : Train the model used in the Kaggle submission.

    3. Choice 3 : Train the VGG16 model + our classifier with random weight
    initialization. An overfit is expected.

    4. Choice 4 : Train an other basic CNN model from scratch which gives less
    accurate results than the pre-trained model.

    **Warning** : All model training takes a lot of time they were
    trained on a Tesla K80 GPU using Amazon Web Service.


### Code description :

  - generateImage.py : Script used to generate 1500 images augmented by rotation,
  shear, horizontal/vertical flip from the original dataset.
  - run.py : Script used to predict or train the models.
  - CNN.py : Files containing the CNN class which contains the model and necessary functions
    to train it.
  - Generator_class.py : Files containing the Generator class which is used to feed the data to
    the CNN.
  - helpers.py : Files containing different functions used to do data processing.
  - mask_to_submission.py : Files containing functions used to output the submission file


### Code for plot and figures :
  - Visualisation folder :
    - analysis.py : Script used to plot the graphs
        Command : python analysis -h
    - Padding_Viz.ipynb : Jupyter notebook which show the mirror padding and window
    - visualize_cnn.py : Script to plot the first layer kernel and convolution output of
    the model I and S. Need to change the path of the model at the beginning of the script.

  - Libraries used :
    - seaborn 0.8.1


### License :
  The VGG16 weights are ported from the ones released by VGG at Oxford
  (http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/) under the Creative Commons Attribution License.
