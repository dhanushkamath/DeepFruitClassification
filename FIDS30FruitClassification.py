import tensorflow as tf
import inception
import numpy as np
import os
import inception
from glob import glob
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2

# Transfer Learning with the pre-trained inception model

# Initialize the inception model and required initializations
inc_model = inception.Inception()
TRAIN_DIR = 'FIDS30'
MODEL_NAME = 'FIDS30_1'
LR = 1e-4

# Define a function for obtaining transfer values
def transfer_values(image_path):
    t_vals = inc_model.transfer_values(image_path)
    return t_vals

def create_train_data():
    training_data = []
    for label_p in tqdm(glob(TRAIN_DIR+'/*')):
        onehot_label = label_image(label_p.split('/')[-1])
        for img_p in tqdm(glob(label_p+'/*')):
            tr_val = transfer_values(image_path = img_p)
            training_data.append([np.array(onehot_label), np.array(tr_val)])
    shuffle(training_data)
    np.save('FIDS30_savedmodel/saved_np/FIDS30_9classes.np',training_data)
    return

def label_image(label): # Returns the training label
    options = { 'apples':      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                'bananas':     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                'coconuts':    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                'grapes':      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                'guava':       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                'mangos':      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                'pineapples':  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                'pomegranates':[0, 0, 0, 0, 0, 0, 0, 1, 0],
                'tomatoes':    [1, 0, 0, 0, 0, 0, 0, 0, 1],}

    return options[label]


def load_train_data():
    if glob("FIDS30_savedmodel/saved_np/FIDS30_9classes.np"):
        train = np.load("training_data/train.npy")
        print("Numpy file of training data exists and has been loaded")
    else:
        print("ERROR : No saved training data. Please run the function create_train_data()")
        exit()
    Y = np.array([i[0] for i in train])
    X = np.array([i[1] for i in train])
    return X,Y


# ======================================================================================================================
# The Fully Connected Layer (To be connected after the inception model)
# This will only have one fully connected layer and a softmax classifier

num_classes = 4
len_transfer = 2048

#Here transfer_layer is the input layer of the new network
transfer_layer = input_data(shape=[None,2048], name='input')

transfer_layer = fully_connected(transfer_layer, 1024, activation='relu')
transfer_layer = fully_connected(transfer_layer, 512, activation='relu')

transfer_layer = dropout(transfer_layer, 0.6)

transfer_layer = fully_connected(transfer_layer,num_classes,activation = 'softmax')
transfer_layer = regression(transfer_layer, optimizer='adam', learning_rate=LR,
                            loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(transfer_layer)

# ======================================================================================================================

def train_the_network():
    X, Y = load_train_data()

    #model.fit({'input': X}, {'targets': Y}, n_epoch=4,
              #snapshot_step=500, show_metric=True, run_id='MODEL_NAME')
    model.fit(X, Y, n_epoch=20, batch_size=1024, show_metric=True)

    model.save('FIDS30_savedmodel/'+MODEL_NAME)

    return

def test_model(path):
    options_num = { 0 : 'Apple' ,
                    1 : 'Orange',
                    2 : 'Pappaya',
                    3 : 'Pineapple',
                     }
    model.load('Attempt2/'+MODEL_NAME)
    tr_val = transfer_values(path)
    tr_val = tr_val.reshape(-1,2048)
    pred = model.predict(tr_val)
    img = cv2.imread(path)
    print(options_num[np.argmax(pred)])
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

create_train_data()
