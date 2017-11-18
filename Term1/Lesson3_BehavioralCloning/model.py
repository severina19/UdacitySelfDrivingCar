import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot  
  
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import os
import cv2
import matplotlib.image as mpimg

np.random.seed(0)
# global parameters
glob_image_height = 66  # this demention is chosen because we are using the nvidia model
glob_image_width = 200
glob_image_channels = 3
glob_image_shape=(glob_image_height, glob_image_width, glob_image_channels)

def load_data(data_dir):
    """
    load training data from csv file, which inclues the pathes to 
    center, left and right images and the steering wheel
    """
    training_data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
    X = training_data[['center', 'left', 'right']].values
    y = training_data['steering'].values
    
    return X, y

def load_image(data_dir, image_file):
    """
    load RGB images from file
    """
    image_path = os.path.join(data_dir, image_file)
    image = mpimg.imread(image_path)
    return image


def choose_random_image(data_dir, center, left, right, steering_angle):
    """
    radomly choose and image out of the entire data set and adjust
    the steering angle according to its position (left, right or center)
    """
    choice_array = ['center', 'left', 'right']
    choice = np.random.choice(choice_array, 1,  p=[0.5,0.25,0.25])
    if choice == 'left':
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 'right':
        return load_image(data_dir, right), steering_angle - 0.2
    elif choice == 'center':
        return load_image(data_dir, center), steering_angle

def augument_data(data_dir, center, left, right, steering_angle):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_random_image(data_dir, center, left, right, steering_angle)
    # Randomly flipt the image horizontally and adjust the steering angle.
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
     
    #translate the object with random distance in x and y direction and adjust the steering angle
    trans_x = np.random.uniform(0, 30)
    trans_y = np.random.uniform(0, 20)
    steering_angle += trans_x * 0.002
    trans_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height= image.shape[0]
    width=image.shape[1]
    image = cv2.warpAffine(image, trans_matrix, (width, height))
    return image, steering_angle

def build_model(keep_prob):
    """
    building a neurol network using the nvidia model
    """
    model=Sequential()
    #normalization
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=glob_image_shape))
    #convolutional layers
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    #drop out to prevent over fitting
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    #fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model


def generator(data_dir, image_paths, steering_angles, batch_size, b_istraining):
    """
    Generate training image give image paths and associated steering angles
    """

    images = np.empty([batch_size, glob_image_height, glob_image_width, glob_image_channels])
    steers = np.empty(batch_size)
    nb_images=image_paths.shape[0]
    while True:
        for i in range(batch_size):
            index = random.randint(0, nb_images-1)
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if b_istraining:
                image, steering_angle = augument_data(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
                
            image_height_orig =image.shape[0]
            # cropping out irrelevant part of the picture
            image = image[60:image_height_orig-30, :, :]
            # resize the image for the nvidia model
            image = cv2.resize(image, (glob_image_width, glob_image_height), cv2.INTER_AREA)
            # convert to yuv space for nvidia model
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            # add image and steering angle to the batch
            images[i] = image
            steers[i] = steering_angle
        yield images, steers

def train_model(model, X_train, X_valid, y_train, y_valid, data_dir, learning_rate, 
                batch_size, samples_per_epoch, nb_epoch, previous_weights):

    # store the trained model 
    checkpoint = ModelCheckpoint('model_test-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only='true',
                                 mode='auto')
    # load previously trained model 
 
    if os.path.isfile(previous_weights) == True:
        model.load_weights(previous_weights)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    train_batch=generator(data_dir, X_train, y_train, batch_size, True)
    validation_batch=generator(data_dir, X_valid, y_valid, batch_size, False)

    model.fit_generator(train_batch,
                        samples_per_epoch,
                        nb_epoch,
                        max_q_size=1,
                        validation_data=validation_batch,
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def main():

    # parameters for training the model
    nb_epoch = 5
    samples_per_epoch = 20000
    batch_size = 50
    learning_rate = 1.0e-4
    test_size = 0.2
    keep_prob = 0.5
    
    # load training data 
    data_dir = os.path.join(os.getcwd(), 'data\mfast')
    X, y = load_data(data_dir)
    # split the data into training set and validation set 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)
    
    model = build_model(keep_prob)
    plot(model, to_file='model.png')
    previous_weights = "model4-001.h5"

    train_model(model, X_train, X_valid, y_train, y_valid, data_dir,
                learning_rate, batch_size, samples_per_epoch, nb_epoch, previous_weights)


if __name__ == '__main__':
    main()

