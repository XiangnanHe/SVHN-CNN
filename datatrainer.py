from dataloader import FileHDF5
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

DIR = 'projdata'
NB_FILTERS = 32
NB_EPOCH = 15

DETECTOR_FILE = 'model_detection.hdf5'
RECOGNIZER_FILE = 'model_clf.hdf5'


class _Preprocessor:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

    def _to_gray(self, image):
        """
        Parameters:
            image (ndarray of shape (n_rows, n_cols, ch) or (n_rows, n_cols))
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image
        else:
            raise ValueError("image dimension is strange")
        return gray_image


class _TrainTimePreprocessor(_Preprocessor):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
    @abstractmethod
    def run(self, images_train, labels_train, images_val, labels_val, nb_classes=2):
        pass


class DataPreproc(_TrainTimePreprocessor):
    
    def __init__(self):
        pass

    def run(self, images_train, labels_train, images_val, labels_val, nb_classes=2):
        
        _, n_rows, n_cols, ch = images_train.shape
        
        # 1. convert to gray images
        X_train = np.array([self._to_gray(patch) for patch in images_train], dtype='float').reshape(-1, n_rows, n_cols, 1)
        X_val = np.array([self._to_gray(patch) for patch in images_val], dtype='float').reshape(-1, n_rows, n_cols, 1)
        
        # convert class vectors to binary class matrices
        y_train = labels_train.astype('int')
        y_val = labels_val.astype('int')

        if nb_classes == 2:
            y_train[y_train > 0] = 1
            y_val[y_val > 0] = 1
        elif nb_classes == 10:
            X_train = X_train[y_train[:,0] > 0, :, :, :]
            X_val = X_val[y_val[:,0] > 0, :, :, :]
            y_train = y_train[y_train > 0]
            y_val = y_val[y_val > 0]
            y_train[y_train == 10] = 0
            y_val[y_val == 10] = 0
            
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_val = np_utils.to_categorical(y_val, nb_classes)

        # 2. calc mean value
        mean_value = X_train.mean()
        X_train -= mean_value
        X_val -= mean_value
     
        return X_train, X_val, Y_train, Y_val, mean_value


class _RunTimePreprocessor(_Preprocessor):
    __metaclass__ = ABCMeta
    
    def __init__(self, mean_value=None):
        self._mean_value = mean_value
    
    @abstractmethod
    def run(self, patches):
        pass
    
    def _substract_mean(self, images):

        images_zero_mean = images - self._mean_value
        return images_zero_mean
    
class GrayPreproc(_RunTimePreprocessor):
    def run(self, patches):

        n_images, n_rows, n_cols, _ = patches.shape
        
        patches = np.array([self._to_gray(patch) for patch in patches], dtype='float')
        patches = self._substract_mean(patches)
        patches = patches.reshape(n_images, n_rows, n_cols, 1)
        return patches
    
class NonePreprocessor(_RunTimePreprocessor):
    def run(self, patches):
        return patches


def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, nb_epoch=5, nb_classes=2, do_augment=False, save_file='models/model_detection.hdf5'):
    """ vgg like deep convolutional network """
    
    np.random.seed(1337) 
      
    # input image 
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3) 
    input_shape = (img_rows, img_cols, 1)


    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
     
    model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
        
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
        
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    if do_augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2)
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))
    else:
        model_hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #plot_model(model, "model/model1.png")
    print(model_hist.history.keys())
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(model_hist.history['acc'])
    plt.plot(model_hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    fig.savefig('model_accuracy.png')
    plt.close('all')

    # summarize history for loss
    fig = plt.figure()
    plt.plot(model_hist.history['loss'])
    plt.plot(model_hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    fig.savefig('model_loss.png')
    plt.close('all')
    model.save(save_file)  




if __name__ == "__main__":

    images_train = FileHDF5().read(os.path.join(DIR, "train.hdf5"), "images")
    labels_train = FileHDF5().read(os.path.join(DIR, "train.hdf5"), "labels")

    images_val = FileHDF5().read(os.path.join(DIR, "val.hdf5"), "images")
    labels_val = FileHDF5().read(os.path.join(DIR, "val.hdf5"), "labels")

    # Train detector to check if contains a number
    
    X_train, X_val, Y_train, Y_val, mean_value = DataPreproc().run(images_train, labels_train, images_val, labels_val, 2)
    print "mean value of the train images : {}".format(mean_value)    # Mean is around 107.524
    print "Train image shape is {}, and Validation image shape is {}".format(X_train.shape, X_val.shape)    # Dimensions: (457723, 32, 32, 1), (113430, 32, 32, 1)
    train_detector(X_train, X_val, Y_train, Y_val, nb_filters = NB_FILTERS, nb_epoch=NB_EPOCH, nb_classes=2, save_file=DETECTOR_FILE)
    
    # Train recognizer to recognize which number from 0-9
    X_train, X_val, Y_train, Y_val, mean_value = DataPreproc().run(images_train, labels_train, images_val, labels_val, 10)
    print "mean value of the train images : {}".format(mean_value)    # Mean is around 112.833
    print "Train image shape is {}, and Validation image shape is {}".format(X_train.shape, X_val.shape)    # Dimensions: (116913, 32, 32, 1), (29456, 32, 32, 1)
    train_detector(X_train, X_val, Y_train, Y_val, nb_filters = NB_FILTERS, nb_epoch=NB_EPOCH, nb_classes=10, save_file=RECOGNIZER_FILE)
    
