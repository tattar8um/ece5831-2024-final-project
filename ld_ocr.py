#Lot of imports -- some may not actually be needed
import pathlib
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import imutils
import cv2
from imutils.contours import sort_contours
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.layers import Rescaling, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD


class LDOCR:

    '''
    Constructor:
    Constructs the class.
    @param batch_size Batch size, defaults to 32
    @param epochs Epochs, defaults to 20
    '''
    def __init__(self, batch_size=32, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_model()
        self._compile()
        self.lut  = ('ا', 'ب','پ','ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز','ژ', 'س', 'ش', 'ص', 'ض','ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك','گ', 'ل', 'م', 'ن', 'و','ه', 'ي')
    
    '''
    _create_lenet:
    Creates the network.
    '''
    def _create_model(self):
        self.model = Sequential([
            Input(shape=(32, 32, 1)),
            Rescaling(1./255),
            Conv2D(filters=32, kernel_size=(9,9),  activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=32, kernel_size=(9,9),  activation='relu',  padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=32, kernel_size=(9,9),  activation='relu',  padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(120, activation='sigmoid'),
            Dropout(0.7),
            Dense(84, activation='sigmoid'),
            Dense(32, activation='softmax')
        ])


    '''
    _compile:  
    Compiles the model
    '''
    def _compile(self):
        if self.model is None:
            print('Error: Create a model first..')
        self.model.compile(optimizer=Adam(lr=0.00001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def _preprocess(self, path):
        '''
        _preprocess:
        Preprocesses the images in the Persian dataset
        Normalizes to 28x28 and saves to the class
        '''
        self.train_db_persian = tf.keras.utils.image_dataset_from_directory(path, image_size=(32, 32), batch_size=32, color_mode='grayscale', validation_split=0.2,  # Reserve 20% of data for validation
            subset="training",     # Specify this is the training subset
            seed=42)
        self.val_db_persian = tf.keras.utils.image_dataset_from_directory(path, image_size=(32, 32), batch_size=32, color_mode='grayscale', validation_split=0.2,  # Reserve 20% of data for validation
            subset="validation",     # Specify this is the training subset
            seed=42)
        

    def train(self):
        '''
        Train:
        Trains the model using the modified Persian alphabet dataset
        '''
        callbacks = [
            #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]

        return self.model.fit(self.train_db_persian, epochs=120, validation_data=self.val_db_persian,
            callbacks=callbacks)
        
    def save(self, fnOut):
        '''
        Save:
        Saves the trained model to the path provided
        @param pathOut the path to store the Keras file
        '''
        self.model.save(fnOut)

    def load(self, fnIn):
        '''
        Load:
        Loads the trained model located at the path provided.
        @param pathIn the path to load the file from
        '''
        self.model = tf.keras.models.load_model(fnIn)

    def predict(self, images):
        '''
        predict:
        Predict the input image(s) using the loaded dataset.
        Input is expected to be a NumPy array of images.
        @param images array of images.
        '''
        pred = self.model.predict(images)
        ret = []
        for i in np.argmax(pred, axis=1):
            ret += [(self.lut[i])]
        #print(pred)
        return ret
    
    def lineOCR(self, lineImgPath):
        '''
        lineOCR:
        Attempts to perform OCR using the model
        on a line of Arabic text. 
        @param lineImgPath: The path to the image of a line of text

        '''
        jimage = cv2.imread("line5.jpg")
        jgray = cv2.cvtColor(jimage, cv2.COLOR_BGR2GRAY)
        jblurred = cv2.GaussianBlur(jgray, (5, 5), 0)

        jedged = cv2.Canny(jblurred, 30, 150)

        jcnts = cv2.findContours(jedged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        jcnts = imutils.grab_contours(jcnts)
        jcnts = sort_contours(jcnts, method="right-to-left")[0]
        jchars = []
        print(len(jcnts))

        rects = [cv2.boundingRect(i) for i in jcnts]
        print("test")

        min_y = min(rects, key=lambda t: t[1])[1]
        min_x = min(rects, key=lambda t: t[0])[0]
        max_h = max(rects, key=lambda t: t[3])[3]
        max_w = max(rects, key=lambda t: t[2])[2]
        max_x = max(rects, key=lambda t: t[0])[0]
        max_y = max(rects, key=lambda t: t[1])[1]

        line_processed = cv2.threshold(jgray[min_y:min_y + max_h, min_x:max_x + max_w], 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        final_line_bounds = cv2.boundingRect(line_processed)
        print(final_line_bounds)

        rlStartPoint = final_line_bounds[2]

        pred_thresh = 9.6e-1

        iterWidth = 15
        startPos = rlStartPoint

        output = ""

        while (startPos > iterWidth):
            preds = []
            print(f'At position {startPos}')
            for i in range(5, 60, 5):
                box = line_processed[final_line_bounds[1]:final_line_bounds[1] + final_line_bounds[3], max(startPos - (iterWidth + i), 0): startPos]
                (tH, tW) = box.shape
                dX = int(max(0, 32 - tW) / 2.0)
                dY = int(max(0, 32 - tH) / 2.0)
                box_pad = cv2.copyMakeBorder(box, top=dY, bottom=dY,
                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0))
                box_pad = cv2.resize(box_pad, (32, 32))

                # if left of window has 15 of blank
                box_left = box[final_line_bounds[1]:final_line_bounds[1] + final_line_bounds[3], 0:15]

                print(np.sum(box_left))
                if (np.sum(box_left) == 0):
                    break
                # plt.imshow(box)
                # plt.show()

                box_pad = np.expand_dims(box_pad, axis=0)  # Add batch dimension -> [1, 32, 32]
                box_pad = np.expand_dims(box_pad, axis=-1)  # Add channel dimension -> [1, 32, 32, 1]

                pred = (i, self.model.predict(box_pad)[0])
                print(max(pred[1]))
                print(np.argmax(pred[1]))
                preds.append(pred)
                print(self.lut[np.argmax(pred[1])])

                if (max(pred[1]) > pred_thresh):
                    print("MAX" + str(max(pred[1])))
                    print(self.lut[np.argmax(pred[1])])
                    #iterWidth += i
                    break
            
            
            #startPos -= iterWidth
            
            try:
                maxes = [(i[0], max(i[1])) for i in preds]
                argmaxes = [(i[0], np.argmax(i[1])) for i in preds]

                index_of_max = max(enumerate(maxes), key=lambda x: x[1][1])[0]
                print(maxes)
                print(index_of_max)
                maxargmax = argmaxes[index_of_max]
                print(maxargmax)
                print(self.lut[maxargmax[1]])

                output += self.lut[maxargmax[1]]


                startPos -= iterWidth + maxargmax[0]
            except:
                output += "" 
                startPos -= iterWidth
        return output



