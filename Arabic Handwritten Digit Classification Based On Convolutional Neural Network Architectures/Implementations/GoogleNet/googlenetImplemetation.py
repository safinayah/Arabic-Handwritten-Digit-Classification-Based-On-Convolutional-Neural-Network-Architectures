# This model based on the architecture mentioned here https://docs.google.com/document/d/1iWhJiMT9pgWqYA_3-iRyvQ1DwlhV3hGdR-pinZiiHfk/edit

from __future__ import division, print_function, absolute_import

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import keras

from keras.utils.np_utils import to_categorical

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU

from tensorflow.python.keras.callbacks import EarlyStopping

# Load your Dataset
trainImg = pd.read_csv("/Your/dataset/training/file/in/csv/form", header=None)
trainLabel = pd.read_csv("/Your/dataset/training/labels/in/csv/form", header=None)
testImg = pd.read_csv("/Your/dataset/teastinf/file/in/csv/form", header=None)
testLabel = pd.read_csv("/Your/dataset/testing/labels/in/csv/form", header=None)

trainImg.head()
testImg.head()

# training images
trainImg = trainImg.values.astype('float32') / 255.0
# training labels
trainLabel = trainLabel.values.astype('int32')

# testing images
testImg = testImg.values.astype('float32') / 255.0
# testing labels
testLabel = testLabel.values.astype('int32')

# In[ ]:
testL = testLabel

trainImg[0]



# One Hot encoding of train labels.
trainLabel = to_categorical(trainLabel, 10)

# One Hot encoding of test labels.
testLabel = to_categorical(testLabel, 10)




trainLabel[0]




print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)



# reshape input images to 28x28x1
trainImg = trainImg.reshape([-1, 28, 28, 1])
testImg = testImg.reshape([-1, 28, 28, 1])



print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)



# reshape input images to 28x28x1
trainImg = trainImg.reshape([-1, 28, 28, 1])
testImg = testImg.reshape([-1, 28, 28, 1])




trainImg[0]




def googlenet(inputShape, nOfClasses):
    def inceptionBlock(x, f):
        t1 = Conv2D(f[0], 1, activation='relu')(x)

        t2 = Conv2D(f[1], 1, activation='relu')(x)
        t2 = Conv2D(f[2], 3, padding='same', activation='relu')(t2)

        t3 = Conv2D(f[3], 1, activation='relu')(x)
        t3 = Conv2D(f[4], 5, padding='same', activation='relu')(t3)

        t4 = MaxPool2D(3, 1, padding='same')(x)
        t4 = Conv2D(f[5], 1, activation='relu')(t4)

        output = Concatenate()([t1, t2, t3, t4])
        return output

    input = Input(inputShape)

    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(192, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(3, strides=2)(x)

    x = inceptionBlock(x, [64, 96, 128, 16, 32, 32])
    x = inceptionBlock(x, [128, 128, 192, 32, 96, 64])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inceptionBlock(x, [192, 96, 208, 16, 48, 64])
    x = inceptionBlock(x, [160, 112, 224, 24, 64, 64])
    x = inceptionBlock(x, [128, 128, 256, 24, 64, 64])
    x = inceptionBlock(x, [112, 144, 288, 32, 64, 64])
    x = inceptionBlock(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inceptionBlock(x, [256, 160, 320, 32, 128, 128])
    x = inceptionBlock(x, [384, 192, 384, 48, 128, 128])

    x = GlobalAvgPool2D()(x)
    x = Dropout(0.5)(x)

    output = Dense(nOfClasses, activation='softmax')(x)
    model = Model(input, output)

    return model



# Model Calling and passing parameters

inputShape = 28, 28, 1
nOfClasses = 10
K.clear_session()
model = googlenet(inputShape, nOfClasses)
model.summary()


# model Compilation

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Model fit with validation, suffling and early stopping to avoid overfittung

model.fit(trainImg, trainLabel,
          shuffle=True,
          validation_split=.16,
          verbose=2,
          batch_size=20,
          epochs=10,
          callbacks=[EarlyStopping(patience=10)])


model.save('Models/ModelName')

print('Predict the classes for training : ')
# prediction = model.predict(trainImg)
classes = model.predict(trainImg)

prediction = classes.argmax(axis=1)
print('Predicted classes: ', prediction)

print('Predict the classes for testing : ')

classess = model.predict(testImg)
prediction = classess.argmax(axis=1)

print('Predicted classes: ', prediction)

score = model.evaluate(testImg, testLabel, verbose=1)
print('Loss accuracy: %2f%%' % (score[0] * 100))
print('Test accuarcy: %2f%%' % (score[1] * 100))
print(model.metrics_names)



