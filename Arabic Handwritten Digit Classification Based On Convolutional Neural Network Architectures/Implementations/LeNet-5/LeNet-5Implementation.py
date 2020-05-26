from __future__ import division, print_function, absolute_import

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load your Dataset
trainImg = pd.read_csv("/Your/dataset/training/file/in/csv/form", header=None)
trainLabel = pd.read_csv("/Your/dataset/training/labels/in/csv/form", header=None)
testImg = pd.read_csv("/Your/dataset/teastinf/file/in/csv/form", header=None)
testLabel = pd.read_csv("/Your/dataset/testing/labels/in/csv/form", header=None)


trainImg.head()

# In[314]:


testImg.head()

# In[315]:


# Split data into training set and validation set
# training images
trainImg = trainImg.values.astype('float32') / 255
# training labels
trainLabel = trainLabel.values.astype('int32')

# testing images
testImg = testImg.values.astype('float32') / 255
# testing labels
testLabel = testLabel.values.astype('int32')

testL = testLabel

testImg

# In[317]:


# One Hot encoding of train labels.
trainLabel = to_categorical(trainLabel, 10)

# One Hot encoding of test labels.
testLabel = to_categorical(testLabel, 10)

# In[318]:


trainLabel[0]

print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)

# reshape input images to 28x28x1
trainImg = trainImg.reshape([-1, 28, 28, 1])
testImg = testImg.reshape([-1, 28, 28, 1])

# In[321]:


print(trainImg.shape, trainLabel.shape, testImg.shape, testLabel.shape)

# In[322]:


trainImg[0]

# In[323]:


# Building convolutional network
model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='tanh'))

model.add(layers.Dense(units=84, activation='tanh'))

model.add(layers.Dense(units=10, activation='softmax'))

model.summary()

# model Compilation

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



# Model fit with validation, suffling and early stopping to avoid overfittung
history = model.fit(trainImg, trainLabel,
                    shuffle=True,
                    validation_split=.16,
                    batch_size=20,
                    verbose=2,
                    epochs=10,
                    callbacks=[EarlyStopping(patience=5)])

# Training classes Prediction
print('Predict the classes for training : ')
prediction = model.predict_classes(trainImg)
print('Predicted classes: ', prediction)

# Testing classes Prediction

print('Predict the classes for testing : ')
prediction = model.predict_classes(testImg)
print('Predicted classes: ', prediction)

# Model Evaluation
score = model.evaluate(testImg, testLabel, verbose=1)
print('Loss accuracy: %2f%%' % (score[0] * 100))
print('Test accuarcy: %2f%%' % (score[1] * 100))
print(model.metrics_names)


model.save('Models/ModelName')


print(prediction)
# print confusion matrix
result = confusion_matrix(y_true=testL, y_pred=prediction)
print('Confusion Matrix :')
print(result)

# print class classification accuracy

print('Accuracy Score :', accuracy_score(testL, prediction))
print('Report : ')
print(classification_report(y_true=testL, y_pred=prediction))
