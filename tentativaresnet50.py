from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D,AveragePooling2D,Dropout,MaxPooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import keras
import os

batch_size = 128
epochs = 250

data = ImageDataGenerator(vertical_flip = True,
                            horizontal_flip = True,
                            brightness_range=[0.2,1.0],
                            validation_split = 0.3,
                            rotation_range = 90
                          )


traindata = data.flow_from_directory(directory="Data/",
                                     target_size = (50,50),
                                     batch_size = batch_size,
                                     class_mode = 'categorical',
                                     subset  = 'training'
                                     shuffle = False
                                     )
                                    

validationdata = data.flow_from_directory(directory="Data/",
                                      target_size=(50,50),
                                      batch_size =batch_size,
                                      class_mode = 'categorical',
                                      subset = 'validation'
                                      
                                      ) 

steps_train = len(traindata.classes)/batch_size
steps_val = len(validationdata.classes)/batch_size

from keras import layers
from keras import Input
from keras.applications.resnet50 import ResNet50

res_model = ResNet50(include_top=False,weights='imagenet',input_shape=(50,50,3))

for layer in res_model.layers[:143]:
  layer.trainable = False

model = Sequential()
model.add(res_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(traindata.num_classes,activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate= 0.01)
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

checkpoint = ModelCheckpoint("pinusresnet50.h5",
                             verbose=1,
                             save_best_only=True,
                             monitor ='val_categorical_accuracy',
                             
                             save_weights_only=True,
                             mode='auto',
                             period=1)
early = EarlyStopping(monitor='val_loss',
                      patience=25,
                                  
                      mode='min')
csv_logger = CSVLogger('pinusresnet50.csv', append=True, separator=',')

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.fit(traindata, epochs = epochs,validation_data=validationdata,callbacks =[checkpoint,csv_logger],steps_per_epoch=steps_train,validation_steps=steps_val)
