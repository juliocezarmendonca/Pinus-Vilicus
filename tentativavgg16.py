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







data = ImageDataGenerator(vertical_flip = True,
                            horizontal_flip = True,
                            brightness_range=[0.2,1.0],
                            validation_split = 0.3,
                            rotation_range = 90
                          )

traindata = data.flow_from_directory(directory="Data/",
                                     target_size = (50,50),
                                     batch_size = 100,
                                     class_mode = 'categorical',
                                     subset  = 'training')
                                 

validationdata = data.flow_from_directory(directory="Data/",
                                      target_size=(50,50),
                                      batch_size =128,
                                      class_mode = 'categorical',
                                      subset = 'validation') 

# Vgg 16


from keras.applications.vgg16 import VGG16

vgg16 = keras.applications.vgg16

conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(50,50,3))

for layer in conv_model.layers:
    layer.trainable = False

    
x = keras.layers.Flatten()(conv_model.output)

x = keras.layers.Dropout(0.5)(x)

l2 = keras.regularizers.l2(0.01)


predictions = keras.layers.Dense(traindata.num_classes, activation='sigmoid',kernel_regularizer=l2,bias_regularizer=l2)(x)

model = keras.models.Model(inputs=conv_model.input, outputs=predictions)



optimizer = keras.optimizers.Adam(learning_rate= 0.01)
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

checkpoint = ModelCheckpoint("pinusvgg16.h5",
                             verbose=1,
                             save_best_only=True,
                             monitor ='val_categorical_accuracy',
                             
                             save_weights_only=True,
                             mode='auto',
                             period=1)

early = EarlyStopping(monitor='val_loss',
          
                      patience=100,
                      
                      
                      mode='min')

csv_logger = CSVLogger('pinusvgg16.csv', append=True, separator=',')

  
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.fit(traindata, epochs = 250,validation_data=validationdata,callbacks =[checkpoint,csv_logger],steps_per_epoch=2,validation_steps=1)
