
from tensorflow.keras.layers import Flatten,Dense,BatchNormalization,Dropout,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import keras


batch_size = 128
epochs = 250


data = ImageDataGenerator(vertical_flip = True,
                            horizontal_flip = True,
                            brightness_range=[0.2,1.0],
                            validation_split = 0.3,
                            rotation_range = 90
                          )



traindata = data.flow_from_directory(directory="Data/",
                                     target_size = (75,75),
                                     batch_size = batch_size,
                                     class_mode = 'categorical',
                                     subset  = 'training'
                                     )
                                    

validationdata = data.flow_from_directory(directory="Data/",
                                      target_size=(75,75),
                                      batch_size =batch_size,
                                      class_mode = 'categorical',
                                      subset = 'validation'
                                      ) 

steps_train = len(traindata.classes)/batch_size
steps_val = len(validationdata.classes)/batch_size

incep_model = InceptionV3(input_shape=(75,75,3), weights='imagenet', include_top= False)

for layer in incep_model.layers:
  layer.trainable = False

incep_model.build((1,50,50,3))
model = Sequential()
model.add(incep_model)
model.add(GlobalAveragePooling2D())
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
model.build((1,75,75,3))

optimizer = keras.optimizers.Adam(learning_rate= 0.01)
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

checkpoint = ModelCheckpoint("pinusinceptionv3.h5",
                             verbose=1,
                             save_best_only=True,
                             monitor ='val_categorical_accuracy',
                             
                             save_weights_only=True,
                             mode='auto',
                             period=1)
early = EarlyStopping(monitor='val_loss',
                      patience=100,     
                      mode='min')
csv_logger = CSVLogger('pinusnceptionv3.csv', append=True, separator=',')

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.fit(traindata, epochs = epochs,validation_data=validationdata,callbacks =[checkpoint,csv_logger,early],steps_per_epoch=steps_train,validation_steps=steps_val)

