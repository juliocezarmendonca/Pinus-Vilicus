
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
                                     target_size = (50,50),
                                     batch_size = batch_size,
                                     class_mode = 'categorical',
                                     subset  = 'training')
                                 

validationdata = data.flow_from_directory(directory="Data/",
                                      target_size=(50,50),
                                      batch_size = batch_size,
                                      class_mode = 'categorical',
                                      subset = 'validation') 

steps_train = len(traindata.classes)/batch_size
steps_val =  len(traindata.classes)/batch_size

# Vgg 16
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
          
                      patience=25,
                      
                      
                      mode='min')

csv_logger = CSVLogger('pinusvgg16.csv', append=True, separator=',')

  
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.fit(traindata, epochs = epochs,validation_data=validationdata,callbacks =[checkpoint,csv_logger,early],steps_per_epoch=steps_train,validation_steps=steps_val)
