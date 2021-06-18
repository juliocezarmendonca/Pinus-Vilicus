#Importação de Bibliotecas
from tensorflow.keras.layers import Flatten,Dense,BatchNormalization,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

#Definição do batch_size e épocas
batch_size = 128
epochs = 250

# Definação do objeto da Classe Image Data Generator que define o Gerador de Imagens e faz o Data Augmentation
#Validation Split de 0.3 separa 70% das imagens para treino e 30% para validação.
data = ImageDataGenerator(vertical_flip = True,
                            horizontal_flip = True,
                            brightness_range=[0.2,1.0],
                            validation_split = 0.3,
                            rotation_range = 90
                          )

# Definição dos gerador de imagens para serem utilizadas no treino
traindata = data.flow_from_directory(directory="Data/",
                                     target_size = (50,50),
                                     batch_size = batch_size,
                                     class_mode = 'categorical',
                                     subset  = 'training',
                                     )
                                    
 # Definição dos gerador de imagens para serem utilizadas durante a validação
validationdata = data.flow_from_directory(directory="Data/",
                                      target_size=(50,50),
                                      batch_size =batch_size,
                                      class_mode = 'categorical',
                                      subset = 'validation'
                                      
                                      ) 

#Definição da quantidade de steps que ocorrerá durante cada época
steps_train = len(traindata.classes)/batch_size
steps_val = len(validationdata.classes)/batch_size

#Import do modelo resnet50 treinado no dataset ImageNet, o input foi modificado para receber a resolução das Imagens
res_model = ResNet50(include_top=False,weights='imagenet',input_shape=(50,50,3))

# Congela o treinamento das camadas Convolucionais
for layer in res_model.layers[:143]:
  layer.trainable = False

#Criação do modelo, com as camadas convolucionais e camadas densas com BatchNormalization e Dropout.
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

# Definição de um Objeto Callback, que salva o Melhor modelo que possui a melhor acurácia de validação.
checkpoint = ModelCheckpoint("pinusresnet50_2.h5",
                             verbose=1,
                             save_best_only=True,
                             monitor ='val_categorical_accuracy',
                             
                             save_weights_only=False,
                             mode='auto',
                             period=1)
#Definição de um objeto Callback responsável por parar o treino se após 100 epocas o modelo não modificar A Loss de validação
early = EarlyStopping(monitor='val_loss',
                      patience=100,
                                  
                      mode='min')
csv_logger = CSVLogger('pinusmodelresnet50_2.csv', append=True, separator=',')

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

model.fit(traindata, epochs = epochs,validation_data=validationdata,callbacks =[checkpoint,csv_logger],steps_per_epoch=steps_train,validation_steps=steps_val)
