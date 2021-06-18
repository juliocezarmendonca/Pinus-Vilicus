# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

from sys import argv

image_data = argv[1]
# Carregando o modelo para predição
model = load_model('pinusmodelresnet50.h5')
optimizer = Adam(learning_rate= 0.01)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])


original = Image.open(image_data)
original = original.resize((50, 50), Image.ANTIALIAS)
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)
   
predictions = model.predict(image_batch)
label = np.argmax(predictions)
    
if label == 0:
    text_classe  = "Essa é uma imagem de Solo."
elif label == 1:
    text_classe = "Essa é uma imagem de um Pinheiro."
    
print("Predição da Imagem Carregada")
print(text_classe)