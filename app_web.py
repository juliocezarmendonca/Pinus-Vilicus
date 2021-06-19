from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from flask import Flask
#from flask_bootstrap import Bootstrap
import os

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

import numpy as np

# Carregando o modelo para predição
model = load_model('pinusmodelresnet50.h5')

class UploadForm(FlaskForm):
    upload = FileField('Selecione uma Imagem:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Classifique')
    
    

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
#bootstrap = Bootstrap(app)

def classify(image_data):
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
    
    return text_classe

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename
        )
        f.save(file_url)
        form = None
        prediction = classify(file_url)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)

app.run()