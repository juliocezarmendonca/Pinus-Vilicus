#Inspirado em:
# https://github.com/jrosebr1/simple-keras-rest-api


# USAGE
# Start the server:
#     python run_keras_server.py
# Submit a request via cURL:
#     curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#    python simple_request.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from keras.models import load_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_mod():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model("pinusmodelresnet50.h5")

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    text_classe = "Predição da Imagem: "
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(50, 50))
            predictions = model.predict(image)
            label = np.argmax(predictions)
            if label == 0:
                text_classe += "Essa é uma Imagem de Solo."
            elif label ==1:
                text_classe += "Essa é uma Imagem de um Pinheiro."
    
    return text_classe

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("* Carregando O modelo Resnet50 e Inicialziando o Server Flask..")
    load_mod()
    app.run(debug=True)

