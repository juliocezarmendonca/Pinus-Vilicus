#Inspirado em:
# https://github.com/jrosebr1/simple-keras-rest-api


from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from keras.models import load_model


app = flask.Flask(__name__)
model = None

def load_mod():
   # Função para carregar o modelo utilizado
    global model
    model = load_model("pinusmodelresnet50.h5")

#Função proveniente do código original, para tratamento das imagens carregadas
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

#Definição do método Post para  a função predição de imagens
@app.route("/predict", methods=["POST"])
def predict():
    text_classe = "Predição da Imagem: "
    # Se  a o método for do tipo Post e a imagem for passada na requisição, fará a rotina de predição.
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

if __name__ == "__main__":
    print("* Carregando O modelo Resnet50 e Inicialziando o Server Flask..")
    # Carregando o modelo antes do app.run, previne que o modelo seja carregado toda vez que o método Post for requisitado
    #evitando que a memória fique sobrecarregada.
    load_mod()
    app.run(debug=True)

