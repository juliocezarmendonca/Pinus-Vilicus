# Inspirado em :
#https://pierpaolo28.github.io/blog/blog40/


import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# Carregando o modelo para predição
model = load_model('pinusmodelresnet50.h5')



# Função responsável por carregar a imagem
def load_img():
    #Trecho não modificado do original
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Escolha Uma Imagem:",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()
    #Fim do trecho
# Função responsável por classificar a imagem carregada
def classify():
    #Trecho não modificado do original
    original = Image.open(image_data)
    original = original.resize((50, 50), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
   
    predictions = model.predict(image_batch)
    label = np.argmax(predictions)
    #Fim do trecho
    if label == 0:
        text_classe  = "Essa é uma imagem de Solo."
    elif label == 1:
        text_classe = "Essa é uma imagem de um Pinheiro."
    
    table = tk.Label(frame, text="Predição da Imagem").pack()
    result = tk.Label(frame,text= text_classe).pack()


#Definição do Objeto Tkinter
root = tk.Tk()
#Título da Aplicação
root.title('Pinus Vilicus!')
#Ícone da aplicação
root.iconbitmap('pine.ico')

root.resizable(False, False)
#Subtitulo
tit = tk.Label(root, text="Pinus Vilicus: Supervisionando seus Pinheiros desde 2021!", padx=25, pady=6, font=("", 14)).pack()
#Define o tamanho da borda interna e a cor
canvas = tk.Canvas(root, height=500, width=500, bg='green')
canvas.pack()
#Define a cor da janela
frame = tk.Frame(root, bg='grey')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
#Definição do Botão que executa a função de carregar a imagem 
chose_image = tk.Button(root, text='Escolha uma Imagem:',
                        padx=35, pady=10,
                        fg="black", bg="green", command=load_img)
chose_image.pack(side=tk.LEFT)

#Definição do Botão que executa a função de Classificar a Imagem
class_image = tk.Button(root, text='Classifique a Imagem!',
                        padx=35, pady=10,
                        fg="black", bg="green", command=classify)
class_image.pack(side=tk.RIGHT)

#Torna possivel refazer o processo de carregar imagens
root.mainloop()
