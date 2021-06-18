import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("pinusvgg16.csv")

plt.plot(log['categorical_accuracy'])
plt.plot(log['val_categorical_accuracy'])
plt.title('Acurácia do treino da Arquitetura VGG16')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()


print("Análise de Resultados da Arquitetura VGG16")

max_accuracy = log[(log['categorical_accuracy']) == (log['categorical_accuracy'].max())]
print("Época com Maior Acurácia de Treino:")
display(max_accuracy)

max_val_accuracy = log[(log['val_categorical_accuracy']) == (log['val_categorical_accuracy'].max())]
print("Época com Maior Acurácia de Validação:")
display(max_val_accuracy)

plt.plot(log['loss'])
plt.plot(log['val_loss'])
plt.title('Loss do treino da Arquitetura VGG16')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')
plt.show()



min_loss = log[(log['loss']==(log['loss'].min()))]
print("Menor Loss do Treino")
display(min_loss)

min_val_loss = log[(log['val_loss']==(log['val_loss'].min()))]
print("Menor Loss de Validação")
display(min_val_loss)

print('\n')


log = pd.read_csv("pinusresnet50.csv")

plt.plot(log['categorical_accuracy'])
plt.plot(log['val_categorical_accuracy'])
plt.title('Acurácia do treino da Arquitetura ResNet50')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()


print("Análise de Resultados da Arquitetura ResNet50")

max_accuracy = log[(log['categorical_accuracy']) == (log['categorical_accuracy'].max())]
print("Época com Maior Acurácia de Treino:")
display(max_accuracy)

max_val_accuracy = log[(log['val_categorical_accuracy']) == (log['val_categorical_accuracy'].max())]
print("Época com Maior Acurácia de Validação:")
display(max_val_accuracy)



plt.plot(log['loss'])
plt.plot(log['val_loss'])

ymin= 0
ymax= 1

ymin, ymax = plt.ylim()
scale_factor = 0.0001

plt.ylim(ymin * scale_factor, ymax * scale_factor)

plt.title('Loss do treino da Arquitetura ResNet50')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')
plt.show()



min_loss = log[(log['loss']==(log['loss'].min()))]
print("Menor Loss do Treino")
display(min_loss)

min_val_loss = log[(log['val_loss']==(log['val_loss'].min()))]
print("Menor Loss de Validação")
display(min_val_loss)

print('\n')




log = pd.read_csv("pinusnceptionv3.csv")

plt.plot(log['categorical_accuracy'])
plt.plot(log['val_categorical_accuracy'])
plt.title('Acurácia do treino da Arquitetura InceptionV3')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()


print("Análise de Resultados da Arquitetura InceptionV3")

max_accuracy = log[(log['categorical_accuracy']) == (log['categorical_accuracy'].max())]
print("Época com Maior Acurácia de Treino:")
display(max_accuracy)

max_val_accuracy = log[(log['val_categorical_accuracy']) == (log['val_categorical_accuracy'].max())]
print("Época com Maior Acurácia de Validação:")
display(max_val_accuracy)

plt.plot(log['loss'])
plt.plot(log['val_loss'])
plt.title('Loss do treino da Arquitetura InceptionV3')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')
plt.show()



min_loss = log[(log['loss']==(log['loss'].min()))]
print("Menor Loss do Treino")
display(min_loss)

min_val_loss = log[(log['val_loss']==(log['val_loss'].min()))]
print("Menor Loss de Validação")
display(min_val_loss)

