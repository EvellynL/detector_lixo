import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import Input, Activation, Add, Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

diretorio = './programacao/dataset/'

trash_types = os.listdir(diretorio)

data = []

for trash_type in trash_types:
    for file in os.listdir(os.path.join(diretorio, trash_type)):
        data.append((os.path.join(diretorio, trash_type, file), trash_type))


df = pd.DataFrame(data, columns=['Caminho', 'label'])

#treinamento e teste

train_df, val_df = train_test_split(df, test_size=0.2, random_state=1, stratify=df['label'])

# print(f'Numero de imagens de treinamento: {len(train_df)}')
# print(f'Numero de imagens de validação: {len(val_df)}')

gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range = 45,
                                         horizontal_flip = True,
                                         zoom_range = 0.2,
                                        width_shift_range=0.15,             
                                        height_shift_range=0.15,
                                        shear_range = 0.05,
                                        brightness_range = [0.9, 1.1],
                                        vertical_flip= True,
                                        channel_shift_range=0.15,
                                        fill_mode='nearest')    
     
dataset_train = gerador_treinamento.flow_from_dataframe(
    dataframe= train_df,
    target_size = (256, 256),
    x_col = 'Caminho',
    y_col = 'label',
    batch_size = 64,
    class_mode = 'categorical',
    seed = 42, 
    shuffle = False,
)

gerador_val = ImageDataGenerator(rescale= 1./255)

dataset_val = gerador_val.flow_from_dataframe(dataframe= val_df,
                                              x_col = 'Caminho',
                                              y_col = 'label',
                                              target_size = (256, 256),
                                              batch_size = 64,
                                              class_mode= 'categorical',
                                              seed = 42,
                                              shuffle = False)

network = Sequential()
network.add(Conv2D(filters=32, kernel_size= (3,3), activation='relu', input_shape=(256, 256, 3)))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Conv2D(filters=32, kernel_size= (3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Conv2D(filters=32, kernel_size= (3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Flatten())
network.add(Dense(units= 2000, activation='relu'))
network.add(Dense(units= 2000, activation='relu'))
network.add(Dropout(0.5))
network.add(Dense(units= 3, activation='softmax'))
network.summary()

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
historico = network.fit(dataset_train, epochs = 350)

model_json = network.to_json()
with open('network5.json', 'w') as json_file:
    json_file.write(model_json)

network_saved = save_model(network, 'pesos5.hdf5')

with open('network5.json', 'r') as json_file:
    json_saved_model = json_file.read()
json_saved_model

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('pesos5.hdf5')
network_loaded.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

previsoes = network_loaded.predict(dataset_val)
previsoes = np.argmax(previsoes, axis =1)

accuracy = accuracy_score(dataset_val.classes, previsoes)
print(accuracy)

# Desempenho

cm = confusion_matrix(dataset_val.classes, previsoes)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(dataset_val.classes, previsoes))
print(dataset_val.class_indices)

# class_names = list(dataset_val.class_indices.keys())

# # Número total de imagens
# num_images = len(val_df)

# # Iniciar a iteração sobre o dataset_val para percorrer todas as imagens
# # Número total de imagens
# num_images = len(val_df)

# # Iniciar a iteração sobre o dataset_val para percorrer todas as imagens
# for i in range(num_images):
#     # Pegar o batch de imagens (dataset_val é dividido em batches)
#     image_batch, label_batch = dataset_val[i]
    
#     # Agora percorremos o batch e mostramos cada imagem individualmente
#     for j in range(image_batch.shape[0]):  # image_batch.shape[0] é o número de imagens no batch
#         image = image_batch[j:j+1]  # Pegando uma imagem individual
#         label = label_batch[j:j+1]  # Pegando o label correspondente

#         # Previsão para a imagem atual
#         previsao = network_loaded.predict(image)
#         previsao_classe = np.argmax(previsao, axis=1)[0]  # Classe prevista para a imagem atual

#         # Classe real
#         classe_real = label[0]

#         # Recuperar o nome das classes
#         classe_real_nome = class_names[np.argmax(classe_real)]  # Classe real da imagem
#         previsao_classe_nome = class_names[previsao_classe]     # Classe prevista para a imagem

#         # Converter a imagem para numpy array (remover a dimensão extra)
#         imagem_exibida = np.array(image[0])

#         # Redimensionar para exibição
#         imagem_exibida = cv2.resize(imagem_exibida, (500, 500))

#         # Adicionar texto com a classe real e a classe prevista na imagem
#         texto = f'Real: {classe_real_nome} | Previsao: {previsao_classe_nome}'
#         cv2.putText(imagem_exibida, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

#         # Exibir a imagem com a previsão
#         cv2.imshow(f'Imagem {i+1}_{j+1}', imagem_exibida)

#         # Esperar a tecla 'q' para avançar para a próxima imagem
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()

# # Fechar as janelas ao final
# cv2.destroyAllWindows()

