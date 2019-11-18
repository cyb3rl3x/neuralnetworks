import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D # 1 e 2 etapa
from keras.layers import Dense, Flatten, Dropout       # 3 e 4 etapa
from keras.utils import np_utils              # mapeamento de variáveis dummy
from keras.layers.normalization import BatchNormalization
# carrega o dataset MNIST
(x_treinamento, y_treinamento), (x_teste, y_teste) = mnist.load_data()

#exibe uma imagem do banco em escala de cinza
plt.imshow(x_treinamento[1], cmap='gray')
#com o título "Classe x"
plt.title('Classe '+str(y_treinamento[5]) )

previsores_treinamento = x_treinamento.reshape(x_treinamento.shape[0], 28, 28, 1)
#PARÂMETROS
#define o formato da imagem
#altura e largura
#canais

previsores_teste = x_teste.reshape(x_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#nomalização dos dados
previsores_treinamento /= 255
previsores_teste /= 255 


#conversão de atributos classe em atrubutos dummy
classe_treinamento = np_utils.to_categorical(y_treinamento,10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()

#camada de convolução 1
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation = 'relu'))
#nomalização
classificador.add(BatchNormalization())
#pooling
classificador.add(MaxPooling2D(pool_size = (2,2)))


#camada de convolução 2
classificador.add(Conv2D(32, (3,3), activation='relu'))
#nomalização
classificador.add(BatchNormalization())
#pooling
classificador.add(MaxPooling2D(pool_size = (2,2)))
#vetorização
classificador.add(Flatten())



#estrutura da rede neural
classificador.add(Dense(units = 128, activation='relu' ))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dense(units = 10, activation = 'softmax'))    
classificador.compile(loss = 'categorical_crossentropy', 
                      optimizer = 'adam', metrics = ['accuracy'])

classificador.fit(previsores_treinamento, 
                      classe_treinamento, 
                      batch_size = 128, 
                      epochs = 5, 
                      validation_data = (previsores_teste, classe_teste))

#resultado = classificador.evaluate(previsores_teste, classe_teste)









