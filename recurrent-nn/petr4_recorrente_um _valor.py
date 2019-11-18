from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base = pd.read_csv('petr4-treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:2].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0]) #0 89
    preco_real.append(base_treinamento_normalizada[i,0])       #90

previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()
#camada 01 e camada de entrada
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))
#camada 02
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))
#camada 03
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))
#camada de saída
regressor.add(Dense(units = 1, activation = 'linear'))
#montagem da rede
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
#treinamento da rede
regressor.fit(previsores, preco_real, epochs=100, batch_size = 32)
    
#carregamento da base de teste    
base_teste = pd.read_csv('petr4-teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)    
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []

for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1],1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes) 

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color = 'red', label = "Preço real")
plt.plot(previsoes, color = 'blue', label = "Previsão" )
plt.title("Previsão de preço das ações")
plt.xlabel("Tempo")
plt.ylabel("Valor Yahoo")
plt.legend()
plt.show()


regressor_json = regressor.to_json()
with open('regressor_previsao.json','w') as json_file:
    json_file.write(regressor_json)

regressor.save_weights('regressor_previsao.h5')




















