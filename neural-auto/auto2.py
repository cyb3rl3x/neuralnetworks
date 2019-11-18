#bibliotecas
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#Load dataset
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

#Limpeza do dataset
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

#Retirar outliers
base = base[base.price > 10]
base = base[base.price < 350000]

#substituir os valores faltantes pelos valores que mais se repetem
valores = {'vehicleType': 'limousine', 
           'gearbox': 'manuell', 
           'model':'golf', 
           'fuelType':'bezin', 
           'notRepairedDamage':'nein'}
base = base.fillna(value = valores)

#divide a base entre atributos e previsões
previsores = base.iloc[:, 1:13].values
#print(previsores)

preco_real = base.iloc[:, 0].values
#print(preco_real)

#transformação de valores categóricos em numéricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_previsores = LabelEncoder()

previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:, 0] )
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:, 1] )
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:, 3] )
previsores[:, 5] = label_encoder_previsores.fit_transform(previsores[:, 5] )
previsores[:, 8] = label_encoder_previsores.fit_transform(previsores[:, 8] )
previsores[:, 9] = label_encoder_previsores.fit_transform(previsores[:, 9] )
previsores[:, 10] = label_encoder_previsores.fit_transform(previsores[:, 10] )

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 3, 5, 8, 9, 10])
previsores = onehotencoder.fit_transform(previsores).toarray() 

#Estrutura da rede neural para regressão (previsão de um valor)
regressor = Sequential()
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 317))
regressor.add(Dense(units = 158, activation = 'relu'))
# para regressão utiliza-se a função linear
regressor.add(Dense(units = 1, activation = 'linear'))
#mean_absolute_error(erro médio absoluto => positivo)
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)







