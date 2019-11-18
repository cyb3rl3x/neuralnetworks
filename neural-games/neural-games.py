import pandas as pd

from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('C:/Users/Lex/jupyter-ds/neural-games/games.csv')

#LIMPEZA DA BASE
#remoção de atributos
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

#remoção de atributos nulos
base = base.dropna(axis=0)

#remoção de valores mínimos
base = base.loc[base['NA_Sales']>1]
base = base.loc[base['EU_Sales']>1]

#remoção de atributos não relacionados
nome_jogos = base.Name
base = base.drop('Name', axis = 1)    

#divisão da base em vendas na américa do norte, eupora e japão
previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

#transformação de valores categóricos em valores numéricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
#codifica atributos em valores numéricos
previsores[:, 0] = labelencoder.fit_transform(previsores[:,0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:,3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:,8])

#codifica atributos numéricos em atributos classificaveis
onehotencoder = OneHotEncoder(categorical_features = [0, 2, 3, 8])
previsores = onehotencoder.fit_transform(previsores).toarray()

#ESTRUTURA DA REDE

camada_entrada = Input(shape=(61, ))
camada_oculta1 = Dense(units = 32, activation = 'sigmoid')(camada_entrada) #units = (61+3)/2
camada_oculta2 = Dense(units = 32, activation = 'sigmoid')(camada_oculta1)
camada_saida1   = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida2   = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida3   = Dense(units = 1, activation = 'linear')(camada_oculta2)


regressor = Model(input = camada_entrada,
                  outputs = [camada_saida1, camada_saida2, camada_saida3])

regressor.compile(optimizer = 'adam', loss = 'mse')
regressor.fit(previsores, [venda_na, venda_eu, venda_jp],
              epochs = 5000, batch_size = 100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)


#salva o último estado da rede 
regressor_json = regressor.to_json()
with open('C:/Users/Lex/jupyter-ds/neural-games/previsor_games.json','w') as json_file:
    json_file.write(regressor_json)

regressor.save_weights('C:/Users/Lex/jupyter-ds/neural-games/regressor_game.h5')

























