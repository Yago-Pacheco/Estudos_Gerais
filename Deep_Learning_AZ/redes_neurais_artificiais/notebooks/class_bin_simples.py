#%%
import pandas as pd
import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#%%
path_entrada = r'C:\Users\Yago Pacheco\Desktop\Projetos\Estudos_Gerais\Deep_Learning_AZ\redes_neurais_artificiais\data\entradas_breast.csv'
path_saida = r'C:\Users\Yago Pacheco\Desktop\Projetos\Estudos_Gerais\Deep_Learning_AZ\redes_neurais_artificiais\data\saidas_breast.csv'

previsores = pd.read_csv(path_entrada)
classe = pd.read_csv(path_saida)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(previsores, 
                                                    classe, 
                                                    test_size=0.25,
                                                    random_state=42)

# %%

'''
Para saber o número de neuronios na camada oculta, a gente pega a nossa entrada (num, de colunas)
soma mais a nossa classe (aqui por ser binario é 1) e divide por dois, pegando o número int.
No nosso caso foi 15.5, logo 16.
'''

# Estrutura da RN
classificador = Sequential()

# Não precisamos fazer nossa camada de entrada, pelo fato de já usarmos o input_dim
classificador.add(Dense(units=16, 
                        activation='relu', 
                        kernel_initializer='random_uniform', 
                        input_dim=30))

# Camada de saída (apenas um neurônio pois estamos com um problema binario)
classificador.add(Dense(units=1, 
                        activation='sigmoid'))
# %%

# Executando a RN
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(X_train, y_train, batch_size=10, epochs=100)
# %%
