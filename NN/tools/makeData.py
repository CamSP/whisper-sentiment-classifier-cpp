import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Preprocesamiento de los datos
# Carga de datos
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# One hot encoder transforma la columna objetivo con n valores unicos, 
# a n columnas con valores de 1 o 0 si corresponden a la salida
encoder = OneHotEncoder(handle_unknown='ignore')

# Se crea un nuevo dataset con el one hot encoder aplicado
encoder_df = pd.DataFrame(encoder.fit_transform(data[['target']]).toarray())

# Se añade al dataset original
data = data.join(encoder_df)

# Se elimina la columna objetivo original
data.drop("target", axis=1, inplace=True)

# Se re organizan de forma aleatoria los datos
data = data.sample(frac=1).reset_index(drop=True)

# Se dividen los datos en entrenamiento y prueba
# 30 % de los datos para prueba
train, test = train_test_split(data, test_size=0.3)

# Se guardan los datos en dos archivos diferentes
# El index y el nombre de las columnas no se guardan
train.to_csv("../data/iris_train.csv", index=False, header=False)
test.to_csv("../data/iris_test.csv", index=False, header=False)