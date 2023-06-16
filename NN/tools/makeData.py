import pandas as pd
import numpy as np
import re
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Preprocesamiento de los datos
# Carga de datos

movies = pd.read_csv("../data/movie.csv")
print("data")
print(movies.columns)
movies['text']=movies['text'].apply(str. lower)
movies['text'] = movies['text'].apply(lambda x: re.sub("\t+"," ", x).strip())
print("head")
# Se dividen los datos en entrenamiento y prueba
# 30 % de los datos para prueba
test, train = train_test_split(movies, test_size=0.8, shuffle=True, stratify=movies['label'])
print("split")

# Se guardan los datos en dos archivos diferentes
# El index y el nombre de las columnas no se guardan
train.to_csv("../data/train.csv", index=False, header=False, sep="\t")
print("save1")
test.to_csv("../data/test.csv", index=False, header=False, sep="\t")
print("save2")