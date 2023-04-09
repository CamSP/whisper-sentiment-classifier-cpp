import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Carga del dataset original
data = pd.read_csv("../data/tweets.csv", encoding="cp437")
# Se toman 100.000 valores, stratify asegura que los datos esten balanceados (# de 0 = # de 1)
_, data = train_test_split(data, test_size=0.0625, shuffle=True, stratify=data[data.columns[0]])
# Se eliminan los \t
data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: re.sub("\t+"," ", x).strip())
# Se convierten los datos objetivo a 0 y 1
data[data.columns[0]] = data[data.columns[0]].apply(lambda x: int(int(x)/4))
# Se eliminan columnas sobrantes
data = data.drop(data.columns[1:-1], axis=1)

newData = data.copy()
#df2 = pd.DataFrame([newData.columns], columns=newData.columns)
# Se reorganiza el dataset
newData[data.columns[-1]] = data[data.columns[0]]
newData[data.columns[0]] = data[data.columns[-1]]
# Se cambian los nombres
newData = newData.rename(columns={newData.columns[0]: 'text', newData.columns[1]: 'value'})
# Se divide el dataset en entrenamiento y prueba
train, test = train_test_split(newData, test_size=0.2, shuffle=True, stratify=newData[newData.columns[-1]])


# Se guardan los datos en dos archivos diferentes
# El index y el nombre de las columnas no se guardan
newData.to_csv("../data/tweets_processed.csv", sep="\t", index=False, header=False)
train.to_csv("../data/train.csv", sep="\t", index=False, header=False)
test.to_csv("../data/test.csv", sep="\t", index=False, header=False)