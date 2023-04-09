import pandas as pd
import re
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/tweets.csv", encoding="cp437")
data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: re.sub("\t+"," ", x).strip())
data[data.columns[0]] = data[data.columns[0]].apply(lambda x: int(int(x)/4))
data = data.drop(data.columns[1:-1], axis=1)
newData = data.copy()
df2 = pd.DataFrame([newData.columns], columns=newData.columns)
newData = pd.concat([df2, newData])
newData[data.columns[-1]] = data[data.columns[0]]
newData[data.columns[0]] = data[data.columns[-1]]
newData = newData.rename(columns={newData.columns[0]: 'text', newData.columns[1]: 'value'})
train, test = train_test_split(newData, test_size=0.2, shuffle=True, stratify=newData[newData.columns[-1]])


# Se guardan los datos en dos archivos diferentes
# El index y el nombre de las columnas no se guardan
newData.to_csv("../data/tweets_processed.csv", sep="\t", index=False, header=False)
train.to_csv("../data/train.csv", sep="\t", index=False, header=False)
test.to_csv("../data/test.csv", sep="\t", index=False, header=False)