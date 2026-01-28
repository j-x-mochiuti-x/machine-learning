import pandas as pd
from sklearn import tree

df = pd.read_excel('dados\dados_frutas.xlsx')
print(df.head())

arvore = tree.DecisionTreeClassifier()
resposta = df['Fruta']