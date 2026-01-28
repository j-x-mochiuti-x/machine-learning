import pandas as pd
import openpyxl
from sklearn import tree
import matplotlib.pyplot as plt

# Abrindo o arquivo
df = pd.read_excel('dados\dados_cerveja.xlsx')


# Separando variaveis e a reposta
variaves = ["temperatura", "copo", "espuma", "cor"]
x=df[variaves]
target = 'classe'
y=df[target]

# Variaveis Dummies
X = x.replace({
    "mud":1,
    "pint":2,
    "sim":1,
    "não":0,
    "clara":0,
    "escura":1
})


# Escolha do modelo
modelo = tree.DecisionTreeClassifier()
modelo.fit(X=X, y=y)

# Arvore de decisão plotada
plt.figure(figsize=(12,8))
tree.plot_tree(modelo, feature_names=variaves, class_names=modelo.classes_, filled=True)
plt.show()


