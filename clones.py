import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_parquet("dados\dados_clones.parquet")

# Separando variaveis e a reposta
variaves = ["Massa(em kilos)", "Estatura(cm)"]
x= df[variaves]
target = 'Status '
y= df[target]


# Escolha do modelo
modelo = tree.DecisionTreeClassifier()
modelo.fit(X=x, y=y)


# Arvore de decisão plotada
plt.figure(figsize=(8,4))
tree.plot_tree(modelo, max_depth=3,feature_names=variaves, class_names=modelo.classes_, filled=True)
plt.show()
