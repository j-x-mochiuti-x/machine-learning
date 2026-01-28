import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

#carregar oa rquivo
df = pd.read_excel('dados\dados_frutas.xlsx')

#arrumando o modelo
arvore = tree.DecisionTreeClassifier(random_state=42)
y = df['Fruta']
caracteristicas = ["Arredondada","Suculenta",'Vermelha','Doce']
z = df[caracteristicas]

# Treinando o modelo
analise_ml = arvore.fit(z.values, y)

# Teste de previsão
previsa = analise_ml.predict([[1,1,1,0]])
print(previsa)

# Teste de probabilidade
probabilidade= arvore.predict_proba([[1,1,1,0]])
proba = pd.Series(probabilidade[0], index=arvore.classes_)
print(proba)

# Plot da arvore de decisão
plt.figure(figsize=(12,8))
tree.plot_tree(analise_ml, feature_names=caracteristicas, class_names=analise_ml.classes_, filled=True, rounded=True)
plt.show()

