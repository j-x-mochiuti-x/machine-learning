# %%
import pandas as pd

df = pd.read_excel("dados\dados_cerveja_nota.xlsx")
df.head()
# %%
# Modelo de Regressão

from sklearn import linear_model
X = df[['cerveja']]
y = df['nota']

regression = linear_model.LinearRegression()
regression.fit(X, y)
# %%
# Predição
predict = regression.predict(X.drop_duplicates())

# %%
import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação CervejaxNota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates()['cerveja'], predict)
# %%
