# %%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("dados\dados_cerveja_nota.xlsx")

df['aprovado'] = (df['nota'] > 5).astype(int)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='red');

# %%
plt.plot(df['cerveja'], df['aprovado'], 'o', color='red');
plt.grid(True)
plt.title('Cervajas x Aprovada')
plt.xlabel('cerveja')
plt.ylabel('Aprovação')

# %%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

# Regressão Logistica
reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_prebabilidade = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

# Arvore de decisão
arvore_dec = tree.DecisionTreeClassifier(random_state=42)
arvore_dec.fit(df[['cerveja']], df['aprovado'])
arvore_dec_predict = arvore_dec.predict(df[['cerveja']].drop_duplicates())
arvore_dec_prob = arvore_dec.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

# Naive Bayes
nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba =nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]


# %%
# Gráfico

plt.grid(True)
plt.title('Cervajas x Aprovada')
plt.xlabel('cerveja')
plt.ylabel('Aprovação')

plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')

plt.plot(df['cerveja'], df['aprovado'], 'o', color='red')

plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color='skyblue')
plt.plot(df['cerveja'].drop_duplicates(), reg_prebabilidade, color='grey')


plt.plot(df['cerveja'].drop_duplicates(), arvore_dec_predict, color='green')
plt.plot(df['cerveja'].drop_duplicates(), arvore_dec_prob, color='tomato')

plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color='pink')
plt.plot(df['cerveja'].drop_duplicates(), nb_proba, color='magenta')

plt.legend(['Reg predict', 'Obervação', 'Reg proba', 'Arvore predict', 'Arvore proba', 'Nb predict', 'Nb proba'])
# %%
