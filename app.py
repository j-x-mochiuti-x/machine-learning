import streamlit as st
import pandas as pd

st.markdown("Descubra a felicidade.app")


redes=['LinkedIn', 'Twitch', 'YouTube', 'Instagram', 'Amigos', 'Twitter / X', 'Outra rede social']
anos=['0', 'Mais que 3', '2', '1', '3']
estado=['MG', 'SC', 'SP', 'CE', 'PE', 'RJ', 'AM', 'PR', 'BA', 'PA', 'MT',
       'RS', 'DF', 'RN', 'ES', 'PB', 'GO', 'MA']
formacao=['Biológicas', 'Exatas', 'Humanas']
temp_dados=['De 0 a 6 meses', 'De 1 ano a 2 anos', 'De 6 meses a 1 ano',
       'Mais de 4 anos', 'Não atuo', 'de 2 anos a 4 anos']
seneriodade=['C-Level', 'Coordenação', 'Diretoria', 'Especialista', 'Gerência',
       'Iniciante', 'Júnior', 'Pleno', 'Sênior']


col1, col2, col3 = st.columns(3)

with col1:
    videogame = st.radio("Curte video games?", ['Sim', 'Não'])
    futebol = st.radio("Curte futebol?", ['Sim', 'Não'])
    idade = st.number_input("Sua idade", 15, 100)
    senior = st.selectbox("Posição da cadeira (senioridade)",options=seneriodade)
with col2:
    livros = st.radio("Curte livros?", ['Sim', 'Não'])
    jogos_de_tabuleiro = st.radio("Curte jogos de tabuleiro?", ['Sim', 'Não'])
    redes_conhece = st.selectbox("Como conheceu o Téo Me Why?", options=redes)
    cursos = st.selectbox("Quantos cursos acompanhou do Téo Me Why?", options=anos)
with col3:
    formula1 = st.radio("Curte fórmula 1?", ['Sim', 'Não'])
    mma = st.radio("Curte MMA?", ['Sim', 'Não'])
    moradia_UF = st.selectbox("Estado que mora atualmente",options=estado)
    area_formacao =st.selectbox("Área de Formação",options=formacao)
    tempo_dados = st.selectbox("Tempo que atua na área de dados",options=temp_dados)


dados = {'Como conheceu o Téo Me Why?': redes_conhece,
         'Quantos cursos acompanhou do Téo Me Why?': anos,
         'Curte games?':videogame,
         'Curte futebol?':futebol,
         'Curte livros?':livros,
         'Curte jogos de tabuleiro?':jogos_de_tabuleiro,
         'Curte jogos de fórmula 1?':formula1,
         'Curte jogos de MMA?':mma,
         'Idade':idade,
         'Estado que mora atualmente':moradia_UF,
         'Área de Formação':area_formacao,
         'Tempo que atua na área de dados':tempo_dados, 
         'Posição da cadeira (senioridade)':senior,
         }

df = pd.DataFrame([dados]).replace({"Sim":1,"Não":0})

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
              
df_template = pd.DataFrame(columns=['Como conheceu o Téo Me Why?_Amigos',
       'Como conheceu o Téo Me Why?_Instagram',
       'Como conheceu o Téo Me Why?_LinkedIn',
       'Como conheceu o Téo Me Why?_Outra rede social',
       'Como conheceu o Téo Me Why?_Twitch',
       'Como conheceu o Téo Me Why?_Twitter / X',
       'Como conheceu o Téo Me Why?_YouTube',
       'Quantos cursos acompanhou do Téo Me Why?_0',
       'Quantos cursos acompanhou do Téo Me Why?_1',
       'Quantos cursos acompanhou do Téo Me Why?_2',
       'Quantos cursos acompanhou do Téo Me Why?_3',
       'Quantos cursos acompanhou do Téo Me Why?_Mais que 3',
       'Estado que mora atualmente_AM', 'Estado que mora atualmente_BA',
       'Estado que mora atualmente_CE', 'Estado que mora atualmente_DF',
       'Estado que mora atualmente_ES', 'Estado que mora atualmente_GO',
       'Estado que mora atualmente_MA', 'Estado que mora atualmente_MG',
       'Estado que mora atualmente_MT', 'Estado que mora atualmente_PA',
       'Estado que mora atualmente_PB', 'Estado que mora atualmente_PE',
       'Estado que mora atualmente_PR', 'Estado que mora atualmente_RJ',
       'Estado que mora atualmente_RN', 'Estado que mora atualmente_RS',
       'Estado que mora atualmente_SC', 'Estado que mora atualmente_SP',
       'Área de Formação_Biológicas', 'Área de Formação_Exatas',
       'Área de Formação_Humanas',
       'Tempo que atua na área de dados_De 0 a 6 meses',
       'Tempo que atua na área de dados_De 1 ano a 2 anos',
       'Tempo que atua na área de dados_De 6 meses a 1 ano',
       'Tempo que atua na área de dados_Mais de 4 anos',
       'Tempo que atua na área de dados_Não atuo',
       'Tempo que atua na área de dados_de 2 anos a 4 anos',
       'Posição da cadeira (senioridade)_C-Level',
       'Posição da cadeira (senioridade)_Coordenação',
       'Posição da cadeira (senioridade)_Diretoria',
       'Posição da cadeira (senioridade)_Especialista',
       'Posição da cadeira (senioridade)_Gerência',
       'Posição da cadeira (senioridade)_Iniciante',
       'Posição da cadeira (senioridade)_Júnior',
       'Posição da cadeira (senioridade)_Pleno',
       'Posição da cadeira (senioridade)_Sênior', 'Curte games?',
       'Curte futebol?', 'Curte livros?', 'Curte jogos de tabuleiro?',
       'Curte jogos de fórmula 1?', 'Curte jogos de MMA?', 'Idade',
       'pessoa feliz'])

pd.concat([df_template, df]).fillna(0)
st.dataframe(df)