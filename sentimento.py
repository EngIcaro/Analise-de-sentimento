#%%
# nltk.download() 
# nltk.download('punkt')
#%% Importando as bibliotecas
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from tqdm import tqdm
#%% Lendo a base de dados 
data_base = pd.read_csv('./input/database.csv', sep=',', encoding = "ISO-8859-1")
#%% Explorar a base de dados
data_base.info() 
print(data_base.nunique())
data_base['Sentimento'] = [x.replace('alegria', 'felicidade') for x in data_base['Sentimento']]
data_base.groupby('Sentimento', as_index=False).count().plot(kind='bar', x='Sentimento')
#%% Removendo sentimentos de desgosto,medo, surpresa 
data_base.drop(data_base[(data_base.Sentimento == 'desgosto') | (data_base.Sentimento == 'medo') | (data_base.Sentimento == 'surpresa')].index, inplace = True)
data_base.info()
# Criando as labels de cada emoção
data_base['felicidade'] = np.where(data_base['Sentimento'] == 'felicidade', 1, 0)
data_base['raiva'] = np.where(data_base['Sentimento'] == 'raiva', 1, 0)
data_base['tristeza'] = np.where(data_base['Sentimento'] == 'tristeza', 1, 0)
data_base.drop(columns= ['Sentimento'], inplace = True)
#%%

#%%
# Capturando as stopwords da língua portuguesa
stopwords=nltk.corpus.stopwords.words('portuguese')
#%% pré-processamento
# Tokenization
treinamento = data_base.iloc[:,0].values
#print(treinamento[:])
aux = word_tokenize(treinamento[:])