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
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
def clean_text(text):
    
    #text = text.encode('ascii', errors = 'ignore').decode() #Decodificando caracteres em ASCII
    text = text.lower() #Apenas caracteres minúsculos
    text = text.strip(' ') #Removendo espaços do começo e fim   
    #text = re.sub("[^A-Za-z0-9]+", ' ',text)
    return text
tqdm.pandas(desc='Limpando o texto')
data_base['text_tokens'] = data_base['Frases'].progress_apply(clean_text)
#%%
# Capturando as stopwords da língua portuguesa
#stopwords=nltk.corpus.stopwords.words('portuguese')
def remove_stopwords(text):
    
  stop_words = set(stopwords.words('portuguese')) 

  word_tokens = word_tokenize(text) 

  filtered_sentence = [w for w in word_tokens if not w in stop_words] 
      
  return filtered_sentence

tqdm.pandas(desc='Removendo as stopwords e tokenizando o texto')
data_base['text_tokens'] = data_base['text_tokens'].progress_apply(remove_stopwords)

#%%
text_tokens = []
 
for vet in data_base['text_tokens'].values:
    text_tokens.extend(vet)
 
num_words = len((list(set(text_tokens))))
print(num_words)

#%%
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(data_base['text_tokens'].values)

data_base['X'] = tok.texts_to_sequences(data_base['text_tokens'])
#%%
data_base['num_words'] = data_base['text_tokens'].apply(lambda x : len(x))
 
max_num_words = data_base['num_words'].max()
#%%
labels = data_base.iloc[:,1:4].columns.values
X = pad_sequences(maxlen=max_num_words, sequences=data_base['X'], value=0, padding='post', truncating='post')
y = data_base[labels].values
 
print('Dimensão do X: {}'.format(X.shape))
print('Dimensão do y: {}'.format(y.shape))
#%% pré-processamento
# Tokenization
treinamento = data_base.iloc[:,0].values
#print(treinamento[:])
aux = word_tokenize(treinamento[:])