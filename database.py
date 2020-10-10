#%%
# Autor: Ícaro Gabriel Paiva Bastos
# Última modificação: 10/10/2020
# Sobre: Este script tem o objetivo de ler a base de dados, fazer o pré-processamento dos dados
# e gerar os arquivos de traino e teste
#%% Importando as bibliotecas
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
import string
import pickle

#%% Declarando algumas funções
# Função responsável por deixar todos os caracteres em minúsculo, remover espaços em branco e pontuação.
def clean_text(text):
    
    text = text.lower() 
    text = text.strip(' ') 
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Capturando as stopwords da língua portuguesa e retirando das frases.
def remove_stopwords(text):  
    stop_words = set(stopwords.words('portuguese')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]     
    return filtered_sentence

# Removendo o tempo gramatical.
def lemmatizer_words(text):
    porter = nltk.PorterStemmer()
    lemma_text = [porter.stem(t) for t in text]
    return lemma_text

#%% Lendo a base de dados 
data_base = pd.read_csv('./input/database.csv', sep=',', encoding = "ISO-8859-1")

#%% Explorando a base de dados
data_base.info() 
print(data_base.nunique())
data_base['Sentimento'] = [x.replace('alegria', 'felicidade') for x in data_base['Sentimento']]
data_base.groupby('Sentimento', as_index=False).count().plot(kind='bar', x='Sentimento')

#%% Removendo sentimentos de desgosto,medo e surpresa 
data_base.drop(data_base[(data_base.Sentimento == 'desgosto') | (data_base.Sentimento == 'medo') | (data_base.Sentimento == 'surpresa')].index, inplace = True)
data_base.info()
# Criando as labels de cada emoção
data_base['felicidade'] = np.where(data_base['Sentimento'] == 'felicidade', 1, 0)
data_base['raiva'] = np.where(data_base['Sentimento'] == 'raiva', 1, 0)
data_base['tristeza'] = np.where(data_base['Sentimento'] == 'tristeza', 1, 0)
# Removendo a coluna sentimento
data_base.drop(columns= ['Sentimento'], inplace = True)

#%% Aplicando a função clean_text nas frases e atribuindo a uma nova coluna
tqdm.pandas(desc='Limpando o texto')
data_base['text_tokens'] = data_base['Frases'].progress_apply(clean_text)

#%% Aplicando a função remove_stopwords na coluna text_tokens
tqdm.pandas(desc='removendo stopwords')
data_base['text_tokens'] = data_base['text_tokens'].progress_apply(remove_stopwords)

#%% Aplicando a função lemmatizer_words na coluna text_tokens
tqdm.pandas(desc='lemmatizer words')
data_base['text_tokens'] = data_base['text_tokens'].progress_apply(lemmatizer_words)

#%% Contando o número de palavras únicas
text_tokens = []
for vet in data_base['text_tokens'].values:
    text_tokens.extend(vet)
num_words = len((list(set(text_tokens))))

#%% Vetorizando um texto com Tokenizer do keras
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(data_base['text_tokens'].values)
# Salvando o Tokenizer para ser utilizado na predição do sistema
with open('./output/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Salvando os vetores em uma nova coluna chamada X
data_base['X'] = tok.texts_to_sequences(data_base['text_tokens'])

#%% Contando o número máximo de palavras de todas as frases  
data_base['num_words'] = data_base['text_tokens'].apply(lambda x : len(x))
max_num_words = data_base['num_words'].max()

#%% Gerando os novos X e Y que serão utilizados nas bases de treino e teste
labels = data_base.iloc[:,1:4].columns.values
X = pad_sequences(maxlen=max_num_words, sequences=data_base['X'], value=0, padding='post', truncating='post')
y = data_base[labels].values

#%% Gerando os cojuntos de treino e teste com a mesma proporção de classes
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
aux1 = sss.split(X, y)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

np.savetxt('./output/X_train.csv', X_train,delimiter=',')
np.savetxt('./output/X_test.csv', X_test,delimiter=',')
np.savetxt('./output/Y_train.csv', y_train,delimiter=',')
np.savetxt('./output/Y_test.csv', y_test,delimiter=',')
np.savetxt('./output/X_all.csv', X,delimiter=',')
np.savetxt('./output/Y_all.csv', y,delimiter=',')