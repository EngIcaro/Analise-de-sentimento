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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import string
from sklearn.svm import SVC
#from xgboost import XGBClassifier
#from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
import pickle
from nltk.stem import WordNetLemmatizer
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
    text = text.translate(str.maketrans('', '', string.punctuation))
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

def lemmatizer_words(text):
    porter = nltk.PorterStemmer()
    lemma_text = [porter.stem(t) for t in text]
    #lemmatizer = WordNetLemmatizer() 
    #lemma_text = lemmatizer.lemmatize(text)
    return lemma_text
tqdm.pandas(desc='Removendo as stopwords e tokenizando o texto')
data_base['text_tokens'] = data_base['text_tokens'].progress_apply(lemmatizer_words)
#%%
text_tokens = []
 
for vet in data_base['text_tokens'].values:
    text_tokens.extend(vet)
 
num_words = len((list(set(text_tokens))))
print(num_words)

#%%
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(data_base['text_tokens'].values)
# saving
with open('./output/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
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
# Teste com o train test split do Scikit-learn
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
aux1 = sss.split(X, y)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
plt.hist(y_test)

#%%
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
tree.plot_tree(model, filled=True)
prediction=model.predict(X_test)
print('The accuracy of the ID3 Regression is',metrics.accuracy_score(prediction,y_test))
#%%
model = KNeighborsClassifier(n_neighbors=2) 
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print('The accuracy of the KNN is:',metrics.accuracy_score(prediction,y_test))
#%%
# Treinando o modelo com toda a base
model = DecisionTreeClassifier()
model.fit(X,y)
#%%
pkl_filename = "./output/arvore.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
    
#%%
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
prediction=pickle_model.predict(X_test)