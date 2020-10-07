import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
#%%
frase = input("Olá, seja bem-vindo!\nO nosso sistema é capaz de detectar o sentimento contido em uma frase.\nDigite uma frase:")
def clean_text(text):
    
    #text = text.encode('ascii', errors = 'ignore').decode() #Decodificando caracteres em ASCII
    text = text.lower() #Apenas caracteres minúsculos
    text = text.strip(' ') #Removendo espaços do começo e fim   
    text = text.translate(str.maketrans('', '', string.punctuation))
    #text = re.sub("[^A-Za-z0-9]+", ' ',text)
    return text

frase = clean_text(frase)

def remove_stopwords(text):
    
  stop_words = set(stopwords.words('portuguese')) 
  word_tokens = word_tokenize(text) 

  filtered_sentence = [w for w in word_tokens if not w in stop_words] 
      
  return filtered_sentence

frase = remove_stopwords(frase)
#%%
def lemmatizer_words(text):
    porter = nltk.PorterStemmer()
    lemma_text = [porter.stem(t) for t in text]
    #lemmatizer = WordNetLemmatizer() 
    #lemma_text = lemmatizer.lemmatize(text)
    return lemma_text
frase = lemmatizer_words(frase)
#%% loading
with open('./output/tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)
#%%
frase = tok.texts_to_sequences(frase)
#%%
frase = [list(x) for x in zip(*frase)]
#%%
x = pad_sequences(maxlen=6, sequences=frase, value=0, padding='post', truncating='post')
pkl_filename = "./output/arvore.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
prediction=pickle_model.predict(x)

if(prediction[0][0] == 1):
    print("Sentimento -> felicidade")
elif(prediction[0][1] == 1):
    print("Sentimento -> raiva")
else:
    print("Sentimento -> tristeza")