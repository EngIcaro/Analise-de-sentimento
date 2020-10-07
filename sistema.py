import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import string
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% Declarando algumas funções
# Função responsável por deixar todos os caracteres em minúsculo, remover espaços em braanco e pontuação, 
def clean_text(text):
    
    text = text.lower() 
    text = text.strip(' ') 
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Capturando as stopwords da língua portuguesa e retirando das frases
def remove_stopwords(text):  
    stop_words = set(stopwords.words('portuguese')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]     
    return filtered_sentence

# Removendo o tempo gramatical
def lemmatizer_words(text):
    porter = nltk.PorterStemmer()
    lemma_text = [porter.stem(t) for t in text]
    return lemma_text

#%% Carregando o tokenizer, modelo
with open('./output/tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)
    
with open('./output/arvore.pkl', 'rb') as file:
    pickle_model = pickle.load(file)
#%%
frase = input("Olá, seja bem-vindo!\nO nosso sistema é capaz de detectar o sentimento contido em uma frase.\nDigite uma frase:")

frase = clean_text(frase)
frase = remove_stopwords(frase)
frase = lemmatizer_words(frase)

frase = tok.texts_to_sequences(frase)
frase = [list(x) for x in zip(*frase)]

x = pad_sequences(maxlen=6, sequences=frase, value=0, padding='post', truncating='post')

prediction=pickle_model.predict(x)

if(prediction[0][0] == 1):
    print("Sentimento -> felicidade")
elif(prediction[0][1] == 1):
    print("Sentimento -> raiva")
else:
    print("Sentimento -> tristeza")