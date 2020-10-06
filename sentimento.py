#%%
# nltk.download() 
# nltk.download('punkt')
#%% Importando as bibliotecas
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
#%%
# Capturando as stopwords da língua portuguesa
stopwords=nltk.corpus.stopwords.words('portuguese')
#%% pré-processamento
icaro = [['ola meu nome é icaro e o seu?', 'raiva'], ['eu estou com raiva', 'alegria']]
print(word_tokenize(icaro[0][0]))