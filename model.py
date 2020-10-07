#%%
# Autor: Ícaro Gabriel Paiva Bastos
# Última modificação: 07/10/2020

#%% Importando as bibliotecas
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords 
from numpy import loadtxt
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import pickle

#%% Carregando os arquivos de treino e de teste

X_train = np.loadtxt('./output/X_train.csv', delimiter=',')
X_test  = np.loadtxt('./output/X_test.csv' , delimiter=',')
y_train = np.loadtxt('./output/Y_train.csv', delimiter=',')
y_test  = np.loadtxt('./output/Y_test.csv',  delimiter=',')
X       = np.loadtxt('./output/X_all.csv', delimiter=',')
y       = np.loadtxt('./output/Y_all.csv',  delimiter=',')
#%% Avaliando uma árvore de decisão
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
tree.plot_tree(model, filled=True)
prediction=model.predict(X_test)
print('Acurácia da árvore de decisão: ',metrics.accuracy_score(prediction,y_test))

#%% Avaliando um modelo KNN
model = KNeighborsClassifier(n_neighbors=2) 
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print('Acurácia do KNN: ',metrics.accuracy_score(prediction,y_test))

#%% Treinando o modelo com toda a base
model = DecisionTreeClassifier()
model.fit(X,y)

#%% Salvando o modelo
pkl_filename = "./output/arvore.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)