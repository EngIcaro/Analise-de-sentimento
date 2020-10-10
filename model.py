#%%
# Autor: Ícaro Gabriel Paiva Bastos
# Última modificação: 10/10/2020
# Sobre: Este script tem o objetivo de ler as bases de treino e teste, construir e avaliar os modelos de árvore de decisão e
# KNN e por último salvar o modelo treinado com toda a base de dados
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
from sklearn.metrics import recall_score,precision_score
#%% Carregando os arquivos de treino e de teste
X_train = np.loadtxt('./output/X_train.csv', delimiter=',')
X_test  = np.loadtxt('./output/X_test.csv' , delimiter=',')
y_train = np.loadtxt('./output/Y_train.csv', delimiter=',')
y_test  = np.loadtxt('./output/Y_test.csv',  delimiter=',')
X       = np.loadtxt('./output/X_all.csv', delimiter=',')
y       = np.loadtxt('./output/Y_all.csv',  delimiter=',')
#%% Construindo e avaliando uma árvore de decisão
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
tree.plot_tree(model, filled=True)
prediction=model.predict(X_test)
print('Acurácia da árvore de decisão: ',metrics.accuracy_score(prediction,y_test))
print('Recall da árvore de decisão: ', recall_score(prediction,y_test, average="macro"))
print('Precisão da árvore de decisão: ', precision_score(prediction,y_test, average="macro"))
#%% Construindo e avaliando um modelo KNN
model = KNeighborsClassifier(n_neighbors=2) 
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print('Acurácia do KNN: ',metrics.accuracy_score(prediction,y_test))
print('Recall   do KNN: ', recall_score(prediction,y_test, average="macro"))
print('Precisão do KNN: ', precision_score(prediction,y_test, average="macro"))
#%% Como a árvore de decisão teve um desempenho melhor que o KNN
# Vamos treinar a árvore de devisão com toda a base de dados
model = DecisionTreeClassifier()
model.fit(X,y)

#%% Salvando o modelo
pkl_filename = "./output/arvore.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)