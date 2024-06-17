import pandas as pd
from keras.src.losses import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

#Funções para o cálculo das métricas
def acuracia(tp, tn, fp, fn):
    return (tp+tn)/(tp+fp+tn+fn)

def precisao(tp, fp):
    return tp/(tp+fp)

def sensibilidade(tp, fn):
    return tp/(tp+fn)

def especificidade(tn, fp):
    return tn/(tn+fp)

def f1__score(precisao, recall):
    return 2*((precisao*recall)/(precisao+recall))

#Função para realizar validação cruzada
def validação_cruzada(model, X, y, n_dobras):
    scores_accuracy = []
    scores_sensibility = []
    scores_especifity = []
    scores_precision = []
    scores_
    mse_scores = []
    #Dividindo o dataset em dobras
    kf = KFold(n_splits=n_dobras, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Treinando o modelo na dobra atual
        history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

        #Predizendo as classes
        predictions = model.predict(X_test)

        y_pred = np.argmax(predictions, axis=1)
        y_test1 = np.argmax(y_test, axis=1)

        #Avaliando o modelo na dobra atual
        #score = modelo.evaluate(X_test, y_test)
        #scores.append(score[1])  # Pegando a métrica de precisão (índice 1)
        #Criar matriz de confusão para obter
        cm = confusion_matrix(y_test1, y_pred)
        tn, fp, fn, tp = cm.ravel()

        #Calcular as métricas
        acc = acuracia(tp, tn, fp, fn)
        precision = precisao(tp, fp)
        recall = sensibilidade(tp, fn)
        Especificidade = especificidade(tn, fp)
        f1_Score = f1__score(precision, recall)


        print('Acurácia:{:.4f}'.format(acc))
        print('Sensibilidade:{:.4f}'.format(recall))
        print('Especificidade:{:.4f}'.format(Especificidade))
        print('Precisão: {:.4f}'.format(precision))
        print('f1_Score: {:.4f}'.format(f1_Score))

        scores_accuracy.append(acc)
        media_accuracy = np.mean(scores_accuracy)
        print('Média da acurácia: {}'.format(media_accuracy))

#Usar o pd.read_excel para ler o arquivo
data = pd.read_excel('mammographic_masses.xlsx')
#print(data.to_string())

#print(data.columns)

#Remove linhas ou colunas com valores ausentes (NaN) de um DataFrame
data_changed = data.dropna()
#print(data_changed.to_string())

#Definição das características
features = ['Age', 'Shape', 'Margin', 'Density'] #definição das características

#Separar X e y
X = data_changed[features].values
y = data_changed['Severity'].values #variável que será prevista

#print(X.shape)
np.unique(y)
#print(y)

#converter o vetor de inteiros das classes para uma matriz hot-encoder
#para classe binária, usar função de perda `binary_crossentropy`
#para vetor de inteiros, usar função de perda `sparse_categorical_crossentropy`
encoded = tf.keras.utils.to_categorical(y)
#print(encoded)

#utilizar o train_test_split do sklearn para dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, encoded, test_size=0.2, random_state=0)
print(x_train.shape)
print(x_test.shape)

#Build the Model
#Utilizar a classe Sequential()
#Definir as camadas InputLayer e Dense
#Mostrar a arquitetura
model = Sequential([InputLayer(input_shape=(4,)),
                    Dense(units=10, activation='relu'),
                    Dense(units=10, activation='relu'),
                    Dense(units=2, activation='softmax')])

print(model.summary())

#Compilar o modelo
#Config otimizador, função de perda e métricas
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the Model
#Chamar a função fit() para treinar
#Passar como parâmetros x_train, y_train, tamanho do batch e quantidade de épocas
#history = model.fit(x_train, y_train, batch_size=32, epochs=20)


# Realizando validação cruzada com 5 dobras
scores = validação_cruzada(model, X, encoded, n_dobras=5)

# Imprimindo os resultados
print("Precisão média:", scores)