import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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

def taxa_de_erro(precisao, recall):
    return 2*((precisao*recall)/(precisao+recall))


#Função para realizar validação cruzada
def validacao_cruzada(model, X, y, n_dobras, n_model):

    #listas para guardar valores das métricas
    scores_accuracy = []
    scores_sensitivity = []
    scores_specificity = []
    scores_precision = []
    scores_f1_score = []
    scores_taxa_de_erro = []

    list_mean_accuracy = []
    list_mean_loss = []

    historico = []
    #Dividindo o dataset em dobras
    kf = KFold(n_splits=n_dobras, random_state=42, shuffle=True)
    fold = 0
    for train_idx, test_idx in kf.split(X):
        #Divisão dos dados para a validação
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Treinando o modelo na dobra atual
        hist = model.fit(X_train, y_train, epochs=150,verbose=0)
        historico.append(hist)
        fold = fold + 1

        #Predizendo as classes
        predictions = model.predict(X_test)

        y_pred = np.argmax(predictions, axis=1)
        y_test1 = np.argmax(y_test, axis=1)

        #Criar matriz de confusão para obter
        # Verdadeiro Positivo (TP),
        # Falso Positivo (FP),
        # Verdadeiro Negativo (TN),
        # Falso Negativo (FN),
        cm = confusion_matrix(y_test1, y_pred)
        tn, fp, fn, tp = cm.ravel()

        #Calcular as métricas
        acc = acuracia(tp, tn, fp, fn)
        precision = precisao(tp, fp)
        recall = sensibilidade(tp, fn)
        Especificidade = especificidade(tn, fp)
        f1_Score = f1__score(precision, recall)
        taxa_de_erro = 1-acc

        #Valores das métricas
        print('Acurácia:{:.4f}'.format(acc))
        print('Sensibilidade:{:.4f}'.format(recall))
        print('Especificidade:{:.4f}'.format(Especificidade))
        print('Precisão: {:.4f}'.format(precision))
        print('f1_Score: {:.4f}'.format(f1_Score))
        print('Taxa de erro: {:.4f}'.format(taxa_de_erro))

        #armazenamento dos valores em listas para média de cada métrica
        scores_accuracy.append(acc)
        mean_accuracy = np.mean(scores_accuracy)
        print('Média da acurácia: {:.4f}'.format(mean_accuracy))

        scores_sensitivity.append(recall)
        mean_sensitivity = np.mean(scores_sensitivity)
        print('Média da sensibilidade: {:.4f}'.format(mean_sensitivity))

        scores_specificity.append(Especificidade)
        mean_specificity = np.mean(scores_specificity)
        print('Média da especificidade: {:.4f}'.format(mean_specificity))

        scores_precision.append(precision)
        mean_precision = np.mean(scores_precision)
        print('Média da precisão: {:.4f}'.format(mean_precision))

        scores_f1_score.append(f1_Score)
        mean_f1_score = np.mean(scores_f1_score)
        print('Média da f1_score: {:.4f}'.format(mean_f1_score))

        scores_taxa_de_erro.append(f1_Score)
        mean_taxa_de_erro = np.mean(scores_taxa_de_erro)
        print('Média da Taxa de erro: {:.4f}'.format(mean_taxa_de_erro))

        accuracy = hist.history['accuracy']
        accuracy_np = np.array(accuracy)
        mean_accuracy_np = np.mean(accuracy_np)
        print('Média do modelo Accuracy: {:.4f}'.format(mean_accuracy_np))
        list_mean_accuracy.append(mean_accuracy_np)

        loss = hist.history['loss']
        loss_np = np.array(loss)
        mean_loss_np = np.mean(loss_np)
        print('Média do model Loss: {:.4f}'.format(mean_loss_np))
        list_mean_loss.append(mean_loss_np)
        plt.figure()

        #Mostrar curva de convergência da acúracia em função das épocas
        for history in historico:
            plt.plot(history.history['accuracy'])
        plt.title(f'Curva de Convergência da Acurácia - Arquitetura {n_model} fold = ' + str(fold))
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.savefig('history_accuracy_fold' + str(fold) + '.png')
        plt.show()

        for history in historico:
            plt.plot(history.history['loss'])
        plt.title(f'Curva de Convergência da Loss - Arquitetura {n_model} fold = ' + str(fold))
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.savefig('history_loss_fold' + str(fold) + '.png')
        plt.show()



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

#Build the Model
#Utilizar a classe Sequential()
#Definir as camadas InputLayer e Dense
#Mostrar a arquitetura
model1 = Sequential([InputLayer(input_shape=(4,)),
                    Dense(units=15, activation='relu'),
                    Dense(units=15, activation='relu'),
                    Dense(units=2, activation='softmax')])

print(model1.summary())

model2 = Sequential([InputLayer(input_shape=(4,)),
                    Dense(units=20, activation='relu'),
                    Dense(units=20, activation='relu'),
                    Dense(units=2, activation='softmax')])

print(model2.summary())

#Compilar o modelo
#Config otimizador, função de perda e métricas
model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy',
                      metrics=['accuracy'])
model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy',
                      metrics=['accuracy'])
#Train the Model
#Chamar a função fit() para treinar
#Passar como parâmetros x_train, y_train, tamanho do batch e quantidade de épocas
#history = model.fit(x_train, y_train, batch_size=32, epochs=20)

fold = 5
# Realizando validação cruzada com 5 dobras
validacao_cruzada(model1, X, encoded, n_dobras= fold, n_model=1)

validacao_cruzada(model2, X, encoded, n_dobras= fold, n_model=2)



