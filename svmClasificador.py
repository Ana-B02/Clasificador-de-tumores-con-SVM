# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:03:50 2024

@author: annie
"""
#svmclasificador cancer

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_curve, auc

 #cargar datos

data = pd.read_csv('C:/Users/annie/OneDrive/Documentos//TopicosPy/pca_resultados_cancer2.csv', header=None)

# Extract features (bin 1 and bin 2)
X = data.iloc[:, [1,2,3]].to_numpy()

# Extract target variable (column 0) and convert to binary labels
y = data.iloc[:, 0].to_numpy()
y = y.reshape(-1, 1)  # Reshape to ensure correct dimensions# 
#% dejamos las etiquetas como 0 y 1
# convertimos la etiqueta 2,3,4 en 0, la 1 se queda igual
y = np.array([(y == 1)]).T  # Transform to binary labels (1 malignos, 0 benignos)
y = y.astype(float)  # Convert to float for compatibility with SVM


#%%

 # Reshape y to a 1D array
y = np.squeeze(y)


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #, random_state=42

 # normalize the data
scaler = StandardScaler()
scaler.fit(X_train)
X = scaler.transform(X)  # Scale the training data

 

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

 

# Definir el clasificador SVM
clf = svm.SVC(kernel='rbf',verbose=True,max_iter = 10000, tol=1e-7)
# radial basis function
# Entrenar el modelo
clf.fit(X_train, y_train)

 

# # indices de los vectores de soporte
isv = clf.support_
# Dibujar la region de decision
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                      np.arange(y_min, y_max, 0.1))

 

XX = np.c_[xx.ravel(), yy.ravel()]
y_pred = clf.predict(XX).reshape(xx.shape)

 

# scores = valores de la hipotesis, previo a la funcion de activacion
y_decision = clf.decision_function(XX).reshape(xx.shape)

 

plt.figure(1)
plt.clf()
plt.plot(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], "bo") # 0 = circulos azules
plt.plot(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], "g^") # 1 = triangulos verdes
# graficamos los vectores de soporte
plt.plot(X_train[isv, 0][:], X_train[isv, 1][:], "rx") 

 

plt.grid(True, which='both')
plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plt.title('Region de evaluacion en el conjunto de Entrenamiento')
plt.contourf(xx, yy, y_pred, cmap=plt.cm.brg, alpha=0.3)
plt.show()

 

# hacer predicciones
predictions = clf.predict(X_test)

 

# evaluar la matriz de confusion
cm = confusion_matrix(y_test, predictions, normalize='all')

 

# la mostramos graficamente
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels= ['Benignos','Malignos'])
plt.figure(2)
plt.clf()
ax1 = plt.gca();
disp.plot(ax = ax1)
plt.show()

 

#%%
plt.figure(5)
plt.clf()
ax = plt.gca()
# plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                      np.arange(y_min, y_max, 0.1))

 

XX = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(XX).reshape(xx.shape)

 

y_decision = clf.decision_function(XX).reshape(xx.shape)
plt.contourf(xx, yy, y_decision, cmap=plt.cm.brg, alpha=0.5)
# ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# graficamos el valor predico
ax.scatter(X_test[:, 0], X_test[:, 1], c=predictions,  s=50,  marker='o', facecolors='none')
# graficamos los marcadores verdaderos
plt.plot(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0],  'bx') # 0 = cruces azules
plt.plot(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], "g^") # 1 =  cruces verdes
ax.set_ylabel('y ')
ax.set_xlabel('x ')
ax.set_title('Region de Decision en el conjunto de Evaluacion')
ax.legend()
plt.show()


#%% ROC curve

# false positive rate
# true positive rate
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
texto = 'Area bajo la curva ROC = %0.2f' % roc_auc
plt.figure(3)
plt.clf()
ax2 = plt.gca();
# ax2 = plt.figure(3)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax = ax2)
plt.title(texto)