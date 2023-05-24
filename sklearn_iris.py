import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, layers, callbacks
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow import keras 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
in this file we implement a simple neural network using forward and backward propagation on the iris dataset, 

optimizer : Adam , lr = 0.001
loss : SparseCategoricalCrossentropy

"""



# Let's load our data
data = pd.read_csv("Projet Alami/iris.csv", encoding='utf-8')
cols = data.columns.to_list()

# Let's encode our classes to numeric values 
classes = data['class'].unique().tolist()
data['class'] = data['class'].apply(lambda cls : classes.index(cls))
data = data.to_numpy()
X, y = data[:, :-1], data[:, -1]

X_train, X_test ,Y_train, Y_test = train_test_split(X, y, test_size= 0.6)

model = Sequential()

model.add(layers.Input(shape=(X.shape[1])))
model.add(layers.Dense(20, activation="relu"))
model.add(layers.Dense(20, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))


print(model.summary())

opt = keras.optimizers.Adam(learning_rate = 0.001)


callback = EarlyStopping(monitor="val_loss", patience=3)

model.compile(optimizer=opt,loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(X_train, Y_train,epochs= 1000 , callbacks=callback, validation_data=(X_test, Y_test))

y_pred = [np.argmax(item) for item in model.predict(X_test)]

print("the f1 score using forward and backwardpropagation is :", f1_score(Y_test, y_pred, average='macro'))


# Let's plot the evolution of losses :
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

































