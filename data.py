import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DFO_NN import NeuralNetwork


# Let's load our data
data = pd.read_csv("/home/amine/Desktop/VS CODE /Projet Alami/iris.csv", encoding='utf-8')
cols = data.columns.to_list()

# Let's encode our classes to numeric values 
classes = data['class'].unique().tolist()
data['class'] = data['class'].apply(lambda cls : classes.index(cls))


'''# now we one hot encode our classes to binary codes
encoded_labels = pd.get_dummies(data['class'], prefix= 'flower', dtype=int)
data[encoded_labels.columns] = encoded_labels
new_cols = cols[:-1]+encoded_labels.columns.tolist()
final_data = data[new_cols]
'''
data = data.to_numpy()
X, y = data[:, :-1], data[:, -1]

print(y)


NN = NeuralNetwork(X, y, {0: (10, 'Relu'), 1:(3, 'Softmax')}, 40, 10, 1,epochs=10,max_iter=10)
hope = NN.train()