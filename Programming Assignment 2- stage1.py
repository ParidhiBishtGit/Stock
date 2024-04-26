#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_excel(r"Folds5x2_pp.xlsx")
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# Split the dataset
X = data[:, :4]  # Input features
y = data[:, 4:]  # Output variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Define the ANN architecture using TensorFlow
input_size = 4
hidden_size = 3
output_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='tanh', input_shape=(input_size,), name='hidden_layer'),
    tf.keras.layers.Dense(output_size, activation='linear', name='output_layer')
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
epochs = 100
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1)

# Evaluate the model on the test set
test_cost = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Cost: {test_cost}")

# Make predictions on new data
predictions = model.predict(X_test)
print(f"Prediction: {predictions[0]}, Actual: {y_test[0]}")
print()

# Define the MAPE loss and accuracy functions using TensorFlow
def mape_loss(y_true, y_pred):
    eps = 0.001
    loss = tf.reduce_mean((tf.abs(y_true - y_pred) * 100) / (tf.abs(y_pred) + eps))
    print(f"MAPE Loss: {loss:.4f}")

def accuracy(y_true, y_pred, threshold=0.1):
    correct_predictions = tf.reduce_sum(tf.cast(tf.abs(y_pred - y_true) <= threshold, tf.float32))
    accuracy = (correct_predictions / len(y_pred)) * 100
    print(f"Accuracy: {accuracy:.4f}%")

# Evaluate MAPE loss and accuracy on the test set
#mape_loss(y_test, predictions)
mape_loss(tf.cast(y_test, tf.float32), predictions)
accuracy(y_test, predictions)
print()

# Plot the convergence graph
plt.figure()
plt.title('Convergence of Training')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




