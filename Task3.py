import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Приховуємо логи TensorFlow

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Приховуємо попередження TensorFlow

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras') # Приховуємо попередження Keras

# Завантаження необхідних бібліотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv('DL/2014_usa_states.csv')

# Перевірка структури даних 
print("Перегляд перших 5 рядків датасету:")
print(data.head())

# Створення категорій для класифікації 
bins = [0, 2000000, 10000000, 40000000]
labels = ['Low', 'Medium', 'High']
data['Population_Category'] = pd.cut(data['Population'], bins=bins, labels=labels)

# Перетворення категорій на числові значення 
label_encoder = LabelEncoder()
data['Population_Category'] = label_encoder.fit_transform(data['Population_Category'])

# Розподіл на підвибірки
X = data[['Rank']]  # Використовуємо 'Rank' як вхідну змінну
y = data['Population_Category']

# Розподіл на тренувальний (70%), валідаційний (20%) і тестовий (10%) набори даних 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

print(f"Розмір тренувальної вибірки: {X_train.shape}")
print(f"Розмір валідаційної вибірки: {X_val.shape}")
print(f"Розмір тестової вибірки: {X_test.shape}")

# Скейлінг даних для покращення навчання моделі та прискорення збіжності алгоритму оптимізації 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Побудова моделі для класифікації 
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Вихідний шар для класифікації (кількість категорій) 
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Компіляція та навчання моделі 
model = build_model(X_train_scaled.shape[1])
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50, batch_size=32, verbose=1
)

# Тестування моделі 
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Тестова втрата: {test_loss}, Тестова точність: {test_accuracy}")

# Візуалізація результатів навчання 
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Функція втрат під час навчання')
plt.xlabel('Епохи')
plt.ylabel('Втрата')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точність під час навчання')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()
plt.grid()
plt.show()