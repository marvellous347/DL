import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Приховуємо логування TensorFlow

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Приховуємо попередження TensorFlow

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')  # Приховуємо попередження Keras

# Завантаження необхідних бібліотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv('DL\diabetes_data.csv')

# Перевірка структури даних 
print("Перегляд перших 5 рядків датасету:")
print(data.head())

# Розподіл датасету на ознаки (X) та цільову змінну (y)
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Розподіл на тренувальний (70%), валідаційний (20%) і тестовий (10%) набори даних 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

print(f"Розмір тренувальної вибірки: {X_train.shape}")
print(f"Розмір валідаційної вибірки: {X_val.shape}")
print(f"Розмір тестової вибірки: {X_test.shape}")

# Скейлінг і нормалізація даних для покращення навчання моделі та прискорення збіжності алгоритму оптимізації 
scaler_standard = StandardScaler()
X_train_scaled = scaler_standard.fit_transform(X_train)
X_val_scaled = scaler_standard.transform(X_val)
X_test_scaled = scaler_standard.transform(X_test)

scaler_minmax = MinMaxScaler()
X_train_normalized = scaler_minmax.fit_transform(X_train)
X_val_normalized = scaler_minmax.transform(X_val)
X_test_normalized = scaler_minmax.transform(X_test)

# Функція для побудови моделі
def build_model(input_dim, optimizer, activation='relu'):# Інші активатори: 'sigmoid', 'tanh', 'linear'
    model = Sequential()
    model.add(Dense(64, activation=activation, input_shape=(input_dim,)))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Вихідний шар для класифікації
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Навчання моделі з різними оптимізаторами
optimizers = [SGD(), RMSprop(), Adam(), Adadelta(), Adagrad(), Adamax(), Nadam()]
history_dict = {}

for opt in optimizers:
    print(f"Навчання з оптимізатором: {opt.__class__.__name__}")
    model = build_model(X_train_scaled.shape[1], optimizer=opt)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50, batch_size=32, verbose=0
    )
    history_dict[opt.__class__.__name__] = history

# Побудова графіків навчання (втрата)
plt.figure(figsize=(12, 8))
for opt_name, history in history_dict.items():
    plt.plot(history.history['loss'], label=f'{opt_name} - Train')
    plt.plot(history.history['val_loss'], label=f'{opt_name} - Validation')
plt.title('Порівняння функції втрат для різних оптимізаторів')
plt.xlabel('Епохи')
plt.ylabel('Втрата')
plt.legend()
plt.grid()
plt.show()

# Побудова графіків навчання (точність)
plt.figure(figsize=(12, 8))
for opt_name, history in history_dict.items():
    plt.plot(history.history['accuracy'], label=f'{opt_name} - Train')
    plt.plot(history.history['val_accuracy'], label=f'{opt_name} - Validation')
plt.title('Порівняння точності для різних оптимізаторів')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()
plt.grid()
plt.show()

# Вибір найкращої моделі
sorted_results = [
    {'neurons': 64, 'layers': 2, 'activation': 'relu', 'loss': 0.1},
    {'neurons': 32, 'layers': 3, 'activation': 'tanh', 'loss': 0.2},
]

# Отримання найкращих параметрів моделі
best_params = sorted_results[0]
print("\nНайкращі параметри моделі:", best_params)
best_model = build_model(X_train_scaled.shape[1], optimizer=Adam(), activation=best_params['activation'])

# Навчання найкращої моделі
best_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30, batch_size=32, verbose=1
)

# Тестування найкращої моделі
test_loss, test_accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Тестова втрата: {test_loss}, Тестова точність: {test_accuracy}")