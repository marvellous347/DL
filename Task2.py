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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
import matplotlib.pyplot as plt

# Завантаження даних 
data = pd.read_csv('DL/hourly_wages_data.csv')

# Перевірка структури даних
print("Перегляд перших 5 рядків датасету:")
print(data.head())

# Розподіл на підвибірки для регресії (визначення зарплати на годину) 
X = data.drop(columns=['wage_per_hour'])
y = (data['wage_per_hour'] > data['wage_per_hour'].median()).astype(int)  # Бінарна класифікація (вища/нижча медіани) 

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

# Побудова базової моделі для порівняння оптимізаторів та активаторів 
def build_model(input_dim, optimizer, activation='relu'):
    model = Sequential()
    model.add(Dense(128, activation=activation, input_shape=(input_dim,)))
    model.add(Dense(64, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Вихідний шар для класифікації (бінарна) 
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Навчання моделі з різними оптимізаторами та збереження історії навчання для кожного оптимізатора 
optimizers = [SGD(), RMSprop(), Adam(), Adadelta(), Adagrad(), Adamax(), Nadam()]
history_dict = {}

for opt in optimizers:
    print(f"Навчання з оптимізатором: {opt.__class__.__name__}")
    model = build_model(X_train_scaled.shape[1], optimizer=opt)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=30, batch_size=32, verbose=0
    )
    history_dict[opt.__class__.__name__] = history

# Побудова графіків навчання 
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

# Тестування моделі з найкращим оптимізатором (Adam) на тестовому наборі даних 
best_model = build_model(X_train_scaled.shape[1], optimizer=Adam())
best_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30, batch_size=32, verbose=1
)

# Оцінка моделі на тестовому наборі 
test_loss, test_accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f"Точність на тестовому наборі: {test_accuracy:.2f}")
