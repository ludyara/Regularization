import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras import layers, regularizers
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Считываем данные
# file_name = "Perfect_All_AnPar.csv"  # Укажите путь к файлу
file_name = "Mini_Perfect_AnPar.csv"  # Укажите путь к файлу
data = pd.read_csv(file_name, delimiter=",")

# Проверяем наличие колонок
main_param = "Tm_outRP"  # Целевая переменная
# input_params = ["Tm_outSP", "Tm_outST"]  # Входные признаки
input_params = ["Tm_outSP", "Tm_outST", "evGt", "evNst", "evNvd",
                "Vibr_1", "Vibr_Z", "Pm_outD", "evTt", "Pa_inD"]

if not all(col in data.columns for col in [main_param] + input_params):
    raise ValueError("Отсутствуют необходимые параметры в данных.")

# Извлечение целевой переменной и признаков
y = data[main_param].values  # Целевая переменная
X = data[input_params].values  # Входные признаки

# Исключение пропусков
valid_indices = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
X = X[valid_indices]
y = y[valid_indices]

# Самый первый вариант (old)
"""
# Нормализация признаков для корректной работы регуляризации
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y.reshape(-1, 1))

# # --- 1. График пути регуляризации ---
# plt.figure(figsize=(12, 8))
# plt.plot(range(len(y)), X, label="X", color="blue")
# plt.plot(range(len(y)), y, label="Y", color="red")
# plt.xlabel("Индекс данных (равен 10 с)")
# plt.ylabel("Значение")
# plt.legend()
# plt.grid()
# plt.show()

# Разделение данных
split_index = int(len(y) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Модели и гиперпараметры
alphas = np.logspace(-3, 1, 50)  # Диапазон alpha
coefficients_lasso = []
coefficients_ridge = []
mse_train_lasso, mse_test_lasso = [], []
mse_train_ridge, mse_test_ridge = [], []

for alpha in alphas:
    # Lasso
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    coefficients_lasso.append(lasso.coef_)
    mse_train_lasso.append(mean_squared_error(y_train, lasso.predict(X_train)))
    mse_test_lasso.append(mean_squared_error(y_test, lasso.predict(X_test)))

    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefficients_ridge.append(ridge.coef_)
    mse_train_ridge.append(mean_squared_error(y_train, ridge.predict(X_train)))
    mse_test_ridge.append(mean_squared_error(y_test, ridge.predict(X_test)))

# --- 1. График пути регуляризации ---
plt.figure(figsize=(12, 8))
for i, param in enumerate(input_params):
    plt.plot(alphas, [coef[i] for coef in coefficients_lasso], label=f"Lasso: {param}")
    plt.plot(alphas, [coef[i] for coef in coefficients_ridge], linestyle='--', label=f"Ridge: {param}")
plt.xscale("log")
plt.xlabel("Alpha (Сила регуляризации)")
plt.ylabel("Значение коэффициента")
# plt.title("Regularization Path for Coefficients (Lasso vs Ridge)")
plt.legend()
plt.grid()
plt.show()

# --- 2. График MSE на тренировочных и тестовых данных ---
plt.figure(figsize=(12, 8))
plt.plot(alphas, mse_train_lasso, label="График MSE на тренировочных данных (Lasso)", color="blue")
plt.plot(alphas, mse_test_lasso, label="График MSE на тестовых данных (Lasso)", color="blue", linestyle="--")
plt.plot(alphas, mse_train_ridge, label="График MSE на тренировочных данных (Ridge)", color="orange")
plt.plot(alphas, mse_test_ridge, label="График MSE на тестовых данных (Ridge)", color="orange", linestyle="--")
plt.xscale("log")
plt.xlabel("Alpha (Сила регуляризации)")
plt.ylabel("Среднеквадратическая ошибка")
# plt.title("MSE vs Alpha for Lasso and Ridge")
plt.legend()

plt.grid()
plt.show()

# --- 3. График предсказаний ---
best_alpha_lasso = alphas[np.argmin(mse_test_lasso)]
best_alpha_ridge = alphas[np.argmin(mse_test_ridge)]

lasso = Lasso(alpha=best_alpha_lasso)
lasso.fit(X_train, y_train)
ridge = Ridge(alpha=best_alpha_ridge)
ridge.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plt.scatter(range(len(y_train)), y_train, label="Тренировочные данные", alpha=0.7, color="blue")
plt.scatter(range(len(y_test)), y_test, label="Тестовые данные", alpha=0.7, color="orange")
plt.plot(range(len(y_train)), lasso.predict(X_train), label="Lasso (трениров. данные)", color="blue", linestyle="--")
plt.plot(range(len(y_test)), lasso.predict(X_test), label="Lasso (тестов. данные)", color="blue")
plt.plot(range(len(y_train)), ridge.predict(X_train), label="Ridge (трениров. данные)", color="orange", linestyle="--")
plt.plot(range(len(y_test)), ridge.predict(X_test), label="Ridge (тестов. данные)", color="orange")
plt.xlabel("Индекс данных (равен 10 с)")
plt.ylabel("Температура масла на выходе из подшипника (°C)")
#plt.title("Best Lasso and Ridge Predictions")
plt.legend()
plt.grid()
plt.show()
"""

# ============================
# Разделение по времени (80/20)
# ============================

split_index = int(len(y) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ============================
# Масштабирование (ТОЛЬКО по train)
# ============================

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

def build_model(reg=None):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu',
                     kernel_regularizer=reg,
                     input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=reg),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Поиск alpha
'''
alphas_L1 = np.logspace(-5, 1, 10)
alphas_L2 = np.logspace(-5, 1, 10)
print(alphas_L1, alphas_L2)
error_matrix = np.zeros((len(alphas_L1), len(alphas_L2)))
best_alpha = None
best_alpha_L1 = None
best_alpha_L2 = None
best_mse = np.inf
best_mse_L1 = np.inf
best_mse_L2 = np.inf

train_errors = []
val_errors = []
train_errors_L1 = []
val_errors_L1 = []
train_errors_L2 = []
val_errors_L2 = []

# for alpha_L2 in alphas_L2:
#     model = build_model(reg=regularizers.l2(l2=alpha_L2))
#     # model = build_model(reg=regularizers.l1_l2(l1=0.00007, l2=0.007))
#     model.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0, shuffle=False)
#     y_tr_pred = model.predict(X_train_scaled)
#     y_test_pred = model.predict(X_test_scaled)
#
#     mse_tr = mean_squared_error(y_train_scaled, y_tr_pred)
#     mse_val = mean_squared_error(y_test_scaled, y_test_pred)
#
#     train_errors_L2.append(mse_tr)
#     val_errors_L2.append(mse_val)
#
#     if mse_val < best_mse_L2:
#         best_mse_L2 = mse_val
#         best_alpha_L2 = alpha_L2
#
# for alpha_L1 in alphas_L1:
#     model = build_model(reg=regularizers.l1(l1=alpha_L1))
#     model.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0, shuffle=False)
#     y_tr_pred = model.predict(X_train_scaled)
#     y_test_pred = model.predict(X_test_scaled)
#
#     mse_tr = mean_squared_error(y_train_scaled, y_tr_pred)
#     mse_val = mean_squared_error(y_test_scaled, y_test_pred)
#
#     train_errors_L1.append(mse_tr)
#     val_errors_L1.append(mse_val)
#
#     if mse_val < best_mse_L1:
#         best_mse_L1 = mse_val
#         best_alpha_L1 = alpha_L1

for i, alpha_L1 in enumerate(alphas_L1):
    for j, alpha_L2 in enumerate(alphas_L2):
        model = build_model(reg=regularizers.l1_l2(l1=alpha_L1, l2=alpha_L2))
        model.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=0, shuffle=False)

        y_test_pred = model.predict(X_test_scaled)
        mse_val = mean_squared_error(y_test_scaled, y_test_pred)
        if mse_val < best_mse:
            best_mse = mse_val
            best_alpha_L1 = alpha_L1
            best_alpha_L2 = alpha_L2
        error_matrix[i, j] = mse_val
print("Лучший alpha L1:", best_alpha_L1)
print("Лучший alpha L2:", best_alpha_L2)
A_L2, A_L1 = np.meshgrid(alphas_L2, alphas_L1)
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(
    A_L2,
    A_L1,
    error_matrix,
    shading='auto'
)
plt.imshow(error_matrix,
           origin='lower',
           aspect='auto',
           extent=[alphas_L2.min(), alphas_L2.max(),
                   alphas_L1.min(), alphas_L1.max()])

plt.colorbar(pcm, label="MSE")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("L2 alpha")
plt.ylabel("L1 alpha")
plt.title("Поверхность ошибки ElasticNet")
plt.show()
# print("Лучший alpha L1:", best_alpha_L1)
# print("Лучший alpha L2:", best_alpha_L2)
# plt.figure(figsize=(8,5))
# plt.semilogx(alphas_L1, val_errors_L1, label="alpha_L1")
# plt.semilogx(alphas_L2, val_errors_L2, label="alpha_L2")
# plt.xlabel("alpha")
# plt.ylabel("MSE")
# plt.title("Подбор коэффициентов L1 и L2-регуляризации")
# plt.legend()
# plt.grid(True)
# plt.show()
'''

# Старые модели предсказания (через MLPRegressor)
'''
# ============================
# 1️ Нейросеть БЕЗ регуляризации
# ============================

mlp_no_reg = MLPRegressor(hidden_layer_sizes=(64, 32),
                          activation='relu',
                          solver='adam',
                          alpha=0.0,              # Без L2
                          max_iter=500,
                          random_state=42)

mlp_no_reg.fit(X_train_scaled, y_train_scaled)

y_pred_train_no_reg = scaler_y.inverse_transform(
    mlp_no_reg.predict(X_train_scaled).reshape(-1, 1)
).ravel()

y_pred_test_no_reg = scaler_y.inverse_transform(
    mlp_no_reg.predict(X_test_scaled).reshape(-1, 1)
).ravel()

# ============================
# 2 Нейросеть С L2-регуляризацией
# ============================

mlp_l2 = MLPRegressor(hidden_layer_sizes=(64, 32),
                      activation='relu',
                      solver='adam',
                      alpha=0.54556,         # L2-регуляризация
                      max_iter=500,
                      random_state=42)

mlp_l2.fit(X_train_scaled, y_train_scaled)

y_pred_train_l2 = scaler_y.inverse_transform(
    mlp_l2.predict(X_train_scaled).reshape(-1, 1)
).ravel()

y_pred_test_l2 = scaler_y.inverse_transform(
    mlp_l2.predict(X_test_scaled).reshape(-1, 1)
).ravel()
'''

# ------------------------------------
# Основная программа (обучение и предсказание)
# ------------------------------------

# Без регуляризации
model_no_reg = build_model(reg=None)
model_no_reg.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0, shuffle=False)

# L2
# model_l2 = build_model(reg=regularizers.l2(0.061585))
model_l2 = build_model(reg=regularizers.l2(0.009)) # 0.01
model_l2.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0, shuffle=False)

# L1
# model_l1 = build_model(reg=regularizers.l1(0.00464))
model_l1 = build_model(reg=regularizers.l1(0.0014)) # 0.002
model_l1.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0, shuffle=False)

# ElasticNet (L1 + L2)
# model_elastic = build_model(reg=regularizers.l1_l2(l1=0.0001585, l2=0.03981))
model_elastic = build_model(reg=regularizers.l1_l2(l1=0.00007, l2=0.007))
model_elastic.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0, shuffle=False)

def predict_and_inverse(model, data):
    return scaler_y.inverse_transform(
        model.predict(data)
    ).ravel()

# Предсказания
y_pred_train_no_reg = predict_and_inverse(model_no_reg, X_train_scaled)
y_pred_train_l2 = predict_and_inverse(model_l2, X_train_scaled)
y_pred_train_l1 = predict_and_inverse(model_l1, X_train_scaled)
y_pred_train_elastic = predict_and_inverse(model_elastic, X_train_scaled)
y_pred_test_no_reg = predict_and_inverse(model_no_reg, X_test_scaled)
y_pred_test_l2 = predict_and_inverse(model_l2, X_test_scaled)
y_pred_test_l1 = predict_and_inverse(model_l1, X_test_scaled)
y_pred_test_elastic = predict_and_inverse(model_elastic, X_test_scaled)

# ============================
# Метрики
# ============================

def print_metrics(name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    print(f"\n{name}")
    print("Train MSE:", mean_squared_error(y_true_train, y_pred_train))
    print("Test  MSE:", mean_squared_error(y_true_test, y_pred_test))
    print("Train MAE:", mean_absolute_error(y_true_train, y_pred_train))
    print("Test  MAE:", mean_absolute_error(y_true_test, y_pred_test))
    print("Train R2 :", r2_score(y_true_train, y_pred_train))
    print("Test  R2 :", r2_score(y_true_test, y_pred_test))

print_metrics("без регуляризации",
              y_train, y_pred_train_no_reg,
              y_test, y_pred_test_no_reg)
print_metrics("c L1-регуляризацией",
              y_train, y_pred_train_l1,
              y_test, y_pred_test_l1)
print_metrics("c L2-регуляризацией",
              y_train, y_pred_train_l2,
              y_test, y_pred_test_l2)
print_metrics("c Elastic",
              y_train, y_pred_train_elastic,
              y_test, y_pred_test_elastic)

# ============================
# во сколько раз L1-регуляризация снизила количество входных признаков
# ============================
weights, biases = model_l1.layers[0].get_weights()
# Норма весов для каждого входного признака
feature_norms = np.linalg.norm(weights, axis=1)

print("feature_norms:", feature_norms)
# Порог "почти ноль"
threshold = 1e-3

active_features = np.sum(feature_norms > threshold)
total_features = weights.shape[0]

print("Всего признаков:", total_features)
print("Активных признаков:", active_features)
print("Сокращение в раз:", total_features / active_features)




# ============================
# График
# ============================
# Создаём массив значений для оси X в минутах
x_minutes = np.arange(len(y_test)) * 0.5  # 30 сек = 0.5 мин

plt.figure(figsize=(14,6))

# Строим графики с новыми значениями по оси X
plt.plot(x_minutes, y_test, label="True meaning", linewidth=2)
plt.plot(x_minutes, y_pred_test_no_reg, label="Without regularization", ls=":")
plt.plot(x_minutes, y_pred_test_l1, label="With L1-regularization")
plt.plot(x_minutes, y_pred_test_l2, label="With L2-regularization")
plt.plot(x_minutes, y_pred_test_elastic, label="With elastic regularization", ls="--")

# Добавляем подписи к осям
plt.xlabel("Minutes")
plt.ylabel("Degrees Celsius (°C)")

# Остальные настройки графика
plt.title("Comparison of gas turbine temperature forecasting")
plt.legend()
plt.grid(True)

plt.show()


