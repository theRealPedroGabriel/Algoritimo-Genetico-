import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import statistics
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# Carregar os dados
data = pd.read_csv(r'C:\Users\pgsmc\Documents\EMGDataset.csv', header=None)

# Organizar os dados em X (N x p) e Y (N x C)
X = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values

# Transformar y em one-hot encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1, 1))

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)


# Mapear classes para cores
colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple', 5: 'orange'}
class_names = {1: 'Neutro', 2: 'Sorriso', 3: 'Sobrancelhas levantadas', 4: 'Surpreso', 5: 'Rabugento'}

plt.figure(figsize=(10, 8))
for label in np.unique(y):
    plt.scatter(X[y == label][:, 0], X[y == label][:, 1], c=colors[label], label=class_names[label], alpha=0.5)

plt.xlabel('Sensor no Corrugador do Supercílio')
plt.ylabel('Sensor no Zigomático Maior')
plt.title('Gráfico de Dispersão dos Dados de EMG')
plt.legend()
plt.show()

# Definir número de rodadas
n_rounds = 10

# Armazenar resultados
acc_ols = []
acc_tikhonov = []
best_alpha = None
best_acc = 0
acc_poly = []
# Definir o modelo de regularização com um valor de alpha mais alto para evitar a matriz mal condicionada
alpha_value = 10.0
# Normalizador
scaler = StandardScaler()

alphas = [0.1, 1, 10, 100, 1000]  # Valores de regularização a serem testados

for _ in range(n_rounds):
    # Embaralhar e dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # MQO tradicional (Regressão Logística)
    model_ols = LogisticRegression(max_iter=1000)
    model_ols.fit(X_train, y_train)
    y_pred_ols = model_ols.predict(X_test)
    acc_ols.append(accuracy_score(y_test, y_pred_ols))

    # MQO regularizado (Ridge Classifier)
    best_alpha_round = None
    best_acc_round = 0
    for alpha in alphas:
        model_tikhonov = RidgeClassifier(alpha=alpha)
        model_tikhonov.fit(X_train, y_train)
        y_pred_tikhonov = model_tikhonov.predict(X_test)
        acc_tikhonov_round = accuracy_score(y_test, y_pred_tikhonov)
        if acc_tikhonov_round > best_acc_round:
            best_acc_round = acc_tikhonov_round
            best_alpha_round = alpha

    acc_tikhonov.append(best_acc_round)
    if best_acc_round > best_acc:
        best_acc = best_acc_round
        best_alpha = best_alpha_round

print("Melhor valor de alpha:", best_alpha)

for _ in range(n_rounds):
    # Embaralhar e dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Normalizar os dados
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Gerar características polinomiais de grau 2
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Treinar o modelo de regressão polinomial com regularização (Ridge)
    model_poly = Ridge(alpha=alpha_value)
    model_poly.fit(X_poly_train, y_train)
    y_pred_poly = model_poly.predict(X_poly_test)

    # Certificar-se de que y_pred_poly é 2D
    if y_pred_poly.ndim == 1:
        y_pred_poly = y_pred_poly[:, np.newaxis]

    # Transformar y_test para one-hot se necessário
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        encoder = OneHotEncoder(sparse_output=False)
        y_test = encoder.fit_transform(y_test.reshape(-1, 1))

    y_pred_classes = np.argmax(y_pred_poly, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    acc_poly.append(accuracy_score(y_test_classes, y_pred_classes))

def compute_statistics(acc_list):
    return {
        "Média": np.mean(acc_list),
        "Desvio Padrão": np.std(acc_list),
        "Máximo": np.max(acc_list),
        "Mínimo": np.min(acc_list),
        "Moda": statistics.mode(acc_list)
    }

# Estatísticas do modelo polinomial
def compute_statistics2(acc_list):
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    acc_max = np.max(acc_list)
    acc_min = np.min(acc_list)
    acc_mode = max(set(acc_list), key=acc_list.count)
    return {
        'Média': acc_mean,
        'Desvio Padrão': acc_std,
        'Máximo': acc_max,
        'Mínimo': acc_min,
        'Moda': acc_mode
    }


stats_ols = compute_statistics(acc_ols)
stats_tikhonov = compute_statistics(acc_tikhonov)

# Exibir resultados
print("Estatísticas do MQO Tradicional:", stats_ols)
print("Estatísticas do MQO Regularizado:", stats_tikhonov)

# Estatísticas do modelo polinomial
stats_poly = compute_statistics2(acc_poly)

# Exibir resultados
print("Estatísticas do Modelo Polinomial (Grau 2):", stats_poly)
