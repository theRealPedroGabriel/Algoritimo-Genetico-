import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# Função para ler os dados do arquivo .dat
def read_dat_file(file_path):
    # Supondo que os dados estão separados por espaços
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    # Nomeando as colunas para facilitar o acesso
    data.columns = ['WindSpeed', 'Power']
    return data

# Caminho para o arquivo aerogerador.dat
file_path = r'C:\Users\pgsmc\Documents\aerogerador.dat'

# Ler os dados
data = read_dat_file(file_path)

# Organizar os dados em matriz X e vetor y
X = data['WindSpeed'].values.reshape(-1, 1)  # Matriz de variáveis regressoras (N x p)
y = data['Power'].values.reshape(-1, 1)      # Vetor de variáveis observadas (N x 1)

# Exibir as dimensões para verificar
print('Dimensão de X:', X.shape)
print('Dimensão de y:', y.shape)

# Definir quantidade de rodadas
num_rounds = 1000

# Inicializar listas para armazenar EQMs
eqm_mqo = []
eqm_mqo_regularizado = []
eqm_media_valores = []

# Loop sobre as rodadas
for _ in range(num_rounds):
    # Embaralhar os dados
    shuffled_data = data.sample(frac=1)

    # Dividir dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(shuffled_data['WindSpeed'],
                                                        shuffled_data['Power'],
                                                        test_size=0.2)

    # MQO Tradicional
    model_mqo = LinearRegression()
    model_mqo.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
    y_pred_mqo = model_mqo.predict(X_test.values.reshape(-1, 1))
    eqm_mqo.append(mean_squared_error(y_test, y_pred_mqo))

    # MQO Regularizado (Ridge)
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
    y_pred_ridge = model_ridge.predict(X_test.values.reshape(-1, 1))
    eqm_mqo_regularizado.append(mean_squared_error(y_test, y_pred_ridge))

    # Média de Valores Observáveis
    mean_power = y_train.mean()
    eqm_media_valores.append(mean_squared_error(y_test, np.full_like(y_test, mean_power)))

# Calcular estatísticas dos EQMs
eqm_stats = {
    'MQO': {
        'Média': np.mean(eqm_mqo),
        'Desvio Padrão': np.std(eqm_mqo),
        'Máximo': np.max(eqm_mqo),
        'Mínimo': np.min(eqm_mqo)
    },
    'MQO Regularizado': {
        'Média': np.mean(eqm_mqo_regularizado),
        'Desvio Padrão': np.std(eqm_mqo_regularizado),
        'Máximo': np.max(eqm_mqo_regularizado),
        'Mínimo': np.min(eqm_mqo_regularizado)
    },
    'Média de Valores': {
        'Média': np.mean(eqm_media_valores),
        'Desvio Padrão': np.std(eqm_media_valores),
        'Máximo': np.max(eqm_media_valores),
        'Mínimo': np.min(eqm_media_valores)
    }
}

# Mostrar estatísticas dos EQMs
eqm_stats_df = pd.DataFrame.from_dict(eqm_stats)
print(eqm_stats_df)

# Visualizar os primeiros registros para entender a estrutura dos dados
#print(data.head())

# Plotar gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(data['WindSpeed'], data['Power'], alpha=0.5)
plt.xlabel('Velocidade do Vento (m/s)')
plt.ylabel('Potência Gerada (kW)')
plt.title('Relação entre Velocidade do Vento e Potência Gerada')
plt.grid(True)
plt.show()
