from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AlgoritmoGenetico:
    def __init__(self, max_generation: int, fitness: Callable[[np.ndarray], float], p: int, N: int, pr: float, pm: float, restricoes_dominio: tuple) -> None:
        self.max_generation = max_generation
        self.l = restricoes_dominio[0]
        self.u = restricoes_dominio[1]
        self.fitness = fitness
        self.p = p
        self.N = N
        self.pr = pr
        self.pm = pm
        self.P = None
        self.S = None
        self.aptidoes = np.zeros(self.N)
        self.total_aptidoes = 0
        self.melhor = []
        self.media = []

    def gerar_populacao(self):
        return np.random.uniform(low=self.l, high=self.u, size=(self.N, self.p))

    def calcular_aptidoes(self):
        for i in range(self.N):
            self.aptidoes[i] = self.fitness(self.P[i])
        self.total_aptidoes = np.sum(self.aptidoes)
        self.melhor.append(np.max(self.aptidoes))
        self.media.append(np.mean(self.aptidoes))

    def roleta(self):
        i = 0
        soma = self.aptidoes[i] / self.total_aptidoes
        r = np.random.uniform()
        while soma < r:
            i += 1
            soma += self.aptidoes[i] / self.total_aptidoes
        return self.P[i, :]

    def selecao(self):
        S = np.empty((0, self.p))
        for i in range(self.N):
            s = self.roleta()
            S = np.concatenate((S, s.reshape(1, self.p)))
        return S

    def recombinacao(self):
        R = np.empty((0, self.p))
        for i in range(0, self.N, 2):
            x1 = self.S[i, :]
            x2 = self.S[i + 1, :]
            x1_t = np.copy(x1)
            x2_t = np.copy(x2)
            if np.random.uniform() <= self.pr:
                alpha = np.random.uniform()
                x1_t = alpha * x1 + (1 - alpha) * x2
                x2_t = (1 - alpha) * x1 + alpha * x2
            R = np.concatenate((R, x1_t.reshape(1, self.p), x2_t.reshape(1, self.p)))
        return R

    def mutacao(self):
        for i in range(self.N):
            for j in range(self.p):
                if np.random.uniform() <= self.pm:
                    self.P[i, j] = np.random.uniform(self.l, self.u)

    def geracoes(self):
        self.P = self.gerar_populacao()
        for _ in range(self.max_generation):
            self.calcular_aptidoes()
            self.S = self.selecao()
            self.P = self.recombinacao()
            self.mutacao()

# Definindo a função de fitness
def fitness_function(x):
    A = 10
    return A * len(x) + np.sum([xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x])

# Configurações do algoritmo genético
max_generation = 100
p = 20
N = 50
pr = 0.85
pm = 0.01
restricoes_dominio = (-10, 10)  # limites do domínio

# Criando uma instância do AlgoritmoGenetico
alg_genetico = AlgoritmoGenetico(max_generation, fitness_function, p, N, pr, pm, restricoes_dominio)

# Executando o algoritmo genético
alg_genetico.geracoes()

# Plotando os resultados
plt.figure(figsize=(10, 5))

# Gráfico da melhor aptidão e média das aptidões ao longo das gerações
plt.subplot(1, 2, 1)
plt.plot(alg_genetico.melhor, label='Melhor Aptidão')
plt.plot(alg_genetico.media, label='Média das Aptidões')
plt.xlabel('Geração')
plt.ylabel('Aptidão')
plt.title('Evolução das Aptidões')
plt.legend()


plt.show()