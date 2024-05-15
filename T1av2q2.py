import csv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular a distância entre duas cidades
def calc_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Função para calcular o comprimento do caminho
def total_distance(path, cities):
    dist = 0
    for i in range(len(path) - 1):
        dist += calc_distance(cities[path[i]], cities[path[i+1]])
    dist += calc_distance(cities[path[-1]], cities[path[0]])  # Volta para a cidade de origem
    return dist

# Função para gerar população inicial
def generate_population(num_cities, pop_size):
    population = []
    for _ in range(pop_size):
        individual = np.random.permutation(num_cities)
        population.append(individual)
    return population

# Função de seleção de pais (torneio)
def select_parents(population, cities, tournament_size):
    parents = []
    while len(parents) < 2:
        tournament = np.random.choice(len(population), tournament_size, replace=False)
        winner = None
        min_dist = float('inf')
        for idx in tournament:
            dist = total_distance(population[idx], cities)
            if dist < min_dist:
                min_dist = dist
                winner = idx
        parents.append(population[winner])
    return parents

# Função de crossover (ordem)
def crossover(parents):
    child = [-1] * len(parents[0])
    start, end = sorted(np.random.choice(len(child), 2, replace=False))
    child[start:end] = parents[0][start:end]
    remaining = [city for city in parents[1] if city not in child]
    idx = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1
    return child

# Função de mutação (swap)
def mutate(individual):
    idx1, idx2 = sorted(np.random.choice(len(individual), 2, replace=False))
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# Função principal do algoritmo genético com elitismo
def genetic_algorithm(cities,ax, lines, pop_size=100, generations=100, tournament_size=5, crossover_rate=0.9, mutation_rate=0.1, num_elites=2):
    num_cities = len(cities)
    population = generate_population(num_cities, pop_size)
    best_distance = float('inf')
    best_path = None
    # Lista para armazenar as distâncias calculadas em cada geração
    distances = []
    mean_distances = []
    best_distances = []


    for gen in range(generations):
        new_population = []

        # Mantenha os melhores indivíduos (elites) da geração anterior
        elites = sorted(population, key=lambda x: total_distance(x, cities))[:num_elites]
        new_population.extend(elites)

        # Preencha o restante da nova população usando seleção, crossover e mutação
        while len(new_population) < pop_size:
            parents = select_parents(population, cities, tournament_size)
            if np.random.rand() < crossover_rate:
                child1 = crossover(parents)
                child2 = crossover(parents[::-1])
            else:
                child1, child2 = parents
            if np.random.rand() < mutation_rate:
                child1 = mutate(child1)
            if np.random.rand() < mutation_rate:
                child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population

        # Atualize o melhor caminho encontrado até agora
        for individual in population:
            distance = total_distance(individual, cities)
            if distance < best_distance:
                best_distance = distance
                best_path = individual

        # Armazene a melhor distância da geração atual
        distances.append(best_distance)
        mean_distances.append(np.mean(distances))
        best_distances.append(np.min(distances))




        # Plotar o melhor caminho a cada geração
        plot_update(ax, lines, pontos, best_path)

        # Calcular métricas
        min_distance = min(distances)
        max_distance = max(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Exibir métricas em uma tabela
        print("Métricas do algoritmo genético:")
        print("Menor distância:", min_distance)
        print("Maior distância:", max_distance)
        print("Média das distâncias:", mean_distance)
        print("Desvio padrão das distâncias:", std_distance)
        print()
        if gen == generations - 1:
            # Plotando os resultados
            plt.figure(figsize=(10, 5))

            # Gráfico da melhor aptidão e média das aptidões ao longo das gerações
            plt.subplot(1, 2, 1)
            plt.plot(distances, label='Menor distancia')
            plt.plot(mean_distances, label='Média das distancia')
            plt.xlabel('Geração')
            plt.ylabel('Distancia')
            plt.title('Evolução da distancia')
            plt.legend()

    return best_path, best_distance

# Leitura dos pontos do arquivo CSV
with open(r'C:\Users\pgsmc\Documentos\CaixeiroSimples.csv', newline='') as csvfile:
    pontos_reader = csv.reader(csvfile, delimiter=',')
    pontos = np.array([list(map(float, row)) for row in pontos_reader])

# Função para plotar atualização em tempo real
def plot_update(ax, lines, pontos, candidato):
    for g in lines:
        if g:
            g.remove()

    for i in range(len(candidato)):
        p1 = pontos[candidato[i], :]
        p2 = pontos[candidato[(i + 1) % len(candidato)], :]
        if i == 0 or i == len(candidato) - 1:
            l = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red')
        else:
            l = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='k')
        lines[i] = l[0]
    plt.pause(.05)


# Gerar cidades aleatórias .a de origem sempre a 1
num_cities = 101
cities = pontos

# Plotar o resultado
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Inicializar linhas para atualização
lines = [None] * num_cities
# Executar algoritmo genético
best_path, best_distance = genetic_algorithm(cities, ax, lines)
# Plotar as cidades
ax.scatter(cities[:,0], cities[:,1], cities[:,2], c='b', marker='o')

# Plotar o melhor caminho
best_path_cities = cities[best_path]

ax.plot(best_path_cities[:,0], best_path_cities[:,1], best_path_cities[:,2], c='r', linestyle='-', marker='o')

plt.show()


