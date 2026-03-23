#importando libs em pyon
import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

## Numero de itens - 10
n = 10
numero_itens = np.arange(1, n+1)

#Gerando randomicamente os pesos
pesos = [2.5, 1.8, 0.7, 2.1, 1.5, 2.2, 0.9, 1.6, 0.5, 1.1]

# Gerando randomicamente os valores de cada item
valores = [2000, 1450, 3400, 1900, 1300, 1000, 600, 1300, 400, 900]

nomes = ['Smartphone Samsung Galaxy S21',
 'Notebook Dell Inspiron 15',
 'Fone de Ouvido Bluetooth JBL',
 'Smartwatch Samsung Galaxy Watch 3',
 'Tablet Apple iPad 10.2',
 'Câmera Digital Canon EOS Rebel T7',
 'Mouse Gamer Logitech G Pro',
 'Teclado Mecânico Redragon Kumara',
 'Caixa de Som Bluetooth JBL GO',
 'Smartband Xiaomi Mi Band 6']

#definindo o peso maximo para mochila
max_peso_mochila = 7

for i in range (numero_itens.shape[0]):
    print('Item: {} \nPeso(kg): {} \nValor($): {} \n' .format(nomes[i], pesos[i], valores[i]))


#Numero de soluções ou individuos por população
solucao_por_populacao = 8
tamanho_populacao = (solucao_por_populacao, numero_itens.shape[0])

print('Tamanho da população = {}'.format(tamanho_populacao))
print('Numero de individuos (solução) = {}'.format(tamanho_populacao[0]))
print('Numero itens (genes) = {}'.format(tamanho_populacao[1]))


#Definido o numero de geração
n_geracoes = 10

#Criando a população onde somente um item sera levado por individuo
populacao_inicial = np.eye(tamanho_populacao[0], tamanho_populacao[1], k=0)

#Convertendo os tipos dos genes para inteiro
populacao_inicial = populacao_inicial.astype(int)

print('População inicial: \n{}'.format(populacao_inicial))

#Função para calcular o fitness de cada individuo
def cal_fitness(peso, valor, populacao, max_peso_mochila):
    fitness = np.zeros(populacao.shape[0])

    for i in range(populacao.shape[0]):
        S1 = np.sum(populacao[i] * valor)
        S2 = np.sum(populacao[i] * peso)

        if S2 <= max_peso_mochila:
            fitness[i] = S1
        else:
            fitness[i] = 0
    return fitness.astype(float)

#Funcao para a seleção dos individuos
def selecao_roleta(fitness, numero_pais, populacao):
    max_fitness = sum(fitness)
    probabilidades = fitness/max_fitness
    selecionados = populacao[np.random.choice(len(populacao), size=numero_pais, p= probabilidades)]
    
    return selecionados

#Funcao para CROSSOVER (Ponto unico)
def crossover(pais, numero_filhos):
    filhos = np.zeros((numero_filhos, pais.shape[1]))

    #O ponto em que o cruzamento ocorre entre dois pais
    ponto_crossover = int(pais.shape[1]/2)

    for k in range(numero_filhos):
        #indice do primeiro a ser fatiado
        pai_1_idx = k%pais.shape[0]
        #indice do segunda a ser fatiado
        pai_2_idx = (k+1)%pais.shape[0]
        #A nova prole tera sua primeira metade de seus genes retirado do primeiro pai
        filhos[k, 0:ponto_crossover] = pais[pai_1_idx, 0:ponto_crossover]
        #A nova prole tera sua segunda metade de seus genes retirado do segundo pai
        filhos[k, ponto_crossover:] = pais[pai_2_idx, ponto_crossover:]

    return filhos

#Função de Mutação (ponto unico)
def mutacao(filhos):
    mutacoes = filhos
    for i in range(mutacoes.shape[0]):
        posicao_gene = randint(0, filhos.shape[1]-1)

        if mutacoes[i,posicao_gene] == 0:
            mutacoes[i,posicao_gene] = 1
        else:
            mutacoes[i,posicao_gene] = 0

    return mutacoes

def rodar_AG(pesos, valores, populacao, tamanho_populacao, n_geracoes, max_peso_mochila):
    historico_fitness, historico_populacao = [] , []
    numero_pais = int(tamanho_populacao[0]/2)
    numero_filhos = tamanho_populacao[0] - numero_pais
    fitness = []

    for i in range(n_geracoes):
        print('--- Começando a geração {} ---'.format(i))
        #Calcula o fitness (aptidao) de cada individuo
        fitness = cal_fitness(pesos, valores, populacao, max_peso_mochila)
        #Armazena na variavel de historico
        historico_fitness.append(fitness.copy())
        historico_populacao.append(populacao.copy())
        pais = selecao_roleta(fitness, numero_pais, populacao)
        #Gerando os filhos
        filhos = crossover(pais, numero_filhos)
        #Mutando os filhos
        filhos_mutados = mutacao(filhos)
        print('População antiga: ')
        print(populacao)
        populacao[0:pais.shape[0], :] = pais
        populacao[pais.shape[0]:, :] = filhos_mutados
        print('População Nova: ')
        print(populacao)
    return historico_populacao, historico_fitness

historico_populacao, historico_fitness = rodar_AG(pesos, valores, populacao_inicial, tamanho_populacao, n_geracoes, max_peso_mochila)

#Mostrando resultados

#Criando o daframe de historico (Geração x fitness individuo)
dataframe = pd.DataFrame(historico_fitness)

#Apresentando o resultado
print(dataframe)

#Encontrando o melho individuo, ou seja, dentro de todas as gerações qual foi o melhor individuo com maior fitness
#Encontra a linha e coluna

max_index = dataframe.values.argmax()
linha, coluna = np.unravel_index(max_index, dataframe.shape)

print("\nValor do fitness (Max); ", dataframe.iloc[linha, coluna])
print('Linha do maior fitness (Geração): ', linha)
print("Coluna do maior fitness (Individuo): ", coluna)

#Armazena melhor individuo
melhor_individuo = historico_populacao[linha][coluna]

#Criando um dataframe apenas com itens "pegos"
itens_selecionados = numero_itens * melhor_individuo
dataframe_itens = pd.DataFrame(columns=['Item', 'Valor', 'Peso'])
for i in itens_selecionados:
    if i != 0:
        posicao = i - 1
        item = {'Item' : nomes[posicao], 'Valor': valores[posicao], 'Peso': pesos[posicao]}
        dataframe_itens.loc[len(dataframe_itens)] = item
print(dataframe_itens)

fitness_medio = [np.mean(fitness) for fitness in historico_fitness]
fitness_max = [np.max(fitness) for fitness in historico_fitness]

plt.plot(list(range(n_geracoes)), fitness_medio, label = 'Fitness Médio')
plt.plot(list(range(n_geracoes)), fitness_max, label = 'Fitness Maximo')
plt.legend()
plt.title('Fitness ao decorrer das gerações')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.show()
print(np.asanyarray(historico_fitness).shape)