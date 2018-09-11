import numpy as np
import argparse
import os
import sys

DATASET_FOLDER = 'datasets'
DATASET_NAMES = list()

"""

initial population


while true
    fitness
    selection
        crossover
        reprodution
        mutation
    new population
    stop criteria

best solution
"""

"""
Estatsticas importantes
Estas estatsticas devem ser coletadas para todas as gerac~oes.
1. Fitness do melhor e pior indivduos
2. Fitness media da populac~ao
3. Numero de indivduos repetidos na populac~ao
4. Numero de indivduos gerados por cruzamento melhores e piores que a tness media
dos pais
O que deve ser entregue...
 Codigo fonte do programa
 Documentac~ao do trabalho:
{ Introduc~ao
{ Implementac~ao: descric~ao sobre a implementac~ao do programa, incluindo detalhes
da representac~ao, tness e operadores utilizados
{ Experimentos: Analise do impacto dos par^ametros no resultado obtido pelo AE.
{ Conclus~oes
{ Bibliograa
"""

class GeneticProgramming(object):

    def __init__(self):
        pass

    def fitness(self):
        pass
    
    def crossover(self):
        pass
    
    def reproduction(self):
        pass
    
    def mutation(self):
        pass
    
    def new_population(self):
        pass


def read_csv(filename):
    train_data = np.matrix([])
    test_data = np.matrix([])
    train_file = DATASET_FOLDER + '/' + filename + '/' + filename + '-train.csv'
    test_file = DATASET_FOLDER + '/' + filename + '/' + filename + '-train.csv'

    def read_file(filename):
        aux_data = list()
        with open(train_file, 'r') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                aux_data.append(list(map(float, line.split(','))))
        return np.matrix(aux_data)
    
    train_data = read_file(train_file)
    test_data = read_file(test_file)
    return train_data, test_data

def search_dataset_names():
    names = os.listdir(DATASET_FOLDER)
    for name in names:
        DATASET_NAMES.append(name)

def proto_debug(set_name):
    train_data, test_data = read_csv(set_name)

    print(train_data)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("echo", help="echo the string you use here")
    # args = parser.parse_args()
    # print(args.echo)
    search_dataset_names()
    proto_debug(DATASET_NAMES[1])

if __name__ == '__main__':
    main()