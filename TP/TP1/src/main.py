import numpy as np
import argparse
import os
import sys

from utils import *

from individual import *

DATASET_FOLDER = 'datasets'
DATASET_NAMES = list()

# """
# Estatísticas importantes
# Estas statísticas devem ser coletadas para todas as gerações.
# 1. Fitness d melhor e pior indivíduos
# 2. Fitness média da população
# 3. Número de indivíduos repetidos na população
# 4. Número de indivíduos gerados por cruzamento melhores e piores que a tness média
# dos pais
# O que deve ser entregue...
#  Código fonte do programa
#  Documentação do trabalho:
# { Introdução
# { Implementação: descrição sobre a implementação do programa, incluindo detalhes
# da representação, tness e operadores utilizados
# { Experimentos: Análise do impacto dos parâmetros no resultado obtido pelo AE.
# { Conclusões
# { Bibliograa
# """

class GeneticProgramming(object):

    def __init__(self, population=50, n_dim=1, depth=7):
        self.population = population
        self.n_dim = n_dim
        self.depth = depth

        self.individuals = list()

        for _ in range(population):
            self.individuals.append(generate_individual(n_dim, depth))

    def fitness(self, train_data):
        """
        NRMSE
        """
        Ymean = train_data[:, -1].mean()
        normalize = train_data[:, -1] - Ymean
        normalize = np.sum(normalize)

        nrmse_ind = list()
        
        index = 0
        for ind in self.individuals:
            real_diff = list()
            for data in train_data:
                eval_value = ind.eval(data[:,:-1])
                real_value = data[:,-1].item(0)
                real_diff.append((eval_value - real_value)**2)
            if float('inf') in real_diff:
                nrmse_ind.append([float('inf'), index])
            else:
                nrmse_ind.append([np.sqrt(np.sum(real_diff)/normalize), index])
            index += 1
        nrmse_ind.sort(key=lambda x: x[0])
        nrmse_ind = np.matrix(nrmse_ind)

        indexes_ord = nrmse_ind[:,1].T.astype(int).tolist()[0]

        # reorder individuals by fitness
        self.individuals = [ self.individuals[i] for i in indexes_ord ]
        
        return nrmse_ind[:,0].T.tolist()

    def reproduction(self):
        pass
    
    def cross(self, choice1, choice2):
        # print_warning('cross')
        parent1, node1, id1, _ = choice1
        parent2, node2, id2, _ = choice2

        # print_warning(node1)
        # print_warning(node2)

        if type(parent1) is Node:
            parent1.children = node2
        elif type(parent1) is Operator:
            idleft = parent1.node_left.get_unique_id()
            idright = parent1.node_right.get_unique_id()
            if idleft == id1:
                parent1.node_left = node2
            elif idright == id1:
                parent1.node_right = node2
            else:
                raise 'Algo deu muito errado'
        elif isinstance(parent1, Function):
            parent1.node = node2
        
        if type(parent2) is Node:
            parent2.children = node1
        elif type(parent2) is Operator:
            idleft = parent2.node_left.get_unique_id()
            idright = parent2.node_right.get_unique_id()
            if idleft == id2:
                parent2.node_left = node1
            elif idright == id2:
                parent2.node_right = node1
            else:
                raise 'Algo deu muito errado'
        elif isinstance(parent2, Function):
            parent2.node = node1
        


    def crossover(self, prob_c=0.5, k = 2):
        """
        Troca sub-arvores
        """
        # print_warning('crossover')
        indexes, inds = self.tournament(k)
        ind1, ind2 = inds[:2]

        if np.random.random() < prob_c:
            ind1_uniques = ind1.walk()
            ind2_uniques = ind2.walk()

            x = np.random.choice(list(range(len(ind1_uniques))))
            y = np.random.choice(list(range(len(ind2_uniques))))
            choice1 = ind1_uniques[x]
            choice2 = ind2_uniques[y]
            self.cross(choice1, choice2)
        
        return ind1, ind2         
    
    def tournament(self, many=2):
        """
        Espera-se que a fitness já foi calculada a espera que os individuos
        já estejam ordernados
        retorna com individuos por ordem de fitness
        """
        # print_warning('tour')
        possible_choices = list(range(0, self.population))
        choices = list()
        for _ in range(many):
            choice = possible_choices.pop(np.random.random_integers(0, len(possible_choices)-1))
            choices.append(choice)
        choices.sort()
        return choices, [self.individuals[i] for i in choices]

    
    def mutation(self, prob_mut=0.5, k = 2):
        # print_warning('mutation')
        indexes, inds = self.tournament(k)
        ind = inds[0]

        if np.random.random() < prob_mut:
            # print(ind)

            ind_uniques = ind.walk()
            x = np.random.choice(list(range(len(ind_uniques))))
            choice1 = ind_uniques[x]
            
            parent1, node1, id1, depth = choice1
            # print_purple(node1)

            if type(parent1) is Node:
                parent1.children = generate_subtree(self.n_dim, self.depth)
            elif type(parent1) is Operator:
                idleft = parent1.node_left.get_unique_id()
                idright = parent1.node_right.get_unique_id()
                if idleft == id1:
                    parent1.node_left = generate_subtree(self.n_dim, self.depth - depth)
                elif idright == id1:
                    parent1.node_right = generate_subtree(self.n_dim, self.depth - depth)
                else:
                    raise 'Algo deu muito errado'
                # add optimizer
                # (((X1 / 3.01) + (exp(X1 - 0.32) + (sin(X1 - 3.76) + ((X0 + 8.34) + X1)))) + 12.53)
                # ((X1 / 3.01) + (exp(X1 - 0.32) + (sin(X1 - 3.76) + ((X0 + 8.34) + X1))))
                # (4.99 + 12.53)
            elif isinstance(parent1, Function):
                parent1.node = generate_subtree(self.n_dim, self.depth - depth)
            
            # print_warning(ind)
            # self.show_ind()
        return ind

    
    def new_population(self, choosen_inds=[]):
        # print_warning('newpop')
        self.individuals = choosen_inds
        for i in range(self.population - len(choosen_inds)):
            # TODO: pais e filhos (nao do Renato Russo)
            self.individuals.append(generate_individual(self.n_dim, self.depth))
    
    def show_ind(self):
        print_green('Unique ID (SHA256)\tExpression')
        for ind in self.individuals:
            print_blue(str(ind.get_unique_id())[:13] + '...\t' + ind.__str__())


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


def main():
    search_dataset_names()
    train_data, test_data = read_csv(DATASET_NAMES[2])

    # description = "Genetic Programming using math expression trees"
    # parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('population', type=int, help="number of individuals per epoch", default=50)
    # args = parser.parse_args()
    # GeneticProgramming(population=args.population, )
    gp = GeneticProgramming(population=50, n_dim=train_data.shape[1]-1, depth=7)
    gp.fitness(train_data)

    count = 0
    while True:
        choosen_inds = list()
        x,y = gp.crossover(prob_c=1.0, k=3)
        choosen_inds.append(x)
        choosen_inds.append(y)
        x = gp.mutation(prob_mut=1.0, k=3)
        choosen_inds.append(x)
        gp.new_population(choosen_inds)
        best = gp.fitness(train_data)[0]
        print_blue(gp.individuals[0])
        print(best[0])
        count += 1
        if count == 50:
            break

    gp.show_ind()


if __name__ == '__main__':
    main()