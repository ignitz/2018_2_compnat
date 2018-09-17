
"""doc String."""
import os
# import sys
# import argparse

from individual import Function, Node, Operator, \
    generate_individual, generate_subtree

import numpy as np

from utils import print_blue, print_green, print_warning, log

DATASET_FOLDER = 'datasets'
DEBUG=False

def get_dataset_names():
    return os.listdir(DATASET_FOLDER)

def load_data(filename):
    if DEBUG:
        log('Read CSV data from', filename)

    train_file = (DATASET_FOLDER + '/' +
                  filename + '/' + filename + '-train.csv')
    test_file = (DATASET_FOLDER + '/' +
                 filename + '/' + filename + '-train.csv')

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
    if DEBUG:
        log('Finish read CSV)',
            '\nShape of train_data =', train_data.shape,
            '\nShape of test_data  =', test_data.shape)
    return train_data, test_data


class GeneticProgramming(object):

    def __init__(self,
                 population=50,
                 data_name='synth1',
                 depth=7,
                 k=2,
                 prob_m=0.5,
                 prob_c=0.5):
        if DEBUG:
            log('====================\n'
                'GP initialization',
                '\npopulation     =', population,
                '\nDATABASE name  =', data_name,
                '\ndepth of tree  =', depth,
                '\n=-=-=-=-=-=-=-=-=-=-')
        self.population = population
        self.depth = depth
        self.k = k
        self.prob_m = prob_m
        self.prob_c = prob_c
        self.train_data, self.test_data = load_data(data_name)

        if self.train_data.shape[1] != self.train_data.shape[1]:
            raise 'Train and Test data are diff dimensions'

        self.n_dim = self.train_data.shape[1] - 1

        self.individuals = list()

        for _ in range(population):
            self.individuals.append(generate_individual(self.n_dim, depth))

    def get_fitness(self):
        if DEBUG:
            log('Run get_fitness')

        nrmse_ind = list()
        index = 0
        for ind in self.individuals:
            nrmse_ind.append([ind.calc_fitness(self.train_data), index])
            index += 1
        nrmse_ind.sort(key=lambda x: x[0])
        nrmse_ind = np.matrix(nrmse_ind)
        indexes_ord = nrmse_ind[:, 1].T.astype(int).tolist()[0]

        # reorder individuals by fitness
        # self.individuals = [self.individuals[i] for i in indexes_ord]

        return nrmse_ind[:, 0].T.tolist(), indexes_ord

    def cross(self, choice1, choice2):
        if DEBUG:
            log(
                '====================\n'
                'Run cross',
                '\nchoice1 =', choice1[1],
                '\n\t parent1 =', choice1[0],
                '\n\t depth1 =', choice1[3],
                '\nchoice2 =', choice2[1],
                '\n\t parent2 =', choice2[0],
                '\n\t depth2 =', choice2[3],
                '\n=-=-=-=-=-=-=-=-=-=-')

        parent1, node1, id1, _ = choice1
        parent2, node2, id2, _ = choice2

        if type(parent1) is Node:
            if DEBUG: log('parent1 is Node')
            parent1.children = node2
        elif type(parent1) is Operator:
            if DEBUG: log('parent1 is Operator')
            idleft = parent1.node_left.get_unique_id()
            idright = parent1.node_right.get_unique_id()
            if idleft == id1:
                parent1.node_left = node2
            elif idright == id1:
                parent1.node_right = node2
            else:
                raise 'Algo deu muito errado'
        elif isinstance(parent1, Function):
            if DEBUG: log('Parent1 is Function')
            parent1.node = node2

        if type(parent2) is Node:
            if DEBUG: log('parent2 is Node')
            parent2.children = node1
        elif type(parent2) is Operator:
            if DEBUG: log('parent2 is Operator')
            idleft = parent2.node_left.get_unique_id()
            idright = parent2.node_right.get_unique_id()
            if idleft == id2:
                parent2.node_left = node1
            elif idright == id2:
                parent2.node_right = node1
            else:
                raise 'Algo deu muito errado'
        elif isinstance(parent2, Function):
            if DEBUG: log('parent2 is Function')
            parent2.node = node1

    def crossover(self, ind1, ind2, prob=0.5):
        """
        Troca sub-arvores
        """
        if DEBUG:
            log(
                '====================\n'
                'Run crossover',
                '\nind1 =', ''.join([ind1.get_unique_id()[:5], '...']), ind1,
                '\nind2 =', ''.join([ind2.get_unique_id()[:5], '...']), ind2,
                '\n=-=-=-=-=-=-=-=-=-=-')

        if np.random.random() < prob:
            if DEBUG: log('True crossover')

            ind1 = ind1.copy()
            ind2 = ind2.copy()
            ind_aux1 = ind1.copy()
            ind_aux2 = ind2.copy()
            ind1_uniques = ind_aux1.walk()
            ind2_uniques = ind_aux2.walk()

            x = np.random.choice(list(range(len(ind1_uniques))))
            y = np.random.choice(list(range(len(ind2_uniques))))
            choice1 = ind1_uniques[x]
            choice2 = ind2_uniques[y]
            self.cross(choice1, choice2)
            # TODO: select best
            arr = []
            arr.append([ind1, ind1.calc_fitness(self.train_data)])
            arr.append([ind2, ind2.calc_fitness(self.train_data)])
            arr.append([ind_aux1, ind_aux1.calc_fitness(self.train_data)])
            arr.append([ind_aux2, ind_aux2.calc_fitness(self.train_data)])
            arr.sort(key=lambda x: x[1])

            if DEBUG: log('result:', arr[0][0], arr[1][0])
            return arr[0][0], arr[1][0]

        return ind1.copy(), ind2.copy()

    def tournament(self, k=2):
        """
        """
        # print_warning('tour')
        n = len(self.individuals)
        possible_choices = list(range(0, n))
        np.random.shuffle(possible_choices)
        choices = possible_choices[:k]
        arr = []
        for choice in choices:
            ind = self.individuals[choice]
            fitness = self.individuals[choice].calc_fitness(self.train_data)
            arr += [[fitness, ind]]

        arr.sort(key=lambda x: x[0])
        return arr[0][1]

    def mutation(self, ind, prob=0.5):
        if DEBUG:
            log(
                '====================\n'
                'Run mutation',
                '\nind =', ''.join([ind.get_unique_id()[:5], '...']), ind,
                '\n=-=-=-=-=-=-=-=-=-=-')

        if np.random.random() < prob:
            if DEBUG: log('True mutation')

            ind_aux = ind.copy()
            ind_uniques = ind_aux.walk()
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
                    parent1.node_left = generate_subtree(
                        self.n_dim, self.depth - depth)
                elif idright == id1:
                    parent1.node_right = generate_subtree(
                        self.n_dim, self.depth - depth)
                else:
                    raise 'Algo deu muito errado'
                # add optimizer
                # (((X1 / 3.01) + (exp(X1 - 0.32) +
                #           (sin(X1 - 3.76) + ((X0 + 8.34) + X1)))) + 12.53)
                # ((X1 / 3.01) + (exp(X1 - 0.32) +
                #           (sin(X1 - 3.76) + ((X0 + 8.34) + X1))))
                # (4.99 + 12.53)
            elif isinstance(parent1, Function):
                parent1.node = generate_subtree(self.n_dim, self.depth - depth)

            # print_warning(ind)
            # self.show_ind()
            # Escolher o melhor
            fit_1 = ind_aux.calc_fitness(self.train_data)
            fit_2 = ind.calc_fitness(self.train_data)
            if DEBUG: log('result:', ind_aux)
            return ind_aux if fit_1 < fit_2 else ind.copy()
        return ind.copy()

    def new_population(self):
        if DEBUG:
            log(
                '====================\n'
                'Run new_population',
                '=-=-=-=-=-=-=-=-=-=-')

        new_inds = []

        while len(new_inds) <= self.population:
            choose = np.random.random_integers(0, 1)
            if choose == 0:
                # Mutation
                ind = self.tournament(k=self.k)
                ind = self.mutation(ind, prob=self.prob_m)
                new_inds += [ind]
            elif choose == 1:
                # Crossover
                ind1 = self.tournament(k=self.k)
                ind2 = self.tournament(k=self.k)
                ind1, ind2 = self.crossover(ind1, ind2, prob=self.prob_c)
                new_inds += [ind1, ind2]
            # elif choose == 2:
                # Reproduction
            else:
                ind = generate_individual(self.n_dim, depth=self.depth)
                new_inds += [ind]

        self.individuals = new_inds

    def show_ind(self):
        """
        Show all individuals
        DEBUG purpose
        """
        print_green('Unique ID (SHA256)\tExpression')
        for ind in self.individuals:
            print_blue(str(ind.get_unique_id())[:13] + '...\t' + ind.__str__())


def main():
    gp = GeneticProgramming(population=25, depth=7, prob_c=1.0, prob_m=1.0)
    
    count = 0
    while True:
        print('Generation', count + 1)
        count+=1
        if count == 500:
            break
        gp.new_population()
        print(gp.get_fitness()[0][0][0])
        # for i in range(5):
        #     print(gp.individuals[i])
        #     print(gp.individuals[i].calc_fitness(gp.train_data))

    gp.show_ind()

    print(gp.individuals[0], gp.individuals[0].calc_fitness(gp.train_data))


if __name__ == '__main__':
    main()
