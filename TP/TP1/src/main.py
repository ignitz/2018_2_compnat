
import argparse
import os
import sys
from individual import *
import numpy as np

# from utils import log

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
                 filename + '/' + filename + '-test.csv')

    def read_file(pathname):
        aux_data = list()
        with open(pathname, 'r') as f:
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

class GeneticProgramming:
    """GP."""

    def __init__(
            self,
            data_name='synth1',
            population=50,
            max_depth=7,
            k=2,
            prob_c=0.5,
            prob_m=0.5
        ):
        """Init a Genetic Programming with params and individuals (equation trees)
        
        Keyword Arguments:
            data_name {str} -- Name of database that contains train and test csv (default: {'synth1'})
            population {int} -- Max number of individuals of each generation (default: {50})
            max_depth {int} -- Max depth of tree (default: {7})
            k {int} -- k individuals to use in tournament (default: {2})
            prob_c {float} -- probability of crossover (default: {0.1})
            prob_m {float} -- probability of mutation (default: {0.1})
        """

        if prob_c + prob_m > 1.0:
            raise 'Probability of Crossver plus Mutation is over then 1'

        self.max_ind = population
        self.max_depth = max_depth
        self.k = k
        self.prob_c = prob_c
        self.prob_m = prob_m
        
        # This dict have a purpose to avoid fitness calculation
        """
        @key: unique id (SHA256) of tree
        @value: fitness already calculated
        """
        self.cache_results = dict()

        # Load database
        self.train_data, self.test_data = load_data(data_name)

        # insanity check
        if self.train_data.shape[1] != self.train_data.shape[1]:
            raise 'Train and Test data are diff dimensions'
        
        # set dimension of domain
        self.n_dim = self.train_data.shape[1] - 1

        # Save to variable for RMSE fitness calculation
        y_mean = self.train_data[:, -1].mean()
        normalize = self.train_data[:, -1] - y_mean
        normalize = np.power(normalize, 2)
        normalize = np.sum(normalize)
        self.normalize = normalize

        # Generate initial population
        """Individuals list is a tuple of
        (unique ID, root tree reference, fitness)
        """
        self.individuals = list()
        for _ in range(population):
            ind = generate_individual(self.n_dim, self.max_depth)
            uniq_id = ind.get_unique_id()
            ind_fit = ind.calc_fitness(self.test_data, self.normalize)
            self.cache_results[uniq_id] = ind_fit

            self.individuals.append((uniq_id, ind, ind_fit))
    
    def selection(self):
        choose = np.random.random()
        response = list()
        if 0 <= choose < self.prob_c:
            response += [*self.crossover()]
        elif self.prob_c <= choose <= self.prob_c + self.prob_m:
            response += [self.mutation()]
        else:
            response += [self.reproduction()]
        return response
    
    def reproduction(self):
        t_ind, _ = self.tournament(self.k)
        return t_ind
    
    def tournament(self, k):
        """Return the best of k random indiduals
        
        Keyword Arguments:
            k {int} -- how many indiduals are random choosen
        
        Returns:
            two tuples {list} -- (unique_id, ind_tree, fitness)
        """
        np.random.shuffle(self.individuals)
        k_individuals = self.individuals[:k]
        k_individuals.sort(key=lambda x: x[2])

        return k_individuals[:2]
    
    def mutation(self):
        t_ind, _ = self.tournament(self.k)

        t_aux = [t_ind[0], t_ind[1].copy(), t_ind[2]]
        ind_uniques = t_aux[1].walk()
        x = np.random.choice(list(range(len(ind_uniques))))
        parent, _, uniq_id, depth = ind_uniques[x]

        if type(parent) is Node:
            parent.children = generate_subtree(self.n_dim, self.max_depth)
        elif type(parent) is Operator:
            idleft = parent.node_left.get_unique_id()
            idright = parent.node_right.get_unique_id()
            if idleft == uniq_id:
                parent.node_left = generate_subtree(
                    self.n_dim, self.max_depth - depth)
            elif idright == uniq_id:
                parent.node_right = generate_subtree(
                    self.n_dim, self.max_depth - depth)
            else:
                raise 'Algo deu muito errado'
        
        t_aux[0] = t_aux[1].get_unique_id()
        
        if t_aux[0] not in self.cache_results:
            self.cache_results[t_aux[0]] = t_aux[1].calc_fitness(
                self.train_data,
                self.normalize
            )
        t_aux[2] = self.cache_results[t_aux[0]]
        
        return t_ind if t_ind[2] < t_aux[2] else tuple(t_aux)
    
    def crossover(self):
        t_ind1, t_ind2 = self.tournament(self.k)
        t_aux1 = [t_ind1[0], t_ind1[1].copy(), t_ind1[2]]
        t_aux2 = [t_ind2[0], t_ind2[1].copy(), t_ind2[2]]

        """Return list of tuples of each node
        (parent node, actual node, depth)
        """
        ind1_uniques = t_aux1[1].walk()
        ind2_uniques = t_aux2[1].walk()

        x = np.random.choice(list(range(len(ind1_uniques))))
        y = np.random.choice(list(range(len(ind2_uniques))))
        choice1 = ind1_uniques[x]
        choice2 = ind2_uniques[y]
        # do cross
        self.cross(choice1, choice2)
        # who is the best
        t_aux1[0] = t_aux1[1].get_unique_id()
        t_aux2[0] = t_aux2[1].get_unique_id()

        if t_aux1[0] not in self.cache_results:
            aux1_depths = np.array(t_aux1[1].walk())[:,-1]
            if np.any(aux1_depths > self.max_depth):
                self.cache_results[t_aux1[0]] = float('inf')
            else:
                self.cache_results[t_aux1[0]] = t_aux1[1].calc_fitness(
                    self.train_data,
                    self.normalize
                )
        t_aux1[2] = self.cache_results[t_aux1[0]]

        if t_aux2[0] not in self.cache_results:
            aux2_depths = np.array(t_aux2[1].walk())[:,-1]
            if np.any(aux2_depths > self.max_depth):
                self.cache_results[t_aux2[0]] = float('inf')
            else:
                self.cache_results[t_aux2[0]] = t_aux2[1].calc_fitness(
                self.train_data,
                self.normalize
            )
        t_aux2[2] = self.cache_results[t_aux2[0]]

        arr =  [[t_aux1, False],
                [t_aux2, False],
                [list(t_ind1), True],
                [list(t_ind2), True]]
        arr.sort(key=lambda x: x[0][2])
        a, b = arr[:2]

        if a[1] == True:
            a[0][1] = a[0][1].copy()
        if b[1] == True:
            b[0][1] = b[0][1].copy()
        
        return tuple(a[0]), tuple(b[0])
    
    def cross(self, choice1, choice2):
        parent1, node1, id1, _ = choice1
        parent2, node2, id2, _ = choice2

        # TODO: Remove get_unique_id later
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
    
    def gen_new_population(self):
        new_inds = []
        while len(new_inds) < self.max_ind:
            new_inds += self.selection()
        
        if len(new_inds) == self.max_ind + 1:
            new_inds = new_inds[:-1]
        
        self.individuals = new_inds    
    
    def get_best_and_worst(self):
        self.individuals.sort(key=lambda x: x[2])
        return self.individuals[0], self.individuals[-1]

    def loop(self, max_generations=50):
        count = 1
        while True:
            print('\nGeneration', count)
            best, worst = self.get_best_and_worst()
            print('Best Individual')
            print('\t error =', best[2])
            print('\t', ''.join([best[0][:7], '...']), best[1])
            print('Worst Individual')
            print('\t error =', worst[2])
            print('\t', ''.join([worst[0][:7], '...']), worst[1])

            if count == max_generations:
                break

            self.gen_new_population()
            count += 1

        # TODO: show median

        print('\nTest data')
        y_mean = self.test_data[:, -1].mean()
        normalize = self.test_data[:, -1] - y_mean
        normalize = np.power(normalize, 2)
        normalize = np.sum(normalize)

        for i in range(len(self.individuals)):
            fit = self.individuals[i][1].calc_fitness(self.test_data, normalize)
            self.individuals[i] = (
                self.individuals[i][0],
                self.individuals[i][1],
                fit)
        best, worst = self.get_best_and_worst()
        print('Best Individual')
        print('\t error =', best[2])
        print('\t', best[0], best[1])
        print('Worst Individual')
        print('\t error =', worst[2])
        print('\t', worst[0], worst[1])

        print('\nShow the five best')
        for ind in self.individuals[:5]:
            print('\t',
                  ''.join([ind[0][:7], '...']),
                  ind[1])
            print('\t', ind[2])
    
def main():
    gp = GeneticProgramming(
        data_name=sys.argv[1],
        population=sys.argv[2],
        prob_c=sys.argv[3],
        prob_m=sys.argv[4],
        k=sys.argv[5],
        max_depth=8
        )
    # gp = GeneticProgramming(
    #     # data_name='keijzer7',
    #     # data_name='keijzer10',
    #     # data_name='synth1',
    #     # data_name='synth2',
    #     data_name='house',
    #     # data_name='concrete',
    #     population=50,
    #     prob_c=0.50,
    #     prob_m=0.50,
    #     k=3,
    #     max_depth=8
    #     )
    gp.loop(50)

if __name__ == '__main__':
    main()
