import argparse
import os
import sys
from individual import *
import numpy as np
from utils import log

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
            prob_m=0.5,
            generations=10
        ):
        """Init a Genetic Programming with params and individuals (equation trees)
        
        Keyword Arguments:
            data_name {str} -- Name of database that contains train and test csv (default: {'synth1'})
            population {int} -- Max number of individuals of each generation (default: {50})
            max_depth {int} -- Max depth of tree (default: {7})
            k {int} -- k individuals to use in tournament (default: {2})
            prob_c {float} -- probability of crossover (default: {0.1})
            prob_m {float} -- probability of mutation (default: {0.1})
            generations {int} -- generations of loop
        """

        if prob_c + prob_m > 1.0:
            raise 'Probability of Crossver plus Mutation is over then 1'

        self.max_ind = population
        self.max_depth = max_depth
        self.k = k
        self.prob_c = prob_c
        self.prob_m = prob_m
        self.generations = generations
        
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
        
        if not os.path.exists(os.getcwd() + '/output'):
            os.mkdir(os.getcwd() + '/output')
        elif not os.path.isdir(os.getcwd() + '/output'):
            raise 'Remove output file, the GP need to write files in output folder'
        
        output_filename = 'output/' + data_name + '_'
        output_filename += 'p' + str(self.max_ind) + '_'
        output_filename += 'g' + str(self.generations) + '_'
        output_filename += 'k' + str(self.k) + '_'
        output_filename += 'c' + str(self.prob_c) + '_'
        output_filename += 'm' + str(self.prob_m) + '_'
        count = 1
        while True:
            if os.path.exists(os.getcwd() + '/'+ output_filename + str(count).zfill(3) + '.csv'):
                count+=1
            else:
                output_filename += str(count).zfill(3)
                break
        self.output_data_filename = output_filename + '.csv'
        self.output_filename = output_filename + '.txt'
        f = open(self.output_data_filename, 'w')
        f.close()
        f = open(self.output_filename, 'w')
        f.close()
    
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
        mean_fitness = 0.0
        count = 0
        for fit in np.array(self.individuals)[:,2]:
            if np.isfinite(fit):
                mean_fitness += fit
                count += 1
        if count > 0:
            mean_fitness /= count
        else:
            mean_fitness = float('inf')
        return self.individuals[0], self.individuals[-1], mean_fitness

    def loop(self):
        count = 1
        while True:
            best, worst, mean_fit = self.get_best_and_worst()
            with open(self.output_filename, 'a') as f:
                f.write('\nGeneration' + str(count) + '\n')
                f.write('Best Individual\n')
                f.write('\t' + ''.join([best[0][:7], '...']) + str(best[1]) + '\n')
                f.write('\terror = ' + str(best[2]) + '\n')
                f.write('Worst Individual\n')
                f.write('\t' + ''.join([worst[0][:7], '...']) + str(worst[1]) + '\n')
                f.write('\terror = ' + str(worst[2]) + '\n')
                f.write('Mean Error\n')
                f.write('\terror = ' + str(mean_fit) + '\n')

            with open(self.output_data_filename, 'a') as f:
                f.write(str(best[2]) + ',' + str(worst[2]) + ',' + str(mean_fit) + '\n')

            if count == self.generations:
                break

            self.gen_new_population()
            count += 1

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
        best, worst, mean_fit = self.get_best_and_worst()
        print('Best Individual')
        print('\t error =', best[2])
        print('\t', ''.join([best[0][:7], '...\t']), best[1])
        print('Worst Individual')
        print('\t error =', worst[2])
        print('\t', ''.join([worst[0][:7], '...\t']), worst[1])
        print('Mean error')
        print('\t error =', mean_fit)

        with open(self.output_filename, 'a') as f:
            f.write('\nTest data\nBest Individual\n')
            f.write('\t' + ''.join([best[0][:7], '...']) + str(best[1]) + '\n')
            f.write('\terror = ' + str(best[2]) + '\n')
            f.write('Worst Individual\n')
            f.write('\t' + ''.join([worst[0][:7], '...']) + str(worst[1]) + '\n')
            f.write('\terror = ' + str(worst[2]) + '\n')
            f.write('Mean Error\n')
            f.write('\terror = ' + str(mean_fit) + '\n')

        print('\nShow the five best')
        with open(self.output_filename, 'a') as f:
            f.write('\nShow the five best\n')
        for ind in self.individuals[:5]:
            print('\t',
                  ''.join([ind[0][:7], '...\t']),
                  ind[1])
            print('\t error =', ind[2])
            with open(self.output_filename, 'a') as f:
                f.write('\t' + ''.join([ind[0][:7], '...\t']) + str(ind[1]) + '\n')
                f.write('\terror = ' + str(ind[2]))
        
        with open(self.output_filename, 'a') as f:
            f.write('\nTest data\nBest Individual\n')
            f.write('\t' + ' '.join([best[0][:7], '...']) + str(best[1]) + '\n')
            f.write('\terror = ' + str(best[2]) + '\n')
            f.write('Worst Individual\n')
            f.write('\t' + ' '.join([worst[0][:7], '...']) + str(worst[1]) + '\n')
            f.write('\t error = ' + str(worst[2]) + '\n')
            f.write('Mean Error\n')
            f.write('\t error = ' + str(mean_fit) + '\n')
        
        with open(self.output_data_filename, 'a') as f:
            f.write(str(best[2]) + ',' + str(worst[2]) + ',' + str(mean_fit) + '\n')
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_name', metavar='data_name', type=str,
        help='Name of training data file inside database folder')
    parser.add_argument('-p', dest='population', default=50, type=int,
        help='Population size for each generation')
    parser.add_argument('-k', dest='k_tournament', default=2, type=int,
        help='Number of individuals to participate in tournaments')
    parser.add_argument('-g', dest='generations', default=10, type=int,
        help='Number of generations')
    parser.add_argument('-c', dest='prob_c', default=0.9, type=float,
        help='Crossover probability')
    parser.add_argument('-m', dest='prob_m', default=0.05, type=float,
        help='Mutation propability')
    
    args = parser.parse_args()
    
    gp = GeneticProgramming(
        data_name=args.data_name,
        population=args.population,
        prob_c=args.prob_c,
        prob_m=args.prob_m,
        k=args.k_tournament,
        max_depth=8,
        generations=args.generations
        )
    gp.loop()

if __name__ == '__main__':
    main()
