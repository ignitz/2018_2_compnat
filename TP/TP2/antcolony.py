import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import socket
import os

OUTPUT_FOLDER = 'output/'


def read_dataset(infilename):
    """Ler arquivos txt e retornam em formato de lista
    
    Arguments:
        infilename {str} -- nome do arquivo
    
    Returns:
        list -- lista multi dimensional
    """

    try:
        with open('datasets/' + infilename) as f:
            content = f.read()
            if ' ' in content:
                content = content.replace(' ', '\t')
            content = content.split('\n')
            # TODO, fix last line
            content = [[int(x) for x in line.split('\t')] for line in content[:-1]]
            return content
    except:
        print('Error in opening file ' + infilename)
        raise

def plot_graph(G):
    """Plota gráfico em formato networkx
    
    Arguments:
        G {instance of Graph} -- Grafo direcionado ou não
    """

    labels = nx.get_edge_attributes(G,'p_xy')
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True)
    # colors = [G[u][v]['color'] for u,v in G.edges()]
    nx.draw_circular(G,node_color='r', with_labels=True, edge_labels=labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Grafo")
    plt.show()

class Ant:
    """Atributos auxiliares para cada formiga
    origin: vértice de origem
    dest: vértice de destino
    path: armazena o caminho
    fitness: o custo total do caminho
    pheromone: custo parciais para a atualização do feromônio em cada aresta
    """
    def __init__(self, origin, dest):
        self.origin = origin
        self.dest = dest
        self.path = []
        self.fitness = 0
        self.pheromone = []

class AntColony(nx.DiGraph):
    """ACO - Ant Colony Optimization
    É uma extensão da classe DiGraph.
    Contém métodos de extensão para o algoritmo de colônias de formigas
    """

    def __init__(self, dataset_file, ants_num=50, it_num=50, origin=None, dest=None, alpha=1.0, beta=1.0, sigma=0.01, init_phero=0.1, k_ants=0):
        """AntColony __init__.
        
        Arguments:
            dataset_file {str} -- Nome do dataset (sem .txt)
        
        Keyword Arguments:
            ants_num {int} -- número de formigas (default: {50})
            it_num {int} -- quantidade de iterações (default: {50})
            origin {int} -- vértice de origem (default: {None})
            dest {int} -- vértice de destino (default: {None})
            alpha {float} -- parâmetro do feromônio (default: {1.0})
            beta {float} -- parâmetro da atratividade (default: {1.0})
            sigma {float} -- taxa de evaporação do feromônio (default: {0.01})
            init_phero {float} -- feromônio inicial (default: {0.1})
            k_ants {int} -- quantidade de formigas para elitismo de cada iteração (default: {0})
        """

        nx.DiGraph.__init__(self)
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma # evap coeficiente
        self.ants_num = ants_num
        self.it_num = it_num
        self.ants = []
        self.k_ants = k_ants

        data = read_dataset(dataset_file)
        self.add_edges(data)

        # if origin and destiny are not set, find minimum and max index to set
        nodes = list(self.nodes())
        self.origin = min(nodes) if origin is None else origin
        self.dest = max(nodes) if dest is None else dest

        # set initial pheromone
        for u in list(self.nodes()):
            for v in self.get_neighbors(u):
                self.set_pheromone(u, v, float(init_phero))
        
        self.update_probs()

        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        hostname = str(socket.gethostname())

        count = 1
        dataset_name = dataset_file.split('.')[0]
        if os.path.exists(OUTPUT_FOLDER + \
                             dataset_name + '_' + \
                             str(self.it_num) +'_' + \
                             str(self.ants_num) +'_' + \
                             str(self.origin) +'_' + \
                             str(self.dest) +'_' + \
                             str(self.alpha) +'_' + \
                             str(self.beta) +'_' + \
                             str(self.sigma) +'_' + \
                             str(init_phero) +'_' + \
                             str(self.k_ants) +'_' + \
                             hostname + '_' + \
                             str(count).zfill(3) + \
                             '.csv'):
            self.already_exists = True
            print(dataset_name + '_' + \
                  str(self.it_num) +'_' + \
                  str(self.ants_num) +'_' + \
                  str(self.origin) +'_' + \
                  str(self.dest) +'_' + \
                  str(self.alpha) +'_' + \
                  str(self.beta) +'_' + \
                  str(self.sigma) +'_' + \
                  str(init_phero) +'_' + \
                  str(self.k_ants) +'_' + \
                  hostname + '_' + \
                  str(count).zfill(3) + \
                  '.csv')
            return
        self.already_exists = False
        
        self.out_filename = '/tmp/' + dataset_name + '_' + \
                             str(self.it_num) +'_' + \
                             str(self.ants_num) +'_' + \
                             str(self.origin) +'_' + \
                             str(self.dest) +'_' + \
                             str(self.alpha) +'_' + \
                             str(self.beta) +'_' + \
                             str(self.sigma) +'_' + \
                             str(init_phero) +'_' + \
                             str(self.k_ants) +'_' + \
                             hostname + '_' + \
                             str(count).zfill(3) + \
                             '.csv'
        self.out_final_filename = OUTPUT_FOLDER + \
                             dataset_name + '_' + \
                             str(self.it_num) +'_' + \
                             str(self.ants_num) +'_' + \
                             str(self.origin) +'_' + \
                             str(self.dest) +'_' + \
                             str(self.alpha) +'_' + \
                             str(self.beta) +'_' + \
                             str(self.sigma) +'_' + \
                             str(init_phero) +'_' + \
                             str(self.k_ants) +'_' + \
                             hostname + '_' + \
                             str(count).zfill(3) + \
                             '.csv'

        with open(self.out_filename, 'w'):
            pass

    
    def add_edge(self, edge):
        """adiciona aresta com pesos e armazena atributos
        
        Arguments:
            edge {tuple} -- (u,v, peso da aresta)
        """

        assert(type(edge) is tuple)
        self.add_weighted_edges_from([edge], t_xy=0.0, p_xy=0.0, color='black')
    
    def add_edges(self, list_edges):
        """adiciona uma lista de arestas e armazenam atributos
        
        Arguments:
            list_edges {list of tuples} -- lista de (u,v, peso da aresta)
        """

        assert(type(list_edges) is list)
        for e in list_edges:
            assert((type(e) is list) or (type(e) is tuple))
        self.add_weighted_edges_from(list_edges, t_xy=0.0, p_xy=0.0, color='black')

    def get_weight(self, u, v):
        """retorna o peso da aresta
        
        Arguments:
            u {int} -- aresta de saída
            v {int} -- aresta de entrada
        
        Returns:
            int -- peso da aresta
        """

        if self.has_edge(u, v):
            return self[u][v]['weight']
        else:
            raise 'Edge ({},{}) not exist'.format(u, v)
    
    def get_edge(self, u, v):
        """retorna dados da aresta
        
        Arguments:
            u {int} -- aresta de saída
            v {int} -- aresta de entrada
        
        Returns:
            dict -- dados da aresta (peso, t_xy, p_xy) além de mais informação da classe nx.Graph
        """

        if self.has_edge(u, v):
            return self[u][v]
        else:
            raise 'Edge ({},{}) not exist'.format(u, v)
    
    def plot_neighbors(self, u):
        """Plota apenas os vértices e arestas vizinhas ao vértice "u" DEBUG PURPOSE
        
        Arguments:
            u {int} -- aresta desejada
        """

        if self.has_node(u):
            nodes = [u]
            nodes += self.neighbors(u)
            plot_graph(self.subgraph(nodes))
        else:
            print('node {} don\'t exists in graph!'.format(u))
    
    def plot_graph(self):
        """Plota o grafo da classe AntColony
        """

        plot_graph(self)
    
    def get_pheromone(self, u, v):
        """Retorna o feromônio da aresta
        
        Arguments:
            u {int} -- aresta de origem
            v {int} -- aresta de entrada
        
        Returns:
            float -- retorna a taxa de feromônio na aresta
        """

        assert(self.has_edge(u, v))
        return self.get_edge(u,v)['t_xy']
    
    def set_pheromone(self, u, v, new_pxy):
        """Seta novo valor do feromônio na aresta
        
        Arguments:
            u {int} -- aresta de origem
            v {int} -- aresta de entrada
            new_pxy {float} -- novo valor para t_xy na aresta
        """

        assert(self.has_edge(u, v))
        assert(type(new_pxy) is float)
        self[u][v]['t_xy'] = new_pxy

    def get_neighbors(self, u):
        """Retorna os nós vizinhos a "u"
        
        Arguments:
            u {int} -- vértice
        
        Returns:
            lista de identificadores do nó -- example [2,3,5 ...] se utilizar inteiros
        """

        assert(self.has_node(u))
        return list(self.neighbors(u))
      
    def fitness(self, path):
        """Retorna a fitness do caminho dado
        
        Arguments:
            path {list} -- lista de nós do caminho
        
        Returns:
            int -- custo total do caminho
        """

        assert(type(path) is list)
        if (len(path) == 0):
            return 0
        
        ttl_weight = 0

        u = path.pop(0)

        for v in path:
            weight = self.get_weight(u, v)
            u = v
            ttl_weight += weight
        
        return ttl_weight
        
    def evaporate_pheromones(self):
        """Evapora a taxa de feromônio de todas as arestas

        A evaporação depende do valor de sigma
        """

        for u in list(self.nodes()):
            for v in self.get_neighbors(u):
                curr_pheromone = self.get_pheromone(u,v)
                self.set_pheromone(u, v, curr_pheromone*(1-self.sigma))
    
    def update_probs(self):
        """Atualiza todos os atributos de probabilidade p_xy em todas as arestas

        A probabilidade é normalizada para que todos os vizinhos.
        """

        for u in list(self.nodes()):
            neighbors = self.get_neighbors(u)
            aux_list = []
            for v in neighbors:
                curr_pher = self.get_pheromone(u, v)
                weight = self.get_weight(u,v)
                aux = math.pow(curr_pher, self.alpha) * math.pow(weight, self.beta)
                aux_list.append(aux)
            
            normalize = math.fsum(aux_list)
            
            for v in neighbors:
                curr_pher = self.get_pheromone(u, v)
                weight = self.get_weight(u,v)
                aux = math.pow(curr_pher, self.alpha) * math.pow(weight, self.beta)
                self[u][v]['p_xy'] = aux/normalize
    
    def choose_new_node(self, u, excl_nodes=[]):
        """Escolhe um novo nó adjancente a "u" utilizando o atributo de probabilidade das arestas
        
        Arguments:
            u {int} -- vértice do estado atual
        
        Keyword Arguments:
            excl_nodes {list} -- lista de nós que deseje excluir da escolha (default: {[]})
        
        Returns:
            vértice -- retorna o vértice escolhido
        """

        all_probs = []
        vs = self.get_neighbors(u)
        if len(vs) == 0: return u
        vs = list(set(vs).difference(excl_nodes))

        for v in vs:
            all_probs.append(self[u][v]['p_xy'])
        normalize = math.fsum(all_probs)
        all_probs = np.array(all_probs) / normalize
        
        choice = np.random.choice(len(all_probs), p=all_probs.tolist())
        return vs[choice]

    def update_pheromone(self, max_pher=None):
        """Atualiza os pheromonios das arestas em que as formigas passaram
        
        Keyword Arguments:
            max_pher {int} -- custo máximo para normalização/penalização, se não for passado é calculado (default: {None})
        """

        if max_pher is None:
            max_pher = 0
            for ant in self.ants:
                max(max_pher, ant.fitness)
    
        for ant in self.ants:
            for i in range(len(ant.pheromone)):
                u = ant.pheromone[i][0]
                v = ant.pheromone[i][1]
                # self[u][v]['t_xy'] += ant.pheromone[i][2] / (15*max_pher)
                self[u][v]['t_xy'] += (ant.fitness + 0.5 - ant.pheromone[i][2]) / max_pher
                # self[u][v]['t_xy'] += 1.0 - 1.0 / ant.fitness
                self[u][v]['t_xy'] = min(self[u][v]['t_xy'], 100.0)
    
    def flux_colony(self):
        """Fluxo do algoritmo pela quantidade de iterações
        No final, o arquivo de dados de saída é movido da pasta temporária para a pasta output/
        """

        for iteration in range(self.it_num):
            print('Iteration', iteration + 1)
            self.iteration()
        import shutil
        shutil.move(self.out_filename, self.out_final_filename)

    def iteration(self):
        """Iteração do algoritmo
        """

        self.ants.sort(key=lambda x: x.fitness, reverse=True)
        
        if self.ants_num < self.k_ants:
            raise Exception('k_ants are bigger then number of ants')
        
        self.ants = self.ants[:self.k_ants]
        
        max_pher = 0
        while len(self.ants) < self.ants_num:
            self.ants.append(Ant(self.origin, self.dest))
        
        for ant in self.ants:
            u = ant.origin
            if len(ant.path) > 0:
                continue
            ant.path.append(u)
            
            while u != ant.dest:
                neighbors = self.get_neighbors(u)
                if set(neighbors).issubset(ant.path):
                    # try again
                    ant.path.clear()
                    ant.pheromone.clear()
                    ant.fitness = 0
                    u = ant.origin
                    ant.path.append(u)
                    continue

                new_u = self.choose_new_node(u, ant.path)
                if new_u == u:
                    ant.path.clear()
                    ant.pheromone.clear()
                    ant.fitness = 0
                    u = ant.origin
                    ant.path.append(u)
                    continue
                
                ant.path.append(new_u)
                ant.fitness += self.get_weight(u, new_u)
                pher = ant.fitness
                ant.pheromone.append((u, new_u, pher))
                u = new_u
            
            max_pher = max(ant.fitness, max_pher)
        
        self.ants.sort(key=lambda x: x.fitness, reverse=True)
        self.update_pheromone(max_pher=max_pher)
        self.update_probs()
        self.evaporate_pheromones()
        # self.print_ants(self.k_ants)
        self.print_fitness_stat()
        # self.print_pheromones()
    
    def print_ants(self, n=None):
        """Imprime todas as formigas DEBUG PURPOSE
        
        Keyword Arguments:
            n {int} -- imprime n formigas (default: {None})
        """ 

        self.ants.sort(key=lambda x: x.fitness, reverse=True)
        
        print('------------------------------\nAnts\n------------------------------')
        for ant in self.ants[:n]:
            print(ant.fitness, ant.path)

    def print_fitness_stat(self):
        """Imprime o Máximo, Mínimo e a Média das fitsness das formigas
        Também é gravado estes dados no arquivo de saída.
        """

        self.ants.sort(key=lambda x: x.fitness, reverse=True)
        fitness = []

        for ant in self.ants:
            fitness.append(ant.fitness)

        print('==============================\n' + self.out_final_filename)    
        print('------------------------------\nMax, Min, Mean\n------------------------------')
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        mean_fitness = np.round(np.mean(fitness), 2)
        var_fitness = np.round(np.var(fitness), 2)

        with open(self.out_filename, 'a') as w:
            content = str(max_fitness) + ';' + str(min_fitness) + ';' + \
                      str(mean_fitness) + ';' + str(var_fitness) + '\n'
            w.write(content)
        print('\t'.join(content.split(';')))
