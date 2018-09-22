import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def read_file(pathname):
    aux_data = list()
    with open(pathname, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            aux_data.append(list(map(float, line.split(','))))
    return np.matrix(aux_data)

def load_data(dataname, p, g, k, c, m):
    count = 1
    filename = 'output/' + dataname
    filename += '_p' + str(p) + '_g' + str(g)
    filename += '_k' + str(k) + '_c' + str(c) + '_m' + str(m)
    filename += '_' + str(count).zfill(3) + '.csv'
    
    count += 1
    data = read_file(filename)
    while count <= 5:
        filename = 'output/' + dataname
        filename += '_p' + str(p) + '_g' + str(g)
        filename += '_k' + str(k) + '_c' + str(c) + '_m' + str(m)
        filename += '_' + str(count).zfill(3) + '.csv'
        data += read_file(filename)
        count += 1
    return (data/5)

# data = load_data('synth1', 50, 50, 2, 0.3, 0.3)
# primeiro tem que testar com 50,50
# variando mutacao e crossover
# depois varia o k

def main():
    EIXOS = [0, 55, 0, 1.5]

<<<<<<< HEAD
    # fig, ax = plt.subplots()

    # for x in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
    #     generations = 50
    #     data = load_data('synth1', 50, generations, 2, x, 0.05)
    #     label = 'P(cross) = ' + str(x)
    #     ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    # title = 'Variação do Crossover'
    # ax.set(xlabel='Gerações', ylabel='Fitness médio',
    #        title=title)
    # ax.axis(EIXOS)
    # ax.legend()
    # ax.grid()

    # fig.savefig("varcrossover.pdf")
    # # plt.show()

    # # AAAAAAAAAAAAAAAAA
    # fig, ax = plt.subplots()

    # for x in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
    #     generations = 50
    #     data = load_data('synth1', 50, generations, 2, 0.05, x)
    #     label = 'P(mut) = ' + str(x)
    #     ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    # title = 'Variação da Mutação'
    # ax.set(xlabel='Gerações', ylabel='Fitness médio',
    #        title=title)
    # ax.axis(EIXOS)
    # ax.legend()
    # ax.grid()

    # fig.savefig("varmutation.pdf")
    # # plt.show()

    # # AAAAAAAAAAAAAAAAA
    # fig, ax = plt.subplots()

    # for x in [2,3,4,5,6,7]:
    #     generations = 50
    #     data = load_data('synth1', 50, generations, x, 0.3, 0.6)
    #     label = 'k = ' + str(x)
    #     ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    # title = 'Variação da K'
    # ax.set(xlabel='Gerações', ylabel='Fitness médio',
    #        title=title)
    # ax.axis(EIXOS)
    # ax.legend()
    # ax.grid()

    # fig.savefig("varK.pdf")


    fig, ax = plt.subplots()

    for x in [50, 100, 500, 1000]:
        generations = 50
        data = load_data('synth1', x, generations, 3, 0.3, 0.6)
        data += load_data('synth2', x, generations, 3, 0.3, 0.6)
        data += load_data('concrete', x, generations, 3, 0.3, 0.6)
        data /= 3
        label = 'Pop = ' + str(x)
        ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    title = 'Variação da população'
=======
    fig, ax = plt.subplots()

    for x in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        generations = 50
        data = load_data('synth1', 50, generations, 2, x, 0.05)
        label = 'P(cross) = ' + str(x)
        ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    title = 'Variação do Crossover'
    ax.set(xlabel='Gerações', ylabel='Fitness médio',
           title=title)
    ax.axis(EIXOS)
    ax.legend()
    ax.grid()

    fig.savefig("varcrossover.pdf")
    # plt.show()

    # AAAAAAAAAAAAAAAAA
    fig, ax = plt.subplots()

    for x in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        generations = 50
        data = load_data('synth1', 50, generations, 2, 0.05, x)
        label = 'P(mut) = ' + str(x)
        ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    title = 'Variação da Mutação'
    ax.set(xlabel='Gerações', ylabel='Fitness médio',
           title=title)
    ax.axis(EIXOS)
    ax.legend()
    ax.grid()

    fig.savefig("varmutation.pdf")
    # plt.show()

    # AAAAAAAAAAAAAAAAA
    fig, ax = plt.subplots()

    for x in [2,3,4,5,6,7]:
        generations = 50
        data = load_data('synth1', 50, generations, x, 0.3, 0.6)
        label = 'k = ' + str(x)
        ax.plot(np.arange(0, generations + 1, 1), data[:,2][:], 'o', label=label)

    title = 'Variação da K'
>>>>>>> master
    ax.set(xlabel='Gerações', ylabel='Fitness médio',
           title=title)
    ax.axis(EIXOS)
    ax.legend()
    ax.grid()

<<<<<<< HEAD
    fig.savefig("img/populations.pdf")
    plt.show()
=======
    fig.savefig("varK.pdf")
    # plt.show()
>>>>>>> master


    # merge data
    # 

if __name__ == '__main__':
    main()