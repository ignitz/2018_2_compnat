import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from tp3 import run_job

dataset = 'datasets/sdss.csv'

def batch_run(dataset):
    
    # possible params
    LEARN_RATE = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    ACT_FUNC = ['tanh', 'sigmoid', 'relu']
    NLAYERS = [1, 2, 3, 4]
    UNITS = [4, 4, 4, 16, 16, 16, 16, 16, 32, 32, 32, 64, 64, 128]
    BATCHES = [1, 10, 20, 30, 100, 200, 300]
    DECAY = [0.0, 0.1, 0.2]

    params = list(itertools.product(LEARN_RATE, NLAYERS, BATCHES, DECAY))
    # np.random.shuffle(params)
    print(len(params))
    return
    for lr, nl, bt, dc in params:
        train_results = np.zeros(100)
        val_results = np.zeros(100)
        final_results = 0.0

        layers = []
        for i in range(nl):
            unit = np.random.choice(UNITS, 1)[0]
            act = np.random.choice(ACT_FUNC, 1)[0]
            layers.append((unit, act))

        n = 10
        for _ in range(n):
            result1, result2, final_val, params = run_job(dataset, lr, layers, bt, dc)
            train_results += np.array(result1)
            val_results += np.array(result2)
            final_results += final_val[1]

        train_results /= 10
        val_results /= 10
        final_results /= 10
        save_figure_auto(train_results, val_results, final_results, params)


def save_figure(title, train_result, val_result, final_result):
    
    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([0, 1])

    axes.set_title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(list(range(1, len(train_result) + 1)), train_result)
    plt.plot(list(range(1, len(val_result) + 1)), val_result)
    plt.plot(len(val_result) + 1, final_result, marker='s')
    axes.legend(['Train', 'Validation', 'Test = ' + str(np.round(final_result, 3))])

    figure_name = 'output/' + title
    print('Saving figure ' + figure_name)
    plt.savefig(figure_name + '.pdf')
    plt.show()
    plt.clf()


def save_figure_auto(train_result, val_result, final_result, params):
    lr = params['lr']
    layers = params['layers']
    batch_size = params['batch_size']
    decay = params['decay']

    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([0, 1])

    axes.set_title('LR: ' + str(lr) + ' Layers: ' + str(layers) + '\n' + 'BZ: ' + str(batch_size) + ' DC: ' + str(decay))

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(list(range(1, len(train_result) + 1)), train_result)
    plt.plot(list(range(1, len(val_result) + 1)), val_result)
    plt.plot(len(val_result) + 1, final_result, marker='s')
    axes.legend(['Train', 'Validation', 'Test = ' + str(np.round(final_result, 3))])

    figure_name = 'output/lr' + str(lr) + '_'
    figure_name += 'layers-'
    for units, act in layers:
        figure_name += str(units) + act[0] + '-'

    figure_name += 'batchsize' + str(batch_size) + '_'
    figure_name += 'decay' + str(decay)
    print('Saving figure ' + figure_name)
    plt.savefig(figure_name + '.pdf')
    plt.clf()
    # plt.show()


def save_figure_compare(title, tag1, tag2,
                        train_result1, val_result1, final_result1,
                        train_result2, val_result2, final_result2):

    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([0, 1])

    print(title)
    axes.set_title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(list(range(1, len(train_result1) + 1)), train_result1)
    plt.plot(list(range(1, len(val_result1) + 1)), val_result1)
    plt.plot(len(val_result1) + 1, final_result1, marker='s')

    plt.plot(list(range(1, len(train_result2) + 1)), train_result2)
    plt.plot(list(range(1, len(val_result2) + 1)), val_result2)
    plt.plot(len(val_result2) + 1, final_result2, marker='s')
    axes.legend(['Train ' + tag1,
                 'Validation ' + tag1,
                 'Test ' + tag1 + ' = ' + str(np.round(final_result1, 3)),
                 'Train ' + tag2,
                 'Validation ' + tag2,
                 'Test ' + tag2 + ' = ' + str(np.round(final_result2, 3))])

    figure_name = 'output/' + title
    print('Saving figure ' + figure_name)
    plt.savefig(figure_name + '.pdf')
    # plt.show()
    plt.clf()

def save_figure_compare_many(title, tags,
                        train_results, val_results, final_results):
    plt.figure(figsize=(10,7))
    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([0, 1])

    axes.set_title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')

    max_n = max(len(train_results), len(val_results), len(final_results))
    flag = {
        'train': True,
        'val': True,
        'final': True
    }

    if len(train_results) < max_n:
        train_results = [None] * max_n
        flag['train'] = False
    
    if len(val_results) < max_n:
        val_results = [None] * max_n
        flag['val'] = False

    if len(final_results) < max_n:
        final_results = [None] * max_n
        flag['final'] = False

    for train_result, val_result, final_result in zip(train_results, val_results, final_results):
        if flag['train']: plt.plot(list(range(1, len(train_result) + 1)), train_result)
        if flag['val']: plt.plot(list(range(1, len(val_result) + 1)), val_result)
        if flag['final']: plt.plot(len(val_result) + 1, final_result, marker='s')
    
    legend = []

    if flag['final']:
        for tag, final_result in zip(tags, final_results):
            if flag['train']: legend.append('Train ' + str(tag))
            if flag['val']: legend.append('Validation ' + str(tag))
            legend.append('Test ' + str(tag) + ' = ' + str(np.round(final_result, 3)))
    else:
        for tag in tags:
            if flag['train']: legend.append('Train ' + str(tag))
            if flag['val']: legend.append('Validation ' + str(tag))

    axes.legend(legend)

    figure_name = 'output/' + title
    print('Saving figure ' + figure_name)
    plt.savefig(figure_name + '.pdf')
    # plt.show()
    plt.clf()

def diff_erro_vs_tempo(parameter_list):
    pass

# 1. O que acontece quando se aumenta o numero de neur^onios da camada escondida da
# rede? Isso afeta o numero de epocas necessarias para converg^encia?
def questao01():
    train_results = []
    val_results = []
    final_results = []

    def do_job(layers):
        train_aux = np.zeros(100)
        val_aux = np.zeros(100)
        final_aux = 0.0
        
        for _ in range(10):
            result1, result2, final_val, params = run_job(dataset, layers=layers)
            train_aux += np.array(result1)
            val_aux += np.array(result2)
            final_aux += final_val[1]
        train_aux /= 10
        val_aux /= 10
        final_aux /= 10

        train_results.append(train_aux)
        val_results.append(val_aux)
        final_results.append(final_aux)
    
    do_job([(4, 'relu')])
    do_job([(16, 'relu')])
    do_job([(32, 'relu')])
    do_job([(64, 'relu')])
    # do_job([(4, 'relu'), (4, 'relu'), (4, 'relu')])
    # do_job([(16, 'relu'), (16, 'relu'), (16, 'relu')])
    # do_job([(32, 'relu'), (32, 'relu'), (32, 'relu')])
    # do_job([(64, 'relu'), (64, 'relu'), (64, 'relu')])

    
    save_figure_compare_many('Comparação de número de neurônios na camada escondida',
                        ['4 neurônios', '16 neurônios', '32 neurônios', '64 neurônios'],
                        train_results, val_results, final_results)

# 2. O que acontece quando se aumenta o numero de camadas escondidas? O ganho no erro
# e grande o suciente para justicar a adic~ao de uma nova camada?
def questao02():
    train_results1 = np.zeros(100)
    val_results1 = np.zeros(100)
    final_results1 = 0.0
    
    for _ in range(10):
        result1, result2, final_val, params = run_job(dataset, layers=[(16, 'relu')])
        train_results1 += np.array(result1)
        val_results1 += np.array(result2)
        final_results1 += final_val[1]
    train_results1 /= 10
    val_results1 /= 10
    final_results1 /= 10

    train_results2 = np.zeros(100)
    val_results2 = np.zeros(100)
    final_results2 = 0.0
    
    for _ in range(10):
        result1, result2, final_val, params = run_job(dataset, layers=[(16, 'relu'), (16, 'relu'), (16, 'relu'), (16, 'relu')])
        train_results2 += np.array(result1)
        val_results2 += np.array(result2)
        final_results2 += final_val[1]
    train_results2 /= 10
    val_results2 /= 10
    final_results2 /= 10
    
    save_figure_compare('Comparação do número de camadas escondidas',
                        '1 camada',
                        '4 camadas',
                        train_results1, val_results1, final_results1,
                        train_results2, val_results2, final_results2)


# 3. Qual o impacto da variac~ao da taxa de aprendizagem na converg^encia da rede? O
# que acontece se esse par^ametro for ajustado automaticamente ao longo das diferentes
# epocas?

def questao03():

    train_results = []
    val_results = []
    final_results = []
    tags = []

    n = 10
    epochs = 1000

    for decay in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        train_aux = np.zeros(epochs)
        val_aux = np.zeros(epochs)
        final_aux = 0.0
        
        tags.append('decay rate = ' + str(decay))
        for _ in range(n):
            result1, result2, final_val, params = run_job(dataset, lr=1.0, layers=[(16, 'relu'), (16, 'relu')], decay=decay, epochs=epochs)
            train_aux += np.array(result1)
            val_aux += np.array(result2)
            final_aux += final_val[1]

        train_aux /= n
        val_aux /= n
        final_aux /= n

        train_results.append(train_aux)
        val_results.append(val_aux)
        final_results.append(final_aux)

    save_figure_compare_many('Comparação de decaimento da taxa de aprendizagem',
                        tags,
                        train_results, [], [])
    
    # Agora testar vários valores
    train_results = []
    val_results = []
    final_results = []
    tags = []

    LEARN_RATE = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    # LEARN_RATE = [0.00001, 0.0001, 0.001, 3.0]

    for lr in LEARN_RATE:
        train_aux = np.zeros(epochs)
        val_aux = np.zeros(epochs)
        final_aux = 0.0

        for _ in range(n):
            result1, result2, final_val, params = run_job(dataset, layers=[(16, 'relu'), (16, 'relu')], lr=lr, epochs=epochs)
            train_aux += np.array(result1)
            val_aux += np.array(result2)
            final_aux += final_val[1]
        train_aux /= n
        val_aux /= n
        final_aux /= n
        
        train_results.append(train_aux)
        val_results.append(val_aux)
        final_results.append(final_aux)

    print(len(train_results), len(val_results), len(final_results))
    save_figure_compare_many('Variação da taxa de aprendizado (Learning Rate)', LEARN_RATE,
                            train_results, [], [])
    

# 4. Compare o treinamento da rede com gradient descent estocastico com o mini-batch.
# A diferenca em erro de treinamento versus tempo computacional indica que qual deles
# deve ser utilizado?

def questao04():
    train_results1 = np.zeros(100)
    val_results1 = np.zeros(100)
    final_results1 = 0.0
    
    for _ in range(10):
        result1, result2, final_val, params = run_job(dataset, layers=[(16, 'relu'), (16, 'relu')], batch_size=1)
        train_results1 += np.array(result1)
        val_results1 += np.array(result2)
        final_results1 += final_val[1]
    train_results1 /= 10
    val_results1 /= 10
    final_results1 /= 10

    train_results2 = np.zeros(100)
    val_results2 = np.zeros(100)
    final_results2 = 0.0
    
    for _ in range(10):
        result1, result2, final_val, params = run_job(dataset, layers=[(16, 'relu'), (16, 'relu')], batch_size=200)
        train_results2 += np.array(result1)
        val_results2 += np.array(result2)
        final_results2 += final_val[1]
    train_results2 /= 10
    val_results2 /= 10
    final_results2 /= 10

    
    save_figure_compare('Comparação de SGD com mini-batch',
                        '(SGD)',
                        '(mini-batch 200)',
                        train_results1, val_results1, final_results1,
                        train_results2, val_results2, final_results2)

# 5. Qual a diferenca do erro encontrado pela rede no conjunto de treinamento ou validac~ao
# em relac~ao ao erro encontrado no teste? Existe overtting? Como ele pode ser evitado?
def questao05():
    train_results = []
    val_results = []
    final_results = []
    tags = []

    n = 10
    epochs = 100

    for drop in [0.5]:
        train_aux = np.zeros(epochs)
        val_aux = np.zeros(epochs)
        final_aux = 0.0
        
        tags.append('Dropout rate = ' + str(drop))
        for _ in range(n):
            result1, result2, final_val, params = run_job(dataset, lr=0.01, layers=[(16, 'relu'), (16, 'relu')], epochs=epochs, dropout_rate=drop)
            train_aux += np.array(result1)
            val_aux += np.array(result2)
            final_aux += final_val[1]

        train_aux /= n
        val_aux /= n
        final_aux /= n

        train_results.append(train_aux)
        val_results.append(val_aux)
        final_results.append(final_aux)

    save_figure_compare_many('Comparação utilizando Dropout Regularization',
                        tags,
                        train_results, val_results, final_results)

# 6. A base com que voc^e trabalhou e um pouco desbalanceada. Voc^e pode tentar contornar
# esse problema usando a tecnica de oversampling, ou seja, fazendo copias dos exemplos
# das classes minoritarias para balancear melhor a base. Por exemplo, a classe QSO tem
# 412 exemplos, e a classe GALAXY 2501. De forma simples, voc^e poderia fazer 6 copias
# de cada exemplo da classe QSO, aumentando o numero de exemplos dessa classe para
# 2472, e utilizando todos eles no treinamento da rede. Fazendo um oversampling das
# classes minoritarias e retreinando a rede com os melhore par^ametros encontrados, o
# erro diminuiu? Por qu^e?

def questao06():
    train_results = []
    val_results = []
    final_results = []

    def do_job(normalize_qso):
        train_aux = np.zeros(100)
        val_aux = np.zeros(100)
        final_aux = 0.0
        
        for _ in range(10):
            result1, result2, final_val, params = run_job(dataset, layers=[(16, 'relu'), (16, 'relu')], normalize_qso=normalize_qso)
            train_aux += np.array(result1)
            val_aux += np.array(result2)
            final_aux += final_val[1]
        train_aux /= 10
        val_aux /= 10
        final_aux /= 10

        train_results.append(train_aux)
        val_results.append(val_aux)
        final_results.append(final_aux)
    
    do_job(False)
    do_job(True)

    save_figure_compare_many('Comparação com a quantidade de QSO normalizado nos dados',
                        ['Sem QSO normalizado', 'Com QSO normalizado'],
                        train_results, val_results, final_results)



def main():
    if not os.path.exists('output'):
        os.mkdir('output')
    # batch_run('datasets/sdss.csv')
    # questao01()
    # questao02()
    # questao03()
    # questao04()
    questao05()
    # questao06()


if __name__ == "__main__":
    main()
