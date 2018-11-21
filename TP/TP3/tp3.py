import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# lib para normalizar os dados (z-score)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def mlp_model(nlayers, layers, input_shape, num_classes, dropout_rate):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        nlayers: int, number of `Dense` layers in the model.
        layers: list of tuples(int, string actvations function).
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = Sequential()
    model.add(Dense(units=layers[0][0], activation=layers[0][1], input_dim=input_shape))
    model.add(Dropout(rate=dropout_rate))

    for i in range(nlayers-1):
        model.add(Dense(units=layers[i][0], activation=layers[i][1]))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def read_data(datafname, normalize=False):
    """Read the dataset from CSV file.

    It's read entire data, remove attr, normalize, factorize classes, shuffle,
    split into input data, output, input shape 1D, number of classes
    and (an extra) index of class
    
    Arguments:
        datafname {str} -- filepath
        normalize {bool} -- If True then replics 5 times the data of QSO class
    
    Returns:
        X {np.array} -- input data (n, 15)
        Y.T {np.array} -- output data in int (n)
        input_shape {int} -- 15 for default
        num_of_classes {int} -- how many uniques Y have
        index_class {classes} -- a reference index to string of classes
    """

    # Leitura do dataset
    data = pd.read_csv(datafname)
    # retirar atributos que não ajudam em nada
    df = data.drop(['objid', 'rerun'], axis=1)

    # Normaliza os dados do QSO para o problema específico
    if normalize:
        qso = df.loc[df['class'] == 'QSO']
        for _ in range(5):
            df = pd.concat([df, qso])

    # quantidade de classes
    num_of_classes = len(df['class'].unique())

    # dimensão de entrada
    # coluna class é para validação
    input_shape = df.drop('class', axis=1).shape[1:]

    d=pd.DataFrame(df)
    
    class_num=pd.DataFrame(LabelEncoder().fit_transform(d['class']), columns=['class'])
    d.drop(['class'], axis=1, inplace=True)
    names=list(d)

    # Normalizar os dados para evitar atributos estrapolados
    scaler = MinMaxScaler()
    d=pd.DataFrame(scaler.fit_transform(d), columns=names)

    d=pd.concat([d, class_num], axis=1)

    df = d

    # Embaralha os dados
    choices = list(range(len(df)))
    np.random.shuffle(choices)
    Y_fac, index_class = pd.factorize(df['class'])
    X, Y = df.drop('class', axis = 1).values[choices], Y_fac[choices]

    return X, Y.T, input_shape, num_of_classes, index_class


def run_job(
                dataset='datasets/sdss.csv',
                lr=0.01,
                layers=[(16, 'relu'), (16, 'relu')],
                batch_size=100,
                decay=0.0,
                momentum=0.9,
                epochs=100,
                normalize_qso=False,
                dropout_rate=0.0
    ):
    """Run a entire job.

    Load the dataset
    Split the dataset in 3 partitions (train, validation, test) as describe in documentation
    Create the Multi Layer Perceptron model
    Training the model with the params
    Return results
    
    Keyword Arguments:
        dataset {str} -- path for the dataset file (default: {'datasets/sdss.csv'})
        lr {float} -- Learning Rate (default: {0.01})
        layers {list of tuples (int, str)} -- units and activation function (default: {[(16, 'relu'), (16, 'relu')]})
        batch_size {int} -- how many samples for each update the weights (default: {100})
        decay {float} -- decay rate for learning rate (default: {0.0})
        momentum {float} -- param to help to choose the best direction for the gradient (default: {0.9})
        epochs {int} -- how many epochs (default: {100})
        normalize_qso {bool} - (Specific) If True then replics 5 times the data of QSO class
        dropout_rate {float} - rate of how many units will be deactivate in the training step
    
    Returns:
        Return the entire history of accuracy of train, validation and the final test and params information
    """

    
    X, Y, input_shape, num_of_classes, index_class = read_data(dataset, normalize_qso)

    assert(len(X) == len(Y))
    n = int(len(X)/3)

    x_train, y_train = X[:n], Y[:n]
    x_validation, y_validation = X[n:2*n], Y[n:2*n]
    x_test, y_test = X[2*n:], Y[2*n:]

    model = mlp_model(len(layers), layers, X.shape[1], num_of_classes, dropout_rate)

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                    optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True), metrics=['acc'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs, batch_size=batch_size)

    loss_and_metrics = model.evaluate(x_test, y_test)

    print('Test accuracy:', np.round(loss_and_metrics[1], 4), '%')

    params = {
        'lr': lr,
        'layers': layers,
        'batch_size': batch_size,
        'decay': decay,
        'epochs': epochs,
        'dropout': dropout_rate
    }

    return history.history['acc'], history.history['val_acc'], loss_and_metrics, params

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.mkdir('output')
    # dataset = 'datasets/sdss.csv'
    dataset = 'data/Skyserver_SQL2_27_2018 6_51_39 PM.csv'
    train_result, val_result, final_result, params = run_job(dataset, lr=0.01, layers=[(16, 'relu'), (16, 'relu')], decay=0.00, epochs=1000, dropout_rate=0.5)
    final_result = final_result[1]

    print('Using params:')
    out_filename = ''
    for k, v in params.items():
        out_filename += str(k) + str(v) + '_'
        print('\t{}: {}'.format(k,v))
    
    out_filename = out_filename[:-1]
        
    axes = plt.gca()
    axes.set_ylim([0, 1])

    axes.set_title('Results')

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(list(range(1, len(train_result) + 1)), train_result)
    plt.plot(list(range(1, len(val_result) + 1)), val_result)
    plt.plot(len(val_result) + 1, final_result, marker='s')
    axes.legend(['Train', 'Validation', 'Test = ' + str(np.round(final_result, 3))])

    figure_name = 'output/' + out_filename
    print('Saving figure ' + figure_name)
    plt.savefig(figure_name + '.pdf')
    plt.show() # comente esta linha se seu S.O. não tiver o python-tk
    plt.clf()
