import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.neural_network import MLPClassifier

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


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = tf.keras.models.Sequential()
    model.add(Dense(units=units, activation='sigmoid'))
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        print('DENSE')
        model.add(Dense(units=units, activation='tanh'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

def read_data():
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    # Leitura do dataset
    data = pd.read_csv('datasets/sdss.csv')
    # retirar atributos que não ajudam em nada
    df = data.drop(['objid', 'rerun'], axis=1)
    # print(df.head())

    # quantidade de classes
    num_of_classes = len(df['class'].unique())
    # print(df['class'].unique(), num_of_classes)

    # dimensão de entrada
    # coluna class é para validação
    input_shape = df.drop('class', axis=1).shape[1:]
    # print('input shape', input_shape)

    d=pd.DataFrame(df)
    
    class_num=pd.DataFrame(LabelEncoder().fit_transform(d['class']), columns=['class'])
    d.drop(['class'], axis=1, inplace=True)
    names=list(d)

    #Data are normalized for better conditioning of the problem
    scaler = MinMaxScaler()
    d=pd.DataFrame(scaler.fit_transform(d), columns=names)

    d=pd.concat([d, class_num], axis=1)

    df = d

    choices = list(range(len(df)))
    np.random.shuffle(choices)
    Y_fac, index_class = pd.factorize(df['class'])
    X, Y = df.drop('class', axis = 1).values[choices], Y_fac[choices]
    
    # train_x, train_y = X[:4000], Y[:4000]
    # test_x, test_y = X[4000:], Y[4000:]

    return X, Y.T, input_shape, num_of_classes, index_class

def try_tf():
    # parametros padrão
    learning_rate=0.1
    epochs=1000
    batch_size=None
    layers=3
    units=4
    dropout_rate=0.2

    X, Y, input_shape, num_of_classes, index_class = read_data()
    train_x, train_y = X[:4000], Y[:4000]
    test_x, test_y = X[4000:], Y[4000:]

    # criação do modelo
    model = mlp_model(layers, units, dropout_rate, input_shape, num_of_classes)

     # Compile model with learning parameters.
    if num_of_classes == 2: # Nunca vai ser
        loss = 'binary_crossentropy'
    else:
        # categorical entropy with integers
        loss = 'categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]
    
    # Train and validate model.
    history = model.fit( train_x, train_y, epochs=epochs, callbacks=None, validation_data=(test_x, test_y), verbose=2, batch_size=batch_size)
    
    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

def try_sklearn():
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.metrics import confusion_matrix, precision_score
    import time

    X, Y, input_shape, num_of_classes, index_class = read_data()
    train_x, train_y = X[:4000], Y[:4000]
    test_x, test_y = X[4000:], Y[4000:]


    nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)

    training_start = time.perf_counter()
    nnc.fit(train_x, train_y)
    training_end = time.perf_counter()

    predict_start = time.perf_counter()
    preds=nnc.predict(test_x)
    predict_end=time.perf_counter()

    acc_nnc = (preds == test_y).sum().astype(float) / len(preds)*100
    print("The first iteration of the Neural Networks gives an accuracy of the %3.2f %%" % (acc_nnc))
    mc=confusion_matrix(test_y, preds)
    mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
    # sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g');
    print(pd.DataFrame(mc_norm))

    nnc_train_t=training_end-training_start;
    nnc_predict_t=predict_end-predict_start;

    scores = cross_val_score(nnc, X, Y, cv=10, scoring = "accuracy")
    score_nnc=scores.mean()
    print("The 10 cross validations of Neural Networks have had an average success rate of %3.2f %%" %(score_nnc*100))
    std_nnc=scores.std()
    print("..and a standar deviation of %8.6f" %(std_nnc))

def keras_example():
    

    X, Y, input_shape, num_of_classes, index_class = read_data()
    x_train, y_train = X[:4000], Y[:4000]
    x_test, y_test = X[4000:], Y[4000:]

    model = Sequential()
    model.add(Dense(units=16, activation='tanh', input_dim=15))
    model.add(Dense(units=16, activation='tanh'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                    optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['acc'])
    
    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=4000)
    
    # for _ in range(100):
    #     model.train_on_batch(x_train, y_train)
    #     result1 = model.evaluate(x_train, y_train)
    #     result2 = model.evaluate(x_test, y_test)
    #     print(result1[1], result2[1])



    # model.train_on_batch(x_batch, y_batch)
    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)

    # classes = model.predict(x_test, batch_size=128)
    # print(len(classes))
    
    # print(history.history['acc'])
    # print(history.history['val_acc'])

    # plt.plot(list(range(1, len(history.history['acc']) + 1)), history.history['acc'])
    # plt.plot(list(range(1, len(history.history['val_acc']) + 1)), history.history['val_acc'])
    # # plt.save_fig('temp.pdf')
    # plt.show()


if __name__ == "__main__":
    # try_tf()
    # try_sklearn()
    keras_example()