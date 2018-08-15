import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, merge, Concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
import seaborn as sns
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
sns.set_style("whitegrid")
sns.set_context("poster")

from sklearn.datasets import load_boston
boston = load_boston()

print(boston.keys())
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print('mse for linear regression:', mse)


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def functional_model():
    # create model
    inputs = Input(shape=(13,))
    x = Dense(20, activation='relu')(inputs)
    prediction = Dense(1)(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='model_plot_functional.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model

def functional_model_only_ten():
    # create model
    inputs = Input(shape=(13,))
    x = Dense(10, activation='relu')(inputs)
    prediction = Dense(1)(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='model_plot_functional.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model

def functional_model_split():
    # create model
    inputs = Input(shape=(13,))
    hidden_one = Dense(10, activation='relu')(inputs)
    hidden_two = Dense(10, activation='relu')(inputs)
    concat = Concatenate()([hidden_one, hidden_two])
    prediction = Dense(1)(concat)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='model_plot_functional.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model

def functional_model_skipconnection():
    # create model
    inputs = Input(shape=(13,))
    x = Dense(20, activation='relu')(inputs)
    z = merge([x, inputs], mode='concat')
    prediction = Dense(1)(z)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='model_plot_skipconnection.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model


results = pd.DataFrame(columns=['linReg', 'hidden20', 'hidden20_split', 'hidden10', 'hidden10+10'])
for i in range(50):
    row = np.array([])

    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    Y_pred = lm.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    row = np.append(row, mse)
    print('mse for linear regression:', mse)

    func_model = functional_model()
    func_model.fit(X_train, Y_train, epochs=1000, verbose=0)
    Y_pred = func_model.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    row = np.append(row, mse)
    print('mse for functional model:', mse)
    # print('weights for final layer of functional model:', func_model.layers[2].get_weights())

    split_model = functional_model_split()
    split_model.fit(X_train, Y_train, epochs=1000, verbose=0)
    Y_pred = split_model.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    row = np.append(row, mse)
    print('mse for split model:', mse)

    func_model_only_ten = functional_model_only_ten()
    func_model_only_ten.fit(X_train, Y_train, epochs=1000, verbose=0)
    Y_pred = func_model_only_ten.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    row = np.append(row, mse)
    print('mse for functional model only ten:', mse)

    layers = [l for l in func_model_only_ten.layers]
    for l in layers:
        l.trainable = False
    added_hidden = Dense(10, activation='relu')(layers[0].output)
    concat = Concatenate()([layers[1].output, added_hidden])
    prediction = Dense(1)(concat)
    added_model = Model(inputs=layers[0].output, outputs=prediction)
    added_model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(added_model, to_file='model_plot_added_hidden.png', show_shapes=True, show_layer_names=True)
    print(added_model.summary())
    added_model.fit(X_train, Y_train, epochs=1000, verbose=0)
    print(added_model.layers[1].get_weights()[0] ==
           func_model_only_ten.layers[1].get_weights()[0])
    Y_pred = added_model.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    row = np.append(row, mse)
    print('mse for added model:', mse)
    results.loc[i] = row

results.to_csv('out.csv')

# z = merge([layers[0].output, layers[1].output], mode='concat')
# # to_extend = [numpy.random.normal(0, 0.1, 1) for i in range(13)]
# to_extend = [[0] for i in range(13)]
# print('to extend:', to_extend)
# prediction_weights = numpy.concatenate((skip_model.layers[2].get_weights()[0], to_extend), axis=0)
# prediction_bias = numpy.array([0])
# print('prediction weights:', prediction_weights)
# prediction_set = MyLayer(1, weights=numpy.array([prediction_weights, prediction_bias]))(z)
# model = Model(inputs=layers[0].input, outputs=prediction_set)
# model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())
# print('final layer weights:', model.layers[3].get_weights())
# # model.layers[3].set_weights(prediction_weights)
#
# model.fit(X_train, Y_train, epochs=1000, verbose=0)
# y_pred = model.predict(X_test)
# mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
# print('mse for skip model:', mse)
# print('weights for final layer of skip model:', model.layers[3].get_weights())
#
# assert(model.layers[1].get_weights().all() == skip_model.layers[1].get_weights().all())

