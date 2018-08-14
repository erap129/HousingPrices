import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import numpy
from keras.models import Sequential
from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams
from sklearn.datasets import load_boston
boston = load_boston()

print(boston.keys())
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
print(bos.head())

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

# seed = 7
# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# seed = 7
# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=functional_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized Functional: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# seed = 7
# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=functional_model_skipconnection(), epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized Functional skip connection: %.2f (%.2f) MSE" % (results.mean(), results.std()))

skip_model = functional_model()
skip_model.fit(X_train, Y_train, epochs=1000, verbose=0)
y_pred = skip_model.predict(X_test)
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print('mse for functional model:', mse)
print('weights for final layer of functional model:', skip_model.layers[2].get_weights())

layers = [l for l in skip_model.layers]
for l in layers:
    l.trainable = False
z = merge([layers[0].output, layers[1].output], mode='concat')
prediction = Dense(1)
# to_extend = [numpy.random.normal(0, 0.1, 1) for i in range(13)]
to_extend = [[0] for i in range(13)]
print('to extend:', to_extend)
prediction_weights = numpy.concatenate((skip_model.layers[2].get_weights()[0], to_extend), axis=0)
print('prediction weights:', prediction_weights)
prediction.set_weights(prediction_weights)
prediction_set = prediction(z)
model = Model(inputs=layers[0].input, outputs=prediction_set)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

model.fit(X_train, Y_train, epochs=1000, verbose=0)
y_pred = model.predict(X_test)
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print('mse for skip model:', mse)
print('weights for final layer of skip model:', model.layers[3].get_weights())

assert((model.layers[1].get_weights() == skip_model.layers[1].get_weights()).all())

