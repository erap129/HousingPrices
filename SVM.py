from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

houses = open('cadata', 'r')
X = []

for line in houses.readlines():
    line = line.split(' ')
    for index,word in enumerate(line):
        if index != 0:
            line[index] = word[2:]
        line[index] = line[index][:-6]
        line[index] = float(line[index])
    X.append(line)

X = np.array(X)
y = X[:,0]
X = np.delete(X, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = svm.SVR()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('svm scores:')
print(clf.score(X_test, y_test))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
