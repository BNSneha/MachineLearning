import pandas
import numpy
import matplotlib.pyplot as plt
import math

# reading data from csv to data frames
X_df = pandas.read_csv("data/boosting/X_train.csv", header=None)
y_df = pandas.read_csv("data/boosting/y_train.csv", header=None)
x_test_df = pandas.read_csv("data/boosting/X_test.csv", header=None)
y_test_df = pandas.read_csv("data/boosting/y_test.csv", header=None)

# converting data frames to numpy arrays
X = X_df.as_matrix()
y = y_df.as_matrix()
x_test = x_test_df.as_matrix()
y_test = y_test_df.as_matrix()

# preprocessing
one_matrix = numpy.ones((len(X), 1))
X = numpy.concatenate((X, one_matrix), axis=1)
one_matrix = numpy.ones((len(x_test), 1))
x_test = numpy.concatenate((x_test, one_matrix), axis=1)

# Least squares Classifier
def LSClassifier(X_Bt, y_Bt):
    temp1 = numpy.linalg.inv(numpy.dot(X_Bt.transpose(), X_Bt))
    temp2 = numpy.dot(X_Bt.transpose(), y_Bt)
    w = numpy.dot(temp1, temp2)
    return w


def calF_tandError_t(X, w, y, wt):
    temp3 = numpy.dot(X, w)
    f_t = numpy.sign(temp3)
    indices, dummy = numpy.where(f_t != y)
    error_t = numpy.sum(wt[indices])
    return f_t, error_t


w = []
alpha_t = []
error_t = []
wts = []
randomRowsArray = []

# AdaBoost
# initial weight vector
wt = numpy.empty(len(X))
wt.fill(1 / len(X))
wt = wt.reshape(len(X), 1)
for t in range(1500):
    wts.append(wt)
    randomRows = numpy.random.choice(X.shape[0], p=wt[:, 0], size=len(
        X), replace=True)  # training sample B_t
    randomRowsArray.append(randomRows)
    currW = LSClassifier(X[randomRows], y[randomRows])
    f_t, currErr = calF_tandError_t(X, currW, y, wt)
    if(currErr > 0.5):
        currW = -currW
        f_t, currErr = calF_tandError_t(X, currW, y, wt)
    w.append(currW)
    error_t.append(currErr)
    currAlpha = numpy.log((1 - currErr) / currErr) / 2
    alpha_t.append(currAlpha)
    exponent = numpy.exp(-currAlpha * (numpy.multiply(y, f_t)))
    wt = numpy.multiply(wt, exponent)
    wtSum = numpy.sum(wt)
    wt = wt / wtSum

# boosted_ft for training
boosted_ft_train_error = []
for i in range(1500):
    summation = 0
    for j in range(i + 1):
        summation += alpha_t[j] * numpy.sign(numpy.dot(X, w[j]))
    boosted_ft_train = numpy.sign(summation)
    error = (len(X) - (boosted_ft_train == y).sum()) / len(X)
    boosted_ft_train_error.append(error)

# boosted_ft for testing
boosted_ft_test_error = []
for i in range(1500):
    summation = 0
    for j in range(i + 1):
        summation += alpha_t[j] * numpy.sign(numpy.dot(x_test, w[j]))
    boosted_ft_test = numpy.sign(summation)
    error = (len(x_test) - (boosted_ft_test == y_test).sum()) / len(x_test)
    boosted_ft_test_error.append(error)

plt.figure()
plt.plot(boosted_ft_train_error, label='train')
plt.plot(boosted_ft_test_error, label='test')
plt.legend()

histArray, dummy = numpy.histogram(randomRowsArray, bins=1036)
plt.figure()
plt.stem(histArray)

upperBound = []
for i in range(1500):
    summation = 0
    for j in range(i + 1):
        summation += pow((0.5 - error_t[j]), 2)
    currIterUpperBound = math.exp(-2 * summation)
    upperBound.append(currIterUpperBound)
plt.figure()
plt.plot(upperBound)