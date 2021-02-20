## Yared Taye
## Lokesh Gangaramaney
## Yash Yadati
import sys
import numpy as np

np.seterr(all='ignore')

def minibatches(X, y, batchsize):

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        ids = indices[start_idx:start_idx + batchsize]
        yield X[ids], y[ids]

def format_labels(y):

    n = y.shape[0]
    Train_y = np.zeros((n, 10))
    for i in range(len(y)):
        Train_y[i, y[i]] = 1
    return Train_y

## Pre Activation Scores = Z = X^T * W
def preactivation_score(X, weight):

    Z = np.dot(X , weight)
    return Z

## Softmax
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

## -1/n * y * log(y^)
def cross_entropy(X, y, weight):

    fce = -np.sum(y * np.log(softmax(preactivation_score(X, weight)))) / y.shape[0]
    return fce

def gradient(X, y, weight, alpha):

    n = y.shape[0]
    diff_y = (softmax(preactivation_score(X, weight)) - y) / n
    unregularized = X.T.dot(diff_y)
    return unregularized + (alpha * weight), np.mean(diff_y, axis=0)

def SGD( X_tr, ytr, StepSize=0.001, alpha=0.01, batchsize=50, epochs=20):

    n, m = X_tr.shape
    n, p = ytr.shape
    init = np.append(np.random.randn(m,p), np.random.randn(1,p), axis=0)
    weight = init[:-1]
    bias = init[-1]

    for e in range(0, epochs):
        for batch in minibatches(X_tr, ytr, batchsize):
            X_tr_batch, ytr_batch = batch
            dw_dx, db_dx = gradient(X_tr_batch, ytr_batch, weight, alpha)
            weight = weight - (StepSize * dw_dx)
            bias = bias - (StepSize * db_dx)
            init = np.vstack((weight, bias))

    return init[:-1]

def predictions(Train_X, Test_X, Train_y, Test_y, aug_w, StepSize, batches, ep, Wd):

    predict = softmax(preactivation_score(Test_X, aug_w))
    train_cost = cross_entropy(Train_X, Train_y, aug_w)
    test_cost = cross_entropy(Test_X, Test_y, aug_w)
    result = np.argmax(predict, axis=1) - np.argmax(Test_y, axis=1)
    acc = len(result[result == 0]) / len(Test_y)

    if acc > 0.80 or test_cost < 0.6:
        print("=" * 50)
        print("                               ACCURACY =", acc)
        print("=" * 50)
        print("*" * 50)
        print('Current Learning Rate: ', StepSize)
        print('Batches: ', batches)
        print('Epochs', ep)
        print('L2-Reg', Wd)
        print('FCE for Training Set: ', train_cost)
        print('FCE for Testing Set: ', test_cost)
        print("*" * 50)

def grid_search():
    ### Grid Search
    LearningRates = [0.0001, 0.001, 0.005, 0.01, 0.1]
    BatchSizes = [10, 50, 100, 200, 500, 1000]
    WeightDecays = [0.01, 0.02, 0.05, 0.1, 0.5, 2]
    num_epochs = [50, 100, 200, 400, 1000]
    for ep in num_epochs:
        for Wd in WeightDecays:
            for batches in BatchSizes:
                for lr in LearningRates:
                    weight = (SGD(X_tr, ytr, alpha=Wd, StepSize=lr, batchsize=batches, epochs=ep))
                    predictions(X_val, X_te, yval, yte, weight, lr, batches, ep, Wd)
        print("Epoch set is done")

def summary():
    ## USED HYPERPARAMETERS
    alpha = 0.02
    epislon = 0.005
    num_batches  = 10
    num_epochs = 50
    weight = (SGD(X_tr, ytr, alpha=alpha, StepSize=epislon, batchsize=num_batches, epochs=num_epochs))
    predictions(X_tr, X_te, ytr, yte, weight, ep=num_epochs, batches=num_batches, StepSize=epislon, Wd=alpha)


##Load Data
X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28))
ytr = np.load("fashion_mnist_train_labels.npy")
X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28 * 28))
yte = np.load("fashion_mnist_test_labels.npy")


## Split Dataset into Validation for Grid Search and Train 80:20
X_tr, X_val = X_tr[:48000, :], X_tr[48000:, :]
ytr, yval = ytr[:48000], ytr[48000:]

## Format Labels
ytr = format_labels(ytr)
yte = format_labels(yte)
yval = format_labels(yval)


## Normalize
X_tr *= 1/255
X_val *= (1/255)
X_te *= 1/255

## Run either summary() for best output or grid_search()
#summary()
grid_search()

