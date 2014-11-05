import scipy
from scipy import optimize
from scipy import misc
import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def nll(x, y, alphas, w0, l, kernel_func):
    kernel_result = np.zeros(len(x))
    for i in range(len(x)):
        kernel_result[i] = np.dot(kernel_func(x, x[i]), alphas)
    exponent = -1 * y * (kernel_result + w0)
    exponent = np.insert(exponent, 0, 0)
    return np.sum(scipy.misc.logsumexp(exponent)) + l * np.linalg.norm(alphas, 1)

def predictor(alphas, w0, kernel_func, x_train, x):
    kernel_result = np.empty(len(x))
    for i in range(len(x)):
        kernel_result[i] = np.dot(kernel_func(x_train, x[i]), alphas)
    return np.sign(w0 + kernel_result)

def error(truth, prediction):
#     print len(truth)
#     print len(prediction)
#     print np.sum(truth != prediction)
    return np.sum(truth != prediction) / float(len(truth))

def sparsity(alphas):
    return np.sum(np.abs(alphas_and_w0) > 1e-6) / len(alphas)

if __name__ == '__main__':
    name = 'bigOverlap'
    train = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_train.csv')
    test = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_test.csv')

    y_train = train[:, -1].copy()
    x_train = train[:, :-1].copy()
    y_test = test[:, -1].copy()
    x_test = test[:, :-1].copy()

    print "num pts = ", len(x_train)

    kernel_func = linear_kernel

    for l in [0, 1e-3, 0.01, 0.1, 1, 10]:
        def nll_xy(params):
            alphas = params[:-1]
            w0 = params[-1]
            # l = 0.1
            # kernel_func = linear_kernel
            x = x_train
            y = y_train
            return nll(x, y, alphas, w0, l, kernel_func)

        # nll(x_train, y_train, np.empty(len(x_train)), 0, 0, linear_kernel)
        alphas_and_w0 = scipy.optimize.fmin_bfgs(nll_xy, np.zeros((len(x_train) + 1, )))
        alphas = alphas_and_w0[:-1]
        w0 = alphas_and_w0[-1]

        y_predict = predictor(alphas, w0, kernel_func, x_train, x_train)
        y_predict_test = predictor(alphas, w0, kernel_func, x_train, x_test)

        print "l = ", l
        print "training error = ", error(y_train, y_predict)
        print "test error = ", error(y_test, y_predict_test)
        print "sparsity = ", sparsity(alphas)