import scipy
from scipy import optimize
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x) for x in np.c_[np.ravel(xx), np.ravel(yy)]])
#     print zz
#     print xx.shape[0] * xx.shape[1]
#     print zz.shape[0] * zz.shape[1]
    zz = np.reshape(zz, xx.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
#     plt.clabel(CS, fontsize=9, inline=1)
#     Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = plt.cm.cool)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')
    plt.show()

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def gaussian_kernel_general(x, y, beta):
    if len(x.shape) > 1:
        return np.exp(-1*beta*np.linalg.norm(x - y.T, axis=1))
    else:
        return np.exp(-1*beta*np.linalg.norm(x - y))

def nll(x, y, alphas, w0, l, kernel_func):
    # return np.sum(np.log(1 + np.exp(-1.0 * y * (np.dot(kernel_func(x, x.T), alphas) + w0)))) + l * np.linalg.norm(alphas, 1)
    return np.sum(scipy.misc.logsumexp(np.concatenate((np.array([0]), -1.0*y*(np.dot(kernel_func(x, x.T), alphas) + w0))))) + l * np.linalg.norm(alphas, 1)
    # return np.sum(np.log(1 + np.exp(-1.0 * y * (np.dot(kernel_func(x, x.T), alphas) + w0)))) + l * np.linalg.norm(alphas, 1)

def predictor(alphas, w0, kernel_func, xt, yt, x_to_predict):
    return np.sign(w0 + np.dot(kernel_func(x_to_predict, xt.T), alphas))

def error(truth, prediction):
    return np.sum(truth != prediction) / float(len(truth))

def sparsity(alphas):
    return np.sum(np.abs(alphas) > 1e-5) / float(len(alphas))

if __name__ == '__main__':
    name = 'stdev4'
    train = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_train.csv')
    test = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_test.csv')

    # name = 'small_example'
    # train = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')
    # test = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')

    y_train = train[:, -1].copy()
    x_train = train[:, :-1].copy()
    y_test = test[:, -1].copy()
    x_test = test[:, :-1].copy()

    print "num pts = ", len(x_train)

    # kernel_func = linear_kernel



    for l in [0, 1e-3, 0.01, 0.1, 1, 10]:
        for beta in [0.1, 0.5, 1.0, 1.5, 2, 2.5]:
            def gaussian_kernel(x, y):
                return gaussian_kernel_general(x, y, beta)

            kernel_func = linear_kernel

            def nll_xy(params):
                alphas = params[:-1]
                w0 = params[-1]
                # l = 0.1
                # kernel_func = linear_kernel
                # x = x_train
                # y = y_train
                return nll(x_train, y_train, alphas, w0, l, kernel_func)

            # nll(x_train, y_train, np.empty(len(x_train)), 0, 0, linear_kernel)
            alphas_and_w0 = scipy.optimize.fmin_bfgs(nll_xy, np.zeros((len(x_train) + 1, )), gtol=1e-12)
            alphas = alphas_and_w0[:-1]
            w0 = alphas_and_w0[-1]

            def score(x_to_predict):
                return predictor(alphas,w0, kernel_func, x_train, y_train, x_to_predict)

            y_predict = predictor(alphas, w0, kernel_func, x_train, y_train, x_train)
            y_predict_test = predictor(alphas, w0, kernel_func, x_train, y_train, x_test)

            print "l = ", l
            print "training error = ", error(y_train, y_predict)
            print "test error = ", error(y_test, y_predict_test)
            print "sparsity = ", sparsity(alphas)
            print "beta = ", beta

            plotDecisionBoundary(x_train, y_train, score, [0])