import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

def to_class_func(w):
    return lambda n: w[0] + w[1] * n[0] + w[2] * n[1]

def error_rate(x, y, w):
    return sum(np.array([np.sign(to_class_func(w)(n)) for n in x]) != y) / float(len(x))

def get_svm_ws(x, y, c):
    n = np.shape(x)[0]
    # print "x data = \n", data[:, :-1]
    k = np.dot(x, x.T) # basically the "kernel matrix"
    # print "kernel = \n", k
    P = matrix(k * np.multiply.outer(y, y))
    # print "P = \n", P
    q = matrix(-1*np.ones(n))
    # print "q = \n", q
    G = matrix(np.concatenate((np.eye(n), -1*np.eye(n))))
    # print "G = \n", G
    h = matrix(np.concatenate((np.tile(c, n), np.tile(0, n))), tc='d')
    A = matrix(y, (1, n))
    b = matrix(0.0)
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x'])
#     print alphas
    w = np.sum((x.T*y)*alphas.flatten(), axis=1)
    zero_thresh = 1e-8
    y = np.reshape(y, (-1, 1))
    x_support = np.where(np.abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0, x, 0)
    k_alpha = np.dot(x_support, x.T)
    w0 = np.array([(np.sum(np.where(np.abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0, y, 0)) - np.sum(k_alpha * y * alphas)) / np.sum(np.where(np.abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0))])
    # print "alphas = ", alphas
    # print "w0 = ", w0
    # print "w = ", w
    return np.concatenate((w0, w))

def train_validate(name):
# name = 'bigOverlap'
    print '======Training======'
    # load data from csv files
    train = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    Y_train = train[:, -1].copy()
    X_train = train[:, :-1].copy()

    # Carry out training, primal and/or dual
    w = get_svm_ws(X_train, Y_train, 1)
    # Define the predictSVM(x) function, which uses trained parameters
    predictSVM = to_class_func(w)

    # plot training results
    print "Error rate train = ", error_rate(X_train, Y_train, w)
#     plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')

    print '======Validation======'
    # load data from csv files
    validate = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_test.csv')
    Y_validate = validate[:, -1].copy()
    X_validate = validate[:, :-1].copy()
    # plot validation results
#     plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
    print "Error rate validate = ", error_rate(X_validate, Y_validate, w)
    
    
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([to_class_func(w)(x) for x in np.c_[np.ravel(xx), np.ravel(yy)]])
#     print zz
#     print xx.shape[0] * xx.shape[1]
#     print zz.shape[0] * zz.shape[1]
    zz = np.reshape(zz, xx.shape)
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121)
    CS = ax1.contour(xx, yy, zz, [-1, 0, 1], colors = 'green', linestyles = 'solid', linewidths = 2)
#     plt.clabel(CS, fontsize=9, inline=1)
#     Plot the training points
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = plt.cm.cool)
    ax1.set_title('Train')
    ax1.axis('tight')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2 = fig.add_subplot(122)
    CS = ax2.contour(xx, yy, zz, [-1, 0, 1], colors = 'green', linestyles = 'solid', linewidths = 2)
#     plt.clabel(CS, fontsize=9, inline=1)
#     Plot the training points
    ax2.scatter(X_validate[:, 0], X_validate[:, 1], c=(1.-Y_validate), s=50, cmap = plt.cm.cool)
    ax2.set_title('Validate')
    ax2.axis('tight')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()