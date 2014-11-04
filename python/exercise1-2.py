import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

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
    zz = np.array([to_class_func(w)(x) for x in c_[np.ravel(xx), np.ravel(yy)]])
#     print zz
#     print xx.shape[0] * xx.shape[1]
#     print zz.shape[0] * zz.shape[1]
    zz = np.reshape(zz, xx.shape)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    CS = ax1.contour(xx, yy, zz, [-1, 0, 1], colors = 'green', linestyles = 'solid', linewidths = 2)
#     plt.clabel(CS, fontsize=9, inline=1)
#     Plot the training points
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = plt.cm.cool)
    ax1.set_title('Train')
    ax1.axis('tight')
    
    ax2 = fig.add_subplot(122)
    CS = ax2.contour(xx, yy, zz, [-1, 0, 1], colors = 'green', linestyles = 'solid', linewidths = 2)
#     plt.clabel(CS, fontsize=9, inline=1)
#     Plot the training points
    ax2.scatter(X_validate[:, 0], X_validate[:, 1], c=(1.-Y_validate), s=50, cmap = plt.cm.cool)
    ax2.set_title('Validate')
    ax2.axis('tight')