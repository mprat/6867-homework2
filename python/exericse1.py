import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

def file_to_x_y(filename):
    data = np.loadtxt('../problemset/HW2_handout/data/' + filename)
    y = data[:, -1] # y (class) vectors
    x = data[:, :-1]
    return x, y

def get_svm_ws(x, y, c):
    n = np.shape(x)[0]
    # print "x data = \n", data[:, :-1]
    k = np.dot(x, x.T) # basically the "kernel matrix"
    # print "kernel = \n", k
    P = matrix(k * np.multiply.outer(y, y))
    print "P = \n", P
    q = matrix(-1*np.ones(n))
    print "q = \n", q
    G = matrix(np.concatenate((np.eye(n), -1*np.eye(n))))
    print "G = \n", G
    h = matrix(np.concatenate((np.tile(c, n), np.tile(0, n))), tc='d')
    print "h = \n", h
    A = matrix(y, (1, n))
    print "A = \n", A
    b = matrix(0.0)
    print "b = \n", b
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

def w_to_plotfunc(w):
    return lambda x: -1*(w[0] + w[1]*x)/w[2]

def to_class_func(w):
    return lambda n: w[0] + w[1] * n[0] + w[2] * n[1]

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot
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

def error_rate(x, y, w):
    return sum(np.array([np.sign(to_class_func(w)(n)) for n in x]) != y) / float(len(x))

def exercise1():
	x, y = file_to_x_y('small_example.csv')
	w = get_svm_ws(x, y, 1)
	plotDecisionBoundary(x, y, to_class_func(w), 1, '4-Point Small Example')
	print "Error = ", error_rate(x, y, w)