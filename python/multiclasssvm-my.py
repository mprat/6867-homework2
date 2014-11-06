import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

def linear_kernel(x, y):
    return np.dot(x, y.T)

def gaussian_kernel_general(x, y, beta):
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
    k =  rbf_kernel(x, y, beta)
    return k.T

def file_to_x_y(filename):
    data = np.loadtxt('../problemset/HW2_handout/data/' + filename)
    y = data[:, -1] # y (class) vectors
    x = data[:, :-1]
    return x, y

def get_svm_ws(x, y, c, kernel_function):
    n = np.shape(x)[0]
    # print "x data = \n", data[:, :-1]
    k = kernel_function(x, x)
#     k = np.dot(x, x.T) # basically the "kernel matrix"
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
    zero_thresh = 1e-6
    y = np.reshape(y, (-1, 1))
    x_support = np.where(np.abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0, x, 0)
    print "Number of support vectors = ", np.sum(np.abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0)
    k_alpha = kernel_function(x_support, x)
    w0 = np.array([(np.sum(np.where(np.abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0, y, 0)) - np.sum(k_alpha * y * alphas)) / np.sum(np.where(abs(alphas - (c + zero_thresh)/2.0) < (c - zero_thresh)/2.0))])
    # print "alphas = ", alphas
    # print "w0 = ", w0
    # print "w = ", w
    return [w0, alphas, x_support]

def to_class_func(w):
    return lambda n: w[0] + w[1] * n[0] + w[2] * n[1]

def predictor(x_train, y_train, alphas, w0, kernel_func, x_support, x_to_predict):
    # print (y_train*alphas).shape
    # print kernel_func(x_train, x_to_predict).shape
    return np.dot((y_train*alphas.T), kernel_func(x_support, x_to_predict).T) + w0

def error_rate(y_pred, y_real):
    return np.sum(y_pred != y_real) / float(len(y_pred))

if __name__ == '__main__':
    name = 'kaggle_train'
    data = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv', delimiter=',', skiprows=1)
    num_pts = len(data)
    cl = 7 # number of classes

    # distribution of classes in the kaggle_train data
    # classes = data[:, -1]
    # print np.bincount(classes.astype(int))

    #split into 3 sets with each type of class 2160 times (they are equal in the original distribution)
    train_indices = []
    validate_indices = []
    test_indices = []
    for i in range(1, cl + 1):
        indices_of_class = np.array(np.where(data[:, -1].astype(int) == i)).flatten()
        total_num_indices = len(indices_of_class)
        num_each = 500 # for all, total_num_indices / 3
        random_indices = np.random.choice(indices_of_class, num_each, replace=False)
        train_indices.extend(random_indices)

        indices_of_class_val = np.delete(indices_of_class, random_indices)
        random_indices = np.random.choice(indices_of_class_val, num_each, replace=False)

        validate_indices.extend(random_indices)

        indices_of_class_test = np.delete(indices_of_class_val, random_indices)
        # random_indices = np.random.choice(indices_of_class, num_each, replace=False)

        test_indices.extend(indices_of_class_test)

    data_train = data[train_indices]
    data_validate = data[validate_indices]
    data_test = data[test_indices]

    x_train = data_train[:, 1:-1]
    y_train = data_train[:, -1]
    x_validate = data_validate[:, 1:-1]
    y_validate = data_validate[:, -1]
    x_test = data_test[:, 1:-1]
    y_test = data_test[:, -1]

    best_c = 0
    min_err = 100

    for c in [100]:
        all_predictors = []

        all_alphas = np.empty((cl, len(x_train)))
        all_w0s = np.empty((cl, ))
        all_support = np.empty((cl, x_train.shape[0], x_train.shape[1]))
        all_y_train_temp = np.empty((cl, len(x_train)))

        for i in range(1, cl + 1):
            y_train_temp = y_train.copy()
            train_indices_pos = np.array(np.where(y_train.astype(int) == i)).flatten()
            train_indices_neg = np.array(np.where(y_train.astype(int) != i)).flatten()
            y_train_temp[train_indices_pos] = 1
            y_train_temp[train_indices_neg] = -1

            # c = 1
            beta = 5
            def gaussian_kernel(x, y):
                return gaussian_kernel_general(x, y, beta)

            kernel_function = gaussian_kernel

            [w0, alphas, x_support] = get_svm_ws(x_train, y_train_temp, c, kernel_function)

            all_alphas[i - 1] = alphas.flatten()
            all_w0s[i - 1] = w0.flatten()
            all_support[i - 1] = x_support
            all_y_train_temp[i - 1] = y_train_temp.flatten()

        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[0], all_alphas[0], all_w0s[0], kernel_function, all_support[0], x_to_predict))
        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[1], all_alphas[1], all_w0s[1], kernel_function, all_support[1], x_to_predict))
        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[2], all_alphas[2], all_w0s[2], kernel_function, all_support[2], x_to_predict))
        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[3], all_alphas[3], all_w0s[3], kernel_function, all_support[3], x_to_predict))
        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[4], all_alphas[4], all_w0s[4], kernel_function, all_support[4], x_to_predict))
        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[5], all_alphas[5], all_w0s[5], kernel_function, all_support[5], x_to_predict))
        all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[6], all_alphas[6], all_w0s[6], kernel_function, all_support[6], x_to_predict))
        
        preds_train = np.empty((cl, len(x_train)))
        preds_val = np.empty((cl, len(x_validate)))

        for i in range(len(all_predictors)):
            preds_train[i] = all_predictors[i](x_train)
            preds_val[i] = all_predictors[i](x_validate)

        preds_train = np.argmax(preds_train, axis = 0) + 1
        preds_val = np.argmax(preds_val, axis = 0) + 1

        train_err = error_rate(preds_train, y_train)
        val_err = error_rate(preds_val, y_validate)
        print "error train = ", train_err
        print "error val = ", val_err
        if val_err < min_err:
            min_err = val_err
            best_c = c

    print "best c = ", best_c

    # re-train on the best C
    c = best_c
    all_predictors = []
    all_alphas = np.empty((cl, len(x_train)))
    all_w0s = np.empty((cl, ))
    all_support = np.empty((cl, x_train.shape[0], x_train.shape[1]))
    all_y_train_temp = np.empty((cl, len(x_train)))

    all_alphas = np.empty((cl, len(x_train)))
    all_w0s = np.empty((cl, ))
    all_support = np.empty((cl, x_train.shape[0], x_train.shape[1]))
    all_y_train_temp = np.empty((cl, len(x_train)))

    for i in range(1, cl + 1):
        y_train_temp = y_train.copy()
        train_indices_pos = np.array(np.where(y_train.astype(int) == i)).flatten()
        train_indices_neg = np.array(np.where(y_train.astype(int) != i)).flatten()
        y_train_temp[train_indices_pos] = 1
        y_train_temp[train_indices_neg] = -1

        # c = 1
        beta = 1
        def gaussian_kernel(x, y):
            return gaussian_kernel_general(x, y, beta)

        kernel_function = gaussian_kernel

        [w0, alphas, x_support] = get_svm_ws(x_train, y_train_temp, c, kernel_function)

        all_alphas[i - 1] = alphas.flatten()
        all_w0s[i - 1] = w0.flatten()
        all_support[i - 1] = x_support
        all_y_train_temp[i - 1] = y_train_temp.flatten()

    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[0], all_alphas[0], all_w0s[0], kernel_function, all_support[0], x_to_predict))
    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[1], all_alphas[1], all_w0s[1], kernel_function, all_support[1], x_to_predict))
    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[2], all_alphas[2], all_w0s[2], kernel_function, all_support[2], x_to_predict))
    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[3], all_alphas[3], all_w0s[3], kernel_function, all_support[3], x_to_predict))
    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[4], all_alphas[4], all_w0s[4], kernel_function, all_support[4], x_to_predict))
    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[5], all_alphas[5], all_w0s[5], kernel_function, all_support[5], x_to_predict))
    all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[6], all_alphas[6], all_w0s[6], kernel_function, all_support[6], x_to_predict))
    

    preds_test = np.empty((cl, len(x_test)))
    x_all = data[:, 1:-1]
    y_all = data[:, -1]
    preds_all = np.empty((cl, len(x_all)))

    for i in range(len(all_predictors)):
        preds_all[i] = all_predictors[i](x_all)
        preds_test[i] = all_predictors[i](x_test)

    preds_all = np.argmax(preds_all, axis=0) + 1
    preds_test = np.argmax(preds_test, axis = 0) + 1

    print "error all = ", error_rate(preds_all, y_all)
    print "error test = ", error_rate(preds_test, y_test)

    

    # for i in range(cl):
        # def predictSVM(x_to_predict):
            # return predictor(x_train, all_y_train_temp[i], all_alphas[i], all_w0s[i], kernel_function, all_support[i], x_to_predict)
        # print "here"
        # all_predictors.append(lambda x_to_predict: predictor(x_train, all_y_train_temp[i], all_alphas[i], all_w0s[i], kernel_function, all_support[i], x_to_predict))


# run_linear_all()

# # def run_gaussian_all():
# for dataset in ['smallOverlap', 'bigOverlap', 'nonSep2', 'ls', 'stdev1', 'stdev2', 'stdev4']:
# 	run1_gaussian(dataset)

# name = 'small_example'
# c = 0.1
# beta = 1
# def gaussian_kernel(x, y):
#     return gaussian_kernel_general(x, y, beta)

# kernel_function = linear_kernel
# print '======Training======'
# # load data from csv files
# train = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')
# # use deep copy here to make cvxopt happy
# Y_train = train[:, -1].copy()
# X_train = train[:, :-1].copy()

# # Carry out training, primal and/or dual
# [w0, alphas, x_support] = get_svm_ws(X_train, Y_train, c, kernel_function)
# # w0 = w[0]
# # alphas = w[1:]
# # Define the predictSVM(x) function, which uses trained parameters
# # predictSVM = to_class_func(w)
# def predictSVM(x_to_predict):
#     return predictor(X_train, Y_train, alphas, w0, kernel_function, x_support, x_to_predict)

# # plot training results
# print "c = ", c
# print "Geometric margin = ", 1.0/np.linalg.norm(np.sum((X_train.T*Y_train)*alphas.flatten(), axis=1))
# print "Error rate train = ", error_rate(X_train, Y_train, X_train, Y_train, w0, alphas, kernel_function, x_support)
# #     plotDecisionBoundary(X_train, Y_train, predictSVM, [-1, 0, 1], title = 'SVM Train')

# print '==Validation=='
# # load data from csv files
# validate = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')
# Y_validate = validate[:, -1].copy()
# X_validate = validate[:, :-1].copy()
# # plot validation results
# #     plotDecisionBoundary(X_validate, Y_validate, predictSVM, [-1, 0, 1], title = 'SVM Validate')
# print "Error rate validate = ", error_rate(X_validate, Y_validate, X_train, Y_train, w0, alphas, kernel_function, x_support)

# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# h = np.max((x_max-x_min)/50., (y_max-y_min)/50.)
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                   np.arange(y_min, y_max, h))
# zz = np.array([predictSVM(x) for x in np.c_[np.ravel(xx), np.ravel(yy)]])
# zz = np.reshape(zz, xx.shape)

# fig = plt.figure(figsize=(8, 3))
# ax1 = fig.add_subplot(121)
# CS = ax1.contour(xx, yy, zz, [0], colors = 'green', linestyles = 'solid', linewidths = 2)
# #     plt.clabel(CS, fontsize=9, inline=1)
# #     Plot the training points
# ax1.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = plt.cm.cool)
# ax1.set_title('Train')
# ax1.axis('tight')
# ax1.set_xticks([])
# ax1.set_yticks([])

# ax2 = fig.add_subplot(122)
# CS = ax2.contour(xx, yy, zz, [0], colors = 'green', linestyles = 'solid', linewidths = 2)
# #     plt.clabel(CS, fontsize=9, inline=1)
# #     Plot the training points
# ax2.scatter(X_validate[:, 0], X_validate[:, 1], c=(1.-Y_validate), s=50, cmap = plt.cm.cool)
# ax2.set_title('Validate')
# ax2.axis('tight')
# ax2.set_xticks([])
# ax2.set_yticks([])

# plt.subplots_adjust(wspace=0)
# plt.show()