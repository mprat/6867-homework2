import scipy
from scipy import optimize
import numpy as np
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy import misc

kaggle = True

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def gaussian_kernel_general(x, y, beta):
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
    k =  rbf_kernel(x, y, beta)
    return k.T

# c is number of classes
def nll_multiclass(x, y, alphas, w0, c, l, kernel_func):
	return np.sum(scipy.misc.logsumexp(np.concatenate((np.zeros((len(x), c)), -1.0*y*(np.dot(kernel_func(x, x), alphas) + w0))))) + l * np.linalg.norm(alphas, 1)
	# return np.sum(np.log(1 + np.exp(-1.0 * y * (np.dot(kernel_func(x, x), alphas) + w0)))) + l * np.linalg.norm(alphas, 1)
	# first_part = np.empty(len(x) * c)
	# index = 0
	# for i in range(len(x)):
	# 	y_i = y[i]
	# 	for c_index in range(c):
	# 		y_ic = y_i[c_index]
	# 		alpha_c = alphas[:, c_index]
	# 		w0c = w0[c_index]
	# 		s = 0
	# 		for j in range(len(x)):
	# 			alpha_cj = alpha_c[j]
	# 			s += alpha_cj * kernel_func(x[j], x[i]) + w0c
	# 		first_part[index] = y_ic * s
	# 		index += 1
	
	# print np.sum(first_part)

	# first_part2 = np.empty(len(x) * c)
	# index = 0
	# for i in range(len(x)):
	# 	y_i = y[i]
	# 	for c_index in range(c):
	# 		y_ic = y_i[c_index]
	# 		alpha_c = alphas[:, c_index]
	# 		w0c = w0[c_index]
	# 		first_part2[index] = y_ic * np.sum(alpha_c * kernel_func(x, x[i]) + w0c)
	# 		index += 1

	# print np.sum(first_part2)

	# first_part = np.empty(len(x))
	# index = 0
	# for i in range(len(x)):
	# 	y_i = y[i]
	# 	first_part[index] = np.sum(np.dot(kernel_func(x, x[i]), y_i * alphas) + len(x[i]) * w0)
	# 	index += 1

	# # print np.sum(first_part)

	# # second_part = np.empty(len(x))
	# # index = 0
	# # for i in range(len(x)):
	# # 	y_i = y[i]
	# # 	outer_outer_sum = 0
	# # 	for c_index_o in range(c):
	# # 		y_ic = y_i[c_index_o]
	# # 		outer_sum = 0
	# # 		for c_index_i in range(c):
	# # 			inner_sum = 0
	# # 			w0c = w0[c_index_i]
	# # 			for j in range(len(x)):
	# # 				alpha_cij = alphas[j, c_index_i]
	# # 				inner_sum += alpha_cij * kernel_func(x[j], x[i]) + w0c
	# # 			outer_sum += np.exp(inner_sum)
	# # 		# print "os = ", outer_sum
	# # 		outer_outer_sum += y_ic * np.log(outer_sum)
	# # 	second_part[index] = outer_outer_sum
	# # 	index += 1

	# # print np.sum(second_part)

	# second_part = np.empty(len(x))
	# for i in range(len(x)):
	# 	y_i = y[i]
	# 	second_part[i] = np.sum(y_i * np.log(np.sum(np.exp(np.dot(kernel_func(x, x[i]), alphas) + len(x) * w0))))
	# 	# for c_index_o in range(c):
	# 		# y_ic = y_i[c_index_o]
	# 		# outer_sum = 0
	# 		# for c_index_i in range(c):
	# 		# 	inner_sum = 0
	# 		# 	w0c = w0[c_index_i]
	# 		# 	for j in range(len(x)):
	# 		# 		alpha_cij = alphas[j, c_index_i]
	# 		# 		inner_sum += alpha_cij * kernel_func(x[j], x[i]) + w0c
	# 		# 	outer_sum += np.exp(inner_sum)
	# 		# print "os = ", outer_sum
	# 		# outer_outer_sum += y_ic * np.log(outer_sum)
	# 	# second_part[index] = outer_outer_sum
	# 	# index += 1

	# # print np.sum(second_part2)

	# return -1*np.sum(first_part) + np.sum(second_part) + l * np.linalg.norm(alphas, 2)


    # first_part = np.empty((len(x), ))
    # for i in range(len(x)):
    #     first_part[i] = np.sum(y[i] * ((kernel_func(x[i], x[i]) * alphas[i]) + w0))

    # second_part = np.empty((len(x), ))
    # for i in range(len(x)):
    #     second_part[i] = np.sum(np.exp(kernel_func(x[i], x[i]) * alphas[i] + w0))
    
    # return -1*np.sum(first_part) + np.sum(np.log(second_part)) + l * np.linalg.norm(alphas, 2)

        # first part
    # first_part = 0
    # for i in range(len(x)):
    #     for c_index in range(c):
    #         first_part += y[i][c_index] * ((kernel_func(x[i], x[i]) * alphas[i, c_index]) + w0[c_index])
    # first_part *= -1
    
    # # second part
    # second_part = 0
    # for i in range(len(x)):
    #     inner_sum = 0
    #     for c_index in range(c):
    #         inner_sum += np.exp(kernel_func(x[i], x[i]) * alphas[i, c_index] + w0[c_index])
    #     second_part += np.log(inner_sum)

    # return first_part + second_part + l * np.linalg.norm(alphas, 2)

    # return 0

# c is number of classes
def transform_y(y, c):
    new_y = np.empty((len(y), c))
    if c == 7:
        for i in range(len(y)):
            if y[i] == 1:
                new_y[i] = np.array([1, 0, 0, 0, 0, 0, 0])
            elif y[i] == 2:
                new_y[i] = np.array([0, 1, 0, 0, 0, 0, 0])
            elif y[i] == 3:
                new_y[i] = np.array([0, 0, 1, 0, 0, 0, 0])
            elif y[i] == 4:
                new_y[i] = np.array([0, 0, 0, 1, 0, 0, 0])
            elif y[i] == 5:
                new_y[i] = np.array([0, 0, 0, 0, 1, 0, 0])
            elif y[i] == 6:
                new_y[i] = np.array([0, 0, 0, 0, 0, 1, 0])
            elif y[i] == 7:
                new_y[i] = np.array([0, 0, 0, 0, 0, 0, 1])
        return new_y
    elif c==2:
        for i in range(len(y)):
            if y[i] == 1:
                new_y[i] = np.array([1, 0])
            elif y[i] == -1:
                new_y[i] = np.array([0, 1])
        return new_y
    elif c == 3:
        for i in range(len(y)):
            if y[i] == 1:
                new_y[i] = np.array([1, 0, 0])
            elif y[i] == 2:
                new_y[i] = np.array([0, 1, 0])
            elif y[i] == 3:
                new_y[i] = np.array([0, 0, 1])
        return new_y

def predictor(alphas, w0, kernel_func, xtrain, x_to_predict):
	# predictions = np.empty((len(x), alphas.shape[1]))

	# for i in range(len(x)):
		# predictions[i] = np.dot(kernel_func(x, x[i]), alphas) + len(x)*w0

	predictions = np.dot(kernel_func(xtrain, x_to_predict), alphas) + w0
	# print predictions
	# print predictions
	# predictions = predictions / np.sum(predictions, axis=1).reshape((-1, 1))
	# print predictions
	max_args = np.argmax(predictions, axis = 1)

	class_pred = np.zeros(predictions.shape)

	index = 0
	for pt in class_pred:
		pt[max_args[index]] = 1
		index += 1
	return class_pred


def error(truth, prediction):
    return (len(truth) - np.sum(np.all(truth == prediction, axis = 1))) / float(len(truth))

# def sparsity(alphas_and_w0):
#     return np.sum(np.abs(alphas_and_w0) < 1e-6)

def bfgs_callback(params):
	print params

def normalize(x):
	scaled = x - x.min(axis = 0)
	return scaled / scaled.max(axis = 0)

if kaggle:
	name = 'kaggle_train'
	data = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv', delimiter=',', skiprows=1)
	num_pts = len(data)
	c = 7 # number of classes

	# distribution of classes in the kaggle_train data
	# classes = data[:, -1]
	# print np.bincount(classes.astype(int))

	#split into 3 sets with each type of class 2160 times (they are equal in the original distribution)
	train_indices = []
	validate_indices = []
	test_indices = []
	for i in range(1, c + 1):
	    indices_of_class = np.array(np.where(data[:, -1].astype(int) == i)).flatten()
	    total_num_indices = len(indices_of_class)
	    num_each = 50 # for all, total_num_indices / 3
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

	# take subset of features
	# x_train = x_train[:, 1:3]
	# x_validate = x_validate[:, 1:3]
	# x_test = x_test[:, 1:3]

	y_train = transform_y(y_train, c)
	y_validate = transform_y(y_validate, c)
	y_test = transform_y(y_test, c)

	# l = 0
	best_l = 0
	best_beta = 0
	min_err = 1e100
	best_alphas = 0
	best_w0s = 0
	l = 0
	for beta in [0.1, 0.5, 1, 1.5, 2]:
		def gaussian(x1, x2):
			return gaussian_kernel_general(x1, x2, beta)

		kernel = gaussian
		def nll_multi_kaggle_train(params):
		    x = x_train
		    y = y_train
		    regroup = np.reshape(params, (-1, c))
		    alphas = regroup[:-1]
		    w0 = regroup[-1]
		#     alphas = params[:, -1]
		#     w0 = params[:, -1]
		    return nll_multiclass(x, y, alphas, w0, c, l, kernel)

		start = np.zeros((len(x_train) + 1, c))
		all_output = scipy.optimize.fmin_bfgs(nll_multi_kaggle_train, start, gtol=1e-12, full_output=True, maxiter=5)
		alphas_and_w0 = all_output[0].reshape((-1, c))
		alphas = alphas_and_w0[:-1]
		w0 = alphas_and_w0[-1]
		# print w0
		train_predictions = predictor(alphas, w0, kernel, x_train, x_train)
		train_err = error(y_train, train_predictions)
		print "train error = ", train_err

		# val_predictions = predictor(alphas, w0, kernel, x_train, x_validate)
		# val_err = error(y_validate, val_predictions)
		# print "validation error = ", val_err
		
		if train_err < min_err:
			best_l = l
			best_b = beta
			best_alphas = alphas
			best_w0s = w0
			print "new best beta b = ", beta


	beta = best_b
	min_err = 1e100

	for l in [1e-3, 1e-2, 0.1, 0.5, 1]:
		def gaussian(x1, x2):
			return gaussian_kernel_general(x1, x2, beta)

		kernel = gaussian
		def nll_multi_kaggle_train(params):
		    x = x_validate
		    y = y_validate
		    regroup = np.reshape(params, (-1, c))
		    alphas = regroup[:-1]
		    w0 = regroup[-1]
		#     alphas = params[:, -1]
		#     w0 = params[:, -1]
		    return nll_multiclass(x, y, alphas, w0, c, l, kernel)

		start = np.zeros((len(x_train) + 1, c))
		all_output = scipy.optimize.fmin_bfgs(nll_multi_kaggle_train, start, gtol=1e-12, full_output=True, maxiter=5)
		alphas_and_w0 = all_output[0].reshape((-1, c))
		alphas = alphas_and_w0[:-1]
		w0 = alphas_and_w0[-1]
		# print w0
		# train_predictions = predictor(alphas, w0, kernel, x_validate, x_train)
		# train_err = error(y_train, train_predictions)
		# print "train error = ", train_err

		val_predictions = predictor(alphas, w0, kernel, x_validate, x_validate)
		val_err = error(y_validate, val_predictions)
		print "validation error = ", val_err
		
		if val_err < min_err:
			min_err = val_err
			best_l = l
			# best_b = beta
			best_alphas = alphas
			best_w0s = w0
			print "new best l = ", l

	test_predictions = predictor(best_alphas, w0, kernel, x_validate, x_test)
	test_err = error(y_test, test_predictions)
	print "test err = ", test_err
else:
	name = 'stdev2'
	train = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_train.csv')
	validate = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_validate.csv')
	test = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_test.csv')

	# name = 'small_example'
	# train = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')
	# test = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')

	y_train = train[:, -1].copy()
	x_train = train[:, :-1].copy()
	# x_train = normalize(x_train)

	# y_validate = validate[:, -1].copy()
	# x_validate = validate[:, :-1].copy()
	# x_validate = normalize(x_validate)

	y_test = test[:, -1].copy()
	x_test = test[:, :-1].copy()
	# x_test = normalize(x_test)

	c = 2

	y_train = transform_y(y_train, c)
	# y_validate = transform_y(y_validate, c)
	y_test = transform_y(y_test, c)

	min_val_error = 100000000

	beta = 0.5
	def gaussian(x1, x2):
		return gaussian_kernel_general(x1, x2, beta)

	kernel = gaussian

	best_alphas = 0
	best_w0 = 0

	# for l in [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 10]:
	for l in [0]:
		def nll_multi_train(params):
		    x = x_train
		    y = y_train
		    regroup = np.reshape(params, (-1, c))
		    alphas = regroup[:-1]
		    w0 = regroup[-1]
		    return nll_multiclass(x, y, alphas, w0, c, l, kernel)

		start = np.zeros((len(x_train) + 1, c))

		# print nll_multi_train(start)
#
		all_output = scipy.optimize.fmin_bfgs(nll_multi_train, start, full_output=True, gtol=1e-12)
		alphas_and_w0 = all_output[0]
		alphas_and_w0 = alphas_and_w0.reshape((-1, c))
		alphas = alphas_and_w0[:-1]
		w0 = alphas_and_w0[-1]
		# print alphas
		# print w0
		# predictor(alphas, w0, kernel, x_train)
		train_predictions = predictor(alphas, w0, kernel, x_train, x_train)
		# print train_predictions
		# test_predictions = predictor(alphas, w0, kernel, x_test)
		err = error(y_train, train_predictions)
		print "error = ", err
		if err < min_val_error:
			min_error = err
			best_alphas = alphas
			best_w0 = w0
			print "smallest error with l = ", l

	test_predictions = predictor(best_alphas, best_w0, kernel, x_train, x_test)
	test_err = error(y_test, test_predictions)
	print "test error = ", test_err