import scipy
from scipy import optimize
import numpy as np

kaggle = False

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def gaussian_kernel_general(x, y, beta):
	if len(x.shape) > 1:
		return np.exp(-1*beta*np.linalg.norm(x - y, axis=1))
	else:
		return np.exp(-1*beta*np.linalg.norm(x - y))

# c is number of classes
def nll_multiclass(x, y, alphas, w0, c, l, kernel_func):
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

	first_part = np.empty(len(x))
	index = 0
	for i in range(len(x)):
		y_i = y[i]
		first_part[index] = np.sum(np.dot(kernel_func(x, x[i]), y_i * alphas) + len(x[i]) * w0)
		index += 1

	# print np.sum(first_part)

	# second_part = np.empty(len(x))
	# index = 0
	# for i in range(len(x)):
	# 	y_i = y[i]
	# 	outer_outer_sum = 0
	# 	for c_index_o in range(c):
	# 		y_ic = y_i[c_index_o]
	# 		outer_sum = 0
	# 		for c_index_i in range(c):
	# 			inner_sum = 0
	# 			w0c = w0[c_index_i]
	# 			for j in range(len(x)):
	# 				alpha_cij = alphas[j, c_index_i]
	# 				inner_sum += alpha_cij * kernel_func(x[j], x[i]) + w0c
	# 			outer_sum += np.exp(inner_sum)
	# 		# print "os = ", outer_sum
	# 		outer_outer_sum += y_ic * np.log(outer_sum)
	# 	second_part[index] = outer_outer_sum
	# 	index += 1

	# print np.sum(second_part)

	second_part = np.empty(len(x))
	for i in range(len(x)):
		y_i = y[i]
		second_part[i] = np.sum(y_i * np.log(np.sum(np.exp(np.dot(kernel_func(x, x[i]), alphas) + len(x) * w0))))
		# for c_index_o in range(c):
			# y_ic = y_i[c_index_o]
			# outer_sum = 0
			# for c_index_i in range(c):
			# 	inner_sum = 0
			# 	w0c = w0[c_index_i]
			# 	for j in range(len(x)):
			# 		alpha_cij = alphas[j, c_index_i]
			# 		inner_sum += alpha_cij * kernel_func(x[j], x[i]) + w0c
			# 	outer_sum += np.exp(inner_sum)
			# print "os = ", outer_sum
			# outer_outer_sum += y_ic * np.log(outer_sum)
		# second_part[index] = outer_outer_sum
		# index += 1

	# print np.sum(second_part2)

	return -1*np.sum(first_part) + np.sum(second_part) + l * np.linalg.norm(alphas, 2)


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

def predictor(alphas, w0, kernel_func, x):
	predictions = np.empty((len(x), alphas.shape[1]))
	# for c_index in range(alphas.shape[1]):
	# 	alpha_c = alphas[:, c_index]
	# 	w0c = w0[c_index]
	# 	for i in range(len(x)):
	# 		predictions[i][c_index] = np.exp(np.sum(alpha_c * kernel_func(x, x[i]) + w0c))

	for i in range(len(x)):
		predictions[i] = np.dot(kernel_func(x, x[i]), alphas) + len(x)*w0

	# print predictions
	# predictions = predictions / np.sum(predictions, axis=1).reshape((-1, 1))
	max_args = np.argmax(predictions, axis = 1)

	class_pred = np.zeros(predictions.shape)

	index = 0
	for pt in class_pred:
		pt[max_args[index]] = 1
		index += 1
	return class_pred


    # kernel_result = np.empty(len(x))
    # for i in range(len(x)):
    #     kernel_result[i] = kernel_func(x[i], x[i])
    # num_classes = alphas.shape[1]
    # predictions = np.empty((num_classes, len(x)))
    # print "kr = ", kernel_result
    # print "alphas= ", alphas
    # for c in range(num_classes):
    #     all_predictions = np.exp(kernel_result * alphas[:, c] + w0[c])
    #     print "c = ", c, "pred = ", all_predictions
    #     predictions[c] = all_predictions / sum(all_predictions)
    # predictions = predictions.T
    # # print predictions
    # class_pred = np.zeros(predictions.shape)
    # max_args = np.argmax(predictions, axis=1)
    # index = 0
    # for pt in class_pred:
    #     pt[max_args[index]] = 1
    #     index += 1
    # return class_pred

    # kernel_result = np.empty(len(x))
    # for i in range(len(x)):
    #     kernel_result[i] = kernel_func(x[i], x[i])
    # num_classes = alphas.shape[1]
    # predictions = np.empty((num_classes, len(x)))
    # for c in range(num_classes):
    #     all_predictions = np.exp(kernel_result * alphas[:, c] + w0[c])
    #     predictions[c] = all_predictions# / sum(all_predictions)
    # predictions = predictions.T
    # print predictions
    # class_pred = np.zeros(predictions.shape)
    # max_args = np.argmax(predictions, axis=1)
    # index = 0
    # for pt in class_pred:
    #     pt[max_args[index]] = 1
    #     index += 1
    # return class_pred

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
	c = 3 # number of classes

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
	    num_each = total_num_indices / 3 # for all, total_num_indices / 3
	    random_indices = np.random.choice(indices_of_class, num_each, replace=False)
	    train_indices.extend(random_indices)

	    indices_of_class_val = np.delete(indices_of_class, random_indices)
	    random_indices = np.random.choice(indices_of_class, num_each, replace=False)

	    validate_indices.extend(random_indices)

	    indices_of_class_test = np.delete(indices_of_class_val, random_indices)
	    random_indices = np.random.choice(indices_of_class, num_each, replace=False)

	    test_indices.extend(random_indices)

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

	def nll_multi_kaggle_train(params):
	    x = x_train
	    y = y_train
	    regroup = np.reshape(params, (-1, c))
	    alphas = regroup[:-1]
	    w0 = regroup[-1]
	#     alphas = params[:, -1]
	#     w0 = params[:, -1]
	    l = 0
	    return nll_multiclass(x, y, alphas, w0, c, l, linear_kernel)

	start = np.empty((len(x_train) + 1, c))
	alphas_and_w0 = scipy.optimize.fmin_bfgs(nll_multi_kaggle_train, start, gtol=1e-12, full_output=True)
	alphas_and_w0 = alphas_and_w0.reshape((-1, c))
	alphas = alphas_and_w0[:-1]
	w0 = alphas_and_w0[-1]
	# print w0
	predictions = predictor(alphas, w0, linear_kernel, x_train)
	print "error = ", error(y_train, predictions)
else:
	name = 'bigOverlap'
	train = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_train.csv')
	test = np.loadtxt('../problemset/HW2_handout/data/data_'+name+'_test.csv')

	# name = 'small_example'
	# train = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')
	# test = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv')

	y_train = train[:, -1].copy()
	x_train = train[:, :-1].copy()
	x_train = normalize(x_train)

	y_test = test[:, -1].copy()
	x_test = test[:, :-1].copy()
	x_test = normalize(x_test)

	c = 2

	y_train = transform_y(y_train, c)
	y_test = transform_y(y_test, c)

	min_val_error = 100000000

	beta = 0.5
	def gaussian(x1, x2):
		return gaussian_kernel_general(x1, x2, beta)

	kernel = linear_kernel

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
		all_output = scipy.optimize.fmin_bfgs(nll_multi_train, start, full_output=True, gtol=1e-12, maxiter=3)
		alphas_and_w0 = all_output[0]
		alphas_and_w0 = alphas_and_w0.reshape((-1, c))
		alphas = alphas_and_w0[:-1]
		w0 = alphas_and_w0[-1]
		# print alphas
		# print w0
		# predictor(alphas, w0, kernel, x_train)
		train_predictions = predictor(alphas, w0, kernel, x_train)
		# print train_predictions
		# test_predictions = predictor(alphas, w0, kernel, x_test)
		err = error(y_train, train_predictions)
		print "error = ", err
		if err < min_val_error:
			min_error = err
			best_alphas = alphas
			best_w0 = w0
			print "smallest error with l = ", l

	test_predictions = predictor(best_alphas, best_w0, kernel, x_test)
	test_err = error(y_test, test_predictions)
	print "test error = ", test_err