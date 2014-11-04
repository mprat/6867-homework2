import sklearn
from sklearn import svm
import numpy as np
name = 'kaggle_train'
data = np.loadtxt('../problemset/HW2_handout/data/'+name+'.csv', delimiter=',', skiprows=1)
num_pts = len(data)
c = 7

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
    # train_indices.extend(indices_of_class[:num_each])
    # validate_indices.extend(indices_of_class[num_each / 3: 2*num_each / 3])
    # test_indices.extend(indices_of_class[2*num_each / 3:])

data_train = data[train_indices]
data_validate = data[validate_indices]
data_test = data[test_indices]

x_train = data_train[:, 1:-1]
y_train = data_train[:, -1]
x_validate = data_validate[:, 1:-1]
y_validate = data_validate[:, -1]
x_test = data_test[:, 1:-1]
y_test = data_test[:, -1]

min_error = 100000
best_l = 0
for l in [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 2e-7, 5e-7, 7e-7, 9e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 7e-2, 0.1, 0.2, 0.5, 0.7, 1, 10]:
	mcsvm = sklearn.svm.LinearSVC(penalty='l1', loss='l2', dual=False, tol=1e-8, multi_class='ovr', C=l)
	mcsvm.fit(x_train, y_train)
	error = mcsvm.score(x_validate, y_validate)
	print "error ", error, " for l = ", l
	if error <= min_error:
		min_error = error
		print "new minimum error ", error, " with l = ", l
		best_l = l

# report test error
print 'report test error'
mcsvm = sklearn.svm.LinearSVC(penalty='l1', loss='l2', dual=False, tol=1e-8, multi_class='ovr', C=best_l)
mcsvm.fit(x_train, y_train)
error = mcsvm.score(x_test, y_test)
print "test error = ", error, " with l = ", best_l