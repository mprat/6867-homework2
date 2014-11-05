import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def margin_with_c():
	c = np.array([0.01, 0.1, 1, 10, 100])
	small_overlap_linear = np.array([1.79, .93, .57, .5611, .5613])
	big_overlap_linear = np.array([2.38, 1.93, 1.88, 1.88, 1.88])
	ls_linear = np.array([.939, .48, .31, .23, .23])
	nonSep2_linear = np.array([3.98, 3.37, 3.29, 3.29, 3.29])

	# small_overlap_gaussian_0001 = np.array([0.76, 0.076, 0.007, 0.0007, 7e-5])
	# small_overlap_gaussian_01 = np.array([0.76, 0.076, 0.007, 0.0007, 7e-5])

	stdev1_beta_0_5 = np.array([0.089, 0.0089, 8e-4, 8e-5, 8e-6])
	stdev1_beta_0_0001 = np.array([8.9e-2, 8.9e-3, 8.9e-4, 8.9e-5, 8.9e-6])
	stdev1_beta_1 = np.array([0.089, 0.0089, 8e-4, 8e-5, 8e-6])

	nonSep2_beta_0_001 = np.array([3, 0.3, 3e-2, 3e-3, 3e-4])
	nonSep2_beta_0_01 = np.array([0.3, 3e-2, 3e-3, 3e-4, 3e-5])


	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(c, small_overlap_linear, label="smallOverlap Linear Kernel")
	ax.plot(c, big_overlap_linear, label="bigOverlap Linear Kernel")
	ax.plot(c, ls_linear, label="ls Linear Kernel")
	ax.plot(c, nonSep2_linear, label="nonSep2 Linear Kernel")
	ax.plot(c, stdev1_beta_0_0001, label=r"stdev1 $\beta = 0.001$")
	ax.plot(c, stdev1_beta_0_5, label=r"stdev1 $\beta = 0.5$")
	ax.plot(c, stdev1_beta_1, label=r"stdev1 $\beta = 1$")
	ax.plot(c, nonSep2_beta_0_001,label=r"nonSep2 $\beta = 0.001$")
	ax.plot(c, nonSep2_beta_0_01,label=r"nonSep2 $\beta = 0.01$")

	ax.set_xlabel('c')
	ax.set_ylabel('Geometric margin')
	ax.set_xscale('log')
	ax.set_yscale('linear')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_ylim([0, 5])

	ax.legend()


	plt.show()

def support_vectors():
	c = np.array([0.01, 0.1, 1, 10, 100])
	small_overlap_linear = np.array([1.79, .93, .57, .5611, .5613])
	big_overlap_linear = np.array([2.38, 1.93, 1.88, 1.88, 1.88])
	ls_linear = np.array([.939, .48, .31, .23, .23])
	nonSep2_linear = np.array([3.98, 3.37, 3.29, 3.29, 3.29])

	# small_overlap_gaussian_0001 = np.array([0.76, 0.076, 0.007, 0.0007, 7e-5])
	# small_overlap_gaussian_01 = np.array([0.76, 0.076, 0.007, 0.0007, 7e-5])

	stdev1_beta_0_5 = np.array([0.089, 0.0089, 8e-4, 8e-5, 8e-6])
	stdev1_beta_0_0001 = np.array([8.9e-2, 8.9e-3, 8.9e-4, 8.9e-5, 8.9e-6])
	stdev1_beta_1 = np.array([0.089, 0.0089, 8e-4, 8e-5, 8e-6])

	nonSep2_beta_0_001 = np.array([3, 0.3, 3e-2, 3e-3, 3e-4])
	nonSep2_beta_0_01 = np.array([0.3, 3e-2, 3e-3, 3e-4, 3e-5])


	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(c, small_overlap_linear, label="smallOverlap Linear Kernel")
	ax.plot(c, big_overlap_linear, label="bigOverlap Linear Kernel")
	ax.plot(c, ls_linear, label="ls Linear Kernel")
	ax.plot(c, nonSep2_linear, label="nonSep2 Linear Kernel")
	ax.plot(c, stdev1_beta_0_0001, label=r"stdev1 $\beta = 0.001$")
	ax.plot(c, stdev1_beta_0_5, label=r"stdev1 $\beta = 0.5$")
	ax.plot(c, stdev1_beta_1, label=r"stdev1 $\beta = 1$")
	ax.plot(c, nonSep2_beta_0_001,label=r"nonSep2 $\beta = 0.001$")
	ax.plot(c, nonSep2_beta_0_01,label=r"nonSep2 $\beta = 0.01$")

	ax.set_xlabel('c')
	ax.set_ylabel('Support vectors')
	ax.set_xscale('linear')
	ax.set_yscale('linear')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_ylim([0, 5])

	ax.legend()


	plt.show()