import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
from ode_solver import RK4Solver
from multiprocessing import Pool

CONSTANT_ACC_G = 9.81


def is_local_maxima(array, i):
	"""
	Tells whether the element at a specified index in a certain array is a local maxima
	:param array: The array
	:param i: The element index
	:return: true iif the element at `array[i]` is a local maxima.
	"""
	return array[i - 1] < array[i] and array[i] > array[i + 1]


def determine_frequency(theta, L):
	"""
	Returns the oscillation frequency of a single pendulum with starting angle `theta` and length `l`
	:param theta: The starting angle
	:param L: The length of the pendulum
	:return: the oscillation frequency
	"""
	def frequency_function(t, y):
		return np.array([y[1], -(CONSTANT_ACC_G / L) * np.sin(y[0])])

	y0 = np.array([theta, 0])
	solver = RK4Solver(0, y0, frequency_function)
	N = 10000
	h = 10 / N
	results = np.array(solver.n_steps(N, h))
	angular_velocities = results[:, 1]
	i_1 = 1
	while not (is_local_maxima(angular_velocities, i_1)):
		i_1 += 1
	i_2 = i_1 + 1
	while not (is_local_maxima(angular_velocities, i_2)):
		i_2 += 1
	period = (i_2 - i_1) * h
	return 1 / period


def get_graph_simple_pendulum():
	"""
	Computes and displays the frequency graph of a single pendulum according to `theta` from 0 to pi/2
	with L = 1.
	Also plots the horizontal line with value (1/2pi)*sqrt(g/L) where L = 1.
	"""
	angles = np.linspace(0, np.pi / 2, 100)
	angles = np.array(list(filter(lambda x: not np.isclose(x, 0), angles)))
	frequencies = []
	for theta in angles:
		frequencies.append(determine_frequency(theta, 1))
	plt.plot(angles, frequencies)
	plt.axhline(y=(1 / (2 * np.pi)) * np.sqrt(CONSTANT_ACC_G / 1), color="orange",
				label="($\\frac{1}{2\pi})\\times\sqrt{g}$")
	plt.legend()
	plt.xlabel("$\\theta$")
	plt.ylabel("Fréquence d'oscillation")
	plt.grid()
	plt.show()


def rk_compatible_2_pendulum_modelisation(l1, l2, m1, m2):
	"""
	Returns the double pendulum model in a format compatible with our
	solvers according to the pendulum's lengths and masses.
	:param l1: The length of the 1rst pendulum
	:param l2: The length of the 2nd pendulum
	:param m1: The mass of the 1rst pendulum
	:param m2: The mass of the 2nd pendulum
	:return: the double pendulum model in a format compatible with our
	solvers
	"""
	def inner_func(t, y):
		theta1 = y[0]
		theta2 = y[1]
		omega1 = y[2]
		omega2 = y[3]

		n_theta1 = y[2]
		n_theta2 = y[3]
		common_denominator = (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
		omega1_denominator = l1 * common_denominator
		omega2_denominator = l2 * common_denominator
		omega1_numerator_p1 = -CONSTANT_ACC_G * (2 * m1 + m2) * np.sin(theta1)
		omega1_numerator_p2 = -m2 * CONSTANT_ACC_G * np.sin(theta1 - 2 * theta2)
		omega1_numerator_p3 = -2 * np.sin(theta1 - theta2) \
							  * m2 \
							  * ((omega2 ** 2) * l2 + (omega1 ** 2) * l1 * np.cos(theta1 - theta2))
		n_omega1 = (omega1_numerator_p1 + omega1_numerator_p2 + omega1_numerator_p3) / omega1_denominator

		omega2_numerator_p1 = (omega1 ** 2) * l1 * (m1 + m2)
		omega2_numerator_p2 = CONSTANT_ACC_G * (m1 + m2) * np.cos(theta1)
		omega2_numerator_p3 = (omega2 ** 2) * l2 * m2 * np.cos(theta1 - theta2)
		n_omega2 = 2 * np.sin(theta1 - theta2) * (
				omega2_numerator_p1 + omega2_numerator_p2 + omega2_numerator_p3) / omega2_denominator

		return np.array([n_theta1, n_theta2, n_omega1, n_omega2])

	return inner_func


def convert_results_to_coordinates(results, l1, l2):
	"""
	Converts the results of our solver result for double pendulum
	 `[[theta1, theta2, omega1, omega2],...]` to a vector of coordinates
	[[[x1, y1], [x2, y2]], ...]
	:param results: The solver results `[[theta1, theta2, omega1, omega2],...]`
	:param l1: The length of the 1rst pendulum
	:param l2: The length of the 2nd pendulum
	:return: A coordinate vector [[[x1, y1], [x2, y2]], ...]
	"""
	def convert_vector_to_xy(y):
		x1 = l1 * np.sin(y[0])
		y1 = -l1 * np.cos(y[0])
		x2 = x1 + l2 * np.sin(y[1])
		y2 = y1 - l2 * np.cos(y[1])

		return np.array([[x1, y1], [x2, y2]])

	return np.array(list(map(convert_vector_to_xy, results)))


def plot_2_pendulum(m1, m2, l1, l2, theta1, theta2, omega1, omega2):
	"""
	Plots the path of the 2nd pendulum in a double pendulum system.
	Does NOT show the plot!
	:param m1: The 1rst mass of the double pendulum
	:param m2: The 2nd mass of the double pendulum
	:param l1: The 1rst length
	:param l2: The 2nd length
	:param theta1: starting angle of first pendulum
	:param theta2: starting angle of second pendulum
	:param omega1: starting angular velocity of first pendulum
	:param omega2: starting angular velocity of second pendulum
	"""
	initial = np.array([theta1, theta2, omega1, omega2])
	N = 1000
	h = 100 / N
	results = RK4Solver(0, initial, rk_compatible_2_pendulum_modelisation(l1, l2, m1, m2)).n_steps(N, h)
	coordinate_results = convert_results_to_coordinates(results, l1, l2)
	first_pendulum = coordinate_results[:, 0, :]
	second_pendulum = coordinate_results[:, 1, :]
	# plt.plot(first_pendulum[:, 0], first_pendulum[:, 1])
	plt.plot(second_pendulum[:, 0], second_pendulum[:, 1])
	plt.plot(first_pendulum[0, 0], first_pendulum[0, 1], marker="o", markersize=4, markerfacecolor="blue",
			 markeredgecolor="blue")
	plt.plot(second_pendulum[0, 0], second_pendulum[0, 1], marker="o", markersize=4, markerfacecolor="red",
			 markeredgecolor="red")


def get_flip_time(m1, m2, l1, l2, theta1, theta2, omega1, omega2):
	"""
	Computes the time it takes for the double pendulum to flip.
	A flip is defined as either one of the pendulum having an angle greater or equal than pi or less or equal than -pi.
	:param m1: The 1rst mass of the double pendulum
	:param m2: The 2nd mass of the double pendulum
	:param l1: The 1rst length
	:param l2: The 2nd length
	:param theta1: starting angle of first pendulum
	:param theta2: starting angle of second pendulum
	:param omega1: starting angular velocity of first pendulum
	:param omega2: starting angular velocity of second pendulum
	:return: inf if the pendulum does not flip in the simulation time, else the flip time
	"""
	initial = np.array([theta1, theta2, omega1, omega2])
	N = int(1000 * np.sqrt(l1 * CONSTANT_ACC_G))
	h = 100 / N
	solver = RK4Solver(0, initial, rk_compatible_2_pendulum_modelisation(l1, l2, m1, m2))
	results = solver.n_steps(N, h)
	results = np.array(results)
	first_pendulum = results[:, (0, -2)]
	second_pendulum = results[:, (1, -1)]
	min_flip_idx = np.inf
	for i, v in enumerate(first_pendulum):
		if abs(v[0]) > np.pi:
			min_flip_idx = i
			break
	for i, v in enumerate(second_pendulum):
		if i >= min_flip_idx:
			break
		if abs(v[0]) > np.pi:
			min_flip_idx = i
			break
	return min_flip_idx * h


def get_graph_2_pendulums():
	"""
	Shows the plot of 50 pendulums with slight variations to their starting conditions
	"""
	for theta in np.linspace(0, 1, 50):
		plot_2_pendulum(1, 1, 1, 1, theta, -theta, 0, 0)
		print(f"2_pendulum for {theta} done")
	plt.show()


def multiprocessing_nightmare(args):
	"""
	Multiprocessing-compatible function to compute flip time.
	:param args: (i, theta2, j, theta1, length)
	:return: the flip time of the double pendulum system described in `args`
	"""
	i, theta2, j, theta1, length = args
	flip_time = get_flip_time(1, 1, length, length, theta1, theta2, 0, 0)
	return flip_time


def get_flip_time_graph():
	"""
	Computes the flip time graph with varying theta1 and theta2.

	Should show a graph similar to https://commons.wikimedia.org/wiki/File:Double_pendulum_flip_time_2021.png
	"""
	print("get_flip_time_graph() will take ~1h with 100% CPU usage to execute. Are you sure you want to do this ? "
		  "CTRL+D if you don't want to execute this.")
	input()
	w = 300
	h = 300
	res_matrix = np.zeros((h, w))
	length = 1
	check_value = np.sqrt(length / CONSTANT_ACC_G)
	counter = 0
	cvals = [np.sqrt(1 / CONSTANT_ACC_G), np.sqrt(10 / CONSTANT_ACC_G), np.sqrt(100 / CONSTANT_ACC_G),
			 np.sqrt(1000 / CONSTANT_ACC_G), np.sqrt(10000 / CONSTANT_ACC_G)]
	colors = ["black", "red", "green", "blue", "white"]

	norm = matplotlib.colors.Normalize(min(cvals), max(cvals))
	tuples = list(zip(map(norm, cvals), colors))
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
	for j, theta1 in enumerate(np.linspace(-np.pi, np.pi, h)):
		iterator = map(lambda x: x + (j, theta1, length),
					   enumerate(np.linspace(-np.pi, np.pi, w)))
		with Pool(50) as p:
			result = p.map(multiprocessing_nightmare, iterator)
			for i, v in enumerate(result):
				res_matrix[i, j] = v
		counter += w
		print(counter)
	# for j, theta2 in enumerate(np.linspace(-np.pi, np.pi, w)):

	plt.xlabel("$\\theta_1$")
	plt.ylabel("$\\theta_2$")
	y_ticks = ["$\\pi$" if x == np.pi else "$-\\pi$" if x == -np.pi else "" for x in np.linspace(-np.pi, np.pi, h)]
	x_ticks = ["$\\pi$" if x == np.pi else "$-\\pi$" if x == -np.pi else "" for x in np.linspace(-np.pi, np.pi, w)]
	plt.xticks(range(w), x_ticks)
	plt.yticks(range(h), y_ticks)
	plt.imshow(res_matrix, cmap=cmap)
	plt.savefig("flip_time_300x300.png")
	plt.show()


def main():
	get_graph_simple_pendulum()
	get_graph_2_pendulums()
	get_flip_time_graph()


# print(get_flip_time(1, 1, 1, 1, 0, 0, 0, 0))


if __name__ == "__main__":
	main()
