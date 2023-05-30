import numpy as np
from matplotlib import pyplot as plt
from ode_solver import *


def display_direction_field_2d(ax, F, minRange, maxRange, N):
	"""
	Draws the direction field of a 2D system
	:param ax: matplotlib Axes object
	:param F: The function to show direction field
	:param minRange:
	:param maxRange:
	:param N: Number of field arrows per line
	"""
	listY1, listY2 = np.meshgrid(np.linspace(minRange, maxRange, N), np.linspace(minRange, maxRange, N))
	u = F(0, (listY1, listY2))
	u = u / np.linalg.norm(u, axis=0)
	ax.quiver(listY1, listY2, u[0], u[1])


def display_direction_field_1d(ax, F, minRange, maxRange, N):
	"""
	Draws the direction field of a 1D system
	:param ax: matplotlib Axes object
	:param F: The function to show direction field
	:param minRange:
	:param maxRange:
	:param N: Number of field arrows per line
	"""
	listY1, listY2 = np.meshgrid(np.linspace(minRange, maxRange, N), np.linspace(minRange, maxRange, N))
	u = F(listY1, listY2)
	ax.quiver(listY1, listY2, np.ones(u.shape), u)


def display_ode_solution_2d(ax, solver_cts, t0, y0, F):
	"""
	Draws a solver's results of a 2D system
	:param ax: matplotlib Axes object
	:param solver_cts: A constructor for a specific Solver
	:param t0: starting time
	:param y0: initial condition for system to plot
	:param F: The system to plot
	"""
	solver = solver_cts(t0, y0, F)
	# res = np.array(solver.n_steps(10000, 1e-1))
	# ax.plot(res[:, 0], res[:, 1])
	resconv, i = solver.epsilon_solve(30, 1e-2)
	resconv = np.array(resconv)
	# print(solver_cts.__name__, " under threshold after ", i, " iterations.")
	ax.plot(resconv[:, 0], resconv[:, 1], label=f"Solveur : {solver_cts.method}")


def display_ode_solution_1d(ax, solver_cts, t0, y0, F):
	"""
	Draws a solver's results of a 1D system
	:param ax: matplotlib Axes object
	:param solver_cts: A constructor for a specific Solver
	:param t0: starting time
	:param y0: initial condition for system to plot
	:param F: The system to plot
	"""
	solver = solver_cts(t0, y0, F)
	# res = np.array(solver.n_steps(10000, 1e-1))
	# ax.plot(res[:, 0], res[:, 1])
	N = 100
	h = 0.1  # 4 / N
	x = np.linspace(0, 10, N)
	resconv = solver.n_steps(N - 1, h)
	resconv = np.array(resconv)
	# resconv = solver.n_steps(N, h)
	# resconv = np.array(resconv)
	# print(solver_cts.__name__, " under threshold after ", i, " iterations.")
	ax.plot(x, resconv, label=f"Solveur : {solver_cts.method}")


def main_2d():
	minRange = -2
	maxRange = 2

	def example_func(t, y):
		return np.array([-y[1], y[0]])

	fig, axs = plt.subplots(2, 2)
	axs = axs.flatten()
	solvers = [EulerSolver, MidpointSolver, HeunSolver, RK4Solver]
	for i, solver in enumerate(solvers):
		axs[i].set_xlim((minRange, maxRange))
		axs[i].set_ylim((minRange, maxRange))
		axs[i].set_title(fr"""Champ des tangentes de ($E$) $(y_1'; y_2')$ = $(−y_2; y_1)$ (dimension 2)
et solutions avec $\mathtt{{{solver.__name__}}}$ de ($E$) pour différentes conditions initiales""")
		display_direction_field_2d(axs[i], example_func, minRange, maxRange, 25)
		for x in np.linspace(0.25, 1.75, 7):
			display_ode_solution_2d(axs[i], solver, 0, np.array([x, 0]), example_func)
		for x in np.linspace(0.25, 1.75, 7):
			axs[i].plot([x], [0], marker="o", markersize=4, markerfacecolor="red", markeredgecolor="red")
	manager = plt.get_current_fig_manager()
	manager.full_screen_toggle()
	plt.tight_layout()
	plt.legend()
	plt.show()


def main_2d_single():
	minRange = -2
	maxRange = 2

	def example_func(t, y):
		return np.array([-y[1], y[0]])

	actual_func = (lambda t: np.array([np.cos(t), np.sin(t)]))
	plt.xlim((-1.5, 1.5))
	plt.ylim((-1.5, 1.5))
	plt.xlabel('y1')
	plt.ylabel('y2')
	display_direction_field_2d(plt, example_func, minRange, maxRange, 25)
	display_ode_solution_2d(plt, EulerSolver, 0, np.array([1, 0]), example_func)
	display_ode_solution_2d(plt, MidpointSolver, 0, np.array([1, 0]), example_func)
	display_ode_solution_2d(plt, HeunSolver, 0, np.array([1, 0]), example_func)
	display_ode_solution_2d(plt, RK4Solver, 0, np.array([1, 0]), example_func)
	plt.plot([1], [0], marker="o", markersize=4, markerfacecolor="red", markeredgecolor="red")
	x = np.linspace(-2, 2, 100)
	plt.plot(actual_func(x)[0], actual_func(x)[1], label='solution exacte')
	plt.legend()
	plt.show()


def main_1d():
	minRange = 0
	maxRange = 10

	def example_func(t, y):
		return y / (1 + t ** 2)

	actual_func = (lambda x: np.exp(np.arctan(x)))
	plt.xlim((0, 10))
	plt.ylim((0, 6))
	plt.xlabel('t')
	plt.ylabel('y')
	display_direction_field_1d(plt, example_func, minRange, maxRange, 25)
	display_ode_solution_1d(plt, EulerSolver, 0, 1, example_func)
	display_ode_solution_1d(plt, MidpointSolver, 0, 1, example_func)
	display_ode_solution_1d(plt, HeunSolver, 0, 1, example_func)
	display_ode_solution_1d(plt, RK4Solver, 0, 1, example_func)
	x = np.linspace(0, 10, 100)
	plt.plot(x, actual_func(x), label='solution exacte')

	plt.legend()
	plt.show()


if __name__ == '__main__':
	print("executing ...")
	main_1d()
	main_2d_single()
	main_2d()
