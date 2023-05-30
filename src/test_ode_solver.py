from ode_solver import *
import numpy as np

def generic_solver(f, fsol, t, y0):
	euler = EulerSolver(t, y0, f).n_steps(99, 0.1)
	middle_point = MidpointSolver(t, y0, f).n_steps(99, 0.1)
	heun = HeunSolver(t, y0, f).n_steps(99, 0.1)
	rk4 = RK4Solver(t, y0, f).n_steps(99, 0.1)

	t = np.linspace(0,10,100)
	for i in range(len(t)):
		assert (np.isclose(euler[i], fsol(t[i]), atol=1e-01).all)
		assert (np.isclose(middle_point[i], fsol(t[i]), atol=1e-01).all)
		assert (np.isclose(heun[i], fsol(t[i]), atol=1e-01).all)
		assert (np.isclose(rk4[i], fsol(t[i]), atol=1e-01).all)

	return None

def test_one_dimension():
	def f(t, y):
		return y / (1 + t**2)
	fsol = (lambda x : np.exp(np.arctan(x)))

	print("Test de f (1 dimension, 100 values)...", end="")
	generic_solver(f, fsol, 0, 1)
	print("\tOK")
	
	return None

def test_two_dimension():
	def f(t, y):
		return np.array([-y[1], y[0]])
	fsol = (lambda t : np.array([ np.cos(t), np.sin(t)]))

	print("Test de f (2 dimensions, 100 values)...", end="")
	generic_solver(f, fsol, 0, np.array([1, 0]))
	print("\tOK")

	return None

if __name__ == '__main__':
	print("=== Test ODE solver ===")
	test_one_dimension()
	test_two_dimension()
