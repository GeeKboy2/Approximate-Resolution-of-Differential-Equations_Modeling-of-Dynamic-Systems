from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg


# def inf_norm(y, ys):
# 	y = np.array(list(y[i] for i in range(1, len(y))))
# 	ys = np.array(list(ys[2*i-1] for i in range(0, len(y))))
# 	return linalg.norm(y-ys)

def inf_norm(y, ys):
	m = linalg.norm(y[0] - ys[0])
	for i in range(len(y)):
		m = max(m, linalg.norm(y[i] - ys[2*i]))
	return m


class OdeSolver(ABC):
	"""
	A solver for ODE.
	"""
	def __init__(self, t0, y0, f) -> None:
		"""
		Constructs a new solver for an ODE
		:param t0: starting time
		:param y0: initial conditions
		:param f: The ODE to solve
		"""
		self.t0 = t0
		self.y0 = y0
		self.f = f

		""" Complex elements of the system"""
		self.y_tab = [y0]
		self.yprime = lambda y: (lambda t: f(y(t), t))

	@abstractmethod
	def step(self, y, t, h):
		"""
		Computes a single step of the ODE solver
		:param y: Previous y value
		:param t: Previous t value
		:param h: Step size
		:return: new y value
		"""
		pass

	def n_steps(self, N, h):
		"""
		Computes N steps of the ODE from t0 to N*h.
		:param N: number of steps
		:param h: step size
		:return: The solution of the ODE
		"""
		y = self.y_tab[-1]
		y_tab = self.y_tab.copy()
		t = self.t0
		for i in range(N):
			y = self.step(y, t, h)
			t += h
			y_tab.append(y)
		self.y_tab = y_tab
		return self.y_tab
	
	def reset_y(self):
		"""
		Resets the internal solution array
		"""
		self.y_tab = [self.y0]

	def epsilon_solve(self, tf, eps):
		"""
		Solves the ODE with a given precision for t in [t0; tf]
		:param tf: Final time
		:param eps: Precision
		:return: The solution of the ODE
		"""
		N = 2
		k = 0
		t = 0
		h = tf / N
		y = self.n_steps(N, h).copy()
		self.reset_y()
		N *= 2
		ys = self.n_steps(N, h).copy()
		self.reset_y()
		i = 0
		while inf_norm(y, ys) > eps:
			t = k
			N *= 2
			h = tf / N
			y = ys.copy()
			self.reset_y()
			ys = self.n_steps(N, h).copy()
			i += 1
		self.y_tab = ys
		return ys, i


class EulerSolver(OdeSolver):
	"""
	The ODE solver using Euler's method
	"""
	method = "Euler"

	def step(self, y, t, h):
		return y + h * self.f(t, y)


class HeunSolver(OdeSolver):
	"""
	The ODE solver using Heun's method
	"""
	method = "Heun"

	def step(self, y, t, h):
		p1 = self.f(t, y)
		y2 = y + h * p1
		p2 = self.f(t + h, y2)
		return y + h * (p1 + p2) / 2


class RK4Solver(OdeSolver):
	"""
	The ODE solver using RK4's method
	"""
	method = "Runge Kutta 4"

	def step(self, y, t, h):
		k1 = self.f(t, y)
		k2 = self.f(t + h / 2, y + (h * k1 / 2))
		k3 = self.f(t + h / 2, y + (h * k2 / 2))
		k4 = self.f(t + h, y + h * k3)
		return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class MidpointSolver(OdeSolver):
	"""
	The ODE solver using the Midpoints method
	"""
	method = "Midpoint"

	def step(self, y, t, h):
		y_half = y + h / 2 * self.f(t, y)
		return y + h * self.f(t + h / 2, y_half)
