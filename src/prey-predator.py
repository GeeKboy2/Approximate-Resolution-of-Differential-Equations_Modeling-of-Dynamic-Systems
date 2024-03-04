from abc import ABC, abstractmethod
from ode_solver import *
from matplotlib import pyplot as plt
import numpy as np
from display_ode_solver import *
from pendulum_n_weights import is_local_maxima

def determine_period(t):
	"""
	Returns the period of the solutions
	:param t: array of solutions
	:return: array's period
	"""
	i_1 = 1
	while not (is_local_maxima(t, i_1)):
		i_1 += 1
	i_2 = i_1 + 1
	while not (is_local_maxima(t, i_2)):
		i_2 += 1
	return i_2 - i_1

def population_malthus(t0,y0,gamma,k,num_steps,step):
	"""
	Returns Malthus's solution
	:param t0: initial t
	:param y0: initial value for t0
	:param gamma: gamma
	:param k: not a parameter of this model (used for genericity)
	:param num_steps: number of iterations
	:param step: iteration's step
	:return: ode solution for Malthus's model
	"""
	f=lambda N,t:gamma*N
	solver=RK4Solver(t0,y0,f)
	return solver.n_steps(num_steps,step)

def population_verhulst(t0,y0,gamma,k,num_steps,step):
	"""
	Returns Verhulst's solution
	:param t0: initial t
	:param y0: initial value for t0
	:param gamma: gamma
	:param k: k
	:param num_steps: number of iterations
	:param step: iteration's step
	:return: ode solution for Malthus's model
	"""
	f=lambda N,t:gamma*N*(1-(N/k))
	solver=RK4Solver(t0,y0,f)
	return solver.n_steps(num_steps,step)

def display_1ode(system, k):
	"""
	Dislays Malthus or Verhulst solutions
	:param system: model function (either population_malthus/population_verhulst)
	:param k: k
	:return: None
	"""
	ic = [np.array([1]), np.array([50]), np.array([100])]

	for i in ic:
		li=system(0,i,1,k,num_steps=300,step=0.1)
		s = [x for x in li]
		x = np.linspace(0, 300, 301)
		plt.plot(x,s)
	plt.xlabel('temps')
	plt.ylabel('population')
	name = 'Malthus' if (system == population_malthus) else 'Verhulst'
	plt.title(f'Résolution du modèle de {name}')
	plt.grid()
	plt.show()

	return None

def prey_predator_function(a,b,c,d):
	"""
	Returns Lotka-Volterra model
	:param a: parameter a
	:param b: parameter b
	:param c: parameter c
	:param d: parameter d
	:return: ode for Lotka-Volterra model
	"""
	def inner_function(t,y):
		N=y[0]
		P=y[1]
		return np.array([N*(a-b*P),P*(c*N-d)])
	return inner_function

def prey_predator(t0,y0,a,b,c,d,num_steps,step):
	"""
	Returns Lotka-Volterra model
	:param t0: initial t
	:param y0: initial value for t0
	:param a: parameter a
	:param b: parameter b
	:param c: parameter c
	:param d: parameter d
	:param num_steps: number of iterations
	:param step: iteration's step
	:return: ode solution for Lotka-Volterra model
	"""
	solver=RK4Solver(t0,y0,prey_predator_function(a,b,c,d))
	return solver.n_steps(num_steps,step)

def display_prey_predator1d():
	"""
	Dislays 1D Lotka-Volterra solutions
	:return: None
	"""
	# Initials conditions:
	ic = [np.array([1, 1]), np.array([0.6, 0.8]), np.array([0.8, 0.6])]

	linestyle=['-','--',':']

	for idx, i in enumerate(ic):
		li=prey_predator(0,i,2/3,4/3,1,1,num_steps=300,step=0.1)
		s1 = [x[0] for x in li]
		s2 = [x[1] for x in li]
		x = np.linspace(0, 300, 301)
		plt.plot(x,s1, linestyle[idx],label=f'Évolution des proies (N(0) = {i[0]}; P(0) = {i[1]})')
		plt.plot(x,s2, linestyle[idx], label=f'Évolution des prédateurs (N(0) = {i[0]}; P(0) = {i[1]})')
		print(f"Période (proies) pour (N(0) = {i[0]}; P(0) = {i[1]}) : {determine_period(s1)}")
		print(f"Période (prédateurs) pour (N(0) = {i[0]}; P(0) = {i[1]}) : {determine_period(s2)}")

	plt.xlabel('temps')
	plt.ylabel('population')
	plt.xlim(0,300)
	plt.ylim(0,2)
	plt.legend()
	plt.grid()
	plt.show()

	return None

def display_prey_predator2d():
	"""
	Dislays 2D Lotka-Volterra solutions
	:return: None
	"""
	display_direction_field_2d(plt, prey_predator_function(2/3,4/3,1,1), 0, 3, 30)

	# Initials conditions:
	ic = [np.array([0.8, 0.8]), np.array([1, 1]), np.array([1.2, 1.2]), np.array([1.4, 1.4]), np.array([1.6, 1.6])]

	# Plotting
	for i in ic:
		li=prey_predator(0,i,2/3,4/3,1,1,num_steps=1000,step=0.1)
		s1 = [x[0] for x in li]
		s2 = [x[1] for x in li]
		plt.plot(s1,s2)
		plt.plot(i[0], i[1], marker='o', markersize="10", color='red')
	plt.xlim(0,3)
	plt.ylim(0,3)
	plt.xlabel('Proies')
	plt.ylabel('Prédateurs')
	plt.grid()
	plt.show()

	return None

def local_prey_predator2d():
	"""
	Dislays local 2D Lotka-Volterra solutions
	:return: None
	"""
	ic = [np.array([0.5, 0.5])]
	for i in range(15):
		ic.append(ic[-1] + np.array([0.02, 0.02]))
	ic.append(ic[0]- np.array([0.02, 0.02]))
	for i in range(14):
		ic.append(ic[-1] - np.array([0.02, 0.02]))
	for i in ic:
		li=prey_predator(0,i,2/3,4/3,1,1,num_steps=1000,step=0.1)
		s1 = [x[0] for x in li]
		s2 = [x[1] for x in li]
		plt.plot(s1,s2)
		plt.plot(i[0], i[1], marker='o', color='red')
	plt.xlim(0.25,0.6)
	plt.ylim(0.25,0.6)
	plt.xlabel('Proies')
	plt.ylabel('Prédateurs')
	plt.grid()
	plt.show()

	return None

if __name__ == '__main__':
	print("executing ...")

	display_1ode(population_malthus, 0)
	display_1ode(population_verhulst, 5)
	display_prey_predator1d()
	display_prey_predator2d()
	local_prey_predator2d()
