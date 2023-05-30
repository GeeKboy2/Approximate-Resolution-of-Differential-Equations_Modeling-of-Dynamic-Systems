import random
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import ipywidgets as ipw


def aprox_period(Y, X):
    number_appearance = 0
    max_value = max(Y)
    times = []
    for i in range(len(Y)):
        # print(Y[i])
        if Y[i] > max_value-0.0009:
            number_appearance += 1
            times.append(X[i])

    if len(times) < 2:
        print("the list is too short to calculate a period")
        return -1
    total_time = times[-1]-times[0]
    return total_time/number_appearance


def derivative(X, t, a, b, c, d):
    x, y = X
    dotx = x * (a - b * y)
    doty = y * (-c + d * x)
    return np.array([dotx, doty])


def display_prey_predator(a, b, c, d, Nt=1000, tmax=30):
    t = np.linspace(0., tmax, Nt)
    X0 = [x0, y0]
    res = integrate.odeint(derivative, X0, t, args=(a, b, c, d))
    x, y = res.T
    print(max(y))
    print("The period is :", aprox_period(x, t), "days for preys")
    print("The period is :", aprox_period(y, t), "days for predators")
    plt.figure()
    plt.grid()
    plt.title("odeint method")
    plt.plot(t, x, 'b-', label='prey')
    plt.plot(t, y, 'r-', label="predator")
    plt.xlabel('Days')
    plt.ylabel('Number of animals')
    plt.legend()

    plt.show()


def display_prey_predators(a, b, c, d, Nt=1000, tmax=30):
    t = np.linspace(0., tmax, Nt)
    X0 = [x0, y0]
    b_s = np.arange(0.9, 1.4, 0.1)

    nums = np.random.random((10, len(b_s)))
    # generate the colors for each data set
    colors = cm.rainbow(np.linspace(0, 1, nums.shape[0]))

    fig, ax = plt.subplots(2, 1)

    for beta, i in zip(b_s, range(len(b_s))):
        res = integrate.odeint(derivative, X0, t, args=(a, beta, c, d))
        ax[0].plot(t, res[:, 0], color=colors[i],  linestyle='-',
                   label=r"$\beta = $" + "{0:.2f}".format(beta))
        ax[1].plot(t, res[:, 1], color=colors[i], linestyle='-',
                   label=r" $\beta = $" + "{0:.2f}".format(beta))
        ax[0].legend()
        ax[1].legend()

    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel('Days')
    ax[0].set_ylabel('Prey')
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel('Predator')

    plt.show()


def display_predator_prey_around_init(a, b, c, d, Nt=1000, tmax=30):
	t = np.linspace(0., tmax, Nt)
	X0 = [x0, y0]
	colors = cm.rainbow(np.linspace(0, 1, 6))
	for i in range(6):
		X0[1]-=0.2
		res = integrate.odeint(derivative, X0, t, args=(a, b, c, d))
		x, y = res.T
		plt.plot(X0[0],X0[1],"x",color=colors[i],label="start_point y0="+"{0:.2f}".format(X0[1]))
		plt.plot(x,y,color=colors[i])
	plt.xlabel("Prey")
	plt.ylabel("Predator")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	x0 = 4.
	y0 = 2.
	display_prey_predator(1, 2.2, 1, 0.2)
	#display_prey_predators(1,2.2,1,0.2)
	display_predator_prey_around_init(1,2.2,1,0.2)
