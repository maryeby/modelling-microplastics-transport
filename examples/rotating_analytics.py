import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import scipy as scp

from transport_framework import particle as prt
from models import rotating_flow as fl
from models import rotating_system as ts

def main():
	"""
	This program computes analytical solutions for the Maxey-Riley equation for
	rigid body rotation, following the approach outlined in Candelier et al.
	(2004), and saves the results to the `data` directory.
	"""
	# initialize variables for the transport systems
	R = 2 / 3 * 0.75
	St = 2 / 3 * 0.3
	my_particle = prt.Particle(stokes_num=St)
	my_flow = fl.RotatingFlow()
	my_system = ts.RotatingTransportSystem(my_particle, my_flow, R)
	gamma = 1 / R - 1 / 2
	S = St / 2
	A_coeff = 1 / (S * (2 * gamma + 1))
	B = (3 * S - 1j) / (S * (2 * gamma + 1))
	C = -3 / ((2 * gamma + 1) * np.sqrt(np.pi * S))
	Y_1 = np.polynomial.polynomial.Polynomial([
			4 * A_coeff * B + C ** 2 * np.pi - C ** 2 * np.pi * B,
			-(1j * C ** 2 * np.pi + 4 * B), -A_coeff, 1]).roots().max()
	P_1 = 1 / 2 * (-C * np.sqrt(np.pi) - np.sqrt(C ** 2 * np.pi - 4 * A_coeff
																+ 4 * Y_1))
	P_2 = 1 / 2 * (-C * np.sqrt(np.pi) + np.sqrt(C ** 2 * np.pi - 4 * A_coeff
																+ 4 * Y_1))
	Q_1 = 1 / 2 * (Y_1 - np.sqrt(Y_1 ** 2 - 4 * B))
	Q_2 = 1 / 2 * (Y_1 + np.sqrt(Y_1 ** 2 - 4 * B))
	X = [(-P_1 + np.sqrt(P_1 ** 2 - 4 * Q_1)) / 2,
		 (-P_1 - np.sqrt(P_1 ** 2 - 4 * Q_1)) / 2,
		 (-P_2 + np.sqrt(P_2 ** 2 - 4 * Q_2)) / 2,
		 (-P_2 - np.sqrt(P_2 ** 2 - 4 * Q_2)) / 2]
#	X = np.roots([1, -C * np.sqrt(np.pi), A_coeff, 1j * C * np.sqrt(np.pi), B])
#	X = np.polynomial.polynomial.Polynomial([B, 1j * C * np.sqrt(np.pi),
#			A_coeff, -C * np.sqrt(np.pi), 1]).roots()

	# get time t from numerics
	numerics = pd.read_csv('data/rotating_numerics.csv')	
	t = numerics['t'][:]

	# compute asymptotics
	x_0, z_0 = 1, 0
	u_0, w_0 = my_flow.velocity(1, 0)
	Z_0, U_0 = x_0 + 1j * z_0, u_0 + 1j * w_0
	A = [0, 0, 0, 0]

	for i in range(4):
		numerator = U_0 * (X[i] ** 2 - C * np.sqrt(np.pi) * X[i]) - B * Z_0
		denominator = 1
		for j in range(4):
			if j != i:
				denominator *= X[i] - X[j]
		A[i] = numerator / denominator

	Z = 0
	for i in range(4):
		Z += A[i] / X[i] * np.exp(X[i] ** 2 * t) * scp.special.erfc(-X[i]
																* np.sqrt(t))

	x, z = np.real(Z), np.imag(Z)
	tol = 1e-16
	print('A3  ', np.abs(A[0] + A[1] + A[2] + A[3]) < tol)
	print('A5  ', np.abs(X[0] + X[1] + X[2] + X[3] - C * np.sqrt(np.pi)) < tol)
	print('A7  ', np.abs(X[0] * X[1] * X[2] * X[3] - B) < tol)
	print('A8  ', np.abs(A[0] / X[0] + A[1] / X[1] + A[2] / X[2] + A[3] / X[3]
			  - Z_0) < tol)
	print('A9  ', np.abs(A[0] * X[0] + A[1] * X[1] + A[2] * X[2] + A[3] * X[3]
			  - U_0) < tol)
	print('A10 ', np.abs(A[0] * X[0] ** 2 + A[1] * X[1] ** 2 + A[2] * X[2] ** 2
						  + A[3] * X[3] ** 2) < tol)

	# store results and write to a csv file
	analytics = pd.DataFrame({'t': t, 'x_ana': x, 'z_ana': z})
	analytics.to_csv('data/rotating_analytics.csv', index=False)

if __name__ == '__main__':
	main()
