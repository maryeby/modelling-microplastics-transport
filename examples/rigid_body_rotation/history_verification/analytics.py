import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import scipy as scp
from tqdm import tqdm
from itertools import chain

from transport_framework import particle as prt
from models import rotating_flow as fl
from models import rotating_system as ts

def main():
	"""
	This program computes analytical solutions for the history force for
	rigid body rotation, following the approach outlined in Candelier et al.
	(2004), and saves the results to the `data/rigid_body_rotation` directory.
	"""
	# initialize translated variables
	R = 2 / 3 * 0.75
	St = 2 / 3 * 0.3		# Stokes number as defined in Haller & Sapsis (2008)
	S = St / 2				# pseudo-Stokes number from Candelier et al. (2004)
	gamma = 1 / R - 1 / 2
	delta_t = 1e-2
	t_final = 10

	# initialize the particle, flow, and transport system
	my_particle = prt.Particle(stokes_num=St)
	my_flow = fl.RotatingFlow()
	my_system = ts.RotatingTransportSystem(my_particle, my_flow, R)
	x_0, z_0 = 1, 0								# initial particle position
	u_0, w_0 = my_flow.velocity(1, 0)			# initial fluid velocity
	Z_0, U_0 = x_0 + 1j * z_0, u_0 + 1j * w_0	# initialize particle position Z
												# and particle velocity U = Zdot
	# initialize coeffs from Candelier et al. (2004) eq (10), compute roots of X
	A_coeff = 1 / (S * (2 * gamma + 1))
	B = (3 * S - 1j) / (S * (2 * gamma + 1))
	C = -3 / ((2 * gamma + 1) * np.sqrt(np.pi * S))
	X = np.roots([1, -C * np.sqrt(np.pi), A_coeff, 1j * C * np.sqrt(np.pi), B])

	# compute A as in Candelier et al. (2004) eq (A2)
	A = [0, 0, 0, 0]
	for i in range(4):
		numerator = U_0 * (X[i] ** 2 - C * np.sqrt(np.pi) * X[i]) - B * Z_0
		denominator = 1
		for j in range(4):
			if j != i:
				denominator *= X[i] - X[j]
		A[i] = numerator / denominator

	# compute Z, U, and F as in Candelier et al. (2004) eqs (12), (A3), and (14)
	t = np.arange(0, t_final, delta_t)
	Z, U, F = 0, 0, 0
	for i in range(4):
		Z += A[i] / X[i] * np.exp(X[i] ** 2 * t) \
				  * scp.special.erfc(-X[i] * np.sqrt(t))
		U += A[i] * X[i] * np.exp(X[i] ** 2 * t) \
				  * scp.special.erfc(-X[i] * np.sqrt(t))
		F += np.sqrt(np.pi) * (1j * A[i] / X[i] - A[i] * X[i]) * X[i] \
							* np.exp(X[i] ** 2 * t) \
							* scp.special.erfc(-X[i] * np.sqrt(t))
	F *= -C * R * (gamma + 1 / 2)
	x, z = np.real(Z), np.imag(Z)			# particle position
	v_x, v_z = np.real(U), np.imag(U)		# particle velocity
	u_x, u_z = my_flow.velocity(x, z, t)	# fluid velocity
	w_x, w_z = v_x - u_x, v_z - u_z			# relative velocity
	F_x, F_z = np.real(F), np.imag(F)		# history force

	# compute history force using the formula for H from Daitche (2013)
	H_x, H_z = [0] * t.size, [0] * t.size
	alpha = ts.compute_alpha(2)
	beta = ts.compute_beta(3, alpha[:, 1])
	gamma = ts.compute_gamma(t.size, beta[:, 2])
	xi = np.sqrt((9 * delta_t) / (2 * np.pi)) * (R / np.sqrt(St))

	for n in tqdm(range(t.size - 1)):
		for j in range(n + 1):
			H_x[n] += gamma[j, n] * w_x[n - j]
			H_z[n] += gamma[j, n] * w_z[n - j]
	H_x = np.array(H_x) * -xi
	H_z = np.array(H_z) * -xi
	history_x, history_z = np.gradient(H_x, t), np.gradient(H_z, t)

	# store results and write to csv files
	data_path = '../../data/rigid_body_rotation/'
	analytics = pd.DataFrame({'t': t, 'x': x, 'z': z, 'v_x': v_x, 'v_z': v_z,
							  'u_x': u_x, 'u_z': u_z, 'w_x': w_x, 'w_z': w_z,
							  'H_x': H_x, 'H_z': H_z,
							  'H\'_x': history_x, 'H\'_z': history_z,
							  'F_x': F_x, 'F_z': F_z})
	analytics.to_csv(data_path + 'analytical_history.csv', index=False)

if __name__ == '__main__':
	main()
