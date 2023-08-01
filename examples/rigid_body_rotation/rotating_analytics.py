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
	This program computes analytical solutions for the Maxey-Riley equation for
	rigid body rotation, following the approach outlined in Candelier et al.
	(2004), and saves the results to the `data` directory.
	"""
	# initialize translated variables
	R = 2 / 3 * 0.75
	St = 2 / 3 * 0.3		# Stokes number as defined in Haller & Sapsis (2008)
	S = St / 2				# pseudo-Stokes number from Candelier et al. (2004)
	gamma = 1 / R - 1 / 2

	# initialize the particle, flow, and transport system
	my_particle = prt.Particle(stokes_num=St)
	my_flow = fl.RotatingFlow()
	my_system = ts.RotatingTransportSystem(my_particle, my_flow, R)
	x_0, z_0 = 1, 0								# initial particle position
	u_0, w_0 = my_flow.velocity(1, 0)			# initial fluid velocity
	Z_0, U_0 = x_0 + 1j * z_0, u_0 + 1j * w_0

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

	# compute asymptotics for various delta_t's
	timesteps = np.linspace(1e-3, 1e-1, 10)
	my_dict = dict.fromkeys(list(chain.from_iterable(('x_%.2e' % delta_t,
													  'z_%.2e' % delta_t) 
							for delta_t in timesteps)))
	for delta_t in tqdm(timesteps):
		x_label = 'x_%.2e' % delta_t
		z_label = 'z_%.2e' % delta_t
		t = np.arange(0, 10 + delta_t, delta_t)
		Z = 0
		for i in range(4):
			Z += A[i] / X[i] * np.exp(X[i] ** 2 * t) \
					  * scp.special.erfc(-X[i] * np.sqrt(t))
		x, z = np.real(Z), np.imag(Z)
		my_dict[x_label] = x
		my_dict[z_label] = z

	# compute asymptotics for the delta_t = 1e-2 case
	numerics = pd.read_csv('../data/rotating_numerics.csv')	
	t = numerics['t'][:]
	Z = 0
	for i in range(4):
		Z += A[i] / X[i] * np.exp(X[i] ** 2 * t) * scp.special.erfc(-X[i]
																* np.sqrt(t))
	x, z = np.real(Z), np.imag(Z)

	# get integer times and the particle position at each integer time	
	int_indices = np.where(t == t.astype(int))
	int_x = np.take(x, int_indices[0])
	int_z = np.take(z, int_indices[0])

	# store results and write to csv files
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	varying_timesteps = pd.DataFrame(my_dict)
	varying_timesteps.to_csv('../data/rotating_analytics_varying_timesteps.csv',
							 index=False)
	analytics = pd.DataFrame({'t': t, 'x_ana': x, 'z_ana': z})
	analytics.to_csv('../data/rotating_analytics.csv', index=False)
	int_times = pd.DataFrame({'int_x': int_x, 'int_z': int_z})
	int_times.to_csv('../data/rotating_int_times.csv', index=False)

if __name__ == '__main__':
	main()
