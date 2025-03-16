import pandas as pd
import numpy as np
import scipy as scp
from tqdm import tqdm

from utils.data_tools import extract_data, update_results
from transport_framework import particle as prt
from models import rotating_flow as fl
from models import rotating_system as ts
from examples.rigid_body_rotation.numerics import R, STOKES_NUM

S = STOKES_NUM / 2			# pseudo-Stokes number from Candelier et al. (2004)
T_FINAL = 10				# total time

# coefficients from Candelier et al. (2004) eq (10)
GAMMA = 1 / R - 1 / 2
A_COEFF = 1 / (S * (2 * GAMMA + 1))
B = (3 * S - 1j) / (S * (2 * GAMMA + 1))
C = -3 / ((2 * GAMMA + 1) * np.sqrt(np.pi * S))
X = np.roots([1, -C * np.sqrt(np.pi), A_COEFF, 1j * C * np.sqrt(np.pi), B])

IN_FILE = '../data/rigid_body_rotation/numerics.csv'
OUT_FILE = '../data/rigid_body_rotation/analytics.csv'

def main():
	"""
	Compute analytical solutions[^1] for a rigid rotating body.

	Analytical solutions are computed following the approach outlined in [1]
	for varying timestep sizes, for the purposes of reproducing the solutions
	shown in [2] Figures 3 and 4. Results are saved to the
	`data/rigid_body_rotation` directory.
	
	References
	----------
	[^1]: [F. Candelier et al. (2004).](https://doi.org/10.1063/1.1689970)
		  On the effect of the Boussinesq–Basset force on the radial migration
		  of a Stokes particle in a vortex. *Physics of Fluids* 16(5),
		  1765–1776.
	[^2]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# create dictionary to store results
	keys = ['t', 'x', 'z', 'history_x', 'history_z', 'delta_t']
	results = {key: [] for key in keys}

	# initialize the Particle, Flow, and TransportSystem objects
	particle = prt.Particle(STOKES_NUM)
	flow = fl.RotatingFlow()
	system = ts.RotatingTransportSystem(particle, flow, R)

	# define the initial position and velocity of the particle
	x_0, z_0 = 1, 0
	u_0, w_0 = flow.velocity(x_0, z_0)
	Z_0, U_0 = x_0 + 1j * z_0, u_0 + 1j * w_0

	# compute A as in Candelier et al. (2004) eq (A2)
	A = [0, 0, 0, 0]
	for i in range(4):
		numerator = U_0 * (X[i] ** 2 - C * np.sqrt(np.pi) * X[i]) - B * Z_0
		denominator = 1
		for j in range(4):
			if j != i:
				denominator *= X[i] - X[j]
		A[i] = numerator / denominator

	# compute analytical solutions for various timestep sizes
	timesteps = np.linspace(1e-3, 1e-1, 10)
	for delta_t in timesteps:
		t = np.arange(0, T_FINAL, delta_t)
		x, z, history_x, history_z = compute_analytics(flow, A, t)
		results = update_results(results, [t, x, z, history_x, history_z],
								[delta_t])
	# compute analytical solutions for delta_t = 1e-2
	numerics = pd.read_csv(IN_FILE)	
	t = extract_data('t', numerics, {'order': 1})
	delta_t = t[1] - t[0]
	x, z, history_x, history_z = compute_analytics(flow, A, t)
	results = update_results(results, [t, x, z, history_x, history_z],
							[delta_t])

	# get integer times and the particle position at each integer time	
	int_indices = np.where(t == t.astype(int))
	int_indices = int_indices[0][:21]
	t = t[int_indices]
	x = x[int_indices]
	z = z[int_indices]
	history_x = history_x[int_indices]
	history_z = history_z[int_indices]
	results = update_results(results, [t, x, z, history_x, history_z],
							[delta_t])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def compute_analytics(flow, A, t):
	"""
	Compute analytical solutions[^1].

	Parameters
	----------
	flow : Flow (obj)
		The flow through which the particle is transported.
	A : list
		A list of `float` elements, computed with eq (A2)[^1].
	t : ndarray
		1D array containing `float` time series data.

	Returns
	-------
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	history_x : ndarray
		1D array of `float` data, the horizontal history force.
	history_z : ndarray
		1D array of `float` data, the vertical history force.

	Notes
	-----
	Relevant equations for this function are (12), (A3), and (14) from [1], as
	well as the formula for H from [2].
	
	References
	----------
	[^1]: [F. Candelier et al. (2004).](https://doi.org/10.1063/1.1689970)
		  On the effect of the Boussinesq–Basset force on the radial migration
		  of a Stokes particle in a vortex. *Physics of Fluids* 16(5),
		  1765–1776.
	[^2]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	delta_t = t[1] - t[0]
	Z, U, F = 0, 0, 0
	# compute Z, U, and F as in Candelier et al. (2004) eqs (12), (A3), and (14)
	for i in range(len(A)):
		Z += A[i] / X[i] * np.exp(X[i] ** 2 * t) \
				  * scp.special.erfc(-X[i] * np.sqrt(t))
		U += A[i] * X[i] * np.exp(X[i] ** 2 * t) \
				  * scp.special.erfc(-X[i] * np.sqrt(t))
		F += np.sqrt(np.pi) * (1j * A[i] / X[i] - A[i] * X[i]) * X[i] \
							* np.exp(X[i] ** 2 * t) \
							* scp.special.erfc(-X[i] * np.sqrt(t))
	F *= -C * R * (GAMMA + 1 / 2)
	x, z = np.real(Z), np.imag(Z)		# particle position
	v_x, v_z = np.real(U), np.imag(U)	# particle velocity
	u_x, u_z = flow.velocity(x, z, t)	# fluid velocity
	w_x, w_z = v_x - u_x, v_z - u_z		# relative velocity
	F_x, F_z = np.real(F), np.imag(F)	# history force

	# compute history force using the formula for H from Daitche (2013)
	H_x, H_z = [0] * t.size, [0] * t.size
	alpha = ts.compute_alpha(2)
	beta = ts.compute_beta(3, alpha[:, 1]) 
	gamma = ts.compute_gamma(t.size, beta[:, 2]) 
	xi = np.sqrt((9 * delta_t) / (2 * np.pi)) * (R / np.sqrt(STOKES_NUM))

	for n in tqdm(range(t.size - 1)):
		for j in range(n + 1):
			H_x[n] += gamma[j, n] * w_x[n - j]
			H_z[n] += gamma[j, n] * w_z[n - j]
	H_x = np.array(H_x) * -xi
	H_z = np.array(H_z) * -xi
	history_x, history_z = np.gradient(H_x, t), np.gradient(H_z, t)
	return x, z, history_x, history_z

if __name__ == '__main__':
	main()
