import pandas as pd
import numpy as np
import scipy as scp
import warnings
from tqdm import tqdm

from utils.data_tools import extract_data, update_results
from transport_framework import particle as prt
from models import rotating_flow as fl
from models import rotating_system as ts
from examples.rigid_body_rotation.numerics import R, STOKES_HAT
from examples.rigid_body_rotation.numerics import OUT_FILE as IN_FILE

S = STOKES_HAT / 2			# pseudo-Stokes number from Candelier et al. (2004)
T_FINAL = 10				# total time

# coefficients from Candelier et al. (2004) eq (10)
X_0 = (1, 0)										# initial particle position
GAMMA = 3 / (2 * R) - 1 / 2
A_COEFF = 1 / (S * (2 * GAMMA + 1))
B = (3 * S - 1j) / (S * (2 * GAMMA + 1))
C = -3 / ((2 * GAMMA + 1) * np.sqrt(np.pi * S))
X = np.roots([1, -C * np.sqrt(np.pi), A_COEFF, 1j * C * np.sqrt(np.pi), B])

KEYS = ['t', 'x', 'z', 'history_x', 'history_z', 'delta_t']
OUT_FILE = '../data/rigid_body_rotation/analytics.csv'

def main():
	"""
	Compute analytical solutions for a rigid rotating body.

	Analytical solutions are computed following the approach outlined in [1]
	for varying timestep sizes, for the purposes of reproducing the solutions
	shown in [2] Figures 3 and 4. Results are saved to the
	`data/rigid_body_rotation` directory.
	"""
	# initialize objects
	particle = prt.Particle(STOKES_HAT)
	flow = fl.RotatingFlow()
	system = ts.RotatingTransportSystem(particle, flow, R)
	results = {key: [] for key in KEYS}

	# define the initial position and velocity of the particle
	ux_0, uz_0 = flow.velocity(X_0[0], X_0[1])
	z_0, u_0 = X_0[0] + 1j * X_0[1], ux_0 + 1j * uz_0

	# compute A as in Candelier et al. (2004) eq (A2)
	warnings.filterwarnings('ignore')
	a = [0, 0, 0, 0]
	for i in range(4):
		numerator = u_0 * (X[i] ** 2 - C * np.sqrt(np.pi) * X[i]) - B * z_0
		denominator = 1
		for j in range(4):
			if j != i:
				denominator *= X[i] - X[j]
		a[i] = numerator / denominator

	# compute analytical solutions for various timestep sizes
	timesteps = np.linspace(1e-3, 1e-1, 10)
	for delta_t in timesteps:
		t = np.arange(0, T_FINAL, delta_t)
		x, z, history_x, history_z = compute_analytics(flow, a, t)
		results = update_results(results, [t, x, z, history_x, history_z],
								[delta_t])
	# compute analytical solutions for delta_t = 1e-2
	numerics = pd.read_csv(IN_FILE)	
	t = extract_data('t', numerics, {'order': 1})
	delta_t = t[1] - t[0]
	x, z, history_x, history_z = compute_analytics(flow, a, t)
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

def compute_analytics(flow, a, t):
	"""
	Compute analytical solutions.

	Parameters
	----------
	flow : Flow (obj)
		The flow through which the particle is transported.
	a : list
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
	well as the formula for **H** from [2].
	
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
	# compute Z, U, and F as in Candelier et al. (2004) eqs (12), (A3), and (14)
	delta_t = t[1] - t[0]
	z, u, f = 0, 0, 0
	for i in range(len(a)):
		z += a[i] / X[i] * np.exp(X[i] ** 2 * t) \
				  * scp.special.erfc(-X[i] * np.sqrt(t))
		u += a[i] * X[i] * np.exp(X[i] ** 2 * t) \
				  * scp.special.erfc(-X[i] * np.sqrt(t))
		f += np.sqrt(np.pi) * (1j * a[i] / X[i] - a[i] * X[i]) * X[i] \
							* np.exp(X[i] ** 2 * t) \
							* scp.special.erfc(-X[i] * np.sqrt(t))
	f *= -C * (GAMMA + 1 / 2)
	x, z = np.real(z), np.imag(z)		# particle position
	v_x, v_z = np.real(u), np.imag(u)	# particle velocity
	u_x, u_z = flow.velocity(x, z, t)	# fluid velocity
	w_x, w_z = v_x - u_x, v_z - u_z		# relative velocity
	f_x, f_z = np.real(f), np.imag(f)	# history force

	# compute history force using the formula for H from Daitche (2013)
	h_x, h_z = [0] * t.size, [0] * t.size
	alpha = ts.compute_alpha(2)
	beta = ts.compute_beta(3, alpha[:, 1]) 
	gamma = ts.compute_gamma(t.size, beta[:, 2]) 
	xi = R / np.sqrt(STOKES_HAT) * np.sqrt(2 * delta_t / np.pi)

	for n in tqdm(range(t.size - 1)):
		for j in range(n + 1):
			h_x[n] += gamma[j, n] * w_x[n - j]
			h_z[n] += gamma[j, n] * w_z[n - j]
	h_x = np.array(h_x) * -xi
	h_z = np.array(h_z) * -xi
	history_x, history_z = np.gradient(h_x, t), np.gradient(h_z, t)
	return x, z, history_x, history_z

if __name__ == '__main__':
	main()
