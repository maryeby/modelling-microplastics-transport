import warnings
import itertools
import pandas as pd
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks, correlate, correlation_lags

from transport_framework import particle as prt
from models import quiescent_flow as qfl
from models import rotating_flow as rfl
from models import deep_linear_wave as dfl
from models import linear_wave as lfl
from models import stokes_wave as sfl
from models import bichromatic_wave as bfl
from models import my_system as ts
from models import haller_system as hs
from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data, match_data
from utils.colors import COLORS, print_success, print_failure

SCALE = 2 / 3 # scale to use for parameter translation
IN_FILE1 = 'examples/data/relaxing_particle/prasath_fig4.csv'
IN_FILE2 = 'examples/data/rigid_body_rotation/daitche_fig3.csv'
IN_FILE3 = 'examples/data/deep_linear_wave/santamaria_fig1.csv'
IN_FILE4 = 'examples/data/deep_linear_wave/cathals_sm_fig1_recreation.csv'
IN_FILE5 = 'examples/data/bichromatic_wave/swash_flat_seabed_data.mat'
IN_FILE6 = 'examples/data/bichromatic_wave/swash_sloped_seabed_data.mat'
IN_FILE7 = 'examples/data/bichromatic_wave/sloped_seabed_coords.mat'
PLOT_IF_SUCCESSFUL = True

def main():
	"""Test various aspects of our model for different flows."""
	warnings.filterwarnings('ignore')
	test_relaxation()
	test_rotation()
	test_tracers()
	test_buoyancy()
	test_boundaries()
	test_stokes_wave()
	test_flat_seabed()
	test_subharmonics()
	if PLOT_IF_SUCCESSFUL: plt.show()

def test_relaxation():
	"""Run tests for a relaxing particle in a quiescent fluid."""
	print('CASE 1: RELAXING PARTICLE')
	success = True
	data = pd.read_csv(IN_FILE1)			# read data
	flow = qfl.QuiescentFlow()				# quiescent flow object
	xdots, ts, asymptotics = [], [], []		# empty lists to store results
	betas = [0.01, 1, 5]					# density ratios

	# run and test a simulation for each beta with & without history effects
	for beta, history in itertools.product(betas, [True, False]):
		params = {'beta': beta, 'history': history, 'asymptotic': False}
		h_str = 'with history' if history else 'without history'

		# run relaxing particle sims, store results, get extracted data
		print('Simulating a relaxing particle for beta =', beta, h_str, '...')
		sols = simulate(flow, beta, history)
		xdot, t = sols[2], sols[4]
		xdots.append(xdot)
		ts.append(t)
		extracted_xdot, extracted_t = extract_data(['xdot', 't'], data, params)

		if history:
			# compute asymptotic results and get extracted asymptotic data
			print('Computing asymptotic results...', end='')
			asymptotics.append(relaxing_asymptotics(beta, t[1:]))
			print('done.')
			params['asymptotic'] = True
			extracted_asymptotics = extract_data('xdot', data, params)

			# compare asymptotic results to the extracted data
			print('\nComparing solutions to extracted data from Prasath ',
				  'et al. 2019) Figure 4...')
			if match_data(asymptotics[-1], extracted_asymptotics):
				print_success('Asymptotic solutions match')
			else:
				print_failure('Asymptotic solutions do not match')
				success = False
		else:
			print('\nComparing solutions to extracted data from Prasath ',
				  'et al. (2019) Figure 4...')

		# compare the numerical results to the extracted data
		if match_data(xdot, extracted_xdot):
			print_success(f'Numerical solutions match {h_str}\n')
		else:
			print_failure(f'Numerical solutions do not match {h_str}\n')
			success = False
	# plot the data and show immediately if there were any failures
	plot_relaxing_case(betas, data, xdots, ts, asymptotics)
	if not success:
		plt.show()
		quit()
	print()

def relaxing_asymptotics(beta, t):
	r"""
	Compute the leading order asymptotic behavior[^1] of the particle velocity.

	Parameters
	----------
	beta : float
		The ratio between the particle and fluid densities.
	t : float or ndarray
		Float or 1D array containing `float` time series data.

	Returns
	-------
	float or ndarray
		The asymptotic particle velocity.

	Notes
	-----
	The computation is based on eq (4.7) from Ref. 1,
	$$q^{(2)}(0, t) \approx c(\alpha, \gamma)
		- \frac{\sigma \gamma}{\alpha^2 \sqrt{\pi t}}
		+ \mathcal{O}(t^{3 / 2}),$$
	with a sign change on the singular term.

	References
	----------
	[^1]: [S. G. Prasath et al. (2019).](https://doi.org/10.1017/jfm.2019.194)
		  Accurate solution method for the Maxey–Riley equation, and the
		  effects of Basset history. *Journal of Fluid Mechanics* 868, 428–460.
	"""
	density_ratio = 2 / (1 + 2 * beta)	# parameter translation
	stokes_num = SCALE
	alpha = density_ratio / stokes_num
	gamma = 3 / 2 * density_ratio * np.sqrt(2 / stokes_num)
	return 1 / (np.sqrt(np.pi) * t ** (3 / 2)) * (gamma / (2 * alpha ** 2))

def plot_relaxing_case(betas, data, xdots, ts, asymptotics):
	"""
	Plot a recreation of Figure 4 from Ref. 1.

	Parameters
	----------
	betas : list
		A list of `float` elements, the density ratios.
	data : DataFrame
		A `DataFrame` containing extracted data from Figure 4 in [1].
	xdots : list
		A list of `ndarray`s, the horizontal particle velocities.
	ts : list
		A list of `ndarray`s, the time series data.
	asymptotics : list
		A list of `ndarray`s, the asymptotic horizontal particle velocities.

	References
	----------
	[^1]: [S. G. Prasath et al. (2019).](https://doi.org/10.1017/jfm.2019.194)
		  Accurate solution method for the Maxey–Riley equation, and the
		  effects of Basset history. *Journal of Fluid Mechanics* 868, 428–460.
	"""
	fig(r'$t$', r'$\dot{x}$', lims=[0, 14.5, 1e-5, 1e1], y_scale='log')
	plt.title('case 1: relaxing particle')
	for i in range(len(betas)):
		# get extracted data
		names = ['xdot', 't']
		params = {'beta': betas[i], 'history': True, 'asymptotic': False}
		ext_xdot_history, ext_t_history = extract_data(names, data, params)
		params['asymptotic'] = True
		ext_asym_xdot, ext_asym_t = extract_data(names, data, params)
		params['history'] = False
		params['asymptotic'] = False
		ext_xdot, ext_t = extract_data(names, data, params)

		# plot extracted data
		plt.plot(ext_t_history, ext_xdot_history, c=COLORS[-1], lw=4)
		plt.plot(ext_t, ext_xdot, c=COLORS[-1], ls='--', lw=4)
		plt.plot(ext_asym_t, ext_asym_xdot, c=COLORS[-1], ls=':', lw=4)

		# plot numerical and asymptotic results
		plt.plot(ts[i * 2], xdots[i * 2], c=COLORS[i],
				 label=r'$\beta =$' + str(betas[i]))
		plt.plot(ts[i * 2 + 1], xdots[i * 2 + 1], ls='--', c=COLORS[i])
		plt.plot(ts[i * 2][1:], asymptotics[i], ls=':', c=COLORS[i])

def test_rotation():
	"""Run tests on a rotating rigid body."""
	print('CASE 2: RIGID BODY ROTATION')
	data = pd.read_csv(IN_FILE2)			# read data
	flow = rfl.RotatingFlow()				# rotating flow object
	r = 0.75								# density ratio
	x_0 = (1, 0)							# initial particle position
	history = True

	# run rigid body rotation simulations
	print('Simulating first order rigid body rotation...')
	x, z = simulate(flow, r, history, x_0, order=1)[:2]
	x1 = np.array((x, z)).T
	print('\nSimulating second order rigid body rotation...')
	x, z = simulate(flow, r, history, x_0, order=2)[:2]
	x2 = np.array((x, z)).T
	print('\nSimulating third order rigid body rotation...')
	sols = simulate(flow, r, history, x_0)
	x, z, t = sols[0], sols[1], sols[4]
	history_x, history_z = sols[-2:]
	x3 = np.array((x, z)).T

	# variables to help slice data for plotting
	history_x = history_x[:-3]
	history_z = history_z[:-3]
	n = t.size // 5
	m = n // 100 + 1

	# compute and store analytical solutions
	print('\nComputing analytical results...')
	x, z, x_int, z_int, exact_hx, exact_hz, analytical, \
	   delta_ts = rotating_analytics(t)
	exact_hx, exact_hz = exact_hx[:-3], exact_hz[:-3]
	exact = np.array((x, z)).T
	x_int = np.array((x_int[:m], z_int[:m]))

	# compute relative error
	print('Computing relative error...', end='')
	e_rel1 = np.linalg.norm(exact - x1, axis=1) / np.linalg.norm(exact, axis=1)
	e_rel2 = np.linalg.norm(exact - x2, axis=1) / np.linalg.norm(exact, axis=1)
	e_rel3 = np.linalg.norm(exact - x3, axis=1) / np.linalg.norm(exact, axis=1)
	print('done.')

	# get extracted data from Daitche (2013) Fig 3
	names = ['first_x', 'first_z', 'exact_x', 'exact_z', 'rel_error1',
			 'rel_error2', 'rel_error3', 't1', 't2', 't3']
	ext_x, ext_z, ext_x_analytical,	ext_z_analytical, ext_e_rel1, ext_e_rel2, \
		   ext_e_rel3, ext_t1, ext_t2, ext_t3 = extract_data(names, data)

	# compare our numerical results to the data extracted from Daitche
	print('\nComparing numerical results to extracted data from Daitche ',
		  '(2013) Figure 3...')
	if match_data(x1[:, 0], ext_x) and match_data(x1[:, 1], ext_z):
		print_success('First order numerical solutions match')
	else:
		print_failure('First order numerical solutions do not match')
		plot_rotating_trajectory(data, exact[:n], x_int, x1[:n])
		plt.legend()
		plt.show()
		quit()

	# compare error analysis to the extracted data
	print('Comparing relative error to extracted data from Daitche (2013)',
		  'Figure 3...')
	if match_data(e_rel1, ext_e_rel1):
		print_success('First order error analysis matches')
	else:
		print_failure('First order error analysis does not match')
		plot_error_analysis(data, t, e_rel1)
		plot_rotating_trajectory(data, exact[:n], x_int, x1[:n])
		plt.legend()
		plt.show()
		quit()
	if match_data(e_rel2, ext_e_rel2):
		print_success('Second order error analysis matches')
	else:
		print_failure('Second order error analysis does not match')
		plot_error_analysis(data, t, e_rel1, e_rel2)
		plot_rotating_trajectory(data, exact[:n], x_int, x1[:n], x2[:n])
		plt.legend()
		plt.show()
		quit()
	plot_error_analysis(data, t, e_rel1, e_rel2, e_rel3)
	plot_rotating_trajectory(data, exact[:n], x_int, x1[:n], x2[:n], x3[:n])
	plt.legend()
	if match_data(e_rel3, ext_e_rel3):
		print_success('Third order error analysis matches')
	else:
		print_failure('Third order error analysis does not match')
		plt.show()
		quit()

	# compare analytical solutions for history force to numerical solutions
	print('Comparing numerical history to analytical history...')
	if match_data(exact_hx, history_x):
		print_success('History force verified in the x direction')
	else:
		print_failure('History force not verified in the x direction')
		plot_history(t[:-3], exact_hx, exact_hz, history_x, history_z)
		plt.show()
		quit()
	plot_history(t[:-3], exact_hx, exact_hz, history_x, history_z)
	if match_data(exact_hz, history_z):
		print_success('History force verified in the z direction')
		print('\n')
	else:
		print_failure('History force not verified in the z direction')
		plt.show()
		quit()

def rotating_analytics(t):
	"""
	Return analytical solutions[^2] for rigid body rotation.
	
	Parameters
	----------
	t : ndarray
		1D array containing `float` time series data.

	Returns
	-------
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	int_x : ndarray
		1D array of `float` data, the horizontal particle positions at `int` t.
	int_z : ndarray
		1D array of `float` data, the vertical particle positions at `int` t.
	history_x : ndarray
		1D array of `float` data, the horizontal history force.
	history_z : ndarray
		1D array of `float` data, the vertical history force.
	results : dict
		The `x` and `z` solutions for various timestep sizes.
	timesteps : ndarray
		An array of `float` timestep sizes used to evaluate the analytics.

	References
	----------
	[^2]: [F. Candelier et al. (2004).](https://doi.org/10.1063/1.1689970)
		  On the effect of the Boussinesq–Basset force on the radial migration
		  of a Stokes particle in a vortex. *Physics of Fluids* 16(5),
		  1765–1776.
	"""
	# translated parameters
	density_ratio = SCALE * 0.75
	stokes_num = SCALE * 0.3	# St as defined in Haller & Sapsis (2008)
	s = stokes_num / 2			# pseudo-Stokes num from Candelier et al. (2004)
	gamma = 1 / density_ratio - 1 / 2

	flow = rfl.RotatingFlow()
	xp_0, zp_0 = 1, 0						# initial particle position
	ux_0, uz_0 = flow.velocity(xp_0, zp_0)	# initial fluid velocity
	z_0, u_0 = xp_0 + 1j * zp_0, ux_0 + 1j * uz_0

	# initialize coeffs from Candelier et al. (2004) eq (10), compute roots of X
	a_coeff = 1 / (s * (2 * gamma + 1))
	b = (3 * s - 1j) / (s * (2 * gamma + 1))
	c = -3 / ((2 * gamma + 1) * np.sqrt(np.pi * s))
	x = np.roots([1, -c * np.sqrt(np.pi), a_coeff, 1j * c * np.sqrt(np.pi), b])

	# compute A as in Candelier et al. (2004) eq (A2)
	a = [0, 0, 0, 0]
	for i in range(4):
		numerator = u_0 * (x[i] ** 2 - c * np.sqrt(np.pi) * x[i]) - b * z_0
		denominator = 1
		for j in range(4):
			if j != i:
				denominator *= x[i] - x[j]
		a[i] = numerator / denominator
	
	# compute analytical solutions for various delta_t's
	timesteps = np.linspace(1e-3, 1e-1, 10)
	results = dict.fromkeys(list(itertools.chain.from_iterable(
						   ('x_%.2e' % delta_t, 'z_%.2e' % delta_t)
							for delta_t in timesteps)))
	for delta_t in tqdm(timesteps):
		x_label = 'x_%.2e' % delta_t
		z_label = 'z_%.2e' % delta_t
		analytical_t = np.arange(0, 10 + delta_t, delta_t)
		z = 0
		for i in range(4):
			z += a[i] / x[i] * np.exp(x[i] ** 2 * analytical_t) \
					  * scp.special.erfc(-x[i] * np.sqrt(analytical_t))
		results[x_label] = np.real(z)
		results[z_label] = np.imag(z)

	# compute Z, U, and F as in Candelier et al. (2004) eqs (12), (A3), and (14)
	delta_t = 1e-2
	z, u, f = 0, 0, 0
	for i in range(len(a)):
		z += a[i] / x[i] * np.exp(x[i] ** 2 * t) * scp.special.erfc(-x[i]
																* np.sqrt(t))
		u += a[i] * x[i] * np.exp(x[i] ** 2 * t) \
				  * scp.special.erfc(-x[i] * np.sqrt(t))
		f += np.sqrt(np.pi) * (1j * a[i] / x[i] - a[i] * x[i]) * x[i] \
							* np.exp(x[i] ** 2 * t) \
							* scp.special.erfc(-x[i] * np.sqrt(t))
	f *= -c * density_ratio * (gamma + 1 / 2)
	x, z = np.real(z), np.imag(z)		# particle position
	v_x, v_z = np.real(u), np.imag(u)   # particle velocity
	u_x, u_z = flow.velocity(x, z, t)   # fluid velocity
	w_x, w_z = v_x - u_x, v_z - u_z		# relative velocity
	f_x, f_z = np.real(f), np.imag(f)   # history force

	# compute history force using the formula for H from Daitche (2013)
	h_x, h_z = [0] * t.size, [0] * t.size
	alpha = ts.compute_alpha(2, hide_progress=False)
	beta = ts.compute_beta(3, alpha[:, 1], hide_progress=False) 
	gamma = ts.compute_gamma(t.size, beta[:, 2], hide_progress=False) 
	xi = density_ratio / np.sqrt(stokes_num) * np.sqrt(9 * delta_t 
														 / (2 * np.pi))
	for n in tqdm(range(t.size - 1)):
		for j in range(n + 1): 
			h_x[n] += gamma[j, n] * w_x[n - j]
			h_z[n] += gamma[j, n] * w_z[n - j]
	h_x = np.array(h_x) * -xi 
	h_z = np.array(h_z) * -xi 
	history_x, history_z = np.gradient(h_x, t), np.gradient(h_z, t)

	# get integer times and the particle position at each integer time  
	int_indices = np.where(t == t.astype(int))
	int_x = np.take(x, int_indices[0])
	int_z = np.take(z, int_indices[0])

	return x, z, int_x, int_z, history_x, history_z, results, timesteps

def plot_rotating_trajectory(data, analytical, x_int, x1, x2=None, x3=None):
	"""
	Plot a recreation of Figure 3(a) from Ref. 1.
	
	Parameters
	----------
	data : DataFrame
		A `DataFrame` containing extracted data from Figure 3(a) in [1].
	analytical : ndarray
		2D array containing `float` data, the analytical solutions for x and z.
	x_int : ndarray
		1D array of `float` data, the horizontal particle positions at `int` t.
	x1 : ndarray
		2D array containing `float` data, the first order solutions for x, z.
	x2 : ndarray, default=None
		2D array containing `float` data, the second order solutions for x, z.
	x3 : ndarray, default=None
		2D array containing `float` data, the third order solutions for x, z.

	References
	----------
	[^4]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# extract data
	names = ['first_x', 'first_z', 'exact_x', 'exact_z']
	ext_x, ext_z, ext_x_analytical, ext_z_analytical = extract_data(names, data)

	# plot
	fig(r'$x$', r'$z$', lims=[-2, 2.5, -2.5, 2], make_square=True)
	plt.title('case 2: rigid body rotation')
	plt.scatter(ext_x, ext_z, c=COLORS[-1], marker='x')	# extracted 1st order
	x, z = analytical.T
	plt.plot(x, z, c='k', label='analytical')			# exact solution
	x, z = x_int
	plt.scatter(x, z, c='k')							# integer times
	x, z = x1.T
	plt.plot(x, z, c='k', ls='--', label='first order')			# 1st order
	if x2 is not None:
		x, z = x2.T
		plt.plot(x, z, c='k', ls='-.', label='second order')	# 2nd order
		if x3 is not None:
			x, z = x3.T
			plt.plot(x, z, c='k', ls=':', label='third order')	# 3rd order

def plot_error_analysis(data, t, e_rel1, e_rel2=None, e_rel3=None):
	"""
	Plot a recreation of Figure 3(b) from Ref. 1.
	
	Parameters
	----------
	data : DataFrame
		A `DataFrame` containing extracted data from Figure 3(b) in [1].
	t : ndarray
		1D array containing `float` time series data.
	e_rel1 : ndarray
		1D array containing `float` data, the first order relative error.
	e_rel2 : ndarray, default=None
		1D array containing `float` data, the second order relative error.
	e_rel3 : ndarray, default=None
		1D array containing `float` data, the third order relative error.

	References
	----------
	[^4]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	fig(r'$t$', r'$E_{rel}$', lims=[0, 100, 1e-7, 1e0], y_scale='log')
	plt.title('case 2: rigid body rotation error analysis')

	# extract data
	names = ['rel_error1', 't1', 'rel_error2', 't2', 'rel_error3', 't3']
	ext_e_rel1, ext_t1, ext_e_rel2, ext_t2, ext_e_rel3, \
				ext_t3 = extract_data(names, data)
	# plot
	plt.plot(ext_t1, ext_e_rel1, c=COLORS[-1])						# 1st order
	plt.plot(t, e_rel1, c='k', ls='--', label='first order')
	if e_rel2 is not None:
		plt.plot(ext_t2, ext_e_rel2, c=COLORS[-1])					# 2nd order	
		plt.plot(t, e_rel2, c='k', ls='-.', label='second order')
		if e_rel3 is not None:
			plt.plot(ext_t3, ext_e_rel3, c=COLORS[-1])				# 3rd order
			plt.plot(t, e_rel3, c='k', ls=':', label='third order')

def plot_history(t, exact_hx, exact_hz, history_x, history_z):
	"""
	Plot the analytical and numerical solutions for the history force.

	Parameters
	----------
	t : ndarray
		1D array containing `float` time series data.
	exact_hx : ndarray
		1D array of `float` data, the horizontal analytical history force.
	exact_hz : ndarray
		1D array of `float` data, the vertical analytical history force.
	history_x : ndarray
		1D array of `float` data, the horizontal numerical history force.
	history_z : ndarray
		1D array of `float` data, the vertical numerical history force.
	"""
	fig(y_label=r"$H'(t)_x$", num=211)
	plt.suptitle('case 2: rigid body rotation history verification')
	plt.plot(t, exact_hx, c=COLORS[-1])
	plt.plot(t, history_x, ':k')
	fig(y_label=r"$H'(t)_z$", num=212)
	plt.plot(t, exact_hz, c=COLORS[-1])
	plt.plot(t, history_z, ':k')

def test_tracers():
	"""Run tests on neutrally buoyant particles in linear waves."""
	print('CASE 3: NEUTRALLY BUOYANT PARTICLE IN A WAVY FLOW')
	success = True
	beta = 1
	depths, amplitude, wavelength = [5, 0.3, 0.15], 0.01, 1.5
	label_values = (np.array(depths) / wavelength).tolist()
	label1 = 'h\'/' +  r'$\lambda$' + f'\' = {wavelength / depths[1]:.2f}'
	label2 = 'h\'/' +  r'$\lambda$' + f'\' = {wavelength / depths[2]:.2f}'
	u_bar, z_bar, analytical_u, analytical_z = [], [], [], []
	m = 1 # simulation number

	# run and test a simulation for each water depth at 4 different z_0s
	for depth in depths:
		wave = lfl.LinearWave(depth, amplitude, wavelength)
		h = wave.wavenum * depth
		z = np.linspace(-0.1, -h, 100)
		analytical_z.append(z / h)
		analytical_u.append(analytical_stokes_drift(wave, z))
		z_0s = np.linspace(-0.1, -h, 4, endpoint=False)
		for z_0 in z_0s:
			print(f'({m}/{len(depths) * len(z_0s):g}) ', end='')
			drift_vel_success, u, z = test_wave(wave, beta, (0, z_0))
			u_bar.append(u)
			z_bar.append(z)
			success &= drift_vel_success
			m += 1

	# plot results and show immediately if any tests failed
	plot_drift_velocity(label_values, analytical_u, analytical_z, u_bar, z_bar)
	plt.legend()
	if not success:
		plt.show()
		quit()
	else:
		print('\n')

def analytical_stokes_drift(wave, z):
	r"""
	Compute the analytical horizontal Stokes drift velocity.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	z : float or ndarray
		The vertical position(s) of the particle.

	Returns
	-------
	u_d : ndarray
		1D array containing `float` data, the horizontal Stokes drift velocity.

	Notes
	-----
	The computation is performed using the expression,
	$$u_d = \frac{\cosh{(2(z + h))}}{2\sinh^2(h)},$$
	based on the dimensional equation[^3],
	$$u'_{SD} = c'(A'k')^2 \frac{\cosh{(2k'(z' + h'))}}{2\sinh^2(k'h')},$$
	and is valid for neutrally buoyant particles transported by a linear wave.

	References
	----------
	[^3]: [T. S. van den Bremer & Ø. Breivik (2018).](
		  https://doi.org/10.1098/rsta.2017.0104) Stokes drift.
		  *Philosophical Transactions of the Royal Society A: Mathematical,
		  Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	h = wave.wavenum * wave.depth	
	return np.cosh(2 * (z + h)) / (2 * np.sinh(h) ** 2)

def plot_drift_velocity(labels, analytical_u, analytical_z, u_bar, z_bar):
	"""
	Plot the drift velocity of particles vs their average vertical position.

	Only the horizontal Stokes drift velocity is plotted, and both the drift
	velocity and average vertical position are normalized.

	Parameters
	----------
	labels : list
		A list of `str` elements to use in the plot legend.
	analytical_u : list
		A list of `float` elements, the analytical horizontal drift velocities.
	analytical_z : list
		A list of `float` elements, the analytical vertical positions.
	u_bar : list
		A list of `float` elements, the average horizontal drift velocities.
	z_bar : list
		A list of `float` elements, the average vertical positions.
	"""
	fig(r'$\bar{u}$', r'$\bar{z}$')
	plt.title('case 3: neutrally buoyant particles in a linear wave')
	label0 = 'h\'/' +  r'$\lambda$' + f'\' = {labels[0]:.2f}'
	label1 = 'h\'/' +  r'$\lambda$' + f'\' = {labels[1]:.2f}'
	label2 = 'h\'/' +  r'$\lambda$' + f'\' = {labels[2]:.2f}'
	plt.plot(analytical_u[0], analytical_z[0], c='k', label=label0)
	plt.plot(analytical_u[1], analytical_z[1], c='k', label=label1, ls='--')
	plt.plot(analytical_u[2], analytical_z[2], c='k', label=label2, ls=':')
	plt.scatter(u_bar, z_bar, ec='k', fc='none')

def test_buoyancy():
	"""Run tests on positively and negatively buoyant particles."""
	# read data
	sm_data = pd.read_csv(IN_FILE3)
	cc_data = pd.read_csv(IN_FILE4)

	# run tests
	print('CASE 4: POSITIVELY BUOYANT PARTICLE IN A WAVY FLOW')
	wave = dfl.DeepLinearWave(depth=5, amplitude=0.026, wavelength=0.5)
	k = wave.wavenum
	x_light, z_light = test_wave(wave, 1.04, x_0=(k * 0.13, k * -0.4))
	print('\n\nCASE 5: NEGATIVELY BUOYANT PARTICLE IN A WAVY FLOW')
	x_heavy, z_heavy = test_wave(wave, 0.96)

	# extract data from Santamaria et al. (2013) Figure 1
	names = ['heavy_x', 'heavy_z', 'light_x', 'light_z']
	sm_heavy_x, sm_heavy_z, sm_light_x, sm_light_z = extract_data(names,
																  sm_data)
	# compare trajectories to extracted data and plot
	print('\nComparing solutions to extracted data from Santamaria et al.',
		  '(2013) Figure 1...')
	plot_wavy_trajectories(sm_data, cc_data, x_heavy, z_heavy, x_light, z_light)
	plt.legend()
	if match_data(x_heavy, sm_heavy_x) and match_data(z_heavy, sm_heavy_z) \
		and match_data(x_light, sm_light_x) \
		and match_data(z_light, sm_light_z):
		print_success('Numerical solutions match')
		print('\n')
	else:
		print_failure('Numerical solutions do not match')
		plt.show()
		quit()

def plot_wavy_trajectories(data1, data2, x_heavy, z_heavy, x_light, z_light):
	"""
	Plot a recreation of Figure 1 from Ref. 1.

	Parameters
	----------
	data1 : DataFrame
		A `DataFrame` containing extracted data from Figure 1 in [1].
	data2 : DataFrame
		A `DataFrame` containing `x`, `z` data from an independent test.
	x_heavy : ndarray
		1D array of `float` data, the horizontal position of the heavy particle.
	z_heavy : ndarray
		1D array of `float` data, the vertical position of the heavy particle.
	x_light : ndarray
		1D array of `float` data, the horizontal position of the light particle.
	z_light : ndarray
		1D array of `float` data, the vertical position of the light particle.

	References
	----------
	[^5]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)* 102(1), 14003.
	"""
	fig(r'$x$', r'$z$', lims=[0, 3.2, -4, 0])
	plt.title(r'cases 4 $\&$ 5: comparison to Santamaria')
	plt.plot('heavy_x', 'heavy_z', c='grey', data=data1, linewidth=4,
			 label='extracted data (Santamaria)')
	plt.plot('heavy_x', 'heavy_z', c=COLORS[-1], data=data2,
			 label='extracted data (Cathal)')
	plt.plot(x_heavy, z_heavy, c='k', ls=':', label='simulation')
	plt.plot('light_x', 'light_z', c='grey', data=data1, linewidth=4, label='')
	plt.plot('light_x', 'light_z', c=COLORS[-1], data=data2, label='')
	plt.plot(x_light, z_light, c='k', ls=':', label='')

def test_boundaries():
	"""Run tests at the surface and seabed boundaries."""
	wave = lfl.LinearWave(depth=0.5, amplitude=0.01, wavelength=1.5)
	print('CASE 6: POSITIVELY BUOYANT PARTICLE REACHING THE SURFACE',
		  'BOUNDARY')
	test_wave(wave, 1.1, x_0=(0, -0.4))
	print('\n\nCASE 7: NEGATIVELY BUOYANT PARTICLE REACHING THE SEABED',
		  'BOUNDARY')
	test_wave(wave, 0.81)
	print('\n')

def test_stokes_wave():
	"""Run tests comparing a linear wave to a fifth order Stokes wave."""
	print('CASE 8: FIFTH ORDER STOKES WAVE')
	stokes_wave = sfl.StokesWave(depth=10, amplitude=0.001, wavelength=1.5)
	linear_wave = lfl.LinearWave(depth=10, amplitude=0.001, wavelength=1.5)
	r = 0.66

	# run wave tests & simulations
	test_wave(stokes_wave, r)
	for h in [False, True]:
		h_string = 'with' if h else 'without'
		print(f'\nComparing to a linear wave {h_string} history effects...')
		xs, zs, xdots, zdots, ts = simulate(stokes_wave, r, h)[:5]
		xl, zl, xdotl, zdotl, tl = simulate(linear_wave, r, h)[:5]
		rtol, atol = 1e-1, 1e-3

		# compare and plot particle trajectories
		plot_stokes_wave(xs, zs, xl, zl, xdots, zdots, xdotl, zdotl, ts, tl)
		plt.legend()
		if np.allclose(xs, xl, rtol, atol) and np.allclose(zs, zl, rtol, atol):
			print_success('Particle trajectories match')
		else:
			print_failure('Particle trajectories do not match')
			plt.show()
			quit()

		# compare particle velocities
		if np.allclose(xdots, xdotl, rtol, atol) and np.allclose(zdots, zdotl,
					   rtol, atol) and np.allclose(ts, tl, rtol, atol):
			print_success('Particle velocities match')
		else:
			print_failure('Particle velocities do not match')
			plt.show()
			quit()
	print('\n')

def plot_stokes_wave(xs, zs, xl, zl, xdots, zdots, xdotl, zdotl, ts, tl):
	"""
	Plot particle trajectories and velocities in linear and fifth order waves.

	Parameters
	----------
	xs, zs : ndarray
		1D arrays of `float` data, the particle positions in a Stokes wave.
	xl, zl : ndarray
		1D arrays of `float` data, the particle positions in a linear wave.
	xdots, zdots : ndarray
		1D arrays of `float` data, the particle velocities in a Stokes wave.
	xdotl, zdotl : ndarray
		1D arrays of `float` data, the particle velocities in a linear wave.
	ts, tl : ndarray
		1D arrays of `float` data, the time series in linear and Stokes waves.
	"""
	# plot particle trajectories
	fig(r'$x$', r'$z$', make_square=True)
	plt.title('case 8: fifth order Stokes wave')
	plt.plot(xl, zl, lw=4, c='silver')
	plt.plot(xs, zs, '-k')

	# plot particle velocities
	fig(y_label=r'$\dot{x}$', num=211)
	plt.suptitle('case 8: fifth order Stokes wave')
	plt.plot(tl, xdotl, lw=4, c='silver')
	plt.plot(ts, xdots, '-k')
	fig(r'$t$', r'$\dot{z}$', 212)
	plt.plot(tl, zdotl, lw=4, c='silver', label='linear wave')
	plt.plot(ts, zdots, '-k', label='Stokes wave')

def test_flat_seabed():
	"""Run tests on a bichromatic wave over a flat seabed."""
	print('CASE 9: BICHROMATIC WAVE OVER A FLAT SEABED')
	x_n, z_n = 3241, 19							# total position data points
	delta_t = 0.2								# time sampling interval
	swash_t_0 = 180								# initial time for SWASH data
	t_0 = 400									# initial time
	t_0_index = int((t_0 - swash_t_0) / delta_t)# index of initial time
	t_n = 2100 - t_0_index						# total time data points
	t_f = t_n * delta_t	+ t_0					# final time
	rtol, atol = 1e-1, 5e-2						# tolerances for comparisons
	depth = 15

	# read SWASH data
	swash = scp.io.loadmat(IN_FILE5)
	swash_x = swash['Xp']
	swash_z = swash['z_E']
	swash_u = swash['u']
	swash_w = swash['w']

	# initialize wave object, variables for scaling and time series
	print('Verifying the velocity field with SWASH data...')
	wave = bfl.BichromaticWave(depth, amplitude=np.array([0.54, 0.18]),
							   wavelength=np.array([60.29039, 80.89962]))
	omega = wave.angular_freq[0]
	c = wave.phase_velocity[0]
	k = wave.wavenum[0]
	t = np.round(np.arange(t_0, t_f, delta_t), 5) * omega

	# compare non-zero fluid velocity for all position points
	for i in range(x_n):
		for j in range(z_n):
			x, z = k * swash_x[0, i], k * (swash_z[i, j] - depth)
			u, w = wave.velocity(x, z, t)
			u_data = swash_u[t_0_index:, i, j] / c
			w_data = swash_w[t_0_index:, i, j] / c
			u_index, w_index = np.nonzero(u_data)[0], np.nonzero(w_data)[0]


			# plot if the velocity field doesn't match SWASH data
			if not (np.allclose(u[u_index], u_data[u_index], rtol, atol) \
				and np.allclose(w[w_index], w_data[w_index], rtol, atol)):
				print_failure('Fluid velocity does not match the SWASH data')
				fig(y_label=r'$u$', num=211, hide_xticks=True)
				plt.title('case 9: bichromatic wave over a flat seabed'
					   + f' (x[{i}], z[{j}])')
				plt.scatter(t, u_data, marker='.', ec='k', fc='none')
				plt.plot(t, u, '-k')
				fig(r'$t$', r'$w$', 212)
				plt.scatter(t, w_data, marker='.', ec='k', fc='none')
				plt.plot(t, w, '-k')
				plt.show()
				quit()
	print_success('Fluid velocity matches the flat seabed SWASH data')
	print('\n')

def test_subharmonics():
	"""Verify subharmonic effects in a bichromatic wave over a sloped seabed."""
	print('CASE 10: BICHROMATIC WAVE OVER A SLOPED SEABED')
	i, j = 100, 10	# indices of x data to test
	t_0 = 600		# initial time
	t_n = 4101		# total number of time points
	delta_t = 0.2	# timestep

	# read data from files
	swash = loadmat(IN_FILE6)
	coord = loadmat(IN_FILE7)
	swash_x = coord['x_E']
	swash_u = swash['u']
	x = swash_x[i, j]

	# create the BichromaticWave object
	print('Verifying the solution for subharmonic effects with SWASH data...')
	depth = 15
	amplitudes = np.array([0.54 / 2, 0.18 / 2])
	wavelengths = np.array([60.2991, 80.8996])
	wave = bfl.BichromaticWave(depth, amplitudes, wavelengths, slope=0.0125)

	# variables for non-dimensional scaling
	omega = wave.angular_freq[0]
	c = wave.phase_velocity[0]
	k = wave.wavenum[0]

	# time series and subharmonic solution
	t = np.round(np.arange(600, t_0 + t_n * delta_t, delta_t), 5)
	u = wave.subharmonic(k * x, t * omega)

	# subtract the time-averaged signal and take the FFT
	swash_u = (swash_u[:, i, j] - np.mean(swash_u[:, i, j]))
	u_fft = fft(swash_u)
	freqs = fftfreq(len(swash_u), delta_t)
	n = len(swash_u) // 2

	# compute the subharmonic and superharmonic frequencies
	peaks, _ = find_peaks(np.abs(u_fft[:n]), height=100)
	p1, p2 = peaks
	f1, f2 = freqs[peaks]
	subharmonic = np.abs(f1 - f2)
	superharmonic = f1 + f2

	# filter out non-subharmonic frequencies and scale results
	subharmonic_fft = u_fft.copy()
	subharmonic_fft[1e-3 < np.abs(np.abs(freqs) - subharmonic)] = 0
	filtered_u = np.real(ifft(subharmonic_fft)) / c
	t *= omega

	# compute the phase shift and shift the computed data from my implementation
	correlation = correlate(u, filtered_u)
	lags = correlation_lags(len(u), len(filtered_u))
	i = np.argmax(correlation)
	phase_shift = int(lags[i])
	indices = range(len(u)) if phase_shift < 0 else np.array(range(len(u))) \
			- phase_shift
	shifted_u = [u[phase_shift + j] for j in indices]

	# plot the frequencies
	fig('Hz', r'$|A|$')
	plt.title('case 10: frequencies of a bichromatic wave over a sloped seabed')
	plt.xlim(0, 0.35)
	plt.axvline(subharmonic, c='silver')
	plt.axvline(superharmonic, c='silver', ls='--')
	plt.plot(np.abs(freqs[:n]), np.abs(u_fft[:n]), '-k')

	# plot the subharmonics over time
	fig(r'$t$', r'$u^{(2)}$')
	plt.title('case 10: subharmonic effects')
	plt.scatter(t, filtered_u, marker='.', ec='k', fc='none')
	plt.plot(t, shifted_u, '-k')

	# compare the computed subharmonic solution and SWASH data
	if not np.allclose(filtered_u, shifted_u, rtol=1e-3, atol=1e-4):
		print_failure('Fluid velocity does not match the SWASH data')
		plt.show()
		quit()
	print_success('Solution for subharmonic effects matches the sloped seabed '
				+ 'SWASH data')

def simulate(flow, density_ratio, include_history, x_0=(0, 0), order=3):
	"""
	Simulate a particle moving through the specified `flow`.

	Parameters
	----------
	flow : Flow (obj)
		The flow through which the particle is transported.
	density_ratio : float
		The ratio between the particle and fluid densities.
	include_history : bool
		Whether to include history effects.
	x_0 : tuple, default=(0, 0)
		The initial horizontal and vertical positions of the particle.
	order : int, default=3
		The order of the integration scheme  (first, second, or third).

	Returns
	-------
	ndarrays
		The position and velocity of the particle over time, and forces.
	"""
	assert order in [1, 2, 3], 'Integration scheme must be' \
							 + ' 1st, 2nd, or 3rd order'
	x_0, z_0 = x_0	# initial particle position
	if isinstance(flow, qfl.QuiescentFlow):
		xdot_0, zdot_0 = 1, 1						# initial particle velocity
		stokes_hat = SCALE
		num_periods, delta_t = 15, 1e-2				# time and timestep
		density_ratio = 2 / (1 + 2 * density_ratio)	# Prasath param translation
	elif isinstance(flow, rfl.RotatingFlow):
		xdot_0, zdot_0 = flow.velocity(x_0, z_0)	# initial particle velocity
		stokes_hat = SCALE * 0.3
		num_periods, delta_t = 100, 1e-2			# time and timestep
		density_ratio *= SCALE						# Daitche param translation
	elif isinstance(flow, sfl.StokesWave):
		xdot_0, zdot_0 = flow.velocity(x_0, z_0, t=0) # initial particle vel
		num_periods, delta_t = 5, 5e-3
		stokes_hat = 0.12
	else:
		xdot_0, zdot_0 = flow.velocity(x_0, z_0, t=0) # initial particle vel
		if density_ratio == 1:
			num_periods, delta_t = 3, 5e-3			# time and timestep
			stokes_hat = 0.15
			density_ratio *= SCALE						# parameter translation
		elif density_ratio == 0.96 or density_ratio == 1.04:
			num_periods = 5 if include_history else 38
			delta_t = 1e-2							# timestep
			stokes_hat = SCALE * density_ratio / 2
			density_ratio *= SCALE						# parameter translation
		elif density_ratio == 0.66:
			xdot_0, zdot_0 = flow.velocity(x_0, z_0, t=0) # initial particle vel
			num_periods, delta_t = 5, 5e-3
			stokes_hat = 0.12
		else:
			num_periods, delta_t = 10, 5e-3
			stokes_hat = 0.15
			density_ratio *= SCALE						# parameter translation

	# create particle and transport system objects, run simulation
	t = np.arange(0, num_periods * flow.period, delta_t)
	y = [x_0, z_0, xdot_0, zdot_0]
	my_particle = prt.Particle(stokes_hat)
	my_system = ts.MyTransportSystem(my_particle, flow, density_ratio)
	return my_system.maxey_riley(t, y, include_history, include_h=True,
								 order=order)

def test_wave(wave, density_ratio, x_0=(0, 0)):
	"""
	Run various tests on a simulated particle in a wavy flow.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	density_ratio : float
		The ratio between the particle and fluid densities.
	x_0 : tuple, default=(0, 0)
		The horizontal and vertical positions of the particle.

	Returns
	-------
	success : bool
		Whether the verification of the Stokes drift velocity succeeded.
	u_bar : float
		The mean horizontal drift velocity of the particle.
	z_bar : float
		The mean vertical position of the particle.
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.

	Notes
	-----
	Returns success, u_bar, z_bar only if the particle is neutrally buoyant,
	returns x, z without history otherwise.
	"""
	# define local variables depending on wave type and density ratio
	if isinstance(wave, sfl.StokesWave):
		message = 'Simulating a negatively buoyant particle'
		r = density_ratio
		num_periods, delta_t, stokes_hat = 5, 5e-3, 0.12
	else:
		r = density_ratio * SCALE
		if density_ratio < 1:
			message = 'Simulating a negatively buoyant particle'
			num_periods = 38 if density_ratio == 0.96 else 10
			delta_t = 1e-2 if density_ratio == 0.96 else 5e-3
			stokes_hat = SCALE * density_ratio / 2 if density_ratio == 0.96 \
						 else 0.15
		elif density_ratio > 1:
			message = 'Simulating a positively buoyant particle'
			num_periods = 38 if density_ratio == 1.04 else 10
			delta_t = 1e-2 if density_ratio == 1.04 else 5e-3
			stokes_hat = SCALE * density_ratio / 2 if density_ratio == 1.04 \
						 else 0.15
		else:
			message = 'Simulating a neutrally buoyant particle'
			num_periods, delta_t, stokes_hat = 3, 5e-3, 0.15

	# run simulation without history
	include_history = False
	print(f'{message} without history...')
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z, \
		drag_x, drag_z, _, _, history_x, history_z = simulate(wave,
		density_ratio, include_history, x_0)
	u_x, u_z = wave.velocity(x, z, t) # compute fluid velocity

	# run numerical integration (Haller system)
	print('Numerically integrating the Maxey-Riley equation...')
	particle = prt.Particle(stokes_hat)
	haller_system = hs.HallerTransportSystem(particle, wave, r)
	x_h, z_h, xdot_h, zdot_h, t_h = haller_system.run_numerics(haller_system
												 .maxey_riley, x_0[0], x_0[1],
												  num_periods, delta_t)
	# truncate data for comparison if necessary
	if len(x) < len(x_h):
		x_h = x_h[:len(x)]
		z_h = z_h[:len(z)]
		xdot_h = xdot_h[:len(xdot)]
		zdot_h = zdot_h[:len(zdot)]
		t_h = t_h[:len(t)]
	# compare non-history solutions to numerical integration
	rtol, atol = 1e-2, 1e-3
	if np.allclose(x, x_h, rtol, atol) & np.allclose(z, z_h, rtol, atol) \
	 & np.allclose(xdot, xdot_h, rtol, atol) \
	 & np.allclose(zdot, zdot_h, rtol, atol) & np.allclose(t, t_h, rtol, atol):
		print_success('Simulated results match numerical integration')
	else:
		print_failure('Simulated results do not match numerical integration')
		# plot particle trajectory
		fig(r'$x$', r'$z$')
		plt.plot(x_h, z_h, c=COLORS[-1], marker='o', linewidth=4,
				 label='Numerical integration')
		plt.plot(x, z, c='k', marker='.', label='Simulation')

		# plot horizontal particle velocity over time
		fig(r'$t$', r'$\dot{x}$')
		plt.plot(t_h, xdot_h, c=COLORS[-1], marker='o', linewidth=4,
				 label='Numerical integration')
		plt.plot(t, xdot, c='k', marker='.', label='Simulation')

		# plot vertical particle velocity over time
		fig(r'$t$', r'$\dot{z}$')
		plt.plot(t_h, zdot_h, c=COLORS[-1], marker='o', linewidth=4,
				 label='Numerical integration')
		plt.plot(t, zdot, c='k', marker='.', label='Simulation')
		plt.legend()
		plt.show()
		quit()

	# verify computed attributes, velocities, and forces
	desk_check_attributes(particle, wave, density_ratio)
	if 0.81 != density_ratio != 1.1: verify_num_periods(x, z, xdot, t,
														num_periods)
	verify_lengths([x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z,
					mass_x, mass_z, drag_x, drag_z, history_x, history_z])
	verify_trajectory_range(x, z, wave.depth * wave.wavenum)
	verify_velocities(density_ratio, x, z, t, xdot, zdot, u_x, u_z)
	verify_forces(wave, density_ratio, stokes_hat, u_x, u_z, x, z, xdot, zdot,
				  t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z,
				  drag_x, drag_z, history_x, history_z, include_history)

	# save non-history results
	x_no_history, z_no_history, xdot_no_history, zdot_no_history, \
		t_no_history = x, z, xdot, zdot, t

	# run simulation with history
	print(f'\n{message} with history...')
	include_history = True
	x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, \
		mass_z, drag_x, drag_z, _, _, history_x, history_z = simulate(wave,
		density_ratio, include_history, x_0)

	if density_ratio == 1:
		# check that simulation without history = simulation with history
		if x.size != x_no_history.size:
			print_failure('Simulation with history ended prematurely')
			fig(r'$x$', r'$z$')
			plt.plot(x_no_history, z_no_history, c='k', label='without history')
			plt.plot(x, z, c='k', ls=':', label='with history')
			plt.legend()
			plt.show()
			quit()
		if np.allclose(x, x_no_history) & np.allclose(z, z_no_history) \
			& np.allclose(xdot, xdot_no_history) \
			& np.allclose(zdot, zdot_no_history) & np.allclose(t, t_no_history):
			print_success('Simulation with history matches simulation without '
						+ 'history')
		else:
			print_failure('Simulation with history does not match simulation '
						+ 'without history')
			fig(r'$x$', r'$z$')
			plt.plot(x_no_history, z_no_history, c='k', label='without history')
			plt.plot(x, z, c='k', ls=':', label='with history')
			plt.legend()
			plt.show()
			quit()
		# verify Stokes drift velocity
		success, u_bar, z_bar = verify_drift_velocity(wave, x, z, xdot, t)
		return success, u_bar, z_bar
	else:
		# verify computed attributes, velocities, and forces
		u_x, u_z = wave.velocity(x, z, t)
		desk_check_attributes(particle, wave, density_ratio)
		if 0.81 != density_ratio != 1.1: verify_num_periods(x, z, xdot, t, 5)
		verify_lengths([x, z, xdot, zdot, t, fpg_x, fpg_z, buoyancy_x,
						buoyancy_z, mass_x, mass_z, drag_x, drag_z, history_x,
						history_z])
		verify_trajectory_range(x, z, wave.depth * wave.wavenum)
		verify_velocities(density_ratio, x, z, t, xdot, zdot, u_x, u_z)
		verify_forces(wave, density_ratio, stokes_hat, u_x, u_z, x, z, xdot,
					  zdot, t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x,
					  mass_z, drag_x, drag_z, history_x, history_z,
					  include_history)
		return x_no_history, z_no_history

def desk_check_attributes(particle, wave, density_ratio):
	r"""
	Desk check computed attributes of the Wave and TransportSystem classes.

	Parameters
	----------
	particle : Particle (obj)
		The particle transported through the wave.
	wave : Wave (obj)
		The wave through which the particle is transported.
	density_ratio : float
		The ratio between the particle and fluid densities.
	"""
	print('\nDesk checking computed attributes of the Wave and TransportSystem',
		  'objects...')
	if isinstance(wave, sfl.StokesWave):
		system = ts.MyTransportSystem(particle, wave, density_ratio)
		g = scp.constants.g
		k = wave.wavenum
		omega = wave.angular_freq
		desk_k = 4 * np.pi / 3 
		desk_omega = 6.4103067
		desk_epsilon = 0.00418879
		desk_c = 1.530348
		desk_period = 2 * np.pi
		desk_fr = 1
		desk_re = 365343.672222675
		desk_re_p = 0.00676556
	else:
		system = ts.MyTransportSystem(particle, wave, SCALE * density_ratio)
		g = scp.constants.g
		k = wave.wavenum
		omega = wave.angular_freq
		if density_ratio == 1:
			desk_k = 4 * np.pi / 3
			desk_period = 2 * np.pi
			desk_epsilon = 0.0418879
			desk_re_p = 0
			if wave.depth == 5:
				desk_omega = 6.4103067
				desk_c = 1.530348
				desk_fr = 1
				desk_re = 365343.67222267
			elif wave.depth == 0.3:
				desk_omega = 5.9104777
				desk_c = 1.4110226
				desk_fr = 0.92202729
				desk_re = 336856.83610078
			elif wave.depth == 0.15:
				desk_omega = 4.7837096
				desk_c = 1.1420265
				desk_fr = 0.74625284
				desk_re = 272638.75436555
		elif density_ratio == 0.96 or density_ratio == 1.04:
			desk_k = 4 * np.pi
			desk_omega = 11.102977
			desk_epsilon = 0.32672564
			desk_c = 0.8835468
			desk_period = 2 * np.pi
			desk_fr = 1
			desk_re = 70310.422501496
			desk_re_p = 4.158477 if density_ratio == 0.96 else 4.3282801
		else:
			desk_k = 4 * np.pi / 3
			desk_omega = 6.3138229
			desk_epsilon = 0.0418879
			desk_c = 1.5073142
			desk_period = 2 * np.pi
			desk_fr = 0.98494864
			desk_re = 359844.75266697
			desk_re_p = 0.87054403 if density_ratio == 1.1 else 2.2462186

	assert np.isclose(k, desk_k), f'Wavenumber {k:.4f}m^-1 computed ' \
								+ f'incorrectly, correct value is {desk_k:.4f}'\
								+ 'm^-1'
	print_success('Wavenumber computed correctly')
	assert np.isclose(omega, desk_omega, rtol=1e-3, atol=1e-5), \
		f'Angular frequency {omega:.4f}s^-1 computed incorrectly,' \
		+ f' the correct value is {desk_omega:.4f}s^-1'
	print_success('Angular frequency computed correctly')
	assert np.isclose(wave.steepness, desk_epsilon, rtol=1e-3, atol=1e-5), \
		f'Wave steepness {wave.steepness:.4f} computed incorrectly, ' \
		+ f'correct value is {desk_epsilon:.4f}'
	print_success('Wave steepness computed correctly')
	assert np.isclose(wave.phase_velocity, desk_c, rtol=1e-3, atol=1e-5), \
		f'Phase velocity {wave.phase_velocity:.4f}m/s computed incorrectly, ' \
		+ f'correct value is {desk_c:.4f}m/s'
	print_success('Phase velocity computed correctly')
	assert np.isclose(wave.period, desk_period, rtol=1e-3, atol=1e-5), \
		f'Wave period {wave.period:.4f}s computed incorrectly,' \
		+ f' the correct value is {desk_period:.4f}s'
	print_success('Wave period computed correctly')
	assert np.isclose(wave.froude_num, desk_fr, rtol=1e-3, atol=1e-5), \
		f'Froude number {wave.froude_num} computed incorrectly,' \
		+ f' the correct value is {desk_fr:.4f}'
	print_success('Froude number computed correctly')
	assert np.isclose(wave.reynolds_num, desk_re, rtol=1e-3, atol=1e-5), \
		f'Reynolds number {wave.reynolds_num} computed incorrectly,' \
		+ f' the correct value is {desk_re:.4f}'
	print_success('Reynolds number computed correctly')
	assert np.isclose(system.reynolds_num, desk_re_p, rtol=1e-3, atol=1e-5),\
		f'Particle Reynolds number {system.reynolds_num} computed' \
		+ f' incorrectly, the correct value is {desk_re_p:.4f}'
	print_success('General particle Reynolds number computed correctly')

def verify_num_periods(x, z, xdot, t, num_periods):
	"""
	Verify that the simulation ran for the prescribed number of periods.

	Parameters
	----------
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	xdot : ndarray
		1D array of `float` data, the horizontal particle velocity.
	t : ndarray
		1D array containing `float` time series data.
	num_periods : int
		The prescribed number of periods.
	"""
	x_crossings, z_crossings, _, _, _ = ts.compute_drift_velocity(x, z, xdot, t)
	if len(x_crossings) in range(num_periods - 1, num_periods + 2):
		print_success('Simulation ran for the prescribed number of periods')
	else:
		print_failure(f'Prescribed {num_periods} periods, but '
					+ f'{len(x_crossings)} periods were simulated')
		fig(r'$x$', r'$z$')
		plt.plot(x, z, c='k')
		plt.scatter(x_crossings, z_crossings, c='k', marker='x')
		plt.show()
		quit()

def verify_lengths(sols):
	"""Verify that all `ndarray` elements in `sols` list are the same length."""
	lengths = [len(sol) for sol in sols]
	verified = len(set(lengths)) == 1
	if verified:
		print_success('All solutions are the same length')
	else:
		fail_str = ''
		for i in range(len(lengths) - 1): fail_str += str(lengths[i]) + ', '
		fail_str += str(lengths[-1])
		print_failure('Not all solutions are the same length\n\t ' + fail_str)
		quit()

def verify_trajectory_range(x, z, h):
	"""
	Verify that vertical particle positions are between the seabed and surface.

	Parameters
	----------
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	h : float
		The dimensionless water depth.
	"""
	if np.min(z[:-1]) < -h:
		print_failure('Particle trajectory penetrated the seabed')
		fig(r'$x$', r'$z$')
		plt.plot(x, z, '-k.')
		plt.axhline(-h, c=COLORS[-1], ls=':')
		plt.show()
		quit()
	elif 1e-5 < np.max(z):
		print_failure('Particle trajectory exceeded the water surface')
		fig(r'$x$', r'$z$')
		plt.plot(x, z, c='k')
		plt.axhline(0, c=COLORS[-1], ls=':')
		plt.show()
		quit()
	else:
		print_success('Particle trajectory is within the surface and seabed')

def verify_velocities(density_ratio, x, z, t, xdot, zdot, u_x=None, u_z=None):
	"""
	Verify the particle and fluid velocities numerically.

	Parameters
	----------
	density_ratio : float
		The ratio between the particle and fluid densities.
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	t : ndarray
		1D array containing `float` time series data.
	xdot : ndarray
		1D array of `float` data, the horizontal particle velocity.
	zdot : ndarray
		1D array of `float` data, the vertical particle velocity.
	u_x : ndarray, default=None
		1D array of `float` data, the horizontal fluid velocity.
	u_z : ndarray, default=None
		1D array of `float` data, the vertical fluid velocity.
	"""
	print('\nChecking the particle and fluid velocities...')
	rtol, atol = 1e-1, 1e-2
	if density_ratio == 1 or density_ratio == SCALE:
		# verify that fluid velocity = particle velocity
		if np.allclose(xdot, u_x, rtol, atol):
			print_success('Particle velocity = fluid velocity in the '
						+ 'x direction')
		else:
			print_failure('Particle velocity != fluid velocity in the '
						+ 'x direction')
			fig(r'$t$', 'horizontal velocity')
			plt.plot(t, xdot, c='k', marker='.', label=r'$\dot{x}$')
			plt.plot(t, u_x, c='k', marker='.', ls=':', label=r'$u_x$')
			plt.legend()
			plt.show()
			quit()
		if np.allclose(zdot, u_z, rtol, atol):
			print_success('Particle velocity = fluid velocity in the '
						+ 'z direction')
		else:
			print_failure('Particle velocity != fluid velocity in the '
						+ 'z direction')
			fig(r'$t$', 'vertical velocity')
			plt.plot(t, zdot, c='k', marker='.', label=r'$\dot{z}$')
			plt.plot(t, u_z, c='k', marker='.', ls=':', label=r'$u_z$')
			plt.legend()
			plt.show()
			quit()

	# verify that particle velocity = numerical derivative
	num_x = np.gradient(x, t)
	num_z = np.gradient(z, t)
	if np.allclose(xdot[1:-1], num_x[1:-1], rtol, atol):
		print_success('Particle velocity matches the numerical derivative '
					+ 'in the x direction')
	else:
		print_failure('Particle velocity does not match the numerical '
					 + 'derivative in the x direction')
		fig(r'$t$', 'horizontal velocity')
		plt.plot(t, xdot, c='k', marker='.', label=r'$\dot{x}$')
		plt.plot(t, num_x, c='k', marker='.', ls=':',
				 label=r'$\partial{x}/\partial{t}$')
		plt.legend()
		plt.show()
		quit()
	if np.allclose(zdot[1:-1], num_z[1:-1], rtol, atol):
		print_success('Particle velocity matches the numerical derivative '
					+ 'in the z direction')
	else:
		print_failure('Particle velocity does not match the numerical '
					+ 'derivative in the z direction')
		fig(r'$t$', 'vertical velocity')
		plt.plot(t, zdot, c='k', marker='.', label=r'$\dot{z}$')
		plt.plot(t, num_z, c='k', marker='.', ls=':',
				 label=r'$\partial{z}/\partial{t}$')
		plt.legend()
		plt.show()
		quit()

def verify_forces(wave, density_ratio, stokes_hat, u_x, u_z, x, z, xdot, zdot,
				  t, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z,
				  drag_x, drag_z, history_x, history_z, include_history):
	r"""
	Verify the horizontal and vertical components of each force.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	density_ratio : float
		The ratio between the particle and fluid densities.
	stokes_hat : float
			The Stokes number $\widehat{St}$.
	u_x : ndarray
		1D array of `float` data, the horizontal fluid velocity.
	u_z : ndarray
		1D array of `float` data, the vertical fluid velocity.
	x, z : float
		The horizontal and vertical position of the particle.
	xdot : ndarray
		1D array of `float` data, the horizontal particle velocity.
	zdot : ndarray
		1D array of `float` data, the vertical particle velocity.
	t : ndarray
		1D array containing `float` time series data.
	fpg_x : ndarray
		1D array of `float` data, the horizontal fluid pressure gradient.
	fpg_z : ndarray
		1D array of `float` data, the vertical fluid pressure gradient.
	buoyancy_x : ndarray
		1D array of `float` data, the horizontal buoyancy force.
	buoyancy_z : ndarray
		1D array of `float` data, the vertical buoyancy force.
	mass_x : ndarray
		1D array of `float` data, the horizontal added mass force.
	mass_z : ndarray
		1D array of `float` data, the vertical added mass force.
	drag_x : ndarray
		1D array of `float` data, the horizontal Stokes drag.
	drag_z : ndarray
		1D array of `float` data, the vertical Stokes drag.
	history_x : ndarray
		1D array of `float` data, the horizontal history force.
	history_z : ndarray
		1D array of `float` data, the vertical history force.
	include_history : bool
		Whether the history force was included in the simulation.
	"""
	print('\nVerifying individual forces...')
	# slightly truncate arrays to avoid numerical errors near t_final
	n = -5
	u_x = u_x[:n]
	u_z = u_z[:n]
	xdot = xdot[:n]
	zdot = zdot[:n]
	x = x[:n]
	z = z[:n]
	t = t[:n]
	fpg_x = fpg_x[:n]
	fpg_z = fpg_z[:n]
	buoyancy_x = buoyancy_x[1:n + 1]
	buoyancy_z = buoyancy_z[1:n + 1]
	mass_x = mass_x[:n]
	mass_z = mass_z[:n]
	drag_x = drag_x[:n]
	drag_z = drag_z[:n]
	history_x = history_x[5:n]
	history_z = history_z[5:n]
	inertial_x = fpg_x + mass_x
	inertial_z = fpg_z + mass_z

	# define local variables for verification
	dudt, dwdt = wave.derivative_along_trajectory(x, z, t, [xdot, zdot])
	w_x, w_z = xdot - u_x, zdot - u_z			# relative velocity w
	A_x = np.gradient(w_x, t)[5:]				# dw_x/dt
	A_z = np.gradient(w_z, t)[5:]				# dw_z/dt
	G_x = (fpg_x + buoyancy_x + mass_x + drag_x - dudt)[5:]
	G_z = (fpg_z + buoyancy_z + mass_z + drag_z - dwdt)[5:]
	success = True
	rtol, atol = 1e-1, 1e-2
	beta = density_ratio if isinstance(wave, lfl.LinearWave) \
		   or isinstance(wave, dfl.DeepLinearWave) else density_ratio / SCALE

	# verify the fluid pressure gradient at t = 0
	mdx, mdz = wave.material_derivative(x, z, t)
	if np.isclose(fpg_x[0], beta * mdx[0], rtol, atol):
		print_success('Fluid pressure gradient verified at t = 0 in the '
					+ 'x direction')
	else:
		print_failure('Fluid pressure gradient not verified at t = 0 in the '
					+ 'x direction')
		success = False
	if np.isclose(fpg_z[0], beta * mdz[0], rtol, atol):
		print_success('Fluid pressure gradient verified at t = 0 in the '
					+ 'z direction')
	else:
		print_failure('Fluid pressure gradient not verified at t = 0 in the '
					+ 'z direction')
		success = False

	# verify the buoyancy force
	g = np.array([0, -scp.constants.g]) * wave.wavenum / (wave.angular_freq
										* wave.angular_freq)
	b_x, b_z = (1 - beta) * g
	if np.allclose(buoyancy_x, b_x, rtol, atol):
		print_success('Buoyancy force is correct in the x direction')
	else:
		print_failure('Buoyancy force is incorrect in the x direction')
		success = False
	if np.allclose(buoyancy_z, b_z, rtol, atol):
		print_success('Buoyancy force is correct in the z direction')
	else:
		print_failure('Buoyancy force is incorrect in the z direction')
		success = False

	# verify the added mass force = 0 at t = 0
	if np.isclose(mass_x[0], 0, rtol, atol):
		print_success('Added mass force = 0 at t = 0 in the x direction')
	else:
		print_failure('Added mass force != 0 at t = 0 in the x direction')
		success = False
	if np.isclose(mass_z[0], 0, rtol, atol):
		print_success('Added mass force = 0 at t = 0 in the z direction')
	else:
		print_failure('Added mass force != 0 at t = 0 in the z direction')
		success = False

	# verify the inertial forces
	if np.allclose(inertial_x, beta * mdx, rtol, atol):
		print_success('Inertial forces verified in the x direction')
	else:
		print_failure('Inertial forces not verified in the x direction')
		success = False
	if np.allclose(inertial_z, beta * mdz, rtol, atol):
		print_success('Inertial forces verified in the z direction')
	else:
		print_failure('Inertial forces not verified in the z direction')
		success = False

	# verify the Stokes drag
	drag_coeff = -2 / 3 * beta / stokes_hat
	drag_est_x = -(beta - 1) * (mdx - wave.gravity[0])
	drag_est_z = -(beta - 1) * (mdz - wave.gravity[1])
	if np.allclose(drag_x, drag_coeff * w_x, rtol, atol):
		print_success('Stokes drag verified in the x direction')
	else:
		print_failure('Stokes drag not verified in the x direction')
		success = False
	if not include_history and np.allclose(drag_x, drag_est_x, rtol=2, atol=1):
		print_success('Stokes drag approximated in the x direction')
	else:
		if not include_history:
			print_failure('Stokes drag not approximated in the x direction')
			success = False
	if np.allclose(drag_z, drag_coeff * w_z, rtol, atol):
		print_success('Stokes drag verified in the z direction')
	else:
		print_failure('Stokes drag not verified in the z direction')
		success = False
	if not include_history and np.allclose(drag_z[20:], drag_est_z[20:],
										   rtol=0.2, atol=0.1):
		print_success('Stokes drag approximated in the z direction')
	else:
		if not include_history:
			print_failure('Stokes drag not approximated in the z direction')
			success = False

	# verify the history force
	if include_history: # A(t) - G(t) = H'(t)
		if np.allclose((A_x - G_x), history_x, rtol, atol):
			print_success('dw_x/dt - the sum of non-history forces = history')
		else:
			print_failure('dw_x/dt - the sum of non-history forces != history')
			success = False
		if np.allclose((A_z - G_z), history_z, rtol, atol):
			print_success('dw_z/dt - the sum of non-history forces = history')
		else:
			print_failure('dw_z/dt - the sum of non-history forces != history')
			success = False
	else:
		# check that history = 0
		if np.allclose(history_x, 0, rtol, atol):
			print_success('History force is zero in the x direction')
		else:
			print_failure('History force is non-zero in the x direction')
			success = False
		if np.allclose(history_z, 0, rtol, atol):
			print_success('History force is zero in the z direction')
		else:
			print_failure('History force is non-zero in the z direction')
			success = False

		# check that A(t) = G(t)
		if np.allclose(A_x[:-1], G_x[:-1], rtol, atol):
			print_success('dw_x/dt = the sum of non-history forces ')
		else:
			print_failure('dw_x/dt != the sum of non-history forces')
			success = False
		if np.allclose(A_z[1:-1], G_z[1:-1], rtol, atol):
			print_success('dw_z/dt = the sum of non-history forces ')
		else:
			print_failure('dw_z/dt != the sum of non-history forces')
			success = False

	if not success:
		# plot verifications for horizontal forces over time
		fig(r'$t$', 'horizontal force')
		plt.scatter(t[0], beta * mdx[0], marker='.', c=COLORS[0])
		plt.axhline(b_x, c=COLORS[4], ls=':')
		plt.plot(t, drag_coeff * w_x, c=COLORS[6], ls=':')
		plt.plot(t, drag_est_x, c=COLORS[6], ls='--')
		plt.plot(t, beta * mdx, c=COLORS[8], ls=':')
		if include_history:
			plt.plot(t[5:], (A_x - G_x), c=COLORS[10], ls=':')
		else:
			plt.axhline(0, c=COLORS[10], ls=':')

		# plot horizontal forces over time
		plt.plot(t, fpg_x, c=COLORS[0], label='fluid pressure gradient')
		plt.plot(t, mass_x, c=COLORS[2], label='added mass')
		plt.plot(t, buoyancy_x, c=COLORS[4], label='buoyancy')
		plt.plot(t, drag_x, c=COLORS[6], label='Stokes drag')
		plt.plot(t, inertial_x, c=COLORS[8], label='inertial forces')
		plt.plot(t[5:], history_x, c=COLORS[10], label='history force')
		plt.legend()

		# plot verifications for vertical forces over time
		fig(r'$t$', 'vertical force')
		plt.scatter(t[0], beta * mdz[0], marker='.', c=COLORS[0])
		plt.axhline(b_z, c=COLORS[4], ls=':')
		plt.plot(t, drag_coeff * w_z, c=COLORS[6], ls=':')
		plt.plot(t, drag_est_z, c=COLORS[6], ls='--')
		plt.plot(t, beta * mdz, c=COLORS[8], ls=':')
		if include_history:
			plt.plot(t[5:], (A_z - G_z), c=COLORS[10], ls=':')
		else:
			plt.axhline(0, c=COLORS[10], ls=':')

		# plot vertical forces over time
		plt.plot(t, fpg_z, c=COLORS[0], label='fluid pressure gradient')
		plt.plot(t, mass_z, c=COLORS[2], label='added mass')
		plt.plot(t, buoyancy_z, c=COLORS[4], label='buoyancy')
		plt.plot(t, drag_z, c=COLORS[6], label='Stokes drag')
		plt.plot(t, inertial_z, c=COLORS[8], label='inertial forces')
		plt.plot(t[5:], history_z, c=COLORS[10], label='history force')
		plt.legend()
		plt.show()
		quit()

def verify_drift_velocity(wave, x, z, xdot, t):
	"""
	Verify that the analytical and numerical Stokes drift velocity match.

	Parameters
	----------
	wave : Wave (obj)
		The wave through which the particle is transported.
	x : ndarray
		1D array of `float` data, the horizontal particle position.
	z : ndarray
		1D array of `float` data, the vertical particle position.
	xdot : ndarray
		1D array of `float` data, the horizontal particle velocity.
	t : ndarray
		1D array containing `float` time series data.

	Returns
	-------
	success : bool
		Whether the verification of the Stokes drift velocity succeeded.
	u_bar : float
		The mean horizontal drift velocity of the particle.
	z_bar : float
		The mean vertical position of the particle.
	"""
	print('\nVerifying the computation of the Stokes drift velocity...')
	x_crossings, z_crossings, u, w, \
				 t_crossings = ts.compute_drift_velocity(x, z, xdot, t)

	# check that all z-crossings are equal
	if np.isclose(np.min(z_crossings), np.max(z_crossings), rtol=1e-3,
															atol=1e-5):
		print_success('Trajectory z-crossings are equal')
	else:
		print_failure('Trajectory z-crossings are not equal, maximum '
			+ f'difference is {np.max(z_crossings) - np.min(z_crossings)}')
		fig(r'$x$', r'$z$')
		plt.plot(x, z, c='k')
		plt.axhline(z_crossings[0], c=COLORS[-1], ls=':')
		plt.scatter(x_crossings, z_crossings, c='k', marker='x')
		plt.show()
		quit()

	# compare the horizontal drift velocity to the analytical solution
	u_bar = np.mean(u) / (wave.steepness * wave.steepness)
	z_bar = np.mean(z_crossings)
	u_d = analytical_stokes_drift(wave, z_bar)
	if match_data(np.array([u_d]), np.array([u_bar])):
		print_success('Horizontal Stokes drift velocity matches the '
					+ 'analytical solution\n')
		success = True
	else:
		print_failure('Horizontal Stokes drift velocity does not match '
					+ 'the analytical solution\n')
		success = False
	return success, u_bar, z_bar / (wave.wavenum * wave.depth)

if __name__ == '__main__':
	main()
