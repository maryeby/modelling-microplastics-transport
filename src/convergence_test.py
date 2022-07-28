import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy import constants
from tqdm import tqdm

import ocean_wave
# TODO refactor & make object-oriented

def main():
	# compute the relative errors for each depth case
	stokes_nums = np.linspace(0.1, 0, num=20, endpoint=False)
	print('Computing relative errors for the shallow case...')
	shallow_errors = relative_errors(stokes_nums, 1)
	print('done. Computing relative errors for the intermediate case...')
	intermediate_errors = relative_errors(stokes_nums, 3)
	print('done. Computing relative errors for the deep case...')
	deep_errors = relative_errors(stokes_nums, 8)

	# plot the Stokes numbers vs relative errors for each depth case
	plt.figure()
	plt.subplot(121)
	plt.xlabel('Stokes Number')
	plt.ylabel('Relative Error')
	plt.suptitle('Convergence Test')
	plt.plot(stokes_nums, shallow_errors, c='deeppink', marker='o',
			 label='Shallow case')
	plt.plot(stokes_nums, intermediate_errors, c='mediumpurple', marker='o',
			 label='Intermediate case')
	plt.plot(stokes_nums, deep_errors, c='cornflowerblue', marker='o',
			 label='Deep case')
	plt.legend()

	# compute the critical stokes number for various depths
	print('done. Computing the set of critical stokes numbers at varying '
		  + 'depths...')
	depths = np.linspace(1, 10, num=20)
	critical_stokes_nums = [optimize.minimize(relative_errors, [0.1],
											  args=(depth),
											  bounds=((1e-100, 0.1),),
											  tol=1e-3).x
							for depth in tqdm(depths)]
	print(depths)
	print(critical_stokes_nums)

	# plot the depths vs critical stokes numbers
	plt.subplot(122)
	plt.xlabel(r'Depth $ h $')
	plt.ylabel('Critical Stokes Number')
	plt.plot(depths, critical_stokes_nums, c='k', marker='o')
	plt.show()

def relative_errors(stokes_nums, depth):
	"""
	Compares drift velocities for various Stokes numbers.

	Parameters
	----------
	stokes_nums : int, float, or array
		Number(s) to be used as the Stokes number, St.
	depth : float
		The depth of the water, h in the correpsonding mathematics.

	Returns
	-------
	array
		An array of floats representing the relative errors.
	"""
	amplitude = 0.1						# A
	wavelength = 10 					# lambda
	wave_num = 2 * np.pi / wavelength 	# k
	# calculate omega using dispersion relation
	angular_freq = np.sqrt(constants.g * wave_num * np.tanh(wave_num * depth))
	period = 2 * np.pi / angular_freq	# period of particle oscillation
	t_span = (0, 10 * period)			# time span
	density = 2 / 3						# R
	x_0 = 0								# initial horizontal position
	initial_depth = [0]					# z_0
	rel_errors = []

	# if the stokes_nums argument is a single number, make it an array
	if isinstance(stokes_nums, int) or isinstance(stokes_nums, float):
		stokes_nums = np.array([stokes_nums])
	
	# compute the relative error for each Stokes number
	for num in stokes_nums:
		numerical_sol, analytical_sol = compare_drift_velocities(
											initial_depth, x_0, period,
											wave_num, depth, angular_freq,
											amplitude, density, num)
		rel_errors.append(np.abs(numerical_sol - analytical_sol) 
						  / analytical_sol)

	return np.array(rel_errors).ravel()

main()
