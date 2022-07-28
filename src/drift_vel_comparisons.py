import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

import ocean_wave

def main():
	# values of h for each depth case
	shallow_h = 1
	intermediate_h = 3
	deep_h = 8	

	# run comparisons for each depth case
	print('Running comparisons for the shallow case...')
	shallow_numerics, shallow_analytics, shallow_z0, shallow_label \
		= run_comparisons(shallow_h)
	print('done. Running comparisons for the intermediate case...')
	intermediate_numerics, intermediate_analytics, intermediate_z0, \
		intermediate_label = run_comparisons(intermediate_h)
	print('done. Running comparisons for the deep case...')
	deep_numerics, deep_analytics, deep_z0, deep_label \
		= run_comparisons(deep_h)
	print('done.')

	# plot results
	# analytical solutions for the shallow case
	plt.plot(shallow_analytics, shallow_z0 / shallow_h, c='deeppink',
			 linestyle='--', label='Shallow analytical solution')

	# numerical solutions for the shallow case
	plt.scatter(shallow_numerics, shallow_z0 / shallow_h, c='k', marker='^',
				label='Shallow numerical solution')	

	# analytical solutions for the intermediate case
	plt.plot(intermediate_analytics, intermediate_z0 / intermediate_h,
			 c='mediumpurple', linestyle='--',
			 label='Intermediate analytical solution')

	# numerical solutions for the intermediate case
	plt.scatter(intermediate_numerics, intermediate_z0 / intermediate_h,
				c='k', marker='o', label='Intermediate numerical solution')

	# analytical solutions for the deep case	
	plt.plot(deep_analytics, deep_z0 / deep_h, c='cornflowerblue',
			 linestyle='--', label='Deep analytical solution')

	# numerical solutions for the deep case
	plt.scatter(deep_numerics, deep_z0 / deep_h, c='k', marker='s',
				label='Deep numerical solution')	

	plt.title('Drift Velocity Comparisons for Varying Depths', fontsize=16)
	plt.xlabel(r'Drift Velocity $ u_d $', fontsize=14)
	plt.ylabel(r'$ \frac{z}{h} $', fontsize=14)
	plt.legend()
	plt.show()

def run_comparisons(depth):
	"""
	Runs drift velocity comparisons for a specified depth.

	Parameters
	----------
	depth : float
		The depth of the water, denoted h in the corresponding mathematics.

	Returns
	-------
	Arrays containing the normalized numerical solutions, analytical solutions,
	and initial depths, and a float representing h over lambda.
	"""
	# initialize ocean wave object and other variables
	my_wave = ocean_wave.OceanWave(amplitude=0.1, depth=depth,
			  stokes_num=1e-10)
	initial_depths = np.linspace(0, -depth, num=10, endpoint=False)
	x_0 = 0								
	# compute solutions
	numerical_sol, analytical_sol = my_wave.compare_drift_velocities(
											my_wave.mr_no_history,
											initial_depths, x_0)
	# normalize drift velocities
	numerical_sol /= np.sqrt(constants.g / my_wave.get_wave_num()
										 * np.tanh(my_wave.get_wave_num()
										 * depth))
	analytical_sol /= np.sqrt(constants.g / my_wave.get_wave_num()
										 * np.tanh(my_wave.get_wave_num()
										 * depth))
	# return normalized solutions
	return numerical_sol, analytical_sol, initial_depths, \
		   depth / my_wave.get_wavelength()

main()
