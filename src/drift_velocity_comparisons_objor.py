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
	shallow_numerics, shallow_analytics, shallow_z0, shallow_label \
		= run_comparisons(shallow_h)
	intermediate_numerics, intermediate_analytics, intermediate_z0, \
		intermediate_label = run_comparisons(intermediate_h)
	deep_numerics, deep_analytics, deep_z0, deep_label \
		= run_comparisons(deep_h)

	# plot results
	# analytical solutions for the shallow case
	plt.plot(shallow_analytics, shallow_z0 / shallow_h, c='deeppink',
			 linestyle='--',
			 label=r'Analytical solution for $ h / \lambda = {:.2f} $'\
					 .format(shallow_label))

	# numerical solutions for the shallow case
	plt.scatter(shallow_numerics, shallow_z0 / shallow_h, c='k', marker='^',
				label=r'Numerical solution for $ h / \lambda = {:.2f} $'\
					    .format(shallow_label))	

	# analytical solutions for the intermediate case
	plt.plot(intermediate_analytics, intermediate_z0 / intermediate_h,
			 c='mediumpurple', linestyle='--',
			 label=r'Analytical solution for $ h / \lambda = {:.2f} $'\
					 .format(intermediate_label))

	# numerical solutions for the intermediate case
	plt.scatter(intermediate_numerics, intermediate_z0 / intermediate_h,
				c='k', marker='o',
				label=r'Numerical solution for $ h / \lambda = {:.2f} $'\
					    .format(intermediate_label))

	# analytical solutions for the deep case	
	plt.plot(deep_analytics, deep_z0 / deep_h, c='cornflowerblue',
			 linestyle='--',
			 label=r'Analytical solution for $ h / \lambda = {:.2f} $'\
					 .format(deep_label))

	# numerical solutions for the deep case
	plt.scatter(deep_numerics, deep_z0 / deep_h, c='k', marker='s',
				label=r'Numerical solution for $ h / \lambda = {:.2f} $'\
					    .format(deep_label))	

	plt.title('Drift Velocity Comparisons for Varying Depths')
	plt.xlabel(r'Drift velocity $ u_d $')
	plt.ylabel(r'$ \frac{z}{h} $')
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
	Arrays containing the numerical solutions, analytical solutions, and
	initial depths, and a float representing h over lambda.
	"""
	my_wave = ocean_wave.OceanWave(depth=depth, stokes_num=1e-10)
	initial_depths = np.linspace(0, -depth, 10)	# list of z_0 values
	x_0 = 0										# initial horizontal position
	numerical_sol, analytical_sol =  my_wave.compare_drift_velocities(
												  initial_depths, x_0)
	return numerical_sol, analytical_sol, initial_depths, \
		   depth / my_wave.get_wavelength()

main()
