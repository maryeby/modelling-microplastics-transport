import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program plots the results of the leading order, first order, and second
	order inertial equations as derived in Santamaria et al. (2013) and Haller &
	Sapsis (2008), as well as the numerical solutions generated using the
	Daitche (2013) method.
	"""
	data_path = '../data/deep_water_wave/'
	numerics = pd.read_csv(data_path + 'inertial_equations.csv')
	global_error = pd.read_csv(data_path + 'global_error.csv')
	computation_times = pd.read_csv(data_path + 'computation_times.csv')

	plt.figure(1)
	plt.title('Comparing Inertial Equations', fontsize=18)
	plt.xlabel('x', fontsize=16)
	plt.ylabel('z', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.plot('x_santamaria', 'z_santamaria', c='grey', marker='o', lw=3,
			 data=numerics, label='Numerical (Santamaria)')
	plt.plot('x_haller', 'z_haller', '.-k', lw=2, data=numerics,
			 label='Numerical (Haller)')
	plt.plot('x_daitche', 'z_daitche', c='mediumpurple', marker='o',
			 data=numerics, label='Numerical (Daitche)')
	plt.plot('x0_santamaria', 'z0_santamaria', c='grey', lw=3, ls='--',
			 marker='o', data=numerics, label='Leading order (Santamaria)')
	plt.plot('x0_haller', 'z0_haller', '.--k', lw=2, data=numerics,
			 label='Leading order (Haller)')
	plt.plot('x1_santamaria', 'z1_santamaria', c='grey', lw=3, ls='-.',
			 marker='o', data=numerics, label='First order (Santamaria)')
	plt.plot('x1_haller', 'z1_haller', '.-.k', lw=2, data=numerics,
			 label='First order (Haller)')
	plt.plot('x2_santamaria', 'z2_santamaria', c='grey', lw=3, ls=':',
			 marker='o', data=numerics, label='Second order (Santamaria)')
	plt.plot('x2_haller', 'z2_haller', '.:k', lw=2, data=numerics,
			 label='Second order (Haller)')
	plt.legend(fontsize=14)
	plt.tight_layout()

	plt.figure(2)
	plt.title('Global Error: Deep Water Waves Without History',
			  fontsize=18)
	plt.xlabel(r'$\Delta t$', fontsize=16)
	plt.ylabel(r'$\mathcal{\epsilon}$', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.minorticks_on()
	plt.axis([1e-3, 5e-2, 1e-7, 10])
	h_scale = np.linspace(2e-3, 2e-2, 10)

	plt.plot(h_scale, h_scale * 2, c='grey', ls='--', label=r'~$h$')
	plt.plot(h_scale, (h_scale ** 2) * 2.3, c='grey', ls='-.', label=r'~$h^2$')
#	plt.plot(h_scale, (h_scale ** 3), c='grey', ls=':', label=r'~$h^3$')
	plt.plot('delta_t', 'global_error1', '.--k', data=global_error,
			 label='first order')
	plt.plot('delta_t', 'global_error2', '.-.k', data=global_error,
			 label='second order')
	plt.plot('delta_t', 'global_error3', '.:k', data=global_error,
			 label='third order')
	plt.legend(fontsize=14)
	plt.tight_layout()

	plt.figure(3)
	plt.title('Timestep Size vs Computation Time: Deep Water Waves',
			  fontsize=18)
	plt.xlabel(r'$\Delta t$', fontsize=16)
	plt.ylabel('computation time (s)', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.minorticks_on()
	plt.plot('delta_t', 'computation_time', '.-k', data=computation_times)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
