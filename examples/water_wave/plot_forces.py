import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import water_wave as fl

DATA_PATH = '../data/water_wave/'

def main():
	"""
	This program plots the numerical solutions for the contributions of various
	forces on the movement of a negatively buoyant inertial particle in a linear
	water wave, not including the history force.
	"""
	numerics = pd.read_csv(f'{DATA_PATH}numerics.csv') # read data

	# initialize force over time figure & top subplot
	plt.figure(1)
	plt.suptitle(r'Forces Acting on a Negatively Buoyant Particle', fontsize=18)
	plt.subplot(211)
	plt.ylabel(r'horizontal force', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# create conditions to help filter through numerical data
	x_0, z_0 = 0, 0
	St, beta = 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	my_wave = fl.WaterWave(depth=h, amplitude=A, wavelength=wavelength)
	history = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									 & (numerics['St'] == St) \
									 & (numerics['beta'] == beta) \
									 & (numerics['history'] == True) \
									 & (numerics['h\''] == h) \
									 & (numerics['A\''] == A) \
									 & (numerics['wavelength\''] == wavelength)\
									 & (numerics['delta_t\''] == delta_t)
	no_history = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									 & (numerics['St'] == St) \
									 & (numerics['beta'] == beta) \
									 & (numerics['history'] == False) \
									 & (numerics['h\''] == h) \
									 & (numerics['A\''] == A) \
									 & (numerics['wavelength\''] == wavelength)\
									 & (numerics['delta_t\''] == delta_t)

	# retrieve relevant numerical results
	x = numerics['x'].where(no_history).dropna()
	z = numerics['z'].where(no_history).dropna()
	xdot = numerics['xdot'].where(no_history).dropna()
	zdot = numerics['zdot'].where(no_history).dropna()
	t = numerics['t'].where(no_history).dropna()
	n = t.shape[0] // 4

	fpg_x = numerics['fluid_pressure_gradient_x'].where(no_history).dropna()
	buoyancy_x = numerics['buoyancy_force_x'].where(no_history).dropna()
	mass_x = numerics['added_mass_force_x'].where(no_history).dropna()
	drag_x = numerics['stokes_drag_x'].where(no_history).dropna()

	u_x, u_z = my_wave.velocity(x, z, t)
	w_x = xdot - u_x
	A_x = np.gradient(w_x, t)
	G_x = fpg_x + buoyancy_x + mass_x + drag_x

	# plot horizontal forces
	plt.plot(t[:n], fpg_x[:n], c='grey', label='fluid pressure gradient')
	plt.plot(t[:n], buoyancy_x[:n], c='darkturquoise', label='buoyancy force')
	plt.plot(t[:n], mass_x[:n], c='forestgreen', label='added mass force')
	plt.plot(t[:n], drag_x[:n], c='limegreen', label='Stokes drag')
	plt.plot(t[10:n], A_x[10:n] - G_x[10:n], c='k', label='verification')
	plt.legend(fontsize=14)

	# initialize bottom subplot
	plt.subplot(212)
	plt.xlabel(r'time', fontsize=16)
	plt.ylabel(r'vertical force', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical results
	fpg_z = numerics['fluid_pressure_gradient_z'].where(no_history).dropna()
	buoyancy_z = numerics['buoyancy_force_z'].where(no_history).dropna()
	mass_z = numerics['added_mass_force_z'].where(no_history).dropna()
	drag_z = numerics['stokes_drag_z'].where(no_history).dropna()

	w_z = zdot - u_z
	A_z = np.gradient(w_z, t)
	G_z = fpg_z + buoyancy_z + mass_z + drag_z

	# plot vertical numerical results
	plt.plot(t[:n], fpg_z[:n], c='grey')
	plt.plot(t[1:n], buoyancy_z[1:n], c='darkturquoise')
	plt.plot(t[:n], mass_z[:n], c='forestgreen')
	plt.plot(t[15:n], drag_z[15:n], c='limegreen')
	plt.plot(t[15:n], A_z[15:n] - G_z[15:n], c='k')

	# initialize particle trajectory figure
	plt.figure(2)
	plt.title(r'Particle Trajectory', fontsize=18)
	plt.xlabel('x', fontsize=14)
	plt.ylabel('z', fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# retrieve relevant numerical results
	x = numerics['x'].where(no_history).dropna()
	z = numerics['z'].where(no_history).dropna()
	x_history = numerics['x'].where(history).dropna()
	z_history = numerics['z'].where(history).dropna()

	# plot particle trajectory with and without history
	plt.plot(x, z, c='k', label='without history')
	plt.plot(x_history, z_history, c='k', ls='--', label='with history')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
