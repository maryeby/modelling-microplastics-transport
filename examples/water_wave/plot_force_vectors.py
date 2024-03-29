import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../data/water_wave/'

def main():
	"""
	This program plots the contributions of various forces on the movement of a
	negatively buoyant inertial particle in a linear water wave.
	"""
	numerics = pd.read_csv(f'{DATA_PATH}numerics.csv') # read data

	# create condition and 'get' function to help filter through numerical data
	x_0, z_0 = 0, 0
	St, beta = 0.01, 0.8
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
								 & (numerics['St'] == St) \
								 & (numerics['beta'] == beta) \
								 & (numerics['history'] == True) \
								 & (numerics['h\''] == h) \
								 & (numerics['A\''] == A) \
								 & (numerics['wavelength\''] == wavelength) \
								 & (numerics['delta_t\''] == delta_t)
	get = lambda name : numerics[name].where(condition).dropna().to_numpy()

	# retrieve relevant numerical results
	x = get('x')
	z = get('z')
	t = get('t')

	fpg_x = get('fluid_pressure_gradient_x')
	buoyancy_x = get('buoyancy_force_x')
	mass_x = get('added_mass_force_x')
	drag_x = get('stokes_drag_x')
	history_x = get('history_force_x')

	fpg_z = get('fluid_pressure_gradient_z')
	buoyancy_z = get('buoyancy_force_z')
	mass_z = get('added_mass_force_z')
	drag_z = get('stokes_drag_z')
	history_z = get('history_force_z')

	# compute positions of various points on the particle trajectory plot
	period = t.shape[0] // 50
	n = period * 3
	A = period
	B = int(period * 1.25)
	C = int(period * 1.5)
	D = int(period * 1.75)

	# initialize force vectors for plt.quiver()
	X = [x[A], x[B], x[C], x[D]]
	Y = [z[A], z[B], z[C], z[D]]
	fpg_U = [fpg_x[A], fpg_x[B], fpg_x[C], fpg_x[D]]
	fpg_V = [fpg_z[A], fpg_z[B], fpg_z[C], fpg_z[D]]
	buoyancy_U = [buoyancy_x[A], buoyancy_x[B], buoyancy_x[C], buoyancy_x[D]]
	buoyancy_V = [buoyancy_z[A], buoyancy_z[B], buoyancy_z[C], buoyancy_z[D]]
	mass_U = [mass_x[A], mass_x[B], mass_x[C], mass_x[D]]
	mass_V = [mass_z[A], mass_z[B], mass_z[C], mass_z[D]]
	drag_U = [drag_x[A], drag_x[B], drag_x[C], drag_x[D]]
	drag_V = [drag_z[A], drag_z[B], drag_z[C], drag_z[D]]
	history_U = [history_x[A], history_x[B], history_x[C], history_x[D]]
	history_V = [history_z[A], history_z[B], history_z[C], history_z[D]]

	# initialize force over time figure & top subplot
	plt.figure(1)
	plt.suptitle(r'Forces Acting on a Negatively Buoyant Particle', fontsize=18)
	plt.subplot(211)
	plt.ylabel(r'horizontal force', fontsize=16)
	plt.axvline(t[A])
	plt.axvline(t[B])
	plt.axvline(t[C])
	plt.axvline(t[D])
	plt.xticks([])
	plt.yticks(fontsize=14)
	plt.minorticks_on()
	labels = ['fluid pressure gradient', 'buoyancy force', 'added mass force',
			  'Stokes drag', 'history force']
	colors= ['grey', 'darkturquoise', 'forestgreen', 'limegreen', 'cyan']

	# plot horizontal forces
	plt.plot(t[:n], fpg_x[:n], c=colors[0], label=labels[0])
	plt.plot(t[:n], buoyancy_x[:n], c=colors[1], label=labels[1])
	plt.plot(t[:n], mass_x[:n], c=colors[2], label=labels[2])
	plt.plot(t[:n], drag_x[:n], c=colors[3], label=labels[3])
	plt.plot(t[:n], history_x[:n], c=colors[4], label=labels[4])
	plt.legend(fontsize=14)

	# initialize bottom subplot
	plt.subplot(212)
	plt.xlabel(r'time', fontsize=16)
	plt.ylabel(r'vertical force', fontsize=16)
	plt.axvline(t[A])
	plt.axvline(t[B])
	plt.axvline(t[C])
	plt.axvline(t[D])
	plt.xticks(ticks=[0, 0.5, t[A], t[B], t[C], t[D], 1.5, 2, 2.5],
			   labels=['0', '0.5', 'A', 'B', 'C', 'D', '1.5', '2', '2.5'],
			   fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	# plot vertical numerical results
	plt.plot(t[:n], fpg_z[:n], c=colors[0])
	plt.plot(t[1:n], buoyancy_z[1:n], c=colors[1])
	plt.plot(t[:n], mass_z[:n], c=colors[2])
	plt.plot(t[15:n], drag_z[15:n], c=colors[3])
	plt.plot(t[5:n], history_z[5:n], c=colors[4])

	# initialize particle trajectory figure
	plt.figure(2)
	plt.title(r'Particle Trajectory with Force Vectors', fontsize=18)
	plt.xlabel('x', fontsize=14)
	plt.ylabel('z', fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()
	w = 5e-3

	# plot particle trajectory with force vectors
	plt.plot(x[:n], z[:n], c='k')
	plt.quiver(X, Y, fpg_U, fpg_V, color=colors[0], label=labels[0], width=w)
	plt.quiver(X, Y, buoyancy_U, buoyancy_V, color=colors[1], label=labels[1],
			   width=w)
	plt.quiver(X, Y, mass_U, mass_V, color=colors[2], label=labels[2], width=w)
	plt.quiver(X, Y, drag_U, drag_V, color=colors[3], label=labels[3], width=w)
	plt.quiver(X, Y, history_U, history_V, color=colors[4], label=labels[4],
			   width=w)
	plt.text(x[A] + 1e-3, z[A] + 1e-3, 'A', ha='left', va='bottom', fontsize=14)
	plt.text(x[B] + 1e-3, z[B] + 1e-3, 'B', ha='left', va='bottom', fontsize=14)
	plt.text(x[C] + 1e-3, z[C] + 1e-3, 'C', ha='left', va='bottom', fontsize=14)
	plt.text(x[D] + 1e-3, z[D] - 1e-3, 'D', ha='left', va='top', fontsize=14)
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
