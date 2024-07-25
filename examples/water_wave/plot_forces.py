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
	This program plots the contributions of various forces on the movement of a
	negatively buoyant inertial particle in a linear water wave.
	"""
	numerics = pd.read_csv(f'{DATA_PATH}numerics.csv') # read data

	# create condition and 'get' function to help filter through numerical data
	x_0, z_0 = 0, 0
	St, beta = 0.01, 0.8
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	base_condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
								 & (numerics['St'] == St) \
								 & (numerics['beta'] == beta) \
								 & (numerics['history'] == True) \
								 & (numerics['h\''] == h) \
								 & (numerics['A\''] == A) \
								 & (numerics['wavelength\''] == wavelength)
	condition1 = base_condition & (numerics['delta_t\''] == 5e-3)
	condition2 = base_condition & (numerics['delta_t\''] == 5e-4)
	get = lambda name : numerics[name].where(condition1).dropna().to_numpy()
	initial = lambda name : numerics[name].where(condition2).dropna().to_numpy()

	# retrieve relevant numerical results
	x = get('x')
	z = get('z')
	t = get('t')

	fpg_x = get('fluid_pressure_gradient_x')
	buoyancy_x = get('buoyancy_force_x')
	mass_x = get('added_mass_force_x')
	drag_x = get('stokes_drag_x')
	history_x = get('history_force_x')
	inertial_x = fpg_x + mass_x

	fpg_z = get('fluid_pressure_gradient_z')
	buoyancy_z = get('buoyancy_force_z')
	mass_z = get('added_mass_force_z')
	drag_z = get('stokes_drag_z')
	history_z = get('history_force_z')
	inertial_z = fpg_z + mass_z

	# verification
	my_wave = fl.WaterWave(depth=h, amplitude=A, wavelength=wavelength)
	xdot = get('xdot')
	zdot = get('zdot')
	u_x, u_z = my_wave.velocity(x, z, t)
	w_x = xdot - u_x
	w_z = zdot - u_z
	A_x = np.gradient(w_x, t)
	A_z = np.gradient(w_z, t)
	G_x = fpg_x + buoyancy_x + mass_x + drag_x
	G_z = fpg_z + buoyancy_z + mass_z + drag_z

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
	inertial_U = [inertial_x[A], inertial_x[B], inertial_x[C], inertial_x[D]]
	inertial_V = [inertial_z[A], inertial_z[B], inertial_z[C], inertial_z[D]]
	buoyancy_U = [buoyancy_x[A], buoyancy_x[B], buoyancy_x[C], buoyancy_x[D]]
	buoyancy_V = [buoyancy_z[A], buoyancy_z[B], buoyancy_z[C], buoyancy_z[D]]
	drag_U = [drag_x[A], drag_x[B], drag_x[C], drag_x[D]]
	drag_V = [drag_z[A], drag_z[B], drag_z[C], drag_z[D]]
	history_U = [history_x[A], history_x[B], history_x[C], history_x[D]]
	history_V = [history_z[A], history_z[B], history_z[C], history_z[D]]

	# scale horizontal and vertical force vectors
	U_forces = [inertial_U, buoyancy_U, drag_U, history_U]
	V_forces = [inertial_V, buoyancy_V, drag_V, history_V]
	U_scale = max(max([np.absolute(i).tolist() for i in U_forces])) * 10
	V_scale = max(max([np.absolute(i).tolist() for i in V_forces])) * 10
	inertial_U, buoyancy_U, drag_U, history_U = np.asarray(U_forces) / U_scale
	inertial_V, buoyancy_V, drag_V, history_V = np.asarray(V_forces) / V_scale

	# initialize figure
	plt.figure()
	zeros = [0, 0, 0, 0]
	labels = ['inertial forces', 'gravity', 'Stokes drag', 'history force']
	colors= ['#dfc27d', '#80cdc1', '#a6611a', '#018571', 'grey']
	os = 1e-3	# offset for text positions
	fs = 14		# fontsize
	lfs = 16	# large fontsize

	# initialize top left subplot
	plt.subplot(221)
	plt.ylabel('z', fontsize=fs)
	plt.gca().set_aspect('equal')
	plt.gca().set_box_aspect(1)
	plt.xticks([])
	plt.yticks(fontsize=lfs)
	plt.minorticks_on()

	# plot particle trajectory with horizontal force vectors
	plt.plot(x[:n], z[:n], c='k')
	plt.quiver(X, Y, inertial_U, zeros, color=colors[0], label=labels[0],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, buoyancy_U, zeros, color=colors[1], label=labels[1],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, drag_U, zeros, color=colors[2], label=labels[2], scale=1,
			   angles='xy', scale_units='xy')
	plt.quiver(X, Y, history_U, zeros, color=colors[3], label=labels[3],
			   scale=1, angles='xy', scale_units='xy')
	
	# add labels
	plt.text(x[A] + os, z[A] + os, 'A', ha='left', va='bottom', fontsize=fs)
	plt.text(x[B] + os, z[B] + os, 'B', ha='left', va='bottom', fontsize=fs)
	plt.text(x[C] + os, z[C] - os, 'C', ha='left', va='top', fontsize=fs)
	plt.text(x[D] - os, z[D] - os, 'D', ha='right', va='bottom', fontsize=fs)

	# initialize top right subplot
	plt.subplot(222)
	plt.ylabel(r'horizontal force', fontsize=lfs)
	plt.axvline(t[A], c=colors[4])
	plt.axvline(t[B], c=colors[4])
	plt.axvline(t[C], c=colors[4])
	plt.axvline(t[D], c=colors[4])
	plt.xticks([])
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot horizontal forces
	plt.plot(t[:n], inertial_x[:n], c=colors[0], label=labels[0])
	plt.plot(t[:n], buoyancy_x[:n], c=colors[1], label=labels[1])
	plt.plot(t[:n], drag_x[:n], c=colors[2], label=labels[2])
	plt.plot(t[:n], history_x[:n], c=colors[3], label=labels[3])
	plt.plot(t[:n], A_x[:n] - G_x[:n], ':k', label='verification')

	# initialize bottom left subplot
	plt.subplot(223)
	plt.xlabel('x', fontsize=lfs)
	plt.ylabel('z', fontsize=lfs)
	plt.gca().set_aspect('equal')
	plt.gca().set_box_aspect(1)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot particle trajectory with vertical force vectors
	plt.plot(x[:n], z[:n], c='k')
	plt.quiver(X, Y, zeros, inertial_V, color=colors[0], label=labels[0],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, zeros, buoyancy_V, color=colors[1], label=labels[1],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, zeros, drag_V, color=colors[2], label=labels[2], scale=1,
			   angles='xy', scale_units='xy')
	plt.quiver(X, Y, zeros, history_V, color=colors[3], label=labels[3],
			   scale=1, angles='xy', scale_units='xy')
	
	# add labels
	plt.text(x[A] + os, z[A] + os, 'A', ha='left', va='bottom', fontsize=fs)
	plt.text(x[B] + os, z[B] + os, 'B', ha='left', va='bottom', fontsize=fs)
	plt.text(x[C] + os, z[C] - os, 'C', ha='left', va='top', fontsize=fs)
	plt.text(x[D] - os, z[D] - os, 'D', ha='right', va='bottom', fontsize=fs)

	# initialize bottom right subplot
	plt.subplot(224)
	plt.xlabel(r'time', fontsize=lfs)
	plt.ylabel(r'vertical force', fontsize=lfs)
	plt.axvline(t[A], c=colors[4])
	plt.axvline(t[B], c=colors[4])
	plt.axvline(t[C], c=colors[4])
	plt.axvline(t[D], c=colors[4])
#	plt.axhline(0, ls=':', c=colors[4])
	plt.xticks(ticks=[0, 0.5, t[A], t[B], t[C], t[D], 2, 2.5],
			   labels=['0', '0.5', 'A', 'B', 'C', 'D', '2', '2.5'], fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot vertical numerical results
	plt.plot(t[:n], inertial_z[:n], c=colors[0], label=labels[0])
	plt.plot(t[1:n], buoyancy_z[1:n], c=colors[1], label=labels[1])
	plt.plot(t[:n], drag_z[:n], c=colors[2], label=labels[2])

	# enforce history = 0 at t = 0 for the verification
	A_x[0], A_z[0], G_x[0], G_z[0] = 0, 0, 0, 0
	plt.plot(t[:n], A_z[:n] - G_z[:n], ':k', label='verification')

	# append initial data with a smaller time step
	initial_history_z = initial('history_force_z')
	initial_t = initial('t')
	initial_t = initial_t[:initial_history_z.size - 2]
	t = np.concatenate((initial_t, t[2:]))
	history_z = np.concatenate((initial_history_z[:-2], history_z[2:]))

	# enforce history = 0 at t = 0 for the numerical solution
	history_x[0], history_z[0] = 0, 0

	plt.plot(t[:n], history_z[:n], c=colors[3], label=labels[3], marker='.')
	plt.legend(fontsize=fs, loc='lower right')
	plt.show()

if __name__ == '__main__':
	main()
