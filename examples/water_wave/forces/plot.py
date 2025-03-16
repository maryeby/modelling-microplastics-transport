import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_subplot as subplot
from utils.plot import FS
from utils.colors import COLORS
from models import water_wave as fl
from examples.water_wave.forces.numerics import DEPTH, AMPLITUDE, WAVELENGTH, \
	 NUM_PERIODS
from examples.water_wave.forces.analysis import ST_TO_SAVE as STOKES_NUM

ZEROS = [0, 0, 0, 0]
LABELS = ['inertial forces', 'gravity', 'Stokes drag', 'history force']
OS = 1e-3			# offset for text positions
IN_FILE = '../../data/water_wave/forces_numerics.csv'

def main():
	"""Plot the trajectory of a particle and the forces over time."""
	numerics = pd.read_csv(IN_FILE)

	# retrieve relevant numerical results
	params = {'St': STOKES_NUM}
	names = ['t', 'x', 'z', 'xdot', 'zdot', 'fluid_pressure_gradient_x',
			 'fluid_pressure_gradient_z', 'buoyancy_force_x',
			 'buoyancy_force_z', 'added_mass_force_x', 'added_mass_force_z',
			 'stokes_drag_x', 'stokes_drag_z', 'history_force_x',
			 'history_force_z']
	data = extract_data(names, numerics, params)
	for s in range(len(data)): data[s] = data[s].to_numpy()[:-5]
	t, x, z, xdot, zdot, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z, \
	   drag_x, drag_z, history_x, history_z = data
	inertial_x = fpg_x + mass_x
	inertial_z = fpg_z + mass_z

	# verification
	wave = fl.WaterWave(DEPTH, AMPLITUDE, WAVELENGTH)
	u_x, u_z = wave.velocity(x, z, t)
	w_x = xdot - u_x
	w_z = zdot - u_z
	A_x = np.gradient(w_x, t)
	A_z = np.gradient(w_z, t)
	G_x = fpg_x + buoyancy_x + mass_x + drag_x
	G_z = fpg_z + buoyancy_z + mass_z + drag_z

	# compute positions of various points on the particle trajectory plot
	period = int(t.shape[0] // NUM_PERIODS)
	A = int(period * 1.27)
	B = int(period * 1.46)
	C = int(period * 1.7)
	D = int(period * 1.93)

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

	# plot particle trajectory with horizontal force vectors (top left)
	plt.figure()
	subplot(221, y_label='z', equal_aspect=True, make_square=True)
	plt.plot(x, z, c='k')
	plt.quiver(X, Y, inertial_U, ZEROS, color=COLORS[0], label=LABELS[0],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, buoyancy_U, ZEROS, color=COLORS[1], label=LABELS[1],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, drag_U, ZEROS, color=COLORS[2], label=LABELS[2], scale=1,
			   angles='xy', scale_units='xy')
	plt.quiver(X, Y, history_U, ZEROS, color=COLORS[3], label=LABELS[3],
			   scale=1, angles='xy', scale_units='xy')
	
	# add labels
	plt.text(x[A] + OS, z[A] + OS, 'A', ha='left', va='bottom', fontsize=FS)
	plt.text(x[B] + OS, z[B] + OS, 'B', ha='left', va='bottom', fontsize=FS)
	plt.text(x[C] + OS, z[C] - OS, 'C', ha='left', va='top', fontsize=FS)
	plt.text(x[D] - OS, z[D] - OS, 'D', ha='right', va='bottom', fontsize=FS)

	# initialize top right subplot
	subplot(222, y_label='horizontal force')
	plt.axvline(t[A], c=COLORS[4])
	plt.axvline(t[B], c=COLORS[4])
	plt.axvline(t[C], c=COLORS[4])
	plt.axvline(t[D], c=COLORS[4])

	# enforce history = 0 at t = 0 for the numerical solution
	history_x[0] = 0
	history_x[1] = A_x[1] - G_x[1]

	# plot horizontal forces
	plt.plot(t, inertial_x, c=COLORS[0], label=LABELS[0])
	plt.plot(t, buoyancy_x, c=COLORS[1], label=LABELS[1])
	plt.plot(t, drag_x, c=COLORS[2], label=LABELS[2])
	plt.plot(t, history_x, c=COLORS[3], label=LABELS[3])
	plt.plot(t, A_x - G_x, ':k', label='verification')

	# plot particle trajectory with vertical force vectors (bottom left)
	subplot(223, 'x', 'z', equal_aspect=True, make_square=True)
	plt.plot(x, z, c='k')
	plt.quiver(X, Y, ZEROS, inertial_V, color=COLORS[0], label=LABELS[0],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, ZEROS, buoyancy_V, color=COLORS[1], label=LABELS[1],
			   scale=1, angles='xy', scale_units='xy')
	plt.quiver(X, Y, ZEROS, drag_V, color=COLORS[2], label=LABELS[2], scale=1,
			   angles='xy', scale_units='xy')
	plt.quiver(X, Y, ZEROS, history_V, color=COLORS[3], label=LABELS[3],
			   scale=1, angles='xy', scale_units='xy')
	
	# add labels
	plt.text(x[A] + OS, z[A] + OS, 'A', ha='left', va='bottom', fontsize=FS)
	plt.text(x[B] + OS, z[B] + OS, 'B', ha='left', va='bottom', fontsize=FS)
	plt.text(x[C] + OS, z[C] - OS, 'C', ha='left', va='top', fontsize=FS)
	plt.text(x[D] - OS, z[D] - OS, 'D', ha='right', va='bottom', fontsize=FS)

	# initialize bottom right subplot
	subplot(224, 'time', 'vertical force')
	plt.xticks(ticks=[0, 0.5, t[A], t[B], t[C], t[D], 2, 2.5],
			   labels=['0', '0.5', 'A', 'B', 'C', 'D', '2', '2.5'], fontsize=FS)
	plt.axvline(t[A], c=COLORS[4])
	plt.axvline(t[B], c=COLORS[4])
	plt.axvline(t[C], c=COLORS[4])
	plt.axvline(t[D], c=COLORS[4])
#	plt.axhline(0, ls=':', c=COLORS[4])

	# plot vertical numerical results
	plt.plot(t, inertial_z, c=COLORS[0], label=LABELS[0])
	plt.plot(t[1:], buoyancy_z[1:], c=COLORS[1], label=LABELS[1])
	plt.plot(t, drag_z, c=COLORS[2], label=LABELS[2])

	# append initial data with a smaller time step
#	initial_history_z = 'history_force_z'
#	initial_t = initial('t')
#	initial_t = initial_t[:initial_history_z.size - 2]
#	t = np.concatenate((initial_t, t[2:]))
#	history_z = np.concatenate((initial_history_z[:-2], history_z[2:]))

	# enforce history = 0 at t = 0 for the numerical solution and plot
	history_z[0] = 0
	plt.plot(t, history_z, c=COLORS[3], label=LABELS[3])

	# enforce history = 0 at t = 0 for the verification
	A_x[0], A_z[0], G_x[0], G_z[0] = 0, 0, 0, 0
	plt.plot(t, A_z - G_z, ':k', label='verification')
	plt.legend(fontsize=FS, loc='lower right')
	plt.show()

if __name__ == '__main__':
	main()
