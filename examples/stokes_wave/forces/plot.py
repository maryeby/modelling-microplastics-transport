import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.colors import COLORS
from models import stokes_wave as fl
from examples.linear_wave.forces.numerics import DEPTH, WAVELENGTH, NUM_PERIODS
from examples.stokes_wave.forces.analysis import ST_TO_SHOW as STOKES_NUM
from examples.stokes_wave.forces.analysis import R_TO_SHOW as R
from examples.stokes_wave.forces.numerics import AMPLITUDE
from examples.stokes_wave.forces.numerics import OUT_FILE as IN_FILE

ZEROS = [0, 0, 0, 0]
LABELS = ['inertial forces', 'gravity', 'Stokes drag', 'history force']
WIDTH = 0.013		# vector arrow width
OS = 1e-3			# offset for text positions

def main():
	"""Plot the trajectory of a particle and the forces over time."""
	# retrieve relevant numerical results
	numerics = pd.read_csv(IN_FILE)
	params = {'St': STOKES_NUM, 'R': R}
	names = ['t', 'x', 'z', 'xdot', 'zdot', 'fluid_pressure_gradient_x',
			 'fluid_pressure_gradient_z', 'buoyancy_force_x',
			 'buoyancy_force_z', 'added_mass_force_x', 'added_mass_force_z',
			 'stokes_drag_x', 'stokes_drag_z', 'history_force_x',
			 'history_force_z']
	data = extract_data(names, numerics, params)
	period = int(data[0].shape[0] // NUM_PERIODS)
	n = int(period * 3) if NUM_PERIODS > 3 else -5
	for s in range(len(data)): data[s] = data[s].to_numpy()[:n]
	t, x, z, xdot, zdot, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z, \
	   drag_x, drag_z, history_x, history_z = data
	inertial_x = fpg_x + mass_x
	inertial_z = fpg_z + mass_z

	# verification
	wave = fl.StokesWave(DEPTH, AMPLITUDE, WAVELENGTH)
	u_x, u_z = wave.velocity(x, z, t)
	dudt, dwdt = wave.derivative_along_trajectory(x, z, t,
												  np.array([xdot, zdot]))
	w_x = xdot - u_x
	w_z = zdot - u_z
	A_x = np.gradient(w_x, t)
	A_z = np.gradient(w_z, t)
	G_x = fpg_x + buoyancy_x + mass_x + drag_x - dudt
	G_z = fpg_z + buoyancy_z + mass_z + drag_z - dwdt

	# compute positions of various points on the particle trajectory plot
	A = int(period * 1.3)
	B = int(period * 1.55)
	C = int(period * 1.78)
	D = int(period * 2)

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

	# plot particle trajectory with horizontal force vectors
	fig(y_label=r'$z$', num=221, make_square=True, equal_aspect=True,
		hide_xticks=True)
	plt.plot(x[:A], z[:A], c=COLORS[-1])
	plt.plot(x[A:D], z[A:D], c='k')
	plt.plot(x[D:], z[D:], c=COLORS[-1])
	plt.quiver(X, Y, inertial_U, ZEROS, color=COLORS[1], label=LABELS[0],
			   scale=1, angles='xy', scale_units='xy', width=WIDTH)
	plt.quiver(X, Y, buoyancy_U, ZEROS, color=COLORS[3], label=LABELS[1],
			   scale=1, angles='xy', scale_units='xy', width=WIDTH)
	plt.quiver(X, Y, history_U, ZEROS, color=COLORS[8], label=LABELS[3],
			   scale=1, angles='xy', scale_units='xy', width=WIDTH)
	plt.quiver(X, Y, drag_U, ZEROS, color=COLORS[6], label=LABELS[2], scale=1,
			   angles='xy', scale_units='xy', width=WIDTH)
	
	# add labels
	plt.text(x[A] + OS, z[A] + OS, 'A', ha='left', va='bottom')
	plt.text(x[B] + OS, z[B] - OS, 'B', ha='left', va='top')
	plt.text(x[C] - OS, z[C] - OS, 'C', ha='right', va='bottom')
	plt.text(x[D] - OS, z[D] - OS, 'D', ha='right', va='bottom')

	# plot particle trajectory with vertical force vectors
	fig(r'$x$', r'$z$', 223, equal_aspect=True, make_square=True)
	plt.xticks([0, 0.1, 0.2])
	plt.plot(x[:A], z[:A], c=COLORS[-1])
	plt.plot(x[A:D], z[A:D], c='k')
	plt.plot(x[D:], z[D:], c=COLORS[-1])
	plt.quiver(X, Y, ZEROS, inertial_V, color=COLORS[1], label=LABELS[0],
			   scale=1, angles='xy', scale_units='xy', width=WIDTH)
	plt.quiver(X, Y, ZEROS, buoyancy_V, color=COLORS[3], label=LABELS[1],
			   scale=1, angles='xy', scale_units='xy', width=WIDTH)
	plt.quiver(X, Y, ZEROS, drag_V, color=COLORS[6], label=LABELS[2], scale=1,
			   angles='xy', scale_units='xy', width=WIDTH)
	plt.quiver(X, Y, ZEROS, history_V, color=COLORS[8], label=LABELS[3],
			   scale=1, angles='xy', scale_units='xy', width=WIDTH)
	
	# add labels
	plt.text(x[A] + OS, z[A] + OS, 'A', ha='left', va='bottom')
	plt.text(x[B] + OS, z[B] - OS, 'B', ha='left', va='top')
	plt.text(x[C] + OS, z[C] - OS, 'C', ha='left', va='top')
	plt.text(x[D] - OS, z[D] - OS, 'D', ha='right', va='bottom')

	# initialize horizontal forces over time subplot
	fig(y_label='horizontal force', num=222, hide_xticks=True)
	plt.axvline(t[A], c=COLORS[-1])
	plt.axvline(t[B], c=COLORS[-1])
	plt.axvline(t[C], c=COLORS[-1])
	plt.axvline(t[D], c=COLORS[-1])

	# enforce history = 0 at t = 0 for the numerical solution and verification
	history_x[0], history_z[0] = 0, 0
	A_x[0], A_z[0], G_x[0], G_z[0] = 0, 0, 0, 0

	# plot horizontal forces
	plt.plot(t, inertial_x, c=COLORS[1], ls='--')
	plt.plot(t, buoyancy_x, c=COLORS[3])
	plt.plot(t, drag_x, c=COLORS[6], ls='-.')
	plt.plot(t[:-1], A_x[:-1] - G_x[:-1], c=COLORS[-1])
	plt.plot(t, history_x, c=COLORS[8], ls=':')

	# initialize vertical forces over time subplot
	fig(r'$t$', 'vertical force', 224)
	plt.xticks(ticks=[0, 5, t[A], t[B], t[C], t[D], 15],
			   labels=['0', '5', 'A', 'B', 'C', 'D', '15'])
	plt.axvline(t[A], c=COLORS[-1])
	plt.axvline(t[B], c=COLORS[-1])
	plt.axvline(t[C], c=COLORS[-1])
	plt.axvline(t[D], c=COLORS[-1])
	plt.axhline(0, ls=':', c=COLORS[-1])

	# plot vertical numerical results
	plt.plot(t, inertial_z, c=COLORS[1], ls='--', label=LABELS[0])
	plt.plot(t[1:], buoyancy_z[1:], c=COLORS[3], label=LABELS[1])
	plt.plot(t, drag_z, c=COLORS[6], ls='-.', label=LABELS[2])
	plt.plot(t[:-1], A_z[:-1] - G_z[:-1], c=COLORS[-1], label='verification')
	plt.plot(t, history_z, c=COLORS[8], ls=':', label=LABELS[3])
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
