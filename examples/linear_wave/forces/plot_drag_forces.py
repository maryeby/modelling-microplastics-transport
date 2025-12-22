import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.colors import COLORS
from examples.linear_wave.forces.numerics import DEPTH, WAVELENGTH, NUM_PERIODS
from examples.linear_wave.forces.numerics import OUT_FILE as IN_FILE
from examples.linear_wave.forces.analysis import ST_TO_SHOW as STOKES_NUM
from examples.linear_wave.forces.analysis import R_TO_SHOW as R

ZEROS = [0, 0, 0, 0]
LABELS = ['inertial forces', 'gravity', 'Stokes drag', 'history force']
LABEL_AEX = 0.04
LABEL_BCDX = 0.52
LABEL_ABY = 0.9
LABEL_CY = 0.62
LABEL_DEY = 0.32
OS = 1e-3 # offset for text positions

def main():
	"""Plot the trajectory of a particle and drag forces over time."""
	numerics = pd.read_csv(IN_FILE)
	gs = gridspec.GridSpec(3, 2)

	# retrieve relevant numerical results
	names = ['t', 'x', 'z', 'xdot', 'zdot', 'fluid_pressure_gradient_x',
			 'fluid_pressure_gradient_z', 'buoyancy_force_x',
			 'buoyancy_force_z', 'added_mass_force_x', 'added_mass_force_z',
			 'stokes_drag_x', 'stokes_drag_z', 'history_force_x',
			 'history_force_z']
	data = extract_data(names, numerics, {'St': STOKES_NUM, 'R': R})
	period = int(data[0].shape[0] // NUM_PERIODS)
	n = int(period * 3) if NUM_PERIODS > 3 else -5
	for s in range(len(data)): data[s] = data[s].to_numpy()[:n]
	t, x, z, xdot, zdot, fpg_x, fpg_z, buoyancy_x, buoyancy_z, mass_x, mass_z, \
	   drag_x, drag_z, history_x, history_z = data

	# compute positions of various points on the particle trajectory plot
	A = int(period * 1.28)
	B = int(period * 1.52)
	C = int(period * 1.77)
	D = int(period * 2.02)

	# plot particle trajectory with horizontal force vectors
	fig(r'$x$', r'$z$', gs[:2, 0], equal_aspect=True,
		lims=[-0.06, 0.16, -0.25, 0.05])
	plt.gcf().text(LABEL_AEX, LABEL_ABY, r'$(a)$')
	plt.gca().xaxis.set_label_position('top')
	plt.gca().xaxis.tick_top()
	plt.plot(x[:A], z[:A], c=COLORS[-1])
	plt.plot(x[A:D], z[A:D], c='k')
	plt.plot(x[D:], z[D:], c=COLORS[-1])
	plt.scatter([x[A], x[B], x[C], x[D]], [z[A], z[B], z[C], z[D]], marker='.',
				c='k')
	# add labels
	plt.text(x[A] + OS, z[A] + OS, 'A', ha='left', va='bottom')
	plt.text(x[B] + OS, z[B] - OS, 'B', ha='left', va='top')
	plt.text(x[C] - OS, z[C] - OS, 'C', ha='right', va='bottom')
	plt.text(x[D] - OS, z[D] - OS, 'D', ha='right', va='bottom')

	# horizontal forces over time subplot
	fig(y_label='horizontal force', num=gs[0, 1], hide_xticks=True)
	plt.gcf().text(LABEL_BCDX, LABEL_ABY, r'$(b)$')
	plt.gca().yaxis.set_label_position('right')
	plt.gca().xaxis.tick_top()
	plt.axvline(t[A], c=COLORS[-1])
	plt.axvline(t[B], c=COLORS[-1])
	plt.axvline(t[C], c=COLORS[-1])
	plt.axvline(t[D], c=COLORS[-1])
	history_x[0], history_z[0] = 0, 0 # enforce history = 0 at t = 0
	plt.plot(t, drag_x, 'k--')
	plt.plot(t, history_x, 'k:')

	# vertical forces over time subplot
	fig(y_label='vertical force', num=gs[1, 1], hide_xticks=True)
	plt.gcf().text(LABEL_BCDX, LABEL_CY, r'$(c)$')
	plt.gca().yaxis.set_label_position('right')
	plt.axvline(t[A], c=COLORS[-1])
	plt.axvline(t[B], c=COLORS[-1])
	plt.axvline(t[C], c=COLORS[-1])
	plt.axvline(t[D], c=COLORS[-1])
	plt.plot(t, drag_z, 'k--', label=LABELS[2])
	plt.plot(t, history_z, 'k:', label=LABELS[3])

	# plot horizontal particle velocity
	fig(r'$t$', r'$\dot{x}$', gs[2, 0])
	plt.gcf().text(LABEL_AEX, LABEL_DEY, r'$(e)$')
	plt.xticks(ticks=[0, 5, t[A], t[B], t[C], t[D], 15],
			   labels=['0', '5', 'A', 'B', 'C', 'D', '15'])
	plt.axvline(t[A], c=COLORS[-1])
	plt.axvline(t[B], c=COLORS[-1])
	plt.axvline(t[C], c=COLORS[-1])
	plt.axvline(t[D], c=COLORS[-1])
	plt.plot(t, xdot, 'k-', label='particle velocity')

	# plot vertical particle velocity
	fig(r'$t$', r'$\dot{z}$', gs[2, 1])
	plt.gcf().text(LABEL_BCDX, LABEL_DEY, r'$(d)$')
	plt.gca().yaxis.set_label_position('right')
	plt.xticks(ticks=[0, 5, t[A], t[B], t[C], t[D], 15],
			   labels=['0', '5', 'A', 'B', 'C', 'D', '15'])
	plt.axvline(t[A], c=COLORS[-1])
	plt.axvline(t[B], c=COLORS[-1])
	plt.axvline(t[C], c=COLORS[-1])
	plt.axvline(t[D], c=COLORS[-1])
	plt.plot(t, zdot, 'k-')
	plt.figlegend(bbox_to_anchor=[0.91, 1])
	plt.show()

if __name__ == '__main__':
	main()
