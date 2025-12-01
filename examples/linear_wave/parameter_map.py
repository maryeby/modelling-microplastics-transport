import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import g

from utils.plot import initialize_figure as fig
from utils.colors import COLORS
from utils.data_tools import extract_data
from examples.linear_wave.forces.compute_st_star import OUT_FILE as IN_FILE1
from examples.linear_wave.forces.analysis import OUT_FILE as IN_FILE2

# constants needed to compute contours
RES = 1000	# resolution
NU = 1e-6	# kinematic viscosity
WAVELENGTH = 200
OMEGA = np.sqrt(g * 2 * np.pi / WAVELENGTH)
R, ST = np.meshgrid(np.linspace(1e-10, 1, RES), np.linspace(0, 1, RES))
GAMMA = 1 / R - 1 / 2
RADIUS = np.sqrt(9 / 2 * ST / GAMMA * NU / OMEGA) * 1e3 # compute a' in mm

# contour plot constants
CONTOURS = [0.5, 1, 1.5, 2, 2.5]
ST_TICKS = [1e-6, 1e-4, 1e-2, 1]
R_TICKS = [0, 0.25, 0.5, 0.75, 1]
R_RANGE = np.linspace(0.5, 0.81, RES)
GAMMA_RANGE = 1 / R_RANGE - 1 / 2
ST_RANGE = 2 / 9 * 2.5 * 2.5 * GAMMA_RANGE * OMEGA
PADDING = 1
LABEL_LOCATIONS = [(0.057, 0.554), (0.173, 0.661), (0.318, 0.740),
				   (0.462, 0.838), (0.598, 0.909)]

# chi vs St plot constants
CHI_R = 0.66
REGIMES = np.array([0.0025, 0.25]) * (1 / CHI_R - 1 / 2) * (1 / CHI_R - 1 / 2)
LABEL_X, LABEL_Y = [0.215, 0.137, 0.427], [0.43, 0.03, 0.001]
CHIS = [1, 0.75, 0.5, 0.1]
MARKERS = ['s', 'v', '*', 'd']
Y = np.linspace(1e-9, 1, 100)
K = 2

def main():
	"""Map the density ratio, Stokes number, and particle size."""
	# read St* data
	st_star_data = pd.read_csv(IN_FILE1[3:])
	r_star, st_star = extract_data(['R', 'St*'], st_star_data)
	r_star, st_star = r_star.to_numpy(), st_star.to_numpy()
	r75, st75 = extract_data(['R', 'St75'], st_star_data)
	r75, st75 = r75.tolist(), st75.tolist()

	# read Sthat and chi data
	chi_data = pd.read_csv(IN_FILE2[3:])
	params = {'force': 'stokes_drag', 'R': CHI_R}
	stokes_drag = extract_data('A', chi_data, params)
	params['force'] = 'history_force'
	history, sthat = extract_data(['A', 'Sthat'], chi_data, params)
	chi = np.array(history) / np.array(stokes_drag)
	st75.append(sthat.iloc[0])
	r75.append(CHI_R)

	# plot contour of the radius for max wavelength and varying St, R
	fig(r'$R$', r'$St$', 121, lims=[R[0, 0], R[-1, -1], ST[0, 0], ST[-1, -1]],
		width='jfm', make_square=True, add_subplot_labels=True)
	for c, m in zip(CHIS[:-1], MARKERS[:-1]):
		plt.scatter(CHI_R, (c / K) ** 2, c='k', marker=m, zorder=2)
	cs = plt.gca().contour(R, ST, RADIUS, levels=CONTOURS, colors='k')
	plt.gca().fill_between(R_RANGE, ST_RANGE, color='silver', zorder=0)
	plt.axvline(CHI_R, c='grey', label=r'$R \approx 2/3$', zorder=1)
	plt.plot(R[0], 2 * GAMMA[0] / 9, '-.k', label=r'$\widehat{St} = 2/9$')
	plt.plot(r_star, st_star, '--k', label=r'$\chi = 1$')
	plt.plot(r75, st75, ':k', label=r'$\chi = 0.75$')
	plt.clabel(cs, manual=LABEL_LOCATIONS, inline_spacing=PADDING)
	plt.legend()

	# plot chi vs St
	fig(r'$\chi$', r'$St$', num=122, width='jfm', make_square=True,
		x_scale='log', y_scale='log', lims=[5e-2, K, 6.5e-4, ST[-1][-1]],
		add_subplot_labels=True)
	plt.gca().fill_between(K * np.sqrt(Y), REGIMES[1], REGIMES[0],
						   color='silver')
	for c, m in zip(CHIS, MARKERS):
		plt.scatter(c, (c / K) ** 2, c='k', marker=m, 
					label=rf'$\chi = {c:g}$')
	plt.plot(K * np.sqrt(Y), Y, '-k', label=r'$\chi \sim \sqrt{\widehat{St}}$')
	plt.scatter(chi, sthat, marker='.', ec='k', fc='none',
				label='measured\npoints')
	plt.text(LABEL_X[0], LABEL_Y[0], 'history dominant', ha='center')
	plt.text(LABEL_X[1], LABEL_Y[1], 'non-negligible\nhistory', ha='center')
	plt.text(LABEL_X[2], LABEL_Y[2], 'Stokes drag dominant', ha='center')
	plt.legend(loc='lower left', bbox_to_anchor=(0.57, 0.1), fontsize='x-small')
	plt.show()

if __name__ == '__main__': main()
