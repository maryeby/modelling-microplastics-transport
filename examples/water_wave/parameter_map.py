import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from scipy.constants import g

from utils.plot import initialize_figure as fig
from utils.colors import COLORS
from utils.data_tools import extract_data
from examples.water_wave.forces.compute_st_star import OUT_FILE as IN_FILE

MIN_STEEPNESS = 0
MAX_STEEPNESS = 0.5
MIN_RADIUS = 0
MAX_RADIUS = 2.5e-3
WAVE_BREAKING = 0.44
LINEAR_LIMIT = 0.1
STEEPNESS_TICKS = [0.1, 0.2, 0.3, 0.4, MAX_STEEPNESS]
RADIUS_TICKS = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025]
ST_TICKS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
RES = 1000
WAVELENGTH = 1.5

def main():
	"""Map the wave steepness, particle size, and Stokes number."""
	warnings.filterwarnings('ignore')
	fig(r'$\epsilon$', 'a\'', make_square=True)
	plt.title(rf'$\lambda = ${WAVELENGTH:g}')
	plt.xticks(np.arange(MIN_STEEPNESS, MAX_STEEPNESS + 0.1, 0.1))

	# compute Stokes numbers for constant wavelength, varying epsilon and a'
	omega = np.sqrt(g * 2 * np.pi / WAVELENGTH)
	epsilon, a = np.meshgrid(np.linspace(MIN_STEEPNESS, MAX_STEEPNESS, RES),
							 np.linspace(MIN_RADIUS, MAX_RADIUS, RES))
	stokes_nums = 2e6 / 9 * a * a * epsilon * omega

	# specify colors of the contours and their limits
	my_map = ListedColormap(COLORS)
	my_norm = LogNorm(ST_TICKS[0], ST_TICKS[-1])

	# plot timestep size vs horizontal displacement
	plt.axvline(LINEAR_LIMIT, c='w', ls='--')
	plt.axvline(WAVE_BREAKING, c='w', ls='--')
#	plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
	plt.contourf(epsilon, a, stokes_nums, ST_TICKS, cmap=my_map,
				 extent=(MIN_STEEPNESS, MAX_STEEPNESS, MIN_RADIUS, MAX_RADIUS),
				 origin='lower', extend='min', norm=my_norm)

	# read and plot St* data
	data = pd.read_csv(IN_FILE[3:])
	epsilon_star, st_star = extract_data(['epsilon', 'St*'], data,
										 {'beta': 0.99})
	epsilon_star, st_star = epsilon_star.to_numpy(), st_star.to_numpy()
	a_star = np.sqrt(9 * st_star / (2e6 * omega * epsilon_star))
	plt.plot(epsilon_star, a_star, c='w', ls=':')

	# add labels
	plt.text(LINEAR_LIMIT / 2, (MAX_RADIUS - MIN_RADIUS) / 2,
			 'moderate St linear wave regime', rotation='vertical', c='w',
			 horizontalalignment='center',
			 verticalalignment='center')
	plt.text(LINEAR_LIMIT / 2, 0.00025, 'low St\nlinear wave\nregime', c='w',
			 horizontalalignment='center', verticalalignment='center')
	plt.text((WAVE_BREAKING + LINEAR_LIMIT) / 2, (MAX_RADIUS - MIN_RADIUS) / 2,
			 'non-linear wave regime', c='w',
			 horizontalalignment='center', verticalalignment='center')
	plt.text(WAVE_BREAKING + (MAX_STEEPNESS - WAVE_BREAKING) / 2,
			 (MAX_RADIUS - MIN_RADIUS) / 4, 'wave breaking',
			 rotation='vertical', c='w',
			 horizontalalignment='center', verticalalignment='center')

	# create colorbar
	cbar = plt.colorbar(ticks=ST_TICKS, boundaries=ST_TICKS, shrink=0.6)
	cbar.set_label(label='St')
	plt.show()

if __name__ == '__main__': main()
