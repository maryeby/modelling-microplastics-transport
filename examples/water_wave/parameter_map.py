import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, Normalize
from scipy.constants import g

from utils.plot import initialize_figure as fig
from utils.colors import COLORS
from utils.data_tools import extract_data
from examples.water_wave.forces.compute_st_star import OUT_FILE as IN_FILE

NU = 1e-6			# kinematic viscosity
ST50 = 0.0592		# 50% value estimated by hand
STEP = 2.5e-4		# used to compute the radiue ticks
NEUTRAL_R = 2 / 3

# lower bounds
MIN_WAVELENGTH = 1.5
MIN_RADIUS = 0
MIN_R = 0
MIN_ST = 0

# upper bounds
MAX_WAVELENGTH = 1000
MAX_RADIUS = 2.5e-3
MAX_R = 1
MAX_ST = 1

RADIUS_TICKS = np.arange(STEP, MAX_RADIUS + STEP, STEP).tolist()
ST_TICKS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
R_TICKS = [0, 0.25, 0.5, 0.75, 1]
RES = 1000 # resolution

def main():
	"""Map the density ratio, Stokes number, and particle size."""
	# specify colors of the contours and their limits, ignore warnings
	my_map = ListedColormap(COLORS)
	fx = [pe.withStroke(linewidth=2, foreground='w')]
	warnings.filterwarnings('ignore')

	# format left subplot
	fig(r'$R$', "$St$", 121, lims=[MIN_R, MAX_R, MIN_ST, MAX_ST], width='jfm')
#	plt.title(rf'$\lambda = ${MIN_WAVELENGTH:g}')
	plt.xticks(R_TICKS)

	# plot contour of the radius for min wavelength and varying St, R
	a, r, stokes_nums = compute_a(MIN_WAVELENGTH)
	plt.axvline(NEUTRAL_R, c='w', ls='--')
	plt.plot(r[0], 2 / (9 * r[0]) - 1 / 9, '-.w')
	plt.contourf(r, stokes_nums, a, RADIUS_TICKS, cmap=my_map, extend='min',
				 extent=(MIN_R, MAX_R, MIN_ST, MAX_ST), origin='lower')

	# read St* data
	data = pd.read_csv(IN_FILE[3:])
	r_star, st_star = extract_data(['R', 'Sthat*'], data)
	r_star, st_star = r_star.to_numpy(), st_star.to_numpy()
	r75, st75 = extract_data(['R', 'Sthat75'], data)
	r75, st75 = r75.to_numpy(), st75.to_numpy()

	# plot St* data and annotations
	plt.plot(r_star, st_star, c='w', ls=':')
	plt.plot(r75, st75, c='w', ls=':')
	plt.plot([r75[-1], r75[-1]], [0, st75[-1]], c='w')
	plt.annotate('', xytext=(0.66, st75[-1]), xy=(0.66, st_star[-1]),
				 arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',
				 color='w'))
	plt.annotate('', xytext=(0.66, ST50), xy=(0.66, ST50 + 1e-7),
				 arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',
				 color='w'))
	plt.annotate(r'100\%', (0.72, st_star[-1]), c='w', va='top', ha='left')
	plt.annotate(r'75\%', (0.72, st75[-1]), c='w', va='center', ha='left')
	plt.annotate(r'50\%', (0.72, ST50), c='w', va='center', ha='left')

	# format right subplot
	fig(r'$R$', num=122, lims=[MIN_R, MAX_R, MIN_ST, MAX_ST], width='jfm',
		hide_yticks=True)
#	plt.title(rf'$\lambda = ${MAX_WAVELENGTH:g}')
	plt.xticks(R_TICKS)

	# plot contour of the radius for max wavelength and varying St, R
	a, r, stokes_nums = compute_a(MAX_WAVELENGTH)
	plt.axvline(NEUTRAL_R, c='w', ls='--')
	plt.plot(r[0], 2 / (9 * r[0]) - 1 / 9, '-.w')
	plt.contourf(r, stokes_nums, a, RADIUS_TICKS, cmap=my_map, extend='min',
				 extent=(MIN_R, MAX_R, MIN_ST, MAX_ST), origin='lower')

	# plot St* data and annotations
	plt.plot(r_star, st_star, c='w', ls=':')
	plt.plot(r75, st75, c='k', ls=':')
	plt.plot([r75[-1], r75[-1]], [0, st75[-1]], c='w')
	plt.annotate('', xytext=(0.66, st75[-1]), xy=(0.66, st_star[-1]),
				 arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',
				 color='w'))
	plt.annotate('', xytext=(0.66, ST50), xy=(0.66, ST50 + 1e-7),
				 arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',
				 color='k'))
	plt.annotate(r'100\%', (0.72, st_star[-1]), c='w', va='top', ha='left')
	plt.annotate(r'75\%', (0.72, st75[-1]), c='w', va='center', ha='left')
	plt.annotate(r'50\%', (0.72, ST50), c='k', va='center', ha='left')

	# create colorbar
	cbar = plt.colorbar(ticks=RADIUS_TICKS, boundaries=RADIUS_TICKS,
						format='%.1e', shrink=0.8)
	cbar.set_label(label=r"$a'$")
	plt.show()

def compute_a(wavelength):
	r"""Compute the particle radius varying $St$ and $R$."""
	omega = np.sqrt(g * 2 * np.pi / wavelength)
	r, st = np.meshgrid(np.linspace(MIN_R, MAX_R, RES),
						np.linspace(MIN_ST, MAX_ST, RES))
	return np.sqrt(st * 9 / 2 * NU / omega / (1 / r - 1 / 2)), r, st

if __name__ == '__main__': main()
