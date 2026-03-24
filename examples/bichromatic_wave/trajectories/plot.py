import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig
from utils.plot import LABEL_AX, LABEL_BX
from utils.data_tools import extract_data, NEUTRAL_R
from examples.bichromatic_wave.trajectories.numerics import SLOPE, DEPTH, \
	 WAVELENGTHS, RS, STOKES_HATS
from examples.bichromatic_wave.trajectories.numerics import OUT_FILE as IN_FILE

SHOW_TITLES = False
SEABED_WIDTH = 0.01
LABEL_ABY = 0.97
LABEL_CDY = 0.66
LABEL_EFY = 0.35

def main():
	"""Plot the position and velocity of particles in a bichromatic wave."""
	numerics = pd.read_csv(IN_FILE)
	names = ['x', 'z', 'seabed_x', 'seabed_z', 'S']
	k1 = 2 * np.pi / WAVELENGTHS[0]
	depth = k1 * DEPTH

	# plot particle trajectories varying S
	r = RS[1]
	for i, (st, m) in zip(range(6), product(STOKES_HATS[1:4], [0, SLOPE])):
		xlabel = r'$x$' if i > 2 else None
		ylabel = r'$z$' if i % 2 == 0 else None
		fig(xlabel, ylabel, 321 + i)
		for h in [True, False]:
			fmt = '--k' if h else '-k'
			title = 'flat seabed' if m == 0 else 'sloped seabed'
			params = {'Sthat': st, 'R': r, 'history': h, 'slope': m}
			x, z, xbed, zbed, s = extract_data(names, numerics, params)
			title += rf' $S = ${s.iloc[0]:g}, $R = ${r:.2g}'
			if SHOW_TITLES: plt.title(title)
			plt.suptitle(' ')
			plt.fill_between(xbed, zbed, -depth - SEABED_WIDTH, color='silver')
			plt.plot(x, z, fmt)
	plt.gcf().text(LABEL_AX, LABEL_ABY, r'$(a)$')
	plt.gcf().text(LABEL_BX, LABEL_ABY, r'$(b)$')
	plt.gcf().text(LABEL_AX, LABEL_CDY, r'$(c)$')
	plt.gcf().text(LABEL_BX, LABEL_CDY, r'$(d)$')
	plt.gcf().text(LABEL_AX, LABEL_EFY, r'$(e)$')
	plt.gcf().text(LABEL_BX, LABEL_EFY, r'$(f)$')

	# plot particle trajectories varying R
	st = STOKES_HATS[-2]
	for i, (r, m) in zip(range(6), product([RS[0]] + RS[-2:], [0, SLOPE])):
		xlabel = r'$x$' if i > 3 else None
		ylabel = r'$z$' if i % 2 == 0 else None
		fig(xlabel, ylabel, 321 + i)
		for h in [True, False]:
			fmt = '--k' if h else '-k'
			title = 'flat seabed' if m == 0 else 'sloped seabed'
			params = {'Sthat': st, 'R': r, 'history': h, 'slope': m}
			x, z, xbed, zbed = extract_data(names[:-1], numerics, params)
			title += rf' $\hat St = ${st:g}, $R = ${r:.2g}'
			if SHOW_TITLES: plt.title(title)
			plt.suptitle(' ')
			plt.fill_between(xbed, zbed, -depth - SEABED_WIDTH, color='silver')
			plt.plot(x, z, fmt)
	plt.gcf().text(LABEL_AX, LABEL_ABY, r'$(a)$')
	plt.gcf().text(LABEL_BX, LABEL_ABY, r'$(b)$')
	plt.gcf().text(LABEL_AX, LABEL_CDY, r'$(c)$')
	plt.gcf().text(LABEL_BX, LABEL_CDY, r'$(d)$')
	plt.gcf().text(LABEL_AX, LABEL_EFY, r'$(e)$')
	plt.gcf().text(LABEL_BX, LABEL_EFY, r'$(f)$')
	plt.show()

if __name__ == '__main__': main()
