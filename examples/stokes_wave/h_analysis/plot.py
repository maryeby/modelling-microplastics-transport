import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig
from utils.plot import LABEL_AX, LABEL_BX
from utils.data_tools import extract_data
from examples.linear_wave.st_analysis.numerics import STOKES_HATS
from examples.linear_wave.st_analysis.numerics import RS as CONSTANT_RS
from examples.linear_wave.r_analysis.numerics import RS
from examples.linear_wave.r_analysis.numerics import STOKES_HAT as \
													 CONSTANT_ST_HAT
from examples.stokes_wave.h_analysis.numerics import DEPTHS
from examples.stokes_wave.h_analysis.numerics import OUT_FILE as IN_FILE1
from examples.stokes_wave.h_analysis.analysis import OUT_FILE as IN_FILE2

# axis limits and whether to display subplot titles
LIMS = [[-0.61, 0, -2.5, -0.2], [-0.61, 0, -2.5, -0.2], [-0.61, 0, -1.7, -0.2],
		[-0.61, 0, -1.7, -0.2], [-0.55, 0, -2.35, -0.2], [-0.63, 0, -2, -0.2]]
SHOW_TITLES = True

# label positioning and properties
LABEL_ABY, LABEL_CDY = 0.95, 0.49
TEXT_POSITION = [[(-0.32, -0.59), (-0.23, -0.64), (-0.175, -0.69)],
				 [(-0.23, -0.6), (-0.3, -0.525), (-0.35, -0.45)],
				 [(-0.41, -0.56), (-0.3, -0.56), (-0.2, -0.56)],
				 [(-0.28, -0.45), (-0.36, -0.39), (-0.41, -0.33)],
				 [(-0.35, -0.66), (-0.35, -0.39), (-0.35, -1)],
				 [(-0.43, -0.55), (-0.51, -0.47), (-0.32, -0.68)]]
PROPERTIES = dict(boxstyle='circle', fc='w', ec='k')
FS = 8

def main():
	"""
	Plot the drift velocity vs vertical position of particles in a wave.

	The numerical horizontal drift velocity and vertical position are averaged
	over each wave period. Solutions with and without history effects are
	included for varying Stokes numbers, density ratios, and water depths.
	"""
	numerics = pd.read_csv(IN_FILE1)
	analysis = pd.read_csv(IN_FILE2)
	names = ['u_d_bar', 'z_bar']

	# create plots for variation in St
	for i in range(len(CONSTANT_RS) * len(DEPTHS)):
		# initialize subplot
		x_label = r'$\bar{u}_d/\epsilon^2$' if i // 2 == 1 else None
		y_label = r'$\bar{z}$' if i % 2 == 0 else None
		hide_x, hide_y = not x_label, not y_label
		fig(x_label, y_label, 221 + i, lims=LIMS[i], make_square=True,
			hide_xticks=hide_x, hide_yticks=hide_y)
		if SHOW_TITLES: plt.title(f'depth = {DEPTHS[i // 2]:g}, '
								+ f'R = {CONSTANT_RS[i % 2]:g}')

		# create variables for data extraction
		st_nums = [STOKES_HATS[0], STOKES_HATS[3], STOKES_HATS[2]] \
				  if i % 2 == 0 else STOKES_HATS[:3]
		params = {'R': CONSTANT_RS[i % 2], 'depth': DEPTHS[i // 2]}

		# extract and plot data
		for st, history in product(st_nums, [False, True]):
			params.update(Sthat=st, history=history)
			style = '--' if history else '-'
			u, z, s = extract_data(names + ['S'], analysis, params)
			u, z = extract_data(names, numerics, params)
			plt.scatter(u, z, marker='.', fc='none', ec='k')
			u, z, s = extract_data(names + ['S'], analysis, params)
			plt.plot(u, z, c='k', ls=style)

			# plot bubble labels
			j = st_nums.index(st)
			l = f'{s.iloc[0]:.1g}' if i % 2 == 0 and j == 2 \
				else f'{s.iloc[0]:.2g}'
			x, y = TEXT_POSITION[i][j]
			if history: plt.text(x, y, l, bbox=PROPERTIES, fontsize=FS)

	# add subplot labels
	plt.suptitle(' ') # add space for labels
	plt.gcf().text(LABEL_AX, LABEL_ABY, r'$(a)$')
	plt.gcf().text(LABEL_BX, LABEL_ABY, r'$(b)$')
	plt.gcf().text(LABEL_AX, LABEL_CDY, r'$(c)$')
	plt.gcf().text(LABEL_BX, LABEL_CDY, r'$(d)$')


	# create plots for variation in R
	params['Sthat'] = CONSTANT_ST_HAT
	for i in range(len(DEPTHS)):
		# initialize subplot
		j = len(CONSTANT_RS) * len(DEPTHS)
		y_label = r'$\bar{z}$' if i == 0 else None
		fig(r'$\bar{u}_d/\epsilon^2$', y_label, 121 + i, make_square=True,
			add_subplot_labels=True, lims=LIMS[i + j])
		if SHOW_TITLES: plt.title(f'depth = {DEPTHS[i]:g}')

		# extract and plot data
		params['depth'] = DEPTHS[i]
		for r, history in product(RS[1:-1], [False, True]):
			params.update(R=r, history=history)
			style = '--' if history else '-'
			u, z = extract_data(names, numerics, params)
			plt.scatter(u, z, marker='.', fc='none', ec='k')
			u, z, s = extract_data(names + ['S'], analysis, params)
			plt.plot(u, z, c='k', ls=style)

			# plot bubble labels
			k = RS.index(r) - 1
			x, y = TEXT_POSITION[i + j][k]
			if history: plt.text(x, y, f'{r:.2g}', bbox=PROPERTIES, fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()
