import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.plot import LABEL_AX
from utils.data_tools import extract_data, print_parameter
from examples.linear_wave.critical_nums.analysis import WAVELENGTH
from examples.stokes_wave.critical_nums.analysis import OUT_FILE as IN_FILE

WAVENUM = 2 * np.pi / WAVELENGTH
NAMES = ['R_c', 'St_c']
YLIMS = [(0.25, 0.85), (0.3, 1), (0.3, 1)]

def main():
	"""Plot the critical density ratio vs the critical Stokes number."""
	analysis = pd.read_csv(IN_FILE)
	amplitudes = analysis['A\''].drop_duplicates()
	for i, j in zip(range(len(amplitudes)), YLIMS):
		# create the subplot
		print_parameter('epsilon', amplitudes.iloc[i] * WAVELENGTH)
		plot_num = 100 * len(amplitudes) + 11 + i
		fig(y_label=r'$St_c$', num=plot_num, hide_xticks=True)
		plt.title(' ') # add space for labels
		plt.ylim(j)
		if i == len(amplitudes) - 1:
			plt.xlabel(r'$R_c$')
			plt.xticks([0.5, 0.6, 0.7, 0.8])

		# extract and plot curves
		for h in [False, True]:
			params = {'history': h, 'A\'': amplitudes.iloc[i]}
			r_c, st_c = extract_data(NAMES, analysis, params)
			fmt = ':k.' if h else '-k.'
			lb = 'with history effects' if h else 'without history effects'
			plt.plot(r_c, st_c, fmt, label=lb)

	# subplot labels
	plt.gcf().text(LABEL_AX, 0.97, r'$(a)$')
	plt.gcf().text(LABEL_AX, 0.67, r'$(b)$')
	plt.gcf().text(LABEL_AX, 0.36, r'$(c)$')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
