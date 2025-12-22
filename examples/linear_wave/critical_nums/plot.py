import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.linear_wave.critical_nums.analysis import AMPLITUDE, WAVELENGTH
from examples.linear_wave.critical_nums.analysis import OUT_FILE as IN_FILE

def main():
	"""Plot the critical density ratio vs the critical Stokes number."""
	analysis = pd.read_csv(IN_FILE) # read data
	fig(r'$R_c$', r'$St_c$', lims=[0.5, 0.81, 0.75, 4.25])

	for h in [False, True]:
		r_c, st_c = extract_data(['R_c', 'St_c'], analysis, {'history': h})
		fmt = ':k.' if h else '-k.'
		lb = 'with history effects' if h else 'without history effects'
		plt.plot(r_c, st_c, fmt, label=lb)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
