import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.critical_nums.analysis import DEPTH

SCALE = 2 / 3
DATA_PATH = '../../data/water_wave/critical_nums.csv'

def main():
	"""Plot the critical density ratio vs the critical Stokes number."""
	analysis = pd.read_csv(DATA_PATH) # read data
	fig(r'$R_c$', r'$St_c$', lims=[0.5, 0.81, 0.09, 1], width='jfm')

	for history in [False, True]:
		beta_c, stokes_c = extract_data(['beta_c', 'St_c'], analysis,
										{'history': history})
		r_c = np.array(beta_c) * SCALE
		fmt = ':k.' if history else '-k.'
		lb = 'with history effects' if history else 'without history effects'
		plt.plot(r_c, stokes_c, fmt, label=lb)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
