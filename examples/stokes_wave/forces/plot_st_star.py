import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from examples.linear_wave.forces.compute_st_star import RS
from examples.stokes_wave.forces.compute_st_star import OUT_FILE as IN_FILE

def main():
	r"""Plot $St^*$ as a function of the density ratio $R$."""
	analysis = pd.read_csv(IN_FILE)
	fig(r'$R$', r'$St^*$', make_square=True)

	# plot 100% lines
	names = ['St*', 'Sthat*', 'R']
	st_star, st_hat_star, r = extract_data(names, analysis)
	plt.plot(r, st_star, '-k.', label=r'100\%')

	# plot 75% lines
	names = ['St75', 'Sthat75', 'R']
	st75, st_hat_75, r = extract_data(names, analysis)
	plt.plot(r, st75, '--k.', label=r'75\%')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
