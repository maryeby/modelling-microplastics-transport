import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_tools import extract_data
from utils.plot import initialize_figure as fig
from utils.plot import plot_quality_control
from examples.linear_wave.forces.compute_st_star import RS
from examples.linear_wave.forces.compute_st_star import OUT_FILE as IN_FILE

def main():
	r"""Plot $St^*$ as a function of the density ratio $R$."""
	analysis = pd.read_csv(IN_FILE)
	fig(r'$R$', r'$St^*$', make_square=True)

	# plot 100% lines
	names = ['St*', 'Sthat*', 'R', 'R^2']
	st_star, st_hat_star, r, rsq = extract_data(names, analysis)
	plot_quality_control(r, st_star, rsq)
	plt.plot(r, st_star, '-k.', label=r'100\%')

	# plot 75% lines
	names = ['St75', 'Sthat75', 'R', 'R^2']
	st75, st_hat_75, r, rsq = extract_data(names, analysis)
	plot_quality_control(r, st75, rsq)
	plt.plot(r, st75, ':k.', label=r'75\%')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
