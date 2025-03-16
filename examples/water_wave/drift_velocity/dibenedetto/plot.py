import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_subplot as subplot
from utils.plot import FS
from utils.data_tools import extract_data

IN_FILE = '../../../data/water_wave/dibenedetto_analytics.csv'

def main():
	"""
	Plot numerical and analytical[^1] solutions for the Stokes drift velocity.

	The horizontal and vertical Stokes drift velocity of a negatively buoyant
	particle in a linear wave of deep water is plotted over time.

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	analytics = pd.read_csv(IN_FILE) # read data
	vslin = analytics['v_s_lin'].iloc[0]

	# create left subplot (horizontal results)
	plt.figure()
	subplot(121, r'$t$', r'$\bar{\bar{u}}$')
	names = ['t', 'u_double_bar', 'v_x_drift']
	t, u_bar, v_x_drift = extract_data(names, analytics, {'history': False})
#	u_bar = u_bar.to_numpy() / vslin
	plt.plot(t, v_x_drift, '-k')
	plt.scatter(t, u_bar, edgecolors='k', facecolors='none')
	t, u_bar, v_x_drift = extract_data(names, analytics, {'history': True})
	plt.plot(t, v_x_drift, ':k')
	plt.scatter(t, u_bar, marker='s', edgecolors='k', facecolors='none')
	
	# create right subplot (vertical results)
	subplot(122, r'$t$', r'$\bar{\bar{w}}$')
	names = ['t', 'w_double_bar', 'v_y_drift']
	t, w_bar, v_y_drift = extract_data(names, analytics, {'history': False})
#	w_bar = w_bar.to_numpy() / vslin
	plt.plot(t, v_y_drift, '-k', label='without history effects')
	plt.scatter(t, w_bar, edgecolors='k', facecolors='none',
				label='without history effects')
	t, w_bar, v_y_drift = extract_data(names, analytics, {'history': True})
	plt.plot(t, v_y_drift, ':k', label='with history effects')
	plt.scatter(t, w_bar, marker='s', edgecolors='k', facecolors='none',
				label='with history effects')
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()
