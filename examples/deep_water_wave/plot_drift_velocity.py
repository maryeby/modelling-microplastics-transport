import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.deep_water_wave.drift_velocity_numerics import STOKES_NUMS

IN_FILE1 = '../data/deep_water_wave/drift_velocity_numerics.csv'
IN_FILE2 = '../data/deep_water_wave/analytics.csv'

def main():
	"""
	Plot solutions for the Stokes drift velocity of particles in a wave.

	Numerical and analytical[^1] solutions are shown for the average horizontal
	Stokes drift velocity of neutrally buoyant particles at different initial
	vertical positions and with various Stokes numbers in a linear wave of
	infinitely deep water.

	References
	----------
	[^1]: [T. S. van den Bremer & Ø. Breivik (2018).](
		  https://doi.org/10.1098/rsta.2017.0104) Stokes drift.
		  *Philosophical Transactions of the Royal Society A: Mathematical,
		  Physical and Engineering Sciences* 376(2111), 20170104.
	"""
	# read data files
	numerics = pd.read_csv(IN_FILE1)
	analytics = pd.read_csv(IN_FILE2)

	# plot results
	fig(r'$\bar{u}$', r'$\bar{z}/h$')
	plt.plot('u_d', 'z/h', c='k', data=analytics, label='exact')
	markers = ['o', '^', 's', 'd']
	for i in range(len(STOKES_NUMS)):
		u_bar, z_bar = extract_data(['u_bar', 'z_bar/h'], numerics,
									{'St': STOKES_NUMS[i]})
		plt.scatter(u_bar, z_bar, marker=markers[i], edgecolors='k',
					facecolors='none', label=f'St = {STOKES_NUMS[i]:g}')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
