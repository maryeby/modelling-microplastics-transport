import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.deep_water_wave.santamaria.fig2_numerics import DELTA_TS

IN_FILE1 = '../../data/deep_water_wave/santamaria_fig2_recreation.csv'
IN_FILE2 = '../../data/deep_water_wave/santamaria_analytics.csv'
LABELS = ['Santamaria', 'fine', 'medium', 'coarse']
MARKERS = ['o', 's', '^', 'v']

def main():
	"""
	Reproduce Figure 2 from [1].

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)*, 102(1), 14003.
	"""
	# read data files
	numerics = pd.read_csv(IN_FILE1)
	analytics = pd.read_csv(IN_FILE2)
	methods = numerics['method'].drop_duplicates().tolist()

	# get timestep sizes
	sm_delta_t = extract_data('delta_t', numerics, {'method': 'Santamaria'})\
							 .drop_duplicates().tolist()
	delta_ts = sm_delta_t + DELTA_TS

	# plot horizontal analytical results
	fig(r'$t$', r'$\bar{u}$', 121, make_square=True)
	plt.plot('t', 'u_d', c='k', data=analytics)
	plt.axhline(0, c='k', ls=':')

	# plot horizontal numerical results
	for i in range(len(LABELS)):
		method = methods[i] if i == 0 else methods[1]
		params = {'method': method, 'delta_t': delta_ts[i]}
		t, u_bar = extract_data(['t', 'u_bar'], numerics, params)
		ec, fc, m = 'k', 'none', MARKERS[i]
		plt.scatter(t, u_bar, edgecolors=ec, facecolors=fc, marker=m)

	# plot vertical analytical results
	fig(r'$t$', r'$\bar{w}$', 122, make_square=True)
	plt.plot('t', 'w_d', c='k', data=analytics, label='analytics')
	plt.axhline(analytics['settling_velocity'].iloc[0], c='k', ls=':',
				label='settling velocity')

	# plot vertical numerical results
	for i in range(len(LABELS)):
		method = methods[i] if i == 0 else methods[1]
		params = {'method': method, 'delta_t': delta_ts[i]}
		t, w_bar = extract_data(['t', 'w_bar'], numerics, params)
		ec, fc, m = 'k', 'none', MARKERS[i]
		l = LABELS[i] + r' ($\Delta t =$' + f'{delta_ts[i]:.0e})'
		plt.scatter(t, w_bar, edgecolors=ec, facecolors=fc, marker=m, label=l)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
