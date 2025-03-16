import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data

IN_FILE = '../../data/deep_water_wave/santamaria_fig1_recreation.csv'

def main():
	"""
	Reproduce Figure 1 from [1].

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)*, 102(1), 14003.
	"""
	numerics = pd.read_csv(IN_FILE)
	betas = numerics['beta'].drop_duplicates()
	methods = numerics['method'].drop_duplicates()

	fig('x', 'z', lims=[0, 3.2, -4, 0])
	for beta, method in itertools.product(betas, methods):
		params = {'beta': beta, 'method': method}
		x = extract_data('x', numerics, params)
		z = extract_data('z', numerics, params)
		lc = 'k' if method == 'Daitche' else 'silver'
		ls = ':' if method == 'Daitche' else '-'
		plt.plot(x, z, c=lc, ls=ls)
	plt.show()
if __name__ == '__main__':
	main()
