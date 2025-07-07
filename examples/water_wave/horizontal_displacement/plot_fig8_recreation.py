import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.data_tools import extract_data
from examples.water_wave.horizontal_displacement.dibenedetto_numerics import \
	 OUT_FILE as IN_FILE

def main():
	"""
	Reproduce Figure 8 from [1].

	References
	----------
	[^1]: [M. H. DiBenedetto et al. (2022).](
		  https://doi.org/10.1017/jfm.2022.95) Enhanced settling and dispersion
		  of inertial particles in surface waves. *Journal of Fluid Mechanics*
		  936, A38.
	"""
	numerics = pd.read_csv(IN_FILE)
	fig(r'$x$', r'$z$', lims=[-0.05, 1.55, -2, 0])
	x, z = extract_data(['x', 'z'], numerics, {'history': False})
	plt.plot(x, z, '-k', label='without history effects')
	x, z = extract_data(['x', 'z'], numerics, {'history': True})
	plt.plot(x, z, '--k', label='with history effects')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
