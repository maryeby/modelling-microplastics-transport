import pandas as pd
import matplotlib.pyplot as plt

from utils.plot import initialize_figure as fig
from utils.plot import FS
from utils.data_tools import extract_data

IN_FILE = '../../data/water_wave/displacement_numerics.csv'

def main():
	"""Plot the trajectory of a particle transported through a wave."""
	numerics = pd.read_csv(IN_FILE)
	fig('x', 'z', lims=[-0.05, 1.55, -2, 0])
	x, z = extract_data(['x', 'z'], numerics, {'history': False})
	plt.plot(x, z, '-k', label='without history effects')
	x, z = extract_data(['x', 'z'], numerics, {'history': True})
	plt.plot(x, z, '--k', label='with history effects')
	plt.legend(fontsize=FS)
	plt.show()

if __name__ == '__main__':
	main()
