import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from utils.plot import initialize_figure as fig 
from utils.data_tools import extract_data
from utils.colors import COLORS

IN_FILE = '../data/water_wave/multi_particles.csv'

def main():
	"""Plot particle position points at regular intervals."""
	numerics = pd.read_csv(IN_FILE)
	x_0s = numerics['x_0'].drop_duplicates().tolist()
	fig('x', 'z', equal_aspect=True, make_square=True)
	for i, history in product(range(len(x_0s)), [False, True]):
		params = {'x_0': x_0s[i], 'history': history}
		mk = 'x' if history else '.'
		x, z = extract_data(['x', 'z'], numerics, params)
		plt.scatter(x, z, c=COLORS[i], marker=mk)
	plt.show()

if __name__ == '__main__':
	main()
