import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../data/water_wave/'

def main():
	"""
	This program plots the contributions of various forces on the movement of a
	negatively buoyant inertial particle in a linear water wave.
	"""
	numerics = pd.read_csv(f'{DATA_PATH}numerics.csv') # read data

	# define relevant variables for finding and plotting data
	St, beta = 0.01, 0.9
	h, A, wavelength = 10, 0.02, 1 # wave parameters
	delta_t = 5e-3
	positions = [(0, 0), (0.126, -0.088), (0.099, -0.226), (-0.017, -0.236),
				 (-0.036, -0.127)]
	colors= ['grey', 'darkturquoise', 'forestgreen', 'limegreen', 'cyan']

	# define lambda functions to help filter through numerical data
	create_condition = lambda x_0, z_0 : (numerics['x_0'] == x_0) \
						   & (numerics['z_0'] == z_0) \
						   & (numerics['St'] == St) \
						   & (numerics['beta'] == beta) \
						   & (numerics['history'] == True) \
						   & (numerics['h\''] == h) \
						   & (numerics['A\''] == A) \
						   & (numerics['wavelength\''] == wavelength) \
						   & (numerics['delta_t\''] == delta_t)
	get = lambda name, condition : numerics[name].where(condition).dropna()\
																  .to_numpy()
	# initialize figure
	plt.figure()
	plt.title('Particle Trajectories', fontsize=18)
	plt.xlabel('x', fontsize=14)
	plt.ylabel('z', fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	for i in range(len(positions)):
		# retrieve relevant numerical results
		x_0, z_0 = positions[i]
		condition = create_condition(x_0, z_0)
		x = get('x', condition)
		z = get('z', condition)
		t = get('t', condition)

		# plot particle trajectory for each initial position
		period = t.shape[0] // 50
		n = period * 10
		plt.plot(x[:n], z[:n], c=colors[i])

	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
