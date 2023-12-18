import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

from models import my_system as ts
DATA_PATH = '../data/water_wave/'

def main():
	"""
	This program checks if the specified simulation exists in the corresponding
	data file, and optionally plots the trajectory of the inertial particle in
	a linear water wave.
	"""
	# initialize variables
	z_0 = float(input('initial vertical position: '))
	St = float(input('Stokes number: '))
	beta = float(input('beta: '))
	history_input = input('history (T/F): ')
	h = int(input('depth: '))
	A = float(input('amplitude: '))
	wavelength = int(input('wavelength: '))
	delta_t = float(input('delta t: '))

	# use history_input to determine the value of boolean variable history
	history = False
	if history_input[0] == 'T' or history_input[0] == 't':
		history = True

	# read data
	filename = 'newmerics.csv'
	numerics = pd.read_csv(DATA_PATH + filename)

	# retrieve relevant numerical results
	cond = (numerics['z_0'] == z_0) & (numerics['St'] == St) \
									& (numerics['beta'] == beta) \
									& (numerics['history'] == history) \
									& (numerics['h'] == h) \
									& (numerics['A'] == A) \
									& (numerics['wavelength'] == wavelength) \
									& (numerics['delta_t'] == delta_t)
	x = numerics['x'].where(cond).dropna().tolist()
	z = numerics['z'].where(cond).dropna().tolist()
	t = numerics['t'].where(cond).dropna().tolist()
	xdot = numerics['xdot'].where(cond).dropna().tolist()

	# print whether the data was found
	answer = 'No'
	if len(x) == 0 and len(z) == 0:
		print(f'Data not found in {filename}')
	else:
		print(f'Data found in {filename}')
		answer = input('Plot particle trajectory? (Y/N): ')

	if answer[0] == 'Y' or answer[0] == 'y':
		print('Plotting...')
		# compute drift velocity to plot period endpoints
		x_crossings, z_crossings, _, _, _ = ts.compute_drift_velocity(x, z,
																	  xdot, t)
		# initialize figure
		plt.figure()
		plt.title(r'Particle Trajectory', fontsize=18)
		plt.xlabel('x', fontsize=14)
		plt.ylabel('z', fontsize=14)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.minorticks_on()

		# plot particle trajectory and period endpoints
		plt.plot(x, z, c='k')
		plt.scatter(x_crossings, z_crossings, c='k', marker='x')
		plt.show()

if __name__ == '__main__':
	main()
