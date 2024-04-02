import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

from models import my_system as ts
DATA_PATH = '../data/water_wave/'

def main():
	"""
	This program gives the user the option to either list unique simulations
	found in the data file or check if a specified simulation exists in the 
	data file. If the user chooses to search for a specific simulation, the
	user is given the option to plot the trajectory of the particle.
	"""
	# read data
	filename = 'numerics.csv'
	numerics = pd.read_csv(DATA_PATH + filename)

	# get user input
	selection = input('(L)ist unique simulations or (S)earch for a '
					  + 'simulation? ')

	# print simulations if user chooses the list option
	if selection[0].upper() == 'L':
		df = numerics[['x_0', 'z_0', 'St', 'beta', 'h\'', 'A\'', 'wavelength\'',
					   'num_periods\'', 'delta_t\'']].drop_duplicates()
		print(df.to_string(index=False))
	elif selection[0].upper() == 'S':
		# initialize variables to use in search
		x_0 = float(input('initial horizontal position: '))
		z_0 = float(input('initial vertical position: '))
		St = float(input('Stokes number: '))
		beta = float(input('beta: '))
		history_input = input('history (T/F): ')
		h = int(input('depth: '))
		A = float(input('amplitude: '))
		wavelength = int(input('wavelength: '))
		delta_t = float(input('delta t: '))

		# use history_input to determine the value of boolean variable history
		history = True if history_input[0].upper() == 'T' else False

		# retrieve relevant numerical results
		condition = (numerics['x_0'] == x_0) & (numerics['z_0'] == z_0) \
									& (numerics['St'] == St) \
									& (numerics['beta'] == beta) \
									& (numerics['history'] == history) \
									& (numerics['h\''] == h) \
									& (numerics['A\''] == A) \
									& (numerics['wavelength\''] == wavelength) \
									& (numerics['delta_t\''] == delta_t)
		get = lambda name : numerics[name].where(condition).dropna().tolist()
		x = get('x')
		z = get('z')
		t = get('t')
		xdot = get('xdot')

		# print whether the data was found
		answer = 'No'
		if len(x) == 0 and len(z) == 0:
			print(f'Data not found in {filename}')
		else:
			print(f'Data found in {filename}')
			answer = input('Plot particle trajectory? (Y/N): ')

		if answer[0].upper() == 'Y':
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
	else:
		print('Selection not recognized.')

if __name__ == '__main__':
	main()
