import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '../../data/water_wave/history_convergence.csv'

def main():
	"""
	This program plots the convergence of the history force at t = 0 of
	inertial particles of varying Stokes numbers in linear water waves.
	"""
	numerics = pd.read_csv(DATA_PATH) # read data
	fs, lfs = 14, 16 # font sizes

	# initialize drift velocity figure & left subplot
	plt.figure()
	plt.subplot(211)
	plt.ylabel(r'$H\'(0)_x$', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.xscale('log')
	plt.minorticks_on()
	plt.plot('delta_t\'', 'H\'(0)_x', '-k.', data=numerics)

	plt.subplot(212)
	plt.xlabel(r'$\Delta t$', fontsize=lfs)
	plt.ylabel(r'$H\'(0)_z$', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.xscale('log')
	plt.minorticks_on()
	plt.plot('delta_t\'', 'H\'(0)_z', '-k.', data=numerics)
	plt.show()

if __name__ == '__main__':
	main()
