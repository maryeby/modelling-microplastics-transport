import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt
import itertools

DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program plots the number of wave period endpoints for each simulation
	as a function of the Stokes number.
	"""
	analysis = pd.read_csv(DATA_PATH + 'st_optimization.csv') # read data
	fs, lfs = 14, 16 # font sizes
	get = lambda name, h : analysis[name].where(analysis['history'] == h)\
										 .dropna().tolist()

	# initialize drift velocity figure & left subplot
	plt.figure(1)
	plt.xlabel(r'$St$', fontsize=lfs)
	plt.ylabel('number of period endpoints', fontsize=lfs)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# plot solutions without history
	St = get('St', False)
	npe = get('num_endpoints', False)
	St_c = St[-1]
	plt.scatter(St[:-1], npe[:-1], edgecolors='k', facecolors='none',
				label='without history effects')
	plt.scatter(St[-1], npe[-1], edgecolors='hotpink', facecolors='none')
	plt.annotate(r'$St_c =$' + f'{St_c:.3f}', (St[-1], npe[-1] + 0.1),
				 ha='center', fontsize=fs)

	# plot solutions with history
	St = get('St', True)
	npe = get('num_endpoints', True)
	St_c = St[-1]
	plt.scatter(St[:-1], npe[:-1], marker='s', edgecolors='k',
				facecolors='none', label='with history effects')
	plt.scatter(St[-1], npe[-1], marker='s', edgecolors='hotpink',
				facecolors='none')
	plt.annotate(r'$St_c =$' + f'{St_c:.3f}', (St[-1], npe[-1] + 0.1),
				 ha='center', fontsize=fs)
	plt.legend(fontsize=fs)
	plt.show()

if __name__ == '__main__':
	main()
