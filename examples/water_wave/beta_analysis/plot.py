import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt
import itertools

DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program plots numerical and analytical solutions for the Stokes drift
	velocity of inertial particles of varying densities in linear water waves.
	"""
	analysis = pd.read_csv(DATA_PATH + 'beta_analysis.csv') # read data

	# initialize drift velocity figure & left subplot
	plt.figure(1)
	plt.title(r'Stokes drift velocity vs depth', fontsize=18)
	plt.xlabel(r'$\frac{u_d}{U\mathrm{Fr}}$', fontsize=16)
	plt.ylabel('kz', fontsize=16)
	plt.axis([0, 1, -4, 0])
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.minorticks_on()

	betas = analysis['beta'].drop_duplicates()
	history = [True, False]
	markers = ['s', 'o', '^', 'X']
	m = 0
	for i in itertools.product(betas, history):
		beta, history = i
		ls, curve_label, scatter_label = '-', '', ''

		# create conditions to help filter through data
		exact = (analysis['beta'] == beta) & (analysis['history'] == history) \
										   & (analysis['exact'] == True)
		estimated = (analysis['beta'] == beta) \
						& (analysis['history'] == history) \
						& (analysis['exact'] == False)

		# retrieve relevant solutions
		u_d = list(analysis['u_d'].where(exact).dropna())
		z = list(analysis['z'].where(exact).dropna())
		estimated_u_d = list(analysis['u_d'].where(estimated).dropna())
		estimated_z = list(analysis['z'].where(estimated).dropna())

		# determine labels and markers depending on what data is being plotted
		if history:
			m += 1
			ls = '--'
			scatter_label = r'$\beta = $ %g' % beta
			curve_label = 'with history' if beta == betas.iloc[-1] else ''
		elif beta == betas.iloc[-1]:
			curve_label = 'without history'

		# plot data
		plt.scatter(u_d, z, marker=markers[m], edgecolors='k',
					facecolors='none', label=scatter_label)
		if beta != 1:
			plt.plot(estimated_u_d, estimated_z, c='k', ls=ls,
					 label=curve_label)
	plt.plot('analytical_u_d', 'analytical_z', c='hotpink', data=analysis,
			 label='analytical')
	plt.legend(fontsize=14)
	plt.show()

if __name__ == '__main__':
	main()
