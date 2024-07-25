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
	fs, lfs = 14, 16
	plt.figure(1)
#	plt.title(r'Stokes drift velocity vs depth', fontsize=18)
#	plt.xlabel(r'$\frac{u_d}{UkA}$', fontsize=lfs)
#	plt.ylabel('kz', fontsize=lfs)
	plt.xlabel(r'$u_d$', fontsize=lfs)
	plt.ylabel('z', fontsize=lfs)
	plt.gca().set_box_aspect(1)
	plt.axis([0, 1, -4, 0.1])
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# initialize lists of beta values and history
	betas = analysis['beta'].drop_duplicates().tolist()
	history = [True, False]

	# position bubble labels
	text_position_x = [0.6, 0.3, 0.4]
	text_position_y = [-0.3, -1.75, -0.76]
	properties = dict(boxstyle='circle', facecolor='w', edgecolor='k')

	for i in itertools.product(betas, history):
		beta, history = i
		ls, label = '-', ''

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
			ls = '--'
			label = 'with history effects' if beta == betas[-1] else ''
		elif beta == betas[-1]:
			label = 'without history effects'

		# plot data
		i = betas.index(beta)
		plt.scatter(u_d, z, marker='.', edgecolors='k', facecolors='none',
					label='', zorder=2)
		plt.text(text_position_x[i], text_position_y[i], f'{beta:g}',
				 fontsize=fs, bbox=properties, zorder=3)
		if beta != 1:
			plt.plot(estimated_u_d, estimated_z, c='k', ls=ls,
					 label=label, zorder=0)
	plt.plot('analytical_u_d', 'analytical_z', c='#018571', data=analysis,
			 label='analytical solution', zorder=1)
#	plt.legend(fontsize=fs)
	plt.show()

if __name__ == '__main__':
	main()
