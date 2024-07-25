import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt
import itertools

DATA_PATH = '../../data/water_wave/'

def main():
	"""
	This program plots numerical solutions for the Stokes drift velocity of
	inertial particles of varying Stokes numbers in linear water waves.
	"""
	analysis = pd.read_csv(DATA_PATH + 'st_analysis.csv') # read data
	fs, lfs = 14, 16 # font sizes

	# initialize drift velocity figure & left subplot
	plt.figure(1)
#	plt.title(r'Stokes drift velocity vs depth', fontsize=18)
	plt.xlabel(r'$u_d$', fontsize=lfs)
	plt.ylabel('z', fontsize=lfs)
	plt.axis([0, 1, -6, 0.1])
	plt.gca().set_box_aspect(1)
	plt.xticks(fontsize=fs)
	plt.yticks(fontsize=fs)
	plt.minorticks_on()

	# initialize lists of beta values and history
	stokes_nums = analysis['St'].drop_duplicates().tolist()
	history = [True, False]

	# position bubble labels
	text_position_x = [0.5, 0.43, 0.3]
	text_position_y = [-0.5, -1.46, -3.65]
	properties = dict(boxstyle='circle', facecolor='w', edgecolor='k')

	for i in itertools.product(stokes_nums, history):
		St, history = i
		ls, label = '-', ''

		# create conditions to help filter through data
		exact = (analysis['St'] == St) & (analysis['history'] == history) \
										   & (analysis['exact'] == True)
		estimated = (analysis['St'] == St) \
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
			label = 'with history effects' if St == stokes_nums[-1] else ''
		elif St == stokes_nums[-1]:
			label = 'without history effects'

		# plot data
		i = stokes_nums.index(St)
		plt.scatter(u_d, z, marker='.', edgecolors='k', facecolors='none',
					label='', zorder=2)
		plt.text(text_position_x[i], text_position_y[i], f'{St:g}',
				 fontsize=fs, bbox=properties, zorder=3)
		plt.plot(estimated_u_d, estimated_z, c='k', ls=ls, marker='.',
				 label=label, zorder=0)
#	plt.legend(fontsize=fs)
	plt.show()

if __name__ == '__main__':
	main()
