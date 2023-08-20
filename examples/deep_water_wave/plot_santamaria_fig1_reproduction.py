import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program reproduces Figure 1 from Santamaria et al. (2013).
	"""
	my_data = pd.read_csv('../data/deep_water_wave/'
						  + 'santamaria_fig1_recreation.csv')
	# plot results
	plt.figure()
	plt.title('Particle Trajectories in Deep Water Waves', fontsize=18)
	plt.xlabel('kx', fontsize=16)
	plt.ylabel('kz', fontsize=16)
	plt.axis([0, 3.2, -4, 0])
	plt.xticks(fontsize=14)
	plt.yticks([-3, -2, -1, 0], fontsize=14)

	plt.plot('sm_heavy_x', 'sm_heavy_z', c='grey', data=my_data,
			 label='heavy particle (Santamaria)')
	plt.plot('sm_light_x', 'sm_light_z', c='grey', data=my_data,
			 label='light particle (Santamaria)')
	plt.plot('my_heavy_x', 'my_heavy_z', '--k', data=my_data,
			 label='heavy particle (Daitche)')
	plt.plot('my_light_x', 'my_light_z', ':k', data=my_data,
			 label='light particle (Daitche)')
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.show()
if __name__ == '__main__':
	main()
