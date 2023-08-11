import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np

def main():
	"""
	This program computes the relative error for the numerical solutions of a
	rotating particle in a flow and saves the results to the `data` directory.
	"""
	# read data
	data_path = '../data/rigid_body_rotation/'
	numerics = pd.read_csv(data_path + 'numerics.csv')
	analytics = pd.read_csv(data_path + 'analytics.csv')

	# compute relative error
	exact = analytics[['x_ana', 'z_ana']].to_numpy()
	x_num1 = numerics[['first_x', 'first_z']].to_numpy()
	x_num2 = numerics[['second_x', 'second_z']].to_numpy()
	x_num3 = numerics[['third_x', 'third_z']].to_numpy()
	my_dict = dict.fromkeys(['e_rel1', 'e_rel2', 'e_rel3', 'global_error1',
							 'global_error2', 'global_error3'])
	my_dict['t'] = numerics['t']
	my_dict['e_rel1'] = np.linalg.norm(exact - x_num1, axis=1) \
						/ np.linalg.norm(exact, axis=1)
	my_dict['e_rel2'] = np.linalg.norm(exact - x_num2, axis=1) \
						/ np.linalg.norm(exact, axis=1)
	my_dict['e_rel3'] = np.linalg.norm(exact - x_num3, axis=1) \
						/ np.linalg.norm(exact, axis=1)
	my_dict['global_error1'] = np.linalg.norm(exact - x_num1, axis=1).max()
	my_dict['global_error2'] = np.linalg.norm(exact - x_num2, axis=1).max()
	my_dict['global_error3'] = np.linalg.norm(exact - x_num3, axis=1).max()

	# write results to data file
	rel_error = pd.DataFrame(my_dict)
	rel_error.to_csv(data_path + 'rel_error.csv', index=False)

if __name__ == '__main__':
	main()
