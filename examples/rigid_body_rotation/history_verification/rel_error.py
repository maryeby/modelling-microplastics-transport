import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np

def main():
	"""
	This program computes the relative error for the numerical solutions of a
	rotating particle in a flow and saves the results to the
	`data/rigid_body_rotation` directory.
	"""
	# read data
	data_path = '../../data/rigid_body_rotation/'
	numerics = pd.read_csv(data_path + 'numerical_history.csv')
	analytics = pd.read_csv(data_path + 'analytical_history.csv')

	# compute relative error
	analytical = analytics[['x', 'z']].to_numpy()
	numerical = numerics[['x', 'z']].to_numpy()
	t = numerics['t']
	e_rel = np.linalg.norm(analytical - numerical, axis=1) \
						/ np.linalg.norm(analytical, axis=1)
	e_abs = np.linalg.norm(analytical - numerical, axis=1)
#	global_error = np.linalg.norm(analytical - numerical, axis=1).max()

	# write results to data file
	rel_error = pd.DataFrame({'t': t, 'E_rel': e_rel, 'E_abs': e_abs})
	rel_error.to_csv(data_path + 'rel_error_history.csv', index=False)

if __name__ == '__main__':
	main()
