import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import matplotlib.pyplot as plt

def main():
	"""
	This program plots the results of the leading order, first order, and second
	order inertial equations as derived in Santamaria et al. (2013) and Haller &
	Sapsis (2008), as well as the numerical solutions generated using the
	Daitche (2013) method.
	"""
	my_data = pd.read_csv('../data/deep_water_wave/inertial_equations.csv')
	plt.figure()
	plt.title('Comparing Inertial Equations', fontsize=18)
	plt.xlabel('x', fontsize=16)
	plt.ylabel('z', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.plot('numerical_x', 'numerical_z', 'k-', data=my_data,
			 label='Numerical (Daitche)')
	plt.plot('x0', 'z0', 'k--', data=my_data, label='Leading order')
	plt.plot('x1', 'z1', 'k-.', data=my_data, label='First order')
	plt.plot('x2_haller', 'z2_haller', 'k:', data=my_data,
			 label='Second order (Haller)')
	plt.plot('x2_santamaria', 'z2_santamaria', c='grey', ls=':', data=my_data,
			 label='Second order (Santamaria)')
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
