import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd

from transport_framework import particle as prt
from models import dim_deep_water_wave as dim_fl
from models import deep_water_wave as fl
from models import santamaria_system as sm
from models import haller_system as h
from models import my_system as my

def main():
	"""
	This program compares the inertial equation as derived in Santamaria et al.
	(2013) and Haller & Sapsis (2008) to the numerical results produced by the
	method outlined in Section 3 of Daitche (2013).
	"""
	# initialize variables for the Santamaria system
	beta = 0.98
	sm_St = 0.5
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=0.026, wavelength=0.5)
	sm_particle = prt.Particle(stokes_num=sm_St)
	sm_system = sm.SantamariaTransportSystem(sm_particle, sm_flow, beta)
	k = sm_flow.wavenum

	# initialize variables for the Haller and Daitche systems
	R = 2 / 3 * beta
	haller_flow = fl.DeepWaterWave(amplitude=0.026, wavelength=0.5)
	haller_St = haller_flow.epsilon * R * sm_St
	haller_particle = prt.Particle(stokes_num=haller_St)
	haller_system = h.HallerTransportSystem(haller_particle, haller_flow, R)
	my_system = my.MyTransportSystem(haller_particle, haller_flow, R)

	my_dict = dict.fromkeys(['numerical_x', 'numerical_z', 'x0', 'z0', 'x1',
							 'z1', 'x2_haller', 'z2_haller', 'x2_santamaria',
							 'z2_santamaria'])
	# generate numerical results
	x, z, _, _, _ = my_system.run_numerics(include_history=False)
	my_dict['numerical_x'] = x
	my_dict['numerical_z'] = z

	# generate leading, first, and second order results using the Haller system
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   order=0)
	my_dict['x0'] = x
	my_dict['z0'] = z
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   order=1)
	my_dict['x1'] = x
	my_dict['z1'] = z
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   order=2)
	my_dict['x2_haller'] = x
	my_dict['z2_haller'] = z

	# generate second order results using the Santamaria system
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=2)
	my_dict['x2_santamaria'] = k * x
	my_dict['z2_santamaria'] = k * z

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	inertial_results = pd.DataFrame(my_dict)
	inertial_results.to_csv('../data/deep_water_wave/inertial_equations.csv')

if __name__ == '__main__':
	main()
