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
	method outlined in Section 3 of Daitche (2013), the method outlined in
	Santamaria et al. (2013), and the method outlined in Haller & Sapsis (2008).
	The results are saved to the `data/deep_water_wave` directory. 
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
	haller_St = haller_flow.froude_num * R * sm_St
	haller_particle = prt.Particle(stokes_num=haller_St)
	haller_system = h.HallerTransportSystem(haller_particle, haller_flow, R)
	my_system = my.MyTransportSystem(haller_particle, haller_flow, R)

	# initialize variables for numerical simulations
	x_0, z_0 = 0, 0
	xdot_0, zdot_0 = haller_flow.velocity(x_0, z_0, 0)
	num_periods = 20
	delta_t = 1e-2 / 5
	T = sm_flow.angular_freq * sm_flow.froude_num	# time scaling
	my_dict = dict.fromkeys(['x_daitche', 'z_daitche', 'x_haller', 'z_haller',
							 'x_santamaria', 'z_santamaria', 'x0_haller',
							 'z0_haller', 'x0_santamaria', 'z0_santamaria',
							 'x1_haller', 'z1_haller', 'x1_santamaria',
							 'z1_santamaria', 'x2_haller', 'z2_haller',
							 'x2_santamaria','z2_santamaria'])

	# generate numerical results
	x, z, _, _, t = my_system.run_numerics(include_history=False,
										   x_0=x_0, z_0=z_0,
										   xdot_0=xdot_0, zdot_0=zdot_0,
										   delta_t=delta_t * T,
										   num_periods=num_periods * T)
	my_dict['x_daitche'] = x
	my_dict['z_daitche'] = z
	my_dict['t_daitche'] = t
	x, z, _, _, t = haller_system.run_numerics(haller_system.maxey_riley,
											   x_0=x_0, z_0=z_0,
											   delta_t=delta_t * T,
											   num_periods=num_periods * T)
	my_dict['x_haller'] = x
	my_dict['z_haller'] = z
	my_dict['t_haller'] = t
	x, z, _, _, t = sm_system.run_numerics(sm_system.maxey_riley,
										   x_0=x_0, z_0=z_0, delta_t=delta_t,
										   num_periods=num_periods)
	my_dict['x_santamaria'] = x * k
	my_dict['z_santamaria'] = z * k
	my_dict['t_santamaria'] = t

	# generate leading order results
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   x_0=x_0, z_0=z_0,
											   order=0, delta_t=delta_t * T,
											   num_periods=num_periods * T)
	my_dict['x0_haller'] = x
	my_dict['z0_haller'] = z
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=0,
										   x_0=x_0, z_0=z_0, delta_t=delta_t,
										   num_periods=num_periods)
	my_dict['x0_santamaria'] = x * k
	my_dict['z0_santamaria'] = z * k

	# generate first order results
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   x_0=x_0, z_0=z_0,
											   order=1, delta_t=delta_t * T,
											   num_periods=num_periods * T)
	my_dict['x1_haller'] = x
	my_dict['z1_haller'] = z
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=1,
										   x_0=x_0, z_0=z_0, delta_t=delta_t,
										   num_periods=num_periods)
	my_dict['x1_santamaria'] = x * k
	my_dict['z1_santamaria'] = z * k

	# generate second order results
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   x_0=x_0, z_0=z_0,
											   order=2, delta_t=delta_t * T,
											   num_periods=num_periods * T)
	my_dict['x2_haller'] = x
	my_dict['z2_haller'] = z
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=2,
										   x_0=x_0, z_0=z_0, delta_t=delta_t,
										   num_periods=num_periods)
	my_dict['x2_santamaria'] = x * k
	my_dict['z2_santamaria'] = z * k

	# write results to data file
	my_dict = dict([(key, pd.Series(value)) for key, value in my_dict.items()])
	inertial_results = pd.DataFrame(my_dict)
	inertial_results.to_csv('../data/deep_water_wave/inertial_equations.csv')

if __name__ == '__main__':
	main()
