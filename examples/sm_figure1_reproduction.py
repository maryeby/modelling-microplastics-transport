import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transport_framework import particle as prt
from models import dim_deep_water_wave as dim_fl
from models import deep_water_wave as fl
from models import santamaria_system as sm
from models import haller_system as h

def main():
	"""
	This program reproduces Figure 1 from Santamaria et al. (2013), comparing my
	results (emulating the Santamaria and Haller systems) to Cathal's results 
	to ensure our solutions agree.
	"""
	# initialize variables for the Santamaria system
	heavy_beta = 0.96
	light_beta = 1.04
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=0.026, wavelength=0.5)
	sm_heavy_particle = prt.Particle(stokes_num=0.5)
	sm_light_particle = prt.Particle(stokes_num=0.5)
	sm_heavy_system = sm.SantamariaTransportSystem(sm_heavy_particle, sm_flow,
												   heavy_beta)
	sm_light_system = sm.SantamariaTransportSystem(sm_light_particle, sm_flow,
												   light_beta)
	k = sm_flow.wavenum

	# initialize variables for the Haller system
	heavy_R = 2 / 3 * 0.96
	light_R = 2 / 3 * 1.04
	x_0 = k * 0.13
	z_0 = k * -0.4
	haller_flow = fl.DeepWaterWave(amplitude=0.026, wavelength=0.5)
	heavy_st = haller_flow.epsilon * heavy_R * 0.5
	light_st = haller_flow.epsilon * light_R * 0.5
	haller_heavy_particle = prt.Particle(stokes_num=heavy_st)
	haller_light_particle = prt.Particle(stokes_num=light_st)
	haller_heavy_system = h.HallerTransportSystem(haller_heavy_particle,
												  haller_flow, heavy_R)
	haller_light_system = h.HallerTransportSystem(haller_light_particle,
												  haller_flow, light_R)

	# generate results for the heavy and light particles
	x, z, _, _, _ = sm_heavy_system.run_numerics(sm_heavy_system.maxey_riley,
												 num_periods=100)
	sm_heavy = {'x': k * x, 'z': k * z}
	x, z, _, _, _ = sm_light_system.run_numerics(sm_light_system.maxey_riley,
												 x_0=0.13, z_0=-0.4,
												 num_periods=100)
	sm_light = {'x': k * x, 'z': k * z}
	x, z, _, _, _ = haller_heavy_system.run_numerics(
					haller_heavy_system.maxey_riley, num_periods=100)
	haller_heavy = {'x': x, 'z': z}
	x, z, _, _, _ = haller_light_system.run_numerics(
					haller_light_system.maxey_riley, x_0=x_0, z_0=z_0,
					num_periods=100)
	haller_light = {'x': x, 'z': z}

	# read data from Cathal's plot
	if __name__ == '__main__':
		cathals_data = pd.read_csv('data/cathals_sm_fig1_recreation.csv')

	# plot results
	plt.figure()
	plt.title('Recreation of Santamaria Fig. 1 with ' \
			  + r'$\beta = ${:.2f} and $\beta = ${:.2f}'.format(heavy_beta,
																light_beta))
	plt.xlabel('kx', fontsize=16)
	plt.ylabel('kz', fontsize=16)
	plt.axis([0, 3.2, -4, 0])
	plt.xticks(fontsize=14)
	plt.yticks([-3, -2, -1, 0], fontsize=14)

	if __name__ == '__main__':
		plt.plot('heavy_x', 'heavy_z', c='mediumpurple', lw=3,
				 data=cathals_data, label='Cathal\'s data (heavy)')
		plt.plot('light_x', 'light_z', c='hotpink', data=cathals_data,
				 label='Cathal\'s data (light)')

	plt.plot('x', 'z', c='k', data=sm_heavy, label='my SM data (heavy)')
	plt.plot('x', 'z', 'k:', data=sm_light, label='my SM data (light)')
	plt.plot('x', 'z', c='cornflowerblue', ls='--',
			 data=haller_heavy, label='my Haller data (heavy)')
	plt.plot('x', 'z', c='grey', ls='-.',
			 data=haller_light, label='my Haller data (light)')
	plt.legend()
	plt.show()
main()
