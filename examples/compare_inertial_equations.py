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
	This program compares the inertial equation as derived in Santamaria et al.
	(2013) and Haller & Sapsis (2008).
	"""
	# initialize variables for the Santamaria system
	beta = 0.98
	sm_St = 0.5
	sm_flow = dim_fl.DimensionalDeepWaterWave(amplitude=0.026, wavelength=0.5)
	sm_particle = prt.Particle(stokes_num=sm_St)
	sm_system = sm.SantamariaTransportSystem(sm_particle, sm_flow, beta)
	k = sm_flow.wavenum

	# initialize variables for the Haller system
	R = 2 / 3 * beta
	haller_flow = fl.DeepWaterWave(amplitude=0.026, wavelength=0.5)
	haller_St = haller_flow.epsilon * R * sm_St
	haller_particle = prt.Particle(stokes_num=haller_St)
	haller_system = h.HallerTransportSystem(haller_particle, haller_flow, R)

	# generate results
	x, z, _, _, _ = sm_system.run_numerics(sm_system.maxey_riley)
	sm_numerics = {'x': k * x, 'z': k * z}
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=0)
	sm0 = {'x': k * x, 'z': k * z}
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=1)
	sm1 = {'x': k * x, 'z': k * z}
	x, z, _, _, _ = sm_system.run_numerics(sm_system.inertial_equation, order=2)
	sm2 = {'x': k * x, 'z': k * z}
	x, z, _, _, _ = haller_system.run_numerics(haller_system.maxey_riley)
	haller_numerics = {'x': x, 'z': z}
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   order=0)
	haller0 = {'x': x, 'z': z}
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   order=1)
	haller1 = {'x': x, 'z': z}
	x, z, _, _, _ = haller_system.run_numerics(haller_system.inertial_equation,
											   order=2)
	haller2 = {'x': x, 'z': z}

	# plot results
	plt.figure()
	plt.title('Comparing Inertial Equations')
	plt.xlabel('x', fontsize=16)
	plt.ylabel('z', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.plot('x', 'z', 'm-', lw=3, data=sm_numerics, label='SM numerics')
	plt.plot('x', 'z', 'm--', lw=3, data=sm0, label='SM leading order')
	plt.plot('x', 'z', 'm-.', lw=3, data=sm1, label='SM first order')
	plt.plot('x', 'z', 'm:', lw=3, data=sm2, label='SM second order')
	plt.plot('x', 'z', 'k-', data=haller_numerics, label='Haller numerics')
	plt.plot('x', 'z', 'k--', data=haller0, label='Haller leading order')
	plt.plot('x', 'z', 'k-.', data=haller1, label='Haller first order')
	plt.plot('x', 'z', 'k:', data=haller2, label='Haller second order')
	plt.legend()
	plt.show()
if __name__ == '__main__':
	main()
