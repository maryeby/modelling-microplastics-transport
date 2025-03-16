import numpy as np
import pandas as pd

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import dim_deep_water_wave as dfl
from models import deep_water_wave as fl
from models import santamaria_system as sts
from models import haller_system as hts
from models import my_system as ts

# wave & particle conditions
AMPLITUDE = 0.026
WAVELENGTH = 0.5
STOKES_NUM = 0.5			# St as specified in Santamaria et al. (2013)
X_0, Z_0 = 0, 0				# initial particle position

# simulation conditions
SCALE = 2 / 3			    # scale used for parameter translation
BETA = 0.98					# density ratio from Santamaria et al. (2013)
R = SCALE * BETA			# density ratio as defined in Haller & Sapsis (2008)
NUM_PERIODS = 5
DELTA_T = 5e-3
INCLUDE_HISTORY = False
OUT_FILE = '../data/deep_water_wave/inertial_numerics.csv'

def main():
	r"""
	Compute numerical solutions of the inertial and M-R equations[^1][^2].

	The solutions for the inertial equations are computed following the
	derivations in [1, 2], and a numerical simulation using the Maxey-Riley
	framework is performed using the TransportSystem objects corresponding to
	[1-3]. The simulation models a negatively buoyant particle in a linear wave
	of infinitely deep water without history effects. Results are saved to the
	`data/deep_water_wave` directory.

	References
	----------
	[^1]: [F. Santamaria et al. (2013).](
		  https://doi.org/10.1209/0295-5075/102/14003)
		  Stokes drift for inertial particles transported by water waves.
		  *EPL (Europhysics Letters)* 102(1), 14003.
	[^2]: [G. Haller & T. Sapsis (2008).](
		  https://doi.org/10.1016/j.physd.2007.09.027)
		  Where do inertial particles go in fluid flows?
		  *Physica D: Nonlinear Phenomena* 237(5), 573–583.
	[^3]: [A. Daitche (2013).](https://doi.org/10.1016/j.jcp.2013.07.024)
		  Advection of inertial particles in the presence of the history force:
		  Higher order numerical schemes. *Journal of Computational Physics*
		  254, 93–106.
	"""
	# create dictonary to store solutions and Wave object
	keys = ['t', 'x', 'z', 'equation', 'order', 'method']
	results = {key: [] for key in keys}

	# run Santamaria simulations and store solutions
	particle = prt.Particle(STOKES_NUM)
	wave = dfl.DimensionalDeepWaterWave(AMPLITUDE, WAVELENGTH)
	transport_system = sts.SantamariaTransportSystem(particle, wave, BETA)
	simulate_equations(transport_system, 0, 'Santamaria', results)
	simulate_equations(transport_system, 1, 'Santamaria', results)
	simulate_equations(transport_system, 2, 'Santamaria', results)

	# run Haller simulations and store solutions
	particle = prt.Particle(STOKES_NUM * R * wave.froude_num)
	wave = fl.DeepWaterWave(AMPLITUDE, WAVELENGTH)
	transport_system = hts.HallerTransportSystem(particle, wave, R)
	simulate_equations(transport_system, 0, 'Haller', results)
	simulate_equations(transport_system, 1, 'Haller', results)
	simulate_equations(transport_system, 2, 'Haller', results)

	# run Daitche simulation and store solutions
	transport_system = ts.MyTransportSystem(particle, wave, R)
	xdot_0, zdot_0 = wave.velocity(X_0, Z_0, t=0)
	t = np.arange(0, NUM_PERIODS * wave.period, DELTA_T)
	y = [X_0, Z_0, xdot_0, zdot_0]
	print('Running Daitche simulation...')
	x, z, _, _, t, _, _, _, _, _, _, _, _, _, \
	   _ = transport_system.maxey_riley(t, y, INCLUDE_HISTORY)
	results = update_results(results, [t, x, z], ['Maxey-Riley', 3, 'Daitche'])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

def simulate_equations(system, order, method, results):
	"""
	Run simulations using the Maxey-Riley and inertial equations.

	Parameters
	----------
	system : TransportSystem (obj)
		The transport system through which the simulation is run.
	order : int
		The order of the inertial equation (0, 1, or 2).
	method : str
		The author name indicating which TransportSystem was used.
	results : dict
		Dictionary used to store the solutions.
	"""
	# create variables for scaling
	omega = system.flow.angular_freq
	Fr = system.flow.froude_num
	k = system.flow.wavenum

	# scale time-related arguments if necessary
	timestep = DELTA_T / (omega * Fr) if method == 'Santamaria' else DELTA_T
	
	# run M-R simulation
	if order == 0:
		print(f'Running Maxey-Riley simulation ({method})...', end='')
		x, z, _, _, t = system.run_numerics(system.maxey_riley, X_0, Z_0,
										   NUM_PERIODS, timestep)
		# scale results if necessary
		if method == 'Santamaria':
			x *= k
			z *= k
			t *= omega
		results = update_results(results, [t, x, z], ['Maxey-Riley', None,
													  method])
		print('done.')

	# run inertial equation simulation
	print(f'Running order {order} inertial equation simulation',
		  f'({method})...', end='')
	x, z, _, _, t = system.run_numerics(system.inertial_equation, X_0, Z_0,
									   NUM_PERIODS, timestep, order)
	# scale results if necessary
	if method == 'Santamaria':
		x *= k
		z *= k
		t *= omega
	results = update_results(results, [t, x, z], ['inertial', order, method])
	print('done.')

if __name__ == '__main__':
	main()
