import pandas as pd
import itertools

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import quiescent_flow as fl
from models import relaxing_system as ts

# particle conditions
STOKES_HAT = 2 / 3		# translated Stokes number from Ref. 1 Figure 4
X_0, Z_0 = 0, 0			# initial particle position
XDOT_0, ZDOT_0 = 1, 1	# initial particle velocity

# simulation conditions
BETAS = [0.01, 1, 5]	# values of beta from Ref. 1 Figure 4
T_FINAL = 15			# total time
DELTA_T = 1e-2			# timestep
OUT_FILE = '../../data/relaxing_particle/velocity_numerics.csv'

def main():
	"""
	Run numerical simulations for a relaxing particle in a quiescent flow.

	This program reproduces the results from Ref. 1 Figure 4, and saves the
	results to the `data/relaxing_particle` directory.

	References
	----------
	[^1]: [S. G. Prasath et al. (2019).](https://doi.org/10.1017/jfm.2019.194)
		  Accurate solution method for the Maxey–Riley equation, and the
		  effects of Basset history. *Journal of Fluid Mechanics* 868, 428–460.
	"""
	# initialize Particle and Flow objects, create dictionary to store sols
	flow = fl.QuiescentFlow()
	results = {'t': [], 'xdot': [], 'beta': [], 'history': []}

	for beta, include_history in itertools.product(BETAS, [False, True]):
		# compute R and initialize TransportSystem object
		particle = prt.Particle(STOKES_HAT)
		system = ts.RelaxingTransportSystem(particle, flow, beta)

		# compute results
		_, _, xdot, _, t = system.run_numerics(X_0, Z_0, XDOT_0, ZDOT_0,
											   T_FINAL, DELTA_T,
											   include_history)
		# store results
		results = update_results(results, [t, xdot], [beta, include_history])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()
