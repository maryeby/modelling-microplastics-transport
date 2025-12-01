import pandas as pd
import numpy as np

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import quiescent_flow as fl
from models import relaxing_system as ts
from examples.relaxing_particle.numerics import STOKES_HAT, BETAS
from examples.relaxing_particle.numerics import OUT_FILE as IN_FILE

OUT_FILE = '../data/relaxing_particle/asymptotics.csv'

def main():
	"""
	Compute the asymptotic velocity of a relaxing particle in a quiescent flow.

	The asymptotic solution[^1] is only computed for the horizontal velocity of
	the particle. Results are saved to the `data/relaxing_particle` directory.

	See Also
	--------
	models.relaxing_system.RelaxingTransportSystem.asymptotic_velocity

	References
	----------
	[^1]: [S. G. Prasath et al. (2019)](https://doi.org/10.1017/jfm.2019.194)
		  Accurate solution method for the Maxey–Riley equation, and the
		  effects of Basset history. *Journal of Fluid Mechanics* 868, 428–460.
	"""
	# get time t and values of beta from numerics
	numerics = pd.read_csv(IN_FILE)	
	t = numerics['t'][1:]
	results = {'t': [], 'xdot': [], 'beta': []}
	flow = fl.QuiescentFlow()

	# calculate the asymptotic velocity of a particle with each beta
	for beta in BETAS:
		particle = prt.Particle(STOKES_HAT)
		system = ts.RelaxingTransportSystem(particle, flow, beta)
		xdot = system.asymptotic_velocity(t)
		results = update_results(results, [t, xdot], [beta])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file
if __name__ == '__main__':
	main()
