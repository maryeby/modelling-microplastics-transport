import sys 
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport')
import pandas as pd
import numpy as np

from transport_framework import particle as prt
from models import quiescent_flow as fl
from models import relaxing_system as ts

def main():
	"""
	This program computes asymptotic solutions for a relaxing particle in a
	quiescent flow, and saves the results to the `data` directory.
	"""
	# initialize variables for the transport systems
	R_light = 2 / (1 + 2 * 0.01)
	R_neutral = 2 / (1 + 2 * 1)
	R_heavy = 2 / (1 + 2 * 5)
	my_particle = prt.Particle(stokes_num=0.66)
	my_flow = fl.QuiescentFlow()
	light_system = ts.RelaxingTransportSystem(my_particle, my_flow, R_light)
	neutral_system = ts.RelaxingTransportSystem(my_particle, my_flow, R_neutral)
	heavy_system = ts.RelaxingTransportSystem(my_particle, my_flow, R_heavy)

	# get time t from numerics
	numerics = pd.read_csv('../data/relaxing_particle/numerics.csv')	
	t = numerics['t'][1:]

	# compute asymptotics
	light = light_system.asymptotic_velocity(t)
	neutral = neutral_system.asymptotic_velocity(t)
	heavy = heavy_system.asymptotic_velocity(t)

	# store results and write to a csv file
	asymptotics = pd.DataFrame({'t': t, 'light': light, 'neutral': neutral,
								'heavy': heavy})
	asymptotics.to_csv('../data/relaxing_particle/asymptotics.csv',
					   index=False)
if __name__ == '__main__':
	main()
