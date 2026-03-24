import pandas as pd
import numpy as np

from utils.data_tools import update_results
from transport_framework import particle as prt
from models import quiescent_flow as fl
from models import my_system as ts

# density ratios
BETA = 8.1
RS = [1 / (BETA + 0.5), 0.625]

# particle conditions
D_PS = np.array([0.00318, 0.0002])					# diameters
L = D_PS / 2										# radii (L' scale)
Z_0S = np.array([-30, 0]) * D_PS / L				# vertical positions
YS = np.array([[0, 0], Z_0S, [0, 0], [0, 0]]).T		# intial x0, xdot0
STOKES_HATS = np.array([0.111969 / BETA, 0.048444455])

# simulation conditions
U_T = np.array([-0.039126062310932797, 0.00218])
G = np.array([9.81]) * L / U_T ** 2
DELTA_T = 1e-2
OUT_FILE = '../../data/relaxing_particle/exp_numerics.csv'

def main():
	"""
	Run simulations to compare to experimental[^1] sedimenting particle data.

	References
	----------
	[^1]: [T. Jaroslawski et al. (2025).](https://doi.org/10.1103/PhysRevFluids.
		  10.L062301) Stokesian settling from quiescence: Experiments and theory
		  on history effects and unsteady flow structures. *Physical Review
		  Fluids* 10(6), L062301-1–L062301-10.
	"""
	results = {'t': [], 'z': [], 'zdot': [], 'history': [], 'd_p': []}
	for st, g, r, y, a, ut, dp in zip(STOKES_HATS, G, RS, YS, L, U_T, D_PS): 
		particle = prt.Particle(st)
		flow = fl.QuiescentFlow(1000, g)
		system = ts.MyTransportSystem(particle, flow, r)

		# run simulations and scale results
		for history in [False, True]:
			t = np.arange(0, 15, DELTA_T)
			_, z, _, zdot, t = system.maxey_riley(t, y, history)[:5]
			if np.any(t < 0): print(t)
			z = np.abs(z - y[1]) * a
			t *= np.abs(a / ut)
			results = update_results(results, [t, z, zdot], [history, dp])
	pd.DataFrame(results).to_csv(OUT_FILE, index=False) # write to data file

if __name__ == '__main__':
	main()
