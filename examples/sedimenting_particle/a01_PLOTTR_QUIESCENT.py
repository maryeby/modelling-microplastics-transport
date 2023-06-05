#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/s2182576/Documents/academia/thesis/'
				+ 'modelling-microplastics-transport/examples/'
				+ 'sedimenting_particle')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

def main():
	"""
	Plots the vertical particle velocity over time and the trajectory of the
	particle on the x-z plane.
	"""
	if __name__ == '__main__':
		# Read data
		#prasath_data = pd.read_csv('prasath_data.csv')
		numerics = pd.read_csv('../data/numerical_results.csv')
		asymptotics = pd.read_csv('../data/asymptotic_results.csv')
		plot_labels = pd.read_csv('../data/plot_labels.csv')

		# Generate plots
		title_fs, label_fs, tick_fs = 16, 14, 12 
		sigma = plot_labels['sigma'].tolist()
		beta  = plot_labels['beta'].tolist()

		plt.figure(figsize=(16, 9))
		plt.suptitle(r'Figure 5 Recreation', fontsize=title_fs)

		# Velocity plot
		plt.subplot(121)
		plt.xlabel(r'$t$', fontsize=label_fs)
		plt.ylabel(r'$ 1 + q^{(2)}(0, t) $', fontsize=label_fs)
		plt.axis([0, 10, 0.95, 1.02])
		plt.xticks([0, 5, 10])
		plt.yticks([0.96, 0.98, 1.00, 1.02])
		plt.tick_params(labelsize=tick_fs)
		plt.plot('t_numerical', 'q01', c='mediumvioletred', data=numerics,
					label=r'my numerics, $\beta = $ {:.2f}'.format(beta[0]))
		plt.plot('t_numerical', 'q02', c='hotpink', data=numerics,
					label=r'my numerics, $\beta = $ {:.2f}'.format(beta[1]))
		plt.plot('t_numerical', 'q03', c='orange', lw=1.5, data=numerics,
				 label=r'my numerics, $\beta = $ {:.2f}'.format(beta[2]))
		plt.axhline(asymptotics['q01_asymp'][0], c='mediumvioletred', ls=':')
		plt.axhline(asymptotics['q02_asymp'][0], c='hotpink', ls=':')
		plt.axhline(asymptotics['q03_asymp'][0], c='orange', ls=':')
		#plt.plot('t_asymp', 'q01_asymp', c='mediumvioletred', ls=':',
		#		  data=asymptotics,
		#		  label=r'eq (4.8), $\beta = $ {:.2f}'.format(beta[0]))
		#plt.plot('t_asymp', 'q02_asymp', c='hotpink', ls=':',
		#		  data=asymptotics,
		#		  label=r'eq (4.8), $\beta = $ {:.2f}'.format(beta[1]))
		#plt.plot('t_asymp', 'q03_asymp', c='orange', ls=':',
		#		  data=asymptotics,
		#		  label=r'eq (4.8), $\beta = $ {:.2f}'.format(beta[2]))
		#plt.plot('t_heavy', 'q_heavy', '.-b', data=prasath_data, 
		#		  label=r'Prasath data, $\beta = 5$')
		#plt.plot('t_med', 'q_med', '.-g', data=prasath_data,
		#		  label=r'Prasath data, $\beta = 1$')
		#plt.plot('t_light', 'q_light', '.-r', data=prasath_data,
		#		  label=r'Prasath data, $\beta = 0.01$')
		#plt.plot('asymp_t_heavy', 'asymp_q_heavy', 'o:b', mfc='none', mec='b',
		#		  data=prasath_data,
		#		  label=r'Prasath asymptotic data, $\beta = 5$')
		#plt.plot('asymp_t_med', 'asymp_q_med', 'o:g', mfc='none', mec='g',
		#		  data=prasath_data,
		#		  label=r'Prasath asymptotic data, $\beta = 1$')
		#plt.plot('asymp_t_light', 'asymp_q_light', 'o:r', mfc='none', mec='r',
		#		  data=prasath_data, label=r'Prasath asymptotic data,
		#		  $\beta = 0.01$')
		plt.legend()

		# Trajectory plot
		plt.subplot(122)
		plt.xlabel('$y^{(1)}(t)$', fontsize=label_fs)
		plt.ylabel('$y^{(2)}(t)$', fontsize=label_fs)
		plt.axis([0, 0.04, 0, 0.035])
		plt.xticks([0, 0.01, 0.02, 0.03, 0.04])
		plt.yticks([0, 0.01, 0.02, 0.03])
		plt.tick_params(labelsize=tick_fs)
		formatter = ticker.ScalarFormatter(useMathText=True)
		formatter.set_scientific(True)
		formatter.set_powerlimits((-2, 2))
		plt.subplot(122).xaxis.set_major_formatter(formatter)
		plt.subplot(122).yaxis.set_major_formatter(formatter)
		plt.plot('x1', 'z1', c='mediumvioletred', data=numerics,
				 label=r'my numerics, $\beta = $ {:.2f}'.format(beta[0]))
		plt.plot('x2', 'z2', c='hotpink', lw=3, data=numerics,
				 label=r'my numerics, $\beta = $ {:.2f}'.format(beta[1]))
		plt.plot('x3', 'z3', c='orange', data=numerics,
				 label=r'my numerics, $\beta = $ {:.2f}'.format(beta[2]))
		#plt.plot('y1_heavy', 'y2_heavy', '.-b', data=prasath_data,
		#		 label=r'Prasath data, $\beta = 5$')
		#plt.plot('y1_med', 'y2_med', '.-g', data=prasath_data,
		#		 label=r'Prasath data, $\beta = 1$')
		#plt.plot('y1_light', 'y2_light', '.-r', data=prasath_data,
		#		 label=r'Prasath data, $\beta = 0.01$')
		#plt.plot('asymp_y1_heavy', 'asymp_y2_heavy', 'o:b', mfc='none', mec='b',
		#		 data=prasath_data, label=r'Prasath data, $\beta = 5$')
		#plt.plot('asymp_y1_med', 'asymp_y2_med', 'o:g', mfc='none', mec='g',
		#		 data=prasath_data, label=r'Prasath data, $\beta = 1$')
		#plt.plot('asymp_y1_light', 'asymp_y2_light', 'o:r', mfc='none', mec='r',
		#		 data=prasath_data, label=r'Prasath data, $\beta = 0.01$')
		plt.legend()
		plt.tight_layout()
		plt.show()
main()
