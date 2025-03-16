import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import initialize_subplot as subplot
from utils.plot import FS
from utils.data_tools import extract_data
from examples.water_wave.forces.numerics import STOKES_NUMS
from examples.water_wave.forces.analysis import ST_TO_SAVE

IN_FILE1 = '../../data/water_wave/forces_numerics.csv'
IN_FILE2 = '../../data/water_wave/forces_curve_fit.csv'
IN_FILE3 = '../../data/water_wave/forces_coeffs.csv'
TOL = 0.96

def main():
	"""
	Plot forces over time with curves fit to the data, and coefficients vs *St*.

	For curves fit to the forces acting on particles of different sizes (Stokes
	numbers) in a linear wave of deep water, the resulting coefficents are
	plotted over the Stokes number. For a selected Stokes number `ST_TO_SAVE`,
	the data for the forces are plotted over time, along with the fitted curves.
	"""
	# read data
	numerics = pd.read_csv(IN_FILE1)
	curves = pd.read_csv(IN_FILE2)
	coefficients = pd.read_csv(IN_FILE3, index_col=0)

	# force over time figure
	plt.figure()
	param = {'St': ST_TO_SAVE}
	subplot(411, y_label='inertial force')
	t, inertial = extract_data(['t', 'inertial'], curves)
	plt.plot(t, inertial, c='k')
	t, fpg, mass = extract_data(['t', 'fluid_pressure_gradient_x',
								 'added_mass_force_x'], numerics, param)
	inertial = fpg + mass
	plt.scatter(t, inertial, marker='.', edgecolors='k', facecolors='none')

	# plot Stokes drag over time
	subplot(412, y_label='Stokes drag')
	t, drag = extract_data(['t', 'stokes_drag'], curves)
	plt.plot(t, drag, c='k')
	t, drag = extract_data(['t', 'stokes_drag_x'], numerics, param)
	plt.scatter(t, drag, marker='.', edgecolors='k', facecolors='none')

	# plot history force over time
	subplot(413, y_label='history force')
	t, history = extract_data(['t', 'history'], curves)
	plt.plot(t, history, c='k')
	t, history = extract_data(['t', 'history_force_x'], numerics, param)
	plt.scatter(t, history, marker='.', edgecolors='k', facecolors='none')

	# plot velocity over time
	subplot(414, 'time', 'particle velocity')
	t, velocity = extract_data(['t', 'velocity'], curves)
	plt.plot(t, velocity, c='k')
	t, xdot = extract_data(['t', 'xdot'], numerics, param)
	plt.scatter(t, xdot, marker='.', edgecolors='k', facecolors='none')

	# St vs A subplot
	plt.figure()
	subplot(221, y_label=r'$A$', make_square=True)
#	subplot(121, r'$St$', r'$A$', make_square=True)
#	plt.gcf().text(0.02, 0.9, '(a)', fontsize=FS, fontfamily='serif',
#				   fontstyle='italic')

	# plot St vs A for each force
	plt.plot(STOKES_NUMS, coefficients.loc['inertial', 'A'], '-k.')
	plt.plot(STOKES_NUMS, coefficients.loc['stokes_drag', 'A'], '--k.')
	plt.plot(STOKES_NUMS, coefficients.loc['history', 'A'], ':k.')

	# plot quality control indicators
	plt.scatter(quality_control(coefficients, 'inertial', 'St')[0],
				quality_control(coefficients, 'inertial', 'A')[1],
				edgecolors='k', facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'stokes_drag', 'St')[0],
				quality_control(coefficients, 'stokes_drag', 'A')[1],
				edgecolors='k', facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'history', 'St')[0],
				quality_control(coefficients, 'history', 'A')[1],
				edgecolors='k', facecolors='none', marker='o')

	# initialize St vs phi subplot
	subplot(222, y_label=r'$\phi$', make_square=True)
#	subplot(122, r'$St$', r'$\phi$', make_square=True)
#	plt.gcf().text(0.525, 0.9, '(b)', fontsize=FS, fontfamily='serif',
#				   fontstyle='italic')
#	ticks = np.arange(-np.pi, 1.01 * np.pi, np.pi / 4)
#	tick_labels = [r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', '0',
#				   r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
#	plt.yticks(ticks=ticks, labels=tick_labels, fontsize=FS)
	
	# get values of phi
	inertial_phi = coefficients.loc['inertial', 'phi']
	stokes_drag_phi = coefficients.loc['stokes_drag', 'phi']
	history_phi = coefficients.loc['history', 'phi']

	# plot St vs phi
	plt.plot(STOKES_NUMS, inertial_phi, '-k.')
	plt.plot(STOKES_NUMS, stokes_drag_phi, '--k.')
	plt.plot(STOKES_NUMS, history_phi, ':k.')

	# plot quality control indicators
	plt.scatter(quality_control(coefficients, 'inertial', 'St')[0],
				quality_control(coefficients, 'inertial', 'phi')[1],
				edgecolors='k', facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'stokes_drag', 'St')[0],
				quality_control(coefficients, 'stokes_drag', 'phi')[1],
				edgecolors='k', facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'history', 'St')[0],
				quality_control(coefficients, 'history', 'phi')[1],
				edgecolors='k',	facecolors='none', marker='o')

	# initialize St vs phi subplot and plot St vs delta for each force
	subplot(223, r'$St$', r'$\delta$', make_square=True)
	plt.plot(STOKES_NUMS, coefficients.loc['inertial', 'delta'], '-k.')
	plt.plot(STOKES_NUMS, coefficients.loc['stokes_drag', 'delta'], '--k.')
	plt.plot(STOKES_NUMS, coefficients.loc['history', 'delta'], ':k.')

	# plot quality control indicators
	plt.scatter(quality_control(coefficients, 'inertial', 'St')[0],
				quality_control(coefficients, 'inertial', 'delta')[1],
				edgecolors='k', facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'stokes_drag', 'St')[0],
				quality_control(coefficients, 'stokes_drag', 'delta')[1],
				edgecolors='k', facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'history', 'St')[0],
				quality_control(coefficients, 'history', 'delta')[1],
				edgecolors='k', facecolors='none', marker='o')

	# plot St vs offset for each force
	print('Max offset:',
		  max(max(np.abs(coefficients.loc['inertial', 'offset'])),
		  max(np.abs(coefficients.loc['stokes_drag', 'offset'])),
		  max(np.abs(coefficients.loc['history', 'offset']))))

	subplot(224, r'$St$', 'offset', make_square=True)
	plt.plot(STOKES_NUMS, coefficients.loc['inertial', 'offset'], '-k.',
			 label='inertial force')
	plt.plot(STOKES_NUMS, coefficients.loc['stokes_drag', 'offset'], '--k.',
			 label='Stokes drag')
	plt.plot(STOKES_NUMS, coefficients.loc['history', 'offset'], '-.k.',
			 label='history force')

	# plot quality control indicators
	plt.scatter(quality_control(coefficients, 'inertial', 'St')[0],
				quality_control(coefficients, 'inertial', 'offset')[1],
				edgecolors='k',	facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'stokes_drag', 'St')[0],
				quality_control(coefficients, 'stokes_drag', 'offset')[1],
				edgecolors='k',	facecolors='none', marker='o')
	plt.scatter(quality_control(coefficients, 'history', 'St')[0],
				quality_control(coefficients, 'history', 'offset')[1],
				edgecolors='k', facecolors='none', marker='o',
				label=r'$R^2 <$' + str(TOL))
	plt.legend(fontsize=FS)
	plt.show()

def quality_control(coefficients, force, coeff):
	r"""
	Return where the $R^2$ of the curve fit to `force` is below a tolerance.

	Parameters
	----------
	coefficients : DataFrame
		A `DataFrame` of float data, coefficients for the curves fit to forces.
	force : str
		The force used to index the `coefficients` `DataFrame`.
	coeff : str
		The coefficient used to index the `coefficients` `DataFrame`.
	"""
	return coefficients.loc[force, 'St']\
					   .where(coefficients.loc[force, 'R^2'] < TOL).dropna(), \
		   coefficients.loc[force, coeff]\
					   .where(coefficients.loc[force, 'R^2'] < TOL).dropna()

if __name__ == '__main__':
	main()
