import matplotlib.pyplot as plt
import numpy as np

from utils.plot import initialize_figure as fig
XMIN, XMAX, YMIN, YMAX = 0, 1, 0, 1
FRAME_LIMS = [-0.1, 1.1, 0, 1.3]
XINT1, XINT2 = 0.318875, 0.444452
RADIUS, CENTER_X, CENTER_Y = 0.1, 0.7, 0.83
FIVE, NINE, TEN = 7 * np.pi / 4, np.pi, 3 * np.pi / 4 # clock positions

def main():
	"""Plot a diagram of a wave propagating in the ocean."""
	fig(lims=FRAME_LIMS, width='jfm', hide_xticks=True, hide_yticks=True)
	plt.axis('off')

	# shade and label seabed and x'
	plt.fill_between([XMIN, XMAX], YMAX / 10, YMAX / 5, color='silver')
	plt.annotate('seabed', (XMIN - 0.01, YMAX / 10 + 0.03), va='center',
				 ha='right')
	plt.annotate(r"$x$", (XMAX / 2, YMIN), va='bottom')

	# add sea surface, wave, and the curve connecting them
	x1 = np.linspace(0, XINT1, 100)
	x2 = np.linspace(XINT1, XINT2, 50)
	wave = np.sin(9 * np.pi * x1) / 6 + YMAX
	curve = 1.35e7 * np.exp(-60 * x2) + YMAX
	plt.plot(x1, wave, c='k')
	plt.plot(x2, curve, c='k')
	plt.plot([XINT2, YMAX], [XMAX, YMAX], c='k')

	# add vertical axis and z' labels at the seabed and surface
	plt.annotate('', xytext=(XMAX, YMAX / 5), xy=(XMAX, YMAX),
				 arrowprops=dict(arrowstyle='<->'))
	plt.annotate(r"$z=0$", (XMAX + 0.02, YMAX), va='center', ha='left')
	plt.annotate(r"$z=-h$", (XMAX + 0.02, YMAX / 5), va='bottom', ha='left')

	# add orbit, particle at period endpoint, and arrows
	circle = lambda theta : (RADIUS * np.cos(theta) + CENTER_X, \
							 RADIUS * 1.7 * np.sin(theta) + CENTER_Y)
	theta = np.linspace(0, 2 * np.pi, 100)
	plt.plot(circle(theta)[0], circle(theta)[1], c='k')
	plt.annotate(r"$\bm{x} = \bm{x}_p$", xytext=(circle(NINE)[0] - 1.23e-1,
				 circle(NINE)[1] - 1e-2), xy=(circle(NINE)[0],
				 circle(NINE)[1]))
	plt.scatter(circle(NINE)[0], circle(NINE)[1], c='k')
	plt.annotate('', xytext=(circle(TEN)[0], circle(TEN)[1]),
				 xy=(circle(TEN + 1e-8)[0], circle(TEN + 1e-8)[1]),
				 arrowprops=dict(arrowstyle='<-'))
	plt.annotate('', xytext=(circle(FIVE)[0], circle(FIVE)[1]),
				 xy=(circle(FIVE - 1e-8)[0], circle(FIVE - 1e-8)[1]),
				 arrowprops=dict(arrowstyle='->'))

	# add and label direction of wave propagation
	plt.annotate('direction of wave propagation', (CENTER_X, 1.27),
				 ha='center', va='bottom')
	plt.annotate('', xytext=(0.55, 1.25), xy=(0.85, 1.25),
				 arrowprops=dict(arrowstyle='->'))

	# add labels of the wavelength and amplitude
	plt.annotate('', xytext=(1 / 18, 1.25), xy=(5 / 18, 1.25),
				 arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5'))
	plt.annotate('', xytext=(-0.03, 1), xy=(-0.03, 7 / 6),
				 arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5'))
	plt.annotate(r"$\lambda$", (3 / 18, 1.26), va='bottom', ha='center')
	plt.annotate(r"$A$", (-0.07, 13 / 12 - 0.01), va='center', ha='left')

	# add axis key
	origin = (XMIN, YMIN + YMAX / 2)
	plt.annotate('', xytext=(origin[0] - 0.005, origin[1]),
				 xy=(origin[0] + XMAX / 10, origin[1]),
				 arrowprops=dict(arrowstyle='->'))
	plt.annotate('', xytext=(origin[0], origin[1] - 0.01),
				 xy=(origin[0], origin[1] + YMAX / 5),
				 arrowprops=dict(arrowstyle='->'))
	plt.annotate('', xytext=(origin[0] + 0.005, origin[1] + 0.01),
				 xy=(origin[0] - XMAX / 18, origin[1] - YMAX / 9),
				 arrowprops=dict(arrowstyle='->'))
	plt.annotate(r"$x$", (origin[0] + XMAX / 10, origin[1]), va='center')
	plt.annotate(r"$z$", (origin[0], origin[1] + YMAX / 5), ha='center')
	plt.annotate(r"$y$", (origin[0] - XMAX / 18, origin[1] - YMAX / 9),
				 va='top')
	plt.show()

if __name__ == '__main__': main()
