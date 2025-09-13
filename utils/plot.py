import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PTS_PER_INCH = 72.27			# used to compute figure size
THESIS_WIDTH = 426.79135		# textwidth for thesis LaTeX template
JFM_WIDTH = 384					# textwidth for JFM LaTeX template
RATIO = (5 ** (1 / 2) - 1) / 2	# ratio of figure width to height

# formatting for subplot (a), (b) labels
LABEL_AX = 0.02
LABEL_BX = 0.525
LABEL_Y = 0.9

def initialize_figure(x_label=None, y_label=None, num=None, make_square=False,
					  equal_aspect=False, width=THESIS_WIDTH, lims=None,
					  x_scale=None, y_scale=None, hide_xticks=False,
					  hide_yticks=False, add_subplot_labels=False):
	"""
	Initialize a pyplot figure according to the provided specifications.

	Parameters
	----------
	x_label, y_label : str, default=None
		The horizontal and vertical axis labels.
	num : int, default=None
		The 3-digit integer used to initialize subplots.
	make_square : bool, default=False
		Whether to make the plot square.
	equal_aspect : bool, default=False
		Whether to make the aspect ratio of the pyplot Axes scaling equal.
	width : str or float, default=THESIS_WIDTH
		The width of the figure in pts, or 'jfm' to use the JFM template width.
	lims : list, default=None
		A list containing the axis limits `[xmin, xmax, ymin, ymax]`.
	x_scale, y_scale : str, default=None
		The axis scale, such as `log`.
	hide_xticks, hide_yticks : bool, default=False
		Whether to hide tick labels on the horizontal or vertical axis.
	add_subplot_labels : bool, default=False
		Whether to include *(a)* and *(b)* labels on subplots.
	"""
	rows, cols = 1, 1
	plt.style.use('tex')

	# skip initialization of the figure for subplots (except the first subplot)
	if num and isinstance(num, int):
		if num % 10 == 1:
			plt.figure(layout='constrained')
			if add_subplot_labels:
				plt.suptitle(' ') # add space for labels
				plt.gcf().text(LABEL_AX, LABEL_Y, r'$(a)$')
				plt.gcf().text(LABEL_BX, LABEL_Y, r'$(b)$')
		plt.subplot(num)
		rows, cols = num // 100, num % 100 // 10
	elif num and isinstance(num, gridspec.SubplotSpec):
		rows, cols, row_index, col_index = num.get_geometry()
		if row_index == 0 and col_index == 0:
			plt.figure(layout='tight')
			if add_subplot_labels:
				plt.suptitle(' ') # add space for labels
				plt.gcf().text(LABEL_AX, LABEL_Y, r'$(a)$')
				plt.gcf().text(LABEL_BX, LABEL_Y, r'$(b)$')
		plt.subplot(num)
	else:
		plt.figure(layout='constrained')
	
	# set aspect ratios, scalings, and axis limits
	if make_square: plt.gca().set_box_aspect(1)
	if equal_aspect: plt.gca().set_aspect('equal')
	if x_scale: plt.xscale(x_scale)
	if y_scale: plt.yscale(y_scale)
	if lims: plt.axis(lims)

	# set axis labels and ticks labels
	if x_label: plt.xlabel(x_label)
	if y_label: plt.ylabel(y_label)
	if hide_xticks: plt.xticks([])
	if hide_yticks: plt.yticks([])
	plt.minorticks_on()

	# set figure size
	if width == 'jfm': width = JFM_WIDTH
	width /= PTS_PER_INCH
	height = width * RATIO * (rows / cols)
	plt.gcf().set_size_inches(width, height)
