import matplotlib.pyplot as plt
FS, LFS = 14, 16 # font sizes

def initialize_figure(x_label, y_label, lims=None, x_scale=None, y_scale=None,
					  make_square=False, equal_aspect=False):
	"""Initialize a pyplot figure according to the provided specifications."""
	plt.figure()
	set_figure_specs(x_label, y_label, lims, x_scale, y_scale, make_square,
					 equal_aspect)

def initialize_subplot(num, x_label=None, y_label=None, lims=None, x_scale=None,
					   y_scale=None, make_square=False, equal_aspect=False):
	"""Initialize a pyplot subplot according to the provided specifications."""
	plt.subplot(num)
	set_figure_specs(x_label, y_label, lims, x_scale, y_scale, make_square,
					 equal_aspect)

def set_figure_specs(x_label, y_label, lims, x_scale, y_scale, make_square,
					 equal_aspect):
	"""Add the provided specifications to a pyplot plot."""
	if x_label: plt.xlabel(x_label, fontsize=LFS)
	if y_label: plt.ylabel(y_label, fontsize=LFS)
	plt.xticks(fontsize=FS)
	plt.yticks(fontsize=FS)
	plt.minorticks_on()
	if not x_label: plt.xticks([])
	if not y_label: plt.yticks([])
	if x_scale: plt.xscale(x_scale)
	if y_scale: plt.yscale(y_scale)
	if lims: plt.axis(lims)
	if make_square: plt.gca().set_box_aspect(1)
	if equal_aspect: plt.gca().set_aspect('equal')
	plt.tight_layout()
