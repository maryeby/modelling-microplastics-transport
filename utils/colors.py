COLORS = ['#003c30', '#01665e', '#35978f', '#80cdc1', '#c7eae5', '#f6e8c3',
		  '#dfc27d', '#bf812d', '#8c510a', '#543005', 'silver']

def print_success(message):
	"""Print `SUCCESS` in colored text followed by the provided `message`."""
	color = int(COLORS[2][1:].upper(), 16)
	print('\x1B[38;2;{};{};{}m{}\x1B[0m'.format(color>>16, color>>8&0xFF,
												color&0xFF, 'SUCCESS'), end='')
	print(':', message)

def print_warning(message):
	"""Print `WARNING` in colored text followed by the provided `message`."""
	color = int(COLORS[6][1:].upper(), 16)
	print('\x1B[38;2;{};{};{}m{}\x1B[0m'.format(color>>16, color>>8&0xFF,
												color&0xFF, 'WARNING'), end='')
	print(':', message)

def print_failure(message):
	"""Print `FAILURE` in colored text followed by the provided `message`."""
	color = int(COLORS[8][1:].upper(), 16)
	print('\x1B[38;2;{};{};{}m{}\x1B[0m'.format(color>>16, color>>8&0xFF,
												color&0xFF, 'FAILURE'), end='')
	print(':', message)
