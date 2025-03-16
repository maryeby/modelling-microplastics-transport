import pandas as pd
import numpy as np

def extract_data(name, df, params=None):
	"""
	Filter through data from the provided DataFrame `df`.
	
	Parameters
	----------
	name : str or list
		The variable(s) to be returned.
	df : DataFrame
		The data to filter through.
	params : dict, default=None
		Parameters used to filter through the data.

	Returns
	-------
	Series or list of Series
		The data specified by the provided name.

	Notes
	-----
	Returns a Series if the provided `name` is a string, or a list of Series
	if `name` is a list. Keywords `none` and `exists` can be used as a `value`
	in the `params` dict to check whether the `key` is a NaN value.
	"""
	series_list = []
	if not params: # if the dictionary is empty
		if isinstance(name, list):
			tuple_list = list(df[name].dropna().items())
			series_list = [t[1] for t in tuple_list]
			return series_list
		else:
			return df[name].dropna()
	elif len(params) == 1:
		key = list(params.keys())[0]
		value = list(params.values())[0]
		if value == 'none':
			condition = df[key].isna()
		elif value == 'exists':
			condition = df[key].notna()
		else:
			condition = df[key] == value
		if isinstance(name, list):
			tuple_list = list(df[name].where(condition).dropna().items())
			series_list = [t[1] for t in tuple_list]
			return series_list
		else:
			return df[name].where(condition).dropna()
	else:
		condition = True
		for key, value in params.items():
			if value == 'none':
				key_condition = df[key].isna()
			elif value == 'exists':
				key_condition = df[key].notna()
			else:
				key_condition = df[key] == value
			condition &= key_condition
		if isinstance(name, list):
			tuple_list = list(df[name].where(condition).dropna().items())
			series_list = [t[1] for t in tuple_list]
			return series_list
		else:
			return df[name].where(condition).dropna()

def match_data(data, extracted_data, rtol=1e-1, atol=1e-3):
	"""Return whether `data` matches `extracted data` within a tolerance."""
	assert extracted_data.size <= data.size, f'Data ({data.size}) must have ' \
		   + f'equal or more points than extracted data ({extracted_data.size})'
	sample = []
	for n in extracted_data:
		i = np.abs(data - n).argmin()
		sample.append(data[i]) 
	return np.allclose(sample, extracted_data, rtol, atol)

def update_results(results, arrays, scalars):
	"""
	Store solutions from `arrays` and `scalars` in `results`.

	Parameters
	----------
	results : dict
		A dictionary containing `str` keys and `list` items.
	arrays : list
		A list of `ndarray` elements.
	scalars : list
		A list of non-array elements (`bool`, `int`, `float`, or `str`).

	Returns
	-------
	dict
		The `results` dictionary with additional solutions.
	"""
	assert len(arrays) + len(scalars) == len(results), \
		   f'Number of solutions ({len(arrays) + len(scalars)}) != number of ' \
		 + f'results to update ({len(results)})'
	keys = list(results.keys())
	if arrays:
		for i in range(len(arrays)):
			results[keys[i]] += arrays[i].tolist()
		for i in range(len(scalars)):
			results[keys[i + len(arrays)]] += [scalars[i]] * len(arrays[0])
	else:
		for i in range(len(scalars)): results[keys[i]].append(scalars[i])
	return results
