#!/usr/bin/env python

# make this file executable: $(chmod +x gauss_err.py)

# automatically generate latex code for gaussian uncertainty propagation using sympy
import sympy
import argparse


# input target function
def error(f, err_vars=None):
	from sympy import Symbol, latex

	s = 0 # summation of squared result
	latex_names = dict() # use latex notation

	# extract variables from input
	if err_vars is None:
		err_vars = f.free_symbols

	# execute calculation for each variable
	for v in err_vars:
		err = Symbol("latex_std_" + v.name)
		s += f.diff(v) ** 2 * err ** 2
#		latex_names[err] = "\\sigma_{" + latex(v) + "}"
		latex_names[err] = "\\left(" + "\\symup{\Delta}" + latex(v) + "\\right)"

	# format output to latex, issues with upright notation
	return latex(sympy.sqrt(s), symbol_names=latex_names)


# parse user input
parser = argparse.ArgumentParser(description='automatically generate latex code for error propagation')
parser.add_argument('f', nargs=1, help='function to be parsed by script')
args = parser.parse_args()

# define target function
f = sympy.parsing.sympy_parser.parse_expr(args.f[0])

# print output
print(f'% error propagation for function f = {f} \n{error(f)}')
