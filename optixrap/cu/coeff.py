#!/usr/bin/env python

from sympy import expand, symbols, simplify, collect, factor, solve, Pow


def get_coeff_(expr, var, n):
    c = range(n)
    for i in range(n-1,-1,-1):
        c[i] = collect(expand(expr), var ).coeff(var,i)
    pass
    return c


def get_coeff(expr, var, n):
    c = range(n)
    for i in range(n-1,-1,-1):
        c[i] = factor(collect(expand(expr), var ).coeff(var,i))
    pass
    return c

def print_coeff(c, msg="coeff"):
    print msg
    for i in range(len(c)):
        print "c[%d]:%r " % (i, c[i])
    pass

def subs_coeff(c, sub):
    for i in range(len(c)):
        c[i] = c[i].subs(sub)
    pass


def expr_coeff(c, var):
    ex = None
    for i in range(len(c)):
        if ex is None:
            ex = Pow(var, i)*c[i]
        else:
            ex += Pow(var, i)*c[i]
        pass
    pass
    return ex


