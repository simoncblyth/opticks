#!/usr/bin/env python
"""
eprint.py
===========

Hmm: actually this debug expression dumping code
needs to be implemented inside the using script 
not brought in from a module to give access to the globals 
for the eval

But nevertheless a central version to copy from is handy to have.  

"""

import numpy as np

def eprint( expr, lprefix="", rprefix="" ):
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, eval(expr) )
    print("%s : %s" % ( lhs, rhs )   )   


def epr(expr, prefix=""):
    ret = eval(expr)
    lhs = "%s%s" % (prefix, expr) 
    print("%-40s : %s " % (lhs, ret)) 
    return ret 


