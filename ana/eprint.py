#!/usr/bin/env python
"""
eprint.py
===========

In order for the below "eval" code to work from an invoking script 
it is necessary to pass globals() and locals() from the script 
into these functions.  

For example::

    from opticks.ana.eprint import eprint 
    ...
    eprint("np.all( t.boundary_lookup_all == t.boundary_lookup_all_src )", globals(), locals() )

"""

def eprint( expr, g, l, lprefix="", rprefix=""):
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, eval(expr, g, l) )
    print("%s : %s" % ( lhs, rhs )   )   


def epr(expr, g, l, prefix=""):
    ret = eval(expr, g, l)
    lhs = "%s%s" % (prefix, expr) 
    print("%-40s : %s " % (lhs, ret)) 
    return ret 


