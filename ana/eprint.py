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

def eprint( expr, g, l, lprefix="", rprefix="", tail="" ):
    """
    Returns formatted line showing:
   
    1. lhs : numpy or python expression 
    2. rhs : evaluated result of the expression 

    :param expr: expression to be evaluated
    :param g: globals() from calling scope
    :param l: locals() from calling scope
    :param lprefix: lhs prefix before the expression
    :param rprefix: rhs prefix before the result
    :param tail: after the result 
    :return line:  formatted line string
    """
    ret = eval(expr, g, l)
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, ret )
    print("%-50s : %s%s" % ( lhs, rhs, tail )   )   
    return ret 

def epr(arg, g, l, **kwa):
    """
    :param arg:
    :param g: globals() from calling scope
    :param l: locals() from calling scope
    :param kwa: named args passed to eprint 
    """
    p = arg.find("=")  
    if p > -1: 
        var_eq = arg[:p+1]
        expr = arg[p+1:]
        label = var_eq
    else:
        label, expr = "", arg 
    pass
    return eprint(expr, g, l,  lprefix=label,  **kwa)




