#!/usr/bin/env python
"""
eprint.py
===========

In order for the below "eval" code to work from an invoking script 
it is necessary to pass globals() and locals() from the script 
into these functions.  

For example::

    from opticks.ana.eprint import eprint, epr
    ...
    eprint("np.all( t.boundary_lookup_all == t.boundary_lookup_all_src )", globals(), locals() )

    # OR returning the evaluation with epr:
    bm = epr("bm = np.all( t.boundary_lookup_all == t.boundary_lookup_all_src )", globals(), locals() ) 

TODO: annotated check asserts

"""

def eprint( expr, g, l, lprefix="", rprefix="", tail="" ):
    """
    Returns the evaluated expression and prints a formatted line showing:
 
    1. lhs : numpy/python expression 
    2. rhs : evaluated result of the expression 

    :param expr: expression to be evaluated
    :param g: globals() from calling scope
    :param l: locals() from calling scope
    :param lprefix: lhs prefix before the expression
    :param rprefix: rhs prefix before the result
    :param tail: after the result 
    :return ret: result of the evaluation 
    """
    ret = eval(expr, g, l)
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, ret )
    print("%-50s : %s%s" % ( lhs, rhs, tail )   )   
    return ret 

def epr(arg, g, l, **kwa):
    """
    Slightly higher level variant of expression printing that invokes eprint 
    with lprefix and expr obtained by parsing the *arg* to find the symbol 
    and expression on either side of equal sign. 

    :param arg: 
    :param g: globals() from calling scope
    :param l: locals() from calling scope
    :param kwa: named args passed to eprint 
    :return ret: evaluated result 
    """
    p = arg.find("=")  
    if p > -1: 
        var_eq = arg[:p+1]  # chars before the first "=" in arg
        expr = arg[p+1:]    # chars following the first "=" in arg 
        lprefix = var_eq
    else:
        lprefix, expr = "", arg 
    pass
    return eprint(expr, g, l,  lprefix=lprefix,  **kwa)


