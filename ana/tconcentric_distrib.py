#!/usr/bin/env python
"""
tconcentric_distrib.py 
=============================================

TODO:

* find way to decouple plotting from distrib chi2 




IPython/ipdb debugging::

    delta:ana blyth$ ip
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks
    /opt/local/bin/ipython --profile=g4opticks

    In [1]: run -d tconcentric_distrib.py 
    *** Blank or comment
    *** Blank or comment
    *** Blank or comment
    NOTE: Enter 'c' at the ipdb>  prompt to continue execution.
    > /Users/blyth/opticks/ana/tconcentric_distrib.py(32)<module>()
         31 
    ---> 32 
         33 import os, sys, logging, numpy as np

    ipdb> c

    ...

    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/_methods.pyc in _amin(a, axis, out, keepdims)
         27 
         28 def _amin(a, axis=None, out=None, keepdims=False):
    ---> 29     return umr_minimum(a, axis, None, out, keepdims)
         30 
         31 def _sum(a, axis=None, dtype=None, out=None, keepdims=False):

    ValueError: zero-size array to reduction operation minimum which has no identity

    In [2]: %debug   ## AT THE ERROR ENTER %debug 

    > /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/_methods.py(29)_amin()
         28 def _amin(a, axis=None, out=None, keepdims=False):
    ---> 29     return umr_minimum(a, axis, None, out, keepdims)
         30 

    ipdb> bt       ## backtrace, up/down to navigate stack

    ...
 
    > /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/_methods.py(29)_amin()
         27 
         28 def _amin(a, axis=None, out=None, keepdims=False):
    ---> 29     return umr_minimum(a, axis, None, out, keepdims)
         30 
         31 def _sum(a, axis=None, dtype=None, out=None, keepdims=False):

    ipdb> p a
    array([], dtype=float64)

    ipdb> up
    > /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/fromnumeric.py(2224)amin()
       2223         return _methods._amin(a, axis=axis,
    -> 2224                             out=out, keepdims=keepdims)
       2225 

    ipdb> up
    > /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/axes.py(8506)hist()
       8505                 for m in n:
    -> 8506                     ymin = np.amin(m[m != 0]) # filter out the 0 height bins
       8507                 ymin = max(ymin*0.9, minimum)

    ipdb> p m
    array([ 0.])

    ipdb> p n
    [array([ 0.])]

    ipdb> p m[m != 0]
    array([], dtype=float64)
    ipdb> 

    ipdb> l    # BIGGER WINDOW 

       8501                 self.dataLim.intervalx = (xmin, xmax)
       8502             elif orientation == 'vertical':
       8503                 ymin0 = max(_saved_bounds[1]*0.9, minimum)
       8504                 ymax = self.dataLim.intervaly[1]
       8505                 for m in n:
    -> 8506                     ymin = np.amin(m[m != 0]) # filter out the 0 height bins
       8507                 ymin = max(ymin*0.9, minimum)
       8508                 ymin = min(ymin0, ymin)
       8509                 self.dataLim.intervaly = (ymin, ymax)
       8510 
       8511         if label is None:

    ipdb> a     # ARGUMENTS

    self = Axes(0.731522,0.354545;0.168478x0.545455)
    x = [ 20.6333  20.6333  20.6776  20.6776  20.6333]
    bins = [ 20.6333  20.6776]
    range = <built-in function range>
    normed = False
    weights = None
    cumulative = False
    bottom = None
    histtype = step
    align = mid
    orientation = vertical
    rwidth = None
    log = False
    color = ['g']
    label = G4 : t[3]
    stacked = False
    kwargs = {}

    ipdb> p __file__
    '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/axes.pyc'




"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 18,10.2   # plt.gcf().get_size_inches()   after maximize
    import matplotlib.gridspec as gridspec
except ImportError:
    print "matplotlib missing : you need this to make plots"
    plt = None 

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt
from opticks.ana.cf import CF 
from opticks.ana.cfplot import cfplot, qwns_plot, qwn_plot, multiplot


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    args = opticks_main(tag="1", src="torch", det="concentric")
    log.info(" args %s " % repr(args))
    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))

    plt.ion()
    plt.close()

    #spawn = slice(0,10)  # pluck top line of seqhis table, needed for multiplot
    spawn = slice(8,9)  # pluck top line of seqhis table, needed for multiplot

    try:
        cf = CF(args, spawn=spawn, seqs=[] )
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    cf.dump()


    st = multiplot(cf, pages=["XYZT","ABCR"])
    #st = multiplot(cf, pages=["XYZT"])
 
    #log_ = False
    #c2_cut = 0 
 
    #scf = cf.ss[0]
    #nrec = scf.nrec()
    #nrec = 1 
    #for irec in range(nrec):
    #    key = scf.suptitle(irec)
    #    page = "XYZT"
    #    qd = qwns_plot( scf, page, irec, log_, c2_cut)
    #    print "qd", qd
    #

    #qwns_plot( cf.ss[0], "XYZT", 0 ) 


    #irec = 6
    #qwn_plot( cf.ss[0], "T", -1, c2_ymax=2000)
    #qwn_plot( cf, "R", irec)
    #qwns_plot( cf, "XYZT", irec)
    #qwns_plot( cf, "ABCR", irec)

    #binsx,ax,bx,lx = cf.rqwn("X",irec)
    #binsy,ay,by,ly = cf.rqwn("Y",irec)
    #binsz,az,bz,lz = cf.rqwn("Z",irec)

   






 
