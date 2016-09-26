OKG4 vizg4 noshow
=======================

Issue
--------

Attempting to load and visualize a G4 event created by OKG4Test
completes, but no propagation visible and no index in GUI

::

    OKG4Test --save
    OKG4Test --load --vizg4


tpmt.py 
--------

Compare with tpmt::

    tpmt-- --okg4

    tpmt-- --okg4 --load    ## succeeds to viz opticks propagation

    tpmt-- --okg4 --load --vizg4 --debugger   ## succeds to viz the g4 propagation, but index missing


tokg4.py
----------

::

    ipython $(which tokg4.py) -i 


Looks like compression domain issue::

    In [14]: b.rx
    Out[14]: 
    A(torch,-1,dayabay)-
    A([[[[-32768, -32768,  30278,     16],
             [ 32766,   5503,      1,   3328]],

            [[-32768, -32768, -32768, -32768],
             [ 27901,   5503,      1,   1536]],



    In [12]: b.rpost_(0)
    Out[12]: 
    A()sliced
    A([[ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           ..., 
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977],
           [ -24230.8603, -809820.8603,      -0.0785,       0.0977]])

    In [13]: b.rpost_(1)
    Out[13]: 
    A()sliced
    A([[ -24230.8603, -809820.8603,  -14835.8603,    -200.0061],
           [ -24230.8603, -809820.8603,  -14835.8603,    -200.0061],
           [ -24230.8603, -809820.8603,  -14835.8603,    -200.0061],
           ..., 
           [ -24230.8603, -809820.8603,  -14835.8603,    -200.0061],
           [ -24230.8603, -809820.8603,  -14835.8603,    -200.0061],
           [ -24230.8603, -809820.8603,  -14835.8603,    -200.0061]])



::

    import matplotlib.pyplot as plt
    plt.plot(a.ht[:,0,0],a.ht[:,0,1])   ## check hits, pattern of PMT positions apparent
    plt.show()

    plt.plot(b.ht[:,0,0],b.ht[:,0,1])   ## all over the place


