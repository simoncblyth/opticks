csg_bounds_gpu_bbox_not_csg_aware_causing_overlarge
=====================================================

ISSUE : GPU side bbox not CSG difference/intersection aware 
---------------------------------------------------------------

* getting overlarge union bbox with difference tree (presumably same problem with intersect)
* CPU side bounds have some rudimentary CSG awareness (done in nnode)
  makeing GPU bounds are greater than the CPU ones ...


Approach
------------

* port the CPU side tree bounds calc GPU side : or copy from CPU ?

TODO
-----

* review the nnode CSG bbox calc
* investigate conseqences of container impingement
* vague recollection of some similar issue with torus, find notes on that


Observations
----------------

For tboolean-cyd the overlarge GPU bounds 
cause no problems it seems, presumably because 
no impingment with container.


tboolean-cyd example
-------------------------

::

    tboolean-;tboolean-cyd
    ...
    // intersect_analytic.cu:bounds buffer sizes pts:   4 pln:   0 trs:   6 
    // csg_bounds_prim CSG_FLAGNODETREE  primIdx   0 partOffset   0  numParts   3 -> height  1 -> numNodes  3  tranBuffer_size   6 
    // csg_bounds_prim CSG_FLAGNODETREE  primIdx   1 partOffset   3  numParts   1 -> height  0 -> numNodes  1  tranBuffer_size   6 
    // csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius 200.000 z1 -100.000 z2 100.000 
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -100.0000  -100.0000     0.0000 )  tbb.max (   100.0000   100.0000   200.0000 ) 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 0  bnd0:123 typ0:  3  min  -200.0000  -200.0000  -100.0000 max   200.0000   200.0000   200.0000 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 1  bnd0:124 typ0:  6  min -1000.0000 -1000.0000 -1000.0000 max  1000.0000  1000.0000  1000.0000 

::

    tboolean-cyd(){ TESTCONFIG=$($FUNCNAME-) TORCHCONFIG=$($FUNCNAME-torch-) tboolean-- $* ; }
    tboolean-cyd-(){  $FUNCNAME- | python $* ; }  
    tboolean-cyd--(){ cat << EOP 
    import numpy as np
    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="$TMP/$FUNCNAME")

    CSG.boundary = args.testobject
    CSG.kwa = dict(verbosity="1", poly="IM", resolution="4" )

    container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution="4", verbosity="0" )

    ra = 200 
    z1 = -100
    z2 = 100

    a = CSG("cylinder", param=[0,0,0,ra], param1=[z1,z2,0,0] )
    b = CSG("sphere", param=[0,0,z2,ra/2]  )

    obj = a - b 

    CSG.Serialize([container, obj], args.csgpath )

    """  

    Expected bbox  (-200,-200,-100)  (200,200,100)

                    

                     _
                   .   .    
                 .       .
             +---.---+---.---+ (200,100) 
             |   .       .   |
             |     .   .     |   
         ----|-------^-------|---- X
             |               |   
             |               |   
             +---------------+

    """

    EOP
    }



