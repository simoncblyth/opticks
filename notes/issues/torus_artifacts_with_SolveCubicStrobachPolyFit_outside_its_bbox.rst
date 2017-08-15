torus Mandel-artifacts with SolveCubicStrobachPolyFit outside its bbox
==========================================================================


Observations with tboolean-torus
-----------------------------------

* ~/opticks_refs/torus_artifact_outside_bbox_with_SolveCubicStrobachPolyFit.png

Artifacts resembling Mandlebrot set manifest above and below 
the torus (outside its bbox).

* not surprised by the Mandelbrot
* very surprised that the artifacts are outside the bbox : how can that happen ?
* potential for a very expensive bug here 

* switching from the usual container + obj to just the obj
  in tboolean-torus makes the artifacts stay within obj bbox : why ?

* suggests I'm misunderstanding bbox mechanics, I thought that 
  intersect routines only get called if the bbox yields a hit ?
  


HUH : Artifacts gone by moving to initial gamma using the SolveCubicPolyDiv method
---------------------------------------------------------------------------------------


* GPU double precision trig appears to be a black art 



TODO
------

* review geometry setup in OGeo.cc



manual
--------

Ray traversal invokes an intersection program when the current ray encounters
one of a Geometry object’s primitives. It is the responsibility of an
intersection program to compute whether the ray intersects with the primitive,
and to report the parametric t-value of the intersection. Additionally, the
intersection program is responsible for computing and reporting any details of
the intersection, such as surface normal vectors, through attribute variables.
Once the intersection program has determined the t-value of a ray-primitive
intersection, it must report the result by calling a pair of OptiX functions,
rtPotentialIntersection and rtReportIntersection:


Investigate
--------------

::

    tboolean-torus(){ TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
    tboolean-torus-(){  $FUNCNAME- | python $* ; } 
    tboolean-torus--(){ cat << EOP 

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="$TMP/$FUNCNAME")

    CSG.boundary = args.testobject

    CSG.kwa = dict(poly="IM", resolution="50")

    container = CSG("box", param=[0,0,0,400], boundary=args.container, poly="MC", nx="20" )
      
    a = CSG.MakeTorus(R=100, r=50)

    #a = CSG.MakeTorus(R=1, r=0.5)
    #a.scale = [100,100,100]

    #CSG.Serialize([container, a], args.csgpath )
    CSG.Serialize([a], args.csgpath )


Is this a bounds bug ?
-------------------------

::

    115 RT_PROGRAM void bounds (int primIdx, float result[6])
    116 {
    117     //if(primIdx == 0) transform_test();
    118     //if(primIdx == 0) solve_callable_test();
    119 
    120     if(primIdx == 0)
    121     {
    122         unsigned partBuffer_size = partBuffer.size() ;
    123         unsigned planBuffer_size = planBuffer.size() ;
    124         unsigned tranBuffer_size = tranBuffer.size() ;
    125 
    126         rtPrintf("// intersect_analytic.cu:bounds pts:%4d pln:%4d trs:%4d \n", partBuffer_size, planBuffer_size, tranBuffer_size );
    127     }
    129 
    130     optix::Aabb* aabb = (optix::Aabb*)result;
    131     *aabb = optix::Aabb();
    132 
    133     uint4 identity = identityBuffer[instance_index] ;  // instance_index from OGeo is 0 for non-instanced
    134 
    135     const Prim prim    = primBuffer[primIdx];
    136     unsigned primFlag    = prim.primFlag() ;
    137 
    138     if(primFlag == CSG_FLAGNODETREE || primFlag == CSG_FLAGINVISIBLE )
    139     {
    140         csg_bounds_prim(primIdx, prim, aabb);
    141     }
    ...
    167     else
    168     {
    169         rtPrintf("## intersect_analytic.cu:bounds ABORT BAD primflag %d \n", primFlag );
    170         return ;
    171     }
    172     rtPrintf("// intersect_analytic.cu:bounds primIdx %d primFlag %d min %10.4f %10.4f %10.4f max %10.4f %10.4f %10.4f \n", primIdx, primFlag,
    173         result[0],
    174         result[1],
    175         result[2],
    176         result[3],
    177         result[4],
    178         result[5]
    179         );
    180 
    181 }



Torus + container box::

    // intersect_analytic.cu:bounds pts:   2 pln:   0 trs:   6 
    //csg_bounds_prim primIdx   0 partOffset   0 numParts   1 height  0 numNodes  1 tranBuffer_size   6 
    //csg_bounds_prim primIdx   1 partOffset   1 numParts   1 height  0 numNodes  1 tranBuffer_size   6 
    //csg_bounds_prim primIdx   0 nodeIdx  1 depth  0 elev  0 typecode 23 tranOffset  0 gtransformIdx  1 complement 0 
    //csg_bounds_prim primIdx   1 nodeIdx  1 depth  0 elev  0 typecode  6 tranOffset  1 gtransformIdx  1 complement 0 

       1.000    0.000    0.000    0.000   (trIdx:  0)[vt]
       0.000    1.000    0.000    0.000

       1.000    0.000    0.000    0.000   (trIdx:  3)[vt]
       0.000    1.000    0.000    0.000

       0.000    0.000    1.000    0.000   (trIdx:  0)[vt]
       0.000    0.000    0.000    1.000

       0.000    0.000    1.000    0.000   (trIdx:  3)[vt]
       0.000    0.000    0.000    1.000
    // csg_bounds_torus rmajor 100.000000 rminor 50.000000 rsum 150.000000  tr 1  
    // intersect_analytic.cu:bounds primIdx 0 primFlag 101 min  -150.0000  -150.0000   -50.0000 max   150.0000   150.0000    50.0000 
    // intersect_analytic.cu:bounds primIdx 1 primFlag 101 min  -400.0000  -400.0000  -400.0000 max   400.0000   400.0000   400.0000 


With only the torus::

    // intersect_analytic.cu:bounds pts:   1 pln:   0 trs:   3 
    //csg_bounds_prim primIdx   0 partOffset   0 numParts   1 height  0 numNodes  1 tranBuffer_size   3 
    //csg_bounds_prim primIdx   0 nodeIdx  1 depth  0 elev  0 typecode 23 tranOffset  0 gtransformIdx  1 complement 0 

       1.000    0.000    0.000    0.000   (trIdx:  0)[vt]
       0.000    1.000    0.000    0.000

       0.000    0.000    1.000    0.000   (trIdx:  0)[vt]
       0.000    0.000    0.000    1.000
    // csg_bounds_torus rmajor 100.000000 rminor 50.000000 rsum 150.000000  tr 1  
    // intersect_analytic.cu:bounds primIdx 0 primFlag 101 min  -150.0000  -150.0000   -50.0000 max   150.0000   150.0000    50.0000 


