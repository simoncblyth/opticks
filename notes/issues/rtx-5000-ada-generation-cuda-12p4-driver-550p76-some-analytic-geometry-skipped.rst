rtx-5000-ada-generation-cuda-12p4-driver-550p76-some-analytic-geometry-skipped
=================================================================================


Follow on from ~/j/issues/opticks-from-scratch-in-blyth-account-on-rtx-5000-ada-generation-machine.rst

Comparing:

+-----------+-------------+----------------+
|           |   A         |    B           |
+===========+=============+================+   
|   CUDA    |   11.7      |  12.4          |
+-----------+-------------+----------------+
|  Driver   |  515.43.04  | 550.76         | 
+-----------+-------------+----------------+
|  Optix    |   7.5       |   7.5          |
+-----------+-------------+----------------+     
|  c.c.     |   70        |   89           |
+-----------+-------------+----------------+     
| GPU       | TITAN RTX   | RTX 5000 Ada   |
+-----------+-------------+----------------+     


Boolean issue ? 
----------------

* simple prim behave as expected in A and B 
* analytic booleans in B have problems 


Create simple GEOM for boolean check
---------------------------------------

1. check U4VolumeMaker U4SolidMaker for valid GEOM strings
2. GEOM func to set the GEOM
3. gxt ; ./G4CX_U4TreeCreateCSGFoundryTest.sh
4. ~/o/cxr_min.sh 


JustOrbOrbUnionSimple 
-----------------------

A and B look OK from inside and out but rendered sizes are different

* TODO: arrange equal screen resolution to check this


JustOrbOrbIntersectionSimple
-----------------------------

A looks as expected, 

B the "lobes"  which should  not be visible, are visible 


JustOrbOrbDifferenceSimple
---------------------------

A looks as expected

B both surfaces that should not be visible, are visible : with a fuzzy edge on one

* clearest signature of issue yet 



TODO: find test that does 2D CPU side tracing, apply that to JustOrbOrbDifferenceSimple in A,B
------------------------------------------------------------------------------------------------









J_2024aug27 : sTarget
-----------------------

A: expected sphere with small chimney 

B: just chimney 

::

   MOI=sTarget ELV=sTarget ~/o/cxr_min.sh


J_2024aug27 : uni1
--------------------

::

   ELV=sWorld,uni1 MOI=sWorld ~/o/cxr_min.sh   # overview of all uni1

   ELV=sWorld,uni1 MOI=uni1:0:-2 ~/o/cxr_min.sh    # target one of them 



A: expected IonRing and columns

B: Bizarre unphysical ray trace render. See IonRing but not cylindrical columns. 
   However rotating around see that the columns are there 
   when view somehow thru the IonRing.  



J_2024aug27 : base_steel
---------------------------

::

    ELV=sWorld,base_steel MOI=sWorld ~/o/cxr_min.sh 
    ELV=sWorld,base_steel MOI=base_steel:0:-2 ~/o/cxr_min.sh 

    ELV=sWorld,base_steel MOI=base_steel:0:-2 ~/o/cxr_min.sh
    ELV=base_steel MOI=base_steel:0:-2 ~/o/cxr_min.sh


* B : shows "clipping" like uni1 


J_2024aug27 : sStrutBallhead : simple sphere looks same in A and B
----------------------------------------------------------------------

::

    ELV=sWorld,sStrutBallhead MOI=sWorld ~/o/cxr_min.sh


J_2024aug27 : sWaterTube : simple cylinder looks same in A and B 
-----------------------------------------------------------------

::

    ELV=sWaterTube MOI=sWaterTube:0:-1 ~/o/cxr_min.sh
    ELV=sWorld,sWaterTube MOI=sWaterTube:0:-1 ~/o/cxr_min.sh


    
J_2024aug27 : HamamatsuR12860sMask
------------------------------------

::

    ELV=sWorld,HamamatsuR12860sMask MOI=HamamatsuR12860sMask:0:-2 ~/o/cxr_min.sh


A : expected hemi-spherical soup bowls 

B : looks OK when viewed from the open face side, but when viewed from the base of the bowl 
    can see through 


J_2024aug27 : svacSurftube_0V1_0  : SIDE ISSUE WITH ELV SELECTION AND TRIANGULATION
-------------------------------------------------------------------------------------

* ELV selection not force triangulation aware ? 

::

    ELV=sWorld,svacSurftube_0V1_0 MOI=svacSurftube_0V1_0:0:-2 ~/o/cxr_min.sh

::

    [blyth@localhost ~]$ ELV=sWorld,svacSurftube_0V1_0 MOI=svacSurftube_0V1_0:0:-2 ~/o/cxr_min.sh
    /home/blyth/o/cxr_min.sh : FOUND B_CFBaseFromGEOM /home/blyth/.opticks/GEOM/J_2024aug27 containing CSGFoundry/prim.npy
                    GEOM : J_2024aug27 
                     MOI : svacSurftube_0V1_0:0:-2 
                    TMIN : 0.5 
                     EYE : 1,0,0 
                    LOOK : 0,0,0 
                      UP : 0,0,1 
                    ZOOM : 1 
                  LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderInteractiveTest 
                    BASE : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderInteractiveTest 
                    PBAS : /data/blyth/opticks/ 
              NAMEPREFIX : cxr_min__eye_1,0,0__zoom_1__tmin_0.5_ 
            OPTICKS_HASH : FAILED_GIT_REV_PARSE 
                 TOPLINE : ESCALE=extent EYE=1,0,0 TMIN=0.5 MOI=svacSurftube_0V1_0:0:-2 ZOOM=1 CAM=perspective ~/opticks/CSGOptiX/cxr_min.sh  
                 BOTLINE : Fri Aug 30 17:06:57 CST 2024 
    CUDA_VISIBLE_DEVICES : 1 
    /home/blyth/o/cxr_min.sh : run : delete prior LOG CSGOptiXRenderInteractiveTest.log
    2024-08-30 17:06:59.712 FATAL [265350] [CSGNode::setAABBLocal@473]  not implemented for tc 116 CSG::Name(tc) torus
    CSGOptiXRenderInteractiveTest: /home/blyth/opticks/CSG/CSGNode.cc:474: void CSGNode::setAABBLocal(): Assertion `0' failed.
    /home/blyth/o/cxr_min.sh: line 271: 265350 Aborted                 (core dumped) $bin
    /home/blyth/o/cxr_min.sh run error
    [blyth@localhost ~]$ 

    Program received signal SIGABRT, Aborted.
    0x00007ffff56b2387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff56b2387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff56b3a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff56ab1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff56ab252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff79ff4c0 in CSGNode::setAABBLocal (this=0x12660eb0) at /home/blyth/opticks/CSG/CSGNode.cc:474
    #5  0x00007ffff7a867d7 in CSGCopy::copyNode (this=0x7fffffff3080, prim_bb=..., nodeIdx=24197) at /home/blyth/opticks/CSG/CSGCopy.cc:351
    #6  0x00007ffff7a863a4 in CSGCopy::copyPrimNodes (this=0x7fffffff3080, prim_bb=..., spr=0x10c86800) at /home/blyth/opticks/CSG/CSGCopy.cc:280
    #7  0x00007ffff7a86023 in CSGCopy::copySolidPrim (this=0x7fffffff3080, solid_bb=..., dPrimOffset=1, sso=0x10a0a410) at /home/blyth/opticks/CSG/CSGCopy.cc:235
    #8  0x00007ffff7a85ba8 in CSGCopy::copy (this=0x7fffffff3080) at /home/blyth/opticks/CSG/CSGCopy.cc:162
    #9  0x00007ffff7a8521d in CSGCopy::Select (src=0xf208490, elv=0xf2d2330) at /home/blyth/opticks/CSG/CSGCopy.cc:54
    #10 0x00007ffff7a1a8f7 in CSGFoundry::CopySelect (src=0xf208490, elv=0xf2d2330) at /home/blyth/opticks/CSG/CSGFoundry.cc:3032
    #11 0x00007ffff7a1a476 in CSGFoundry::Load () at /home/blyth/opticks/CSG/CSGFoundry.cc:2995
    #12 0x000000000044538c in main (argc=1, argv=0x7fffffff4b48) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:54
    (gdb) 


    CSGFoundry::Load_[/home/blyth/.opticks/GEOM/J_2024aug27]
    2024-08-30 20:58:47.202 INFO  [262795] [main@66] standard CSGFoundry::Load has scene : no need to kludge OverrideScene 
    2024-08-30 20:58:47.605 FATAL [262795] [SBT::_getOffset@715]  UNEXPECTED trimesh with   UNEQUAL:  num_bi 5 numPrim 1 gas_idx 1 mmlabel 322:solidSJCLSanchor
    CSGOptiXRenderInteractiveTest: /home/blyth/opticks/CSGOptiX/SBT.cc:723: int SBT::_getOffset(unsigned int, unsigned int) const: Assertion `num_bi == numPrim' failed.
    /home/blyth/o/cxr_min.sh: line 275: 262795 Aborted                 (core dumped) $bin
    /home/blyth/o/cxr_min.sh run error





