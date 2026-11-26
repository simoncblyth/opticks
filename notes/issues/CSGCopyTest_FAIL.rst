CSGCopyTest_FAIL prim mismatch
===============================


::

    [lo] A[blyth@localhost tests]$ ./CSGCopyTest.sh 
             BASH_SOURCE : ./CSGCopyTest.sh
                    GEOM : J26_4_1_opticks_Release_g411
                    name : CSGCopyTest
                     bin : CSGCopyTest
                  script : CSGCopyTest.py
                     arg : info_run_ana
                    BASE : /tmp/blyth/opticks/CSGCopyTest
                   AFOLD : /tmp/blyth/opticks/CSGCopyTest/src
                   BFOLD : /tmp/blyth/opticks/CSGCopyTest/dst
    ./CSGTestRunner.sh - using external config for GEOM J26_4_1_opticks_Release_g411 J26_4_1_opticks_Release_g411_CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /home/blyth/opticks/CSG/tests
                    GEOM : J26_4_1_opticks_Release_g411
             BASH_SOURCE : ./CSGTestRunner.sh
              EXECUTABLE : CSGCopyTest



    2026-11-26 18:03:39.612 INFO  [2296909] [CSGCopy::copySolidInstances@421]  sNumInst 49186
    2026-11-26 18:03:39.614 INFO  [2296909] [CSGCopy::Select@55] 
    src:CSGFoundry  num_total 13 num_solid 13 num_prim 3650 num_node 29173 num_plan 0 num_tran 9587 num_itra 9587 num_inst 49186 gas 0 meshname 472 mmlabel 13 mtime 1795575419 mtimestamp 20261125_105659 sim Y
    dst:CSGFoundry  num_total 13 num_solid 13 num_prim 3650 num_node 29173 num_plan 0 num_tran 9587 num_itra 9587 num_inst 49186 gas 0 meshname 472 mmlabel 13 mtime 1795575419 mtimestamp 20261125_105659 sim Y

    2026-11-26 18:03:39.614 INFO  [2296909] [CSGFoundryCompare::CompareVec@120] prim sizeof(T) 64 data_match FAIL 
    2026-11-26 18:03:39.615 INFO  [2296909] [CSGFoundryCompare::CompareVec@124] prim sizeof(T) 64 byte_match FAIL 
    2026-11-26 18:03:39.615 FATAL [2296909] [CSGFoundryCompare::CompareVec@127]  mismatch FAIL for prim
     mismatch FAIL for prim a.size 3650 b.size 3650
    2026-11-26 18:03:39.620 FATAL [2296909] [CSGFoundryCompare::Compare@31]  mismatch FAIL 
    2026-11-26 18:03:39.620 INFO  [2296909] [CSGFoundry::save_@2618] /tmp/blyth/opticks/CSGCopyTest/src/CSGFoundry
    2026-11-26 18:03:39.748 INFO  [2296909] [CSGFoundry::save_@2642]  SSim::save /tmp/blyth/opticks/CSGCopyTest/src/CSGFoundry
    [stree::save_desc_ fold [/tmp/blyth/opticks/CSGCopyTest/src/CSGFoundry/SSim/stree/desc]
    -stree::save_desc_ name [NRT.txt]
    -stree::save_desc_ name [_csg.txt]
    -stree::save_desc_ name [bd.txt]
    -stree::save_desc_ name [factor.txt]




recently added boundary not copied ? or stomped ?
--------------------------------------------------

::


    [lo] A[blyth@localhost tests]$ ./CSGCopyTest.sh pdb
    ...

    In [3]: np.where( a.prim != b.prim )
    Out[3]: 
    (array([   0,    1,    2,    3,    4, ..., 3645, 3646, 3647, 3648, 3649], shape=(3650,)),
     array([3, 3, 3, 3, 3, ..., 3, 3, 3, 3, 3], shape=(3650,)),
     array([2, 2, 2, 2, 2, ..., 2, 2, 2, 2, 2], shape=(3650,)))

    In [4]: a.prim[:,3,2]
    Out[4]: array([0., 0., 0., 0., 0., ..., 0., 0., 0., 0., 0.], shape=(3650,), dtype=float32)

    In [5]: a.prim[:,3,2].view(np.uint32)
    Out[5]: array([  0,   1,   2,   1,   3, ..., 439, 440, 441, 442, 443], shape=(3650,), dtype=uint32)

    In [6]: b.prim[:,3,2].view(np.uint32)
    Out[6]: array([4294967295, 4294967295, 4294967295, 4294967295, 4294967295, ..., 4294967295, 4294967295, 4294967295, 4294967295, 4294967295], shape=(3650,), dtype=uint32)

    In [7]: b.prim[:,3,2].view(np.int32)
    Out[7]: array([-1, -1, -1, -1, -1, ..., -1, -1, -1, -1, -1], shape=(3650,), dtype=int32)




Thats due to omission in CSGCopy
-----------------------------------


::

    [lo] A[blyth@localhost opticks]$ git diff -b CSG/CSGCopy.cc
    diff --git a/CSG/CSGCopy.cc b/CSG/CSGCopy.cc
    index 555c2a1fc..7902c335a 100644
    --- a/CSG/CSGCopy.cc
    +++ b/CSG/CSGCopy.cc
    @@ -232,6 +232,8 @@ void CSGCopy::copySolidPrim(AABB& solid_bb, int dPrimOffset, const CSGSolid* sso
              const CSGPrim* spr = src->getPrim(primIdx);
              unsigned meshIdx = spr->meshIdx() ;
              unsigned repeatIdx = spr->repeatIdx() ;
    +         unsigned boundary = spr->boundary();
    +
              bool selected = elv == nullptr ? true : elv->is_set(meshIdx) ;
              if( selected == false ) continue ;
     
    @@ -245,6 +247,7 @@ void CSGCopy::copySolidPrim(AABB& solid_bb, int dPrimOffset, const CSGSolid* sso
              dpr->setMeshIdx(meshIdx);
              dpr->setRepeatIdx(repeatIdx);
              dpr->setPrimIdx(dPrimIdx_local);
    +         dpr->setBoundary(boundary);
     
     #ifdef WITH_S_BB
              s_bb prim_bb = {} ;



