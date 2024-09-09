new_two_CSGTest_fails_on_P_but_not_A
=====================================

FIXED WITH : CSGSolid::CopyIntent 
---------------------------------------- 


From P
-------

::

    FAILS:  2   / 214   :  Fri Sep  6 17:24:10 2024   
      13 /43  Test #13 : CSGTest.CSGFoundry_IntersectPrimTest          ***Failed                      0.25   
      32 /43  Test #32 : CSGTest.CSGCopyTest                           ***Failed                      4.43   



::

    13/43 Test #13: CSGTest.CSGFoundry_IntersectPrimTest ..................***Failed    0.26 sec
    /data/blyth/opticks_Debug/bin/CSGTestRunner.sh : FOUND B_CFBaseFromGEOM /home/blyth/.opticks/GEOM/J_2024aug27 containing CSGFoundry/prim.npy
                    HOME : /home/blyth
                     PWD : /data/blyth/opticks_Debug/build/CSG/tests
                    GEOM : J_2024aug27
             BASH_SOURCE : /data/blyth/opticks_Debug/bin/CSGTestRunner.sh
              EXECUTABLE : CSGFoundry_IntersectPrimTest
                    ARGS : 
    FATAL : cannot make J_2024aug27
    CSGFoundry_IntersectPrimTest: /home/blyth/opticks/CSG/tests/CSGFoundry_IntersectPrimTest.cc:52: void CSGFoundry_IntersectPrimTest::init(): Assertion `can_make' failed.
    /data/blyth/opticks_Debug/bin/CSGTestRunner.sh: line 55: 353460 Aborted                 (core dumped) $EXECUTABLE $@
    /data/blyth/opticks_Debug/bin/CSGTestRunner.sh : FAIL from CSGFoundry_IntersectPrimTest




::

    32/43 Test #32: CSGTest.CSGCopyTest ...................................***Failed    4.46 sec
    /data/blyth/opticks_Debug/bin/CSGTestRunner.sh : FOUND B_CFBaseFromGEOM /home/blyth/.opticks/GEOM/J_2024aug27 containing CSGFoundry/prim.npy
                    HOME : /home/blyth
                     PWD : /data/blyth/opticks_Debug/build/CSG/tests
                    GEOM : J_2024aug27
             BASH_SOURCE : /data/blyth/opticks_Debug/bin/CSGTestRunner.sh
              EXECUTABLE : CSGCopyTest
                    ARGS : 
    2024-09-09 10:05:04.656 INFO  [354404] [main@16]  mode [K]
    2024-09-09 10:05:06.146 INFO  [354404] [main@28] env->desc()
     ELV         t 302 : 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
    src->descELV(elv)
    CSGFoundry::descELV elv.num_bits 302 num_include 302 num_exclude 0 is_all_set 1
    INCLUDE:302

    p:  0:midx:  0:mn:sTopRock_domeAir
    p:  1:midx: -1:mn:sTopRock_dome
    p:  2:midx:  2:mn:sDomeRockBox
    p:  3:midx:  3:mn:PoolCoversub
    p:  4:midx:  4:mn:Upper_LS_tube
    p:  5:midx:  5:mn:Upper_Steel_tube
    p:  6:midx:  6:mn:Upper_Tyvek_tube
    p:  7:midx:  7:mn:Upper_Chimney
    p:  8:midx:  8:mn:sBar_0
    p:  9:midx:  9:mn:sBar_1
    p: 10:midx: 10:mn:sPanelTape
    p: 11:midx: -1:mn:sPanel
    p: 12:midx: 12:mn:sPlane
    p: 13:midx: 13:mn:sWall
    p: 14:midx: 14:mn:sAirTT
    p: 15:midx: 15:mn:sExpHall
    p: 16:midx: 16:mn:sExpRockBox
    p: 17:midx: -1:mn:sTopRock
    ...
    p:290:midx:290:mn:svacSurftube_38V1_1
    p:291:midx:291:mn:sSurftube_38V1_1
    p:292:midx:292:mn:sInnerWater
    p:293:midx:293:mn:sReflectorInCD
    p:294:midx: -1:mn:mask_PMT_20inch_vetosMask
    p:295:midx:295:mn:PMT_20inch_veto_inner_solid_1_2
    p:296:midx:296:mn:PMT_20inch_veto_pmt_solid_1_2
    p:297:midx:297:mn:mask_PMT_20inch_vetosMask_virtual
    p:298:midx:298:mn:sAirGap
    p:299:midx:299:mn:sPoolLining
    p:300:midx:300:mn:sBottomRock
    p:301:midx:301:mn:sWorld
    EXCLUDE:0



    2024-09-09 10:05:06.310 INFO  [354404] [CSGFoundry::CompareVec@652] solid sizeof(T) 48 data_match FAIL 
    2024-09-09 10:05:06.310 INFO  [354404] [CSGFoundry::CompareVec@656] solid sizeof(T) 48 byte_match FAIL 
    2024-09-09 10:05:06.310 FATAL [354404] [CSGFoundry::CompareVec@659]  mismatch FAIL for solid
     mismatch FAIL for solid a.size 11 b.size 11
    2024-09-09 10:05:06.326 FATAL [354404] [CSGFoundry::Compare@565]  mismatch FAIL 
    2024-09-09 10:05:06.326 INFO  [354404] [main@46]  src 0x10c1e000 dst 0x10c1f4c0 cf 2
    2024-09-09 10:05:06.326 FATAL [354404] [main@54]  UNEXPECTED DIFFERENCE  DEBUG WITH :
     ~/opticks/CSG/tests/CSGCopyTest.sh ana 
    CSGCopyTest: /home/blyth/opticks/CSG/tests/CSGCopyTest.cc:61: int main(int, char**): Assertion `cf == 0' failed.
    /data/blyth/opticks_Debug/bin/CSGTestRunner.sh: line 55: 354404 Aborted                 (core dumped) $EXECUTABLE $@
    /data/blyth/opticks_Debug/bin/CSGTestRunner.sh : FAIL from CSGCopyTest








::

    In [3]: w = np.where( a.solid != b.solid ) ; w 
    Out[3]: 
    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
     array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
     array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]))

    In [5]: a.solid[w]
    Out[5]: array([82, 70, 70, 70, 70, 70, 70, 70, 70, 70, 84], dtype=int32)

    In [6]: b.solid[w]
    Out[6]: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)



    n [7]: a.solid[0]
    Out[7]: 
    array([[     12370,          0,          0,          0],
           [      2896,          0,          0,         82],
           [         0,          0,          0, 1198153728]], dtype=int32)

    In [8]: b.solid[0]
    Out[8]: 
    array([[     12370,          0,          0,          0],
           [      2896,          0,          0,          0],
           [         0,          0,          0, 1198153728]], dtype=int32)



