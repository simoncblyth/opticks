opticks-t-CSG-fails
======================




V1J011 FAIL (no fail with RaindropRockAirWater)::

    98% tests passed, 1 tests failed out of 42

    Total Test time (real) =  32.54 sec

    The following tests FAILED:
         30 - CSGTest.CSGCopyTest (Subprocess aborted)


    2023-11-04 17:24:29.224 INFO  [131637] [CSGFoundry::CompareVec@494] prim sizeof(T) 64 data_match FAIL 
    2023-11-04 17:24:29.224 INFO  [131637] [CSGFoundry::CompareVec@498] prim sizeof(T) 64 byte_match FAIL 
    2023-11-04 17:24:29.224 FATAL [131637] [CSGFoundry::CompareVec@501]  mismatch FAIL for prim
     mismatch FAIL for prim a.size 3088 b.size 3088
    2023-11-04 17:24:29.224 INFO  [131637] [CSGFoundry::CompareVec@494] node sizeof(T) 64 data_match FAIL 
    2023-11-04 17:24:29.227 INFO  [131637] [CSGFoundry::CompareVec@498] node sizeof(T) 64 byte_match FAIL 
    2023-11-04 17:24:29.227 FATAL [131637] [CSGFoundry::CompareVec@501]  mismatch FAIL for node
     mismatch FAIL for node a.size 15968 b.size 15968
    2023-11-04 17:24:29.238 FATAL [131637] [CSGFoundry::Compare@442]  mismatch FAIL 
    2023-11-04 17:24:29.238 INFO  [131637] [main@33]  src 0xda9d40 dst 0xdaa780 cf 4
    2023-11-04 17:24:29.238 FATAL [131637] [main@41]  UNEXPECTED DIFFERENCE  DEBUG WITH :
     ~/opticks/CSG/tests/CSGCopyTest.sh ana 
    CSGCopyTest: /data/blyth/junotop/opticks/CSG/tests/CSGCopyTest.cc:48: int main(int, char**): Assertion `cf == 0' failed.


::

    CTESTARG="-R CSGCopyTest" om-test


::


    ~/opticks/CSG/tests/CSGCopyTest.sh
    ~/opticks/CSG/tests/CSGCopyTest.sh ana 

    In [3]: np.where(a.node != b.node)
    Out[3]: 
    (array([   27,    28,    29,    30,    31, ..., 15956, 15960, 15962, 15964, 15966]),
     array([2, 2, 2, 2, 2, ..., 2, 3, 2, 3, 2]),
     array([1, 1, 1, 1, 1, ..., 1, 0, 1, 0, 1]))

    In [4]: w = np.where(a.node != b.node)

    In [5]: a.node[w]
    Out[5]: array([-10141.8  , -10096.351, -10141.8  , -10141.8  , -10096.351, ...,    686.45 ,    765.55 ,    765.65 ,    818.35 ,    818.45 ], dtype=float32)

    In [6]: b.node[w]
    Out[6]: array([-10141.801, -10096.35 , -10141.801, -10141.801, -10096.35 , ...,    686.45 ,    765.55 ,    765.65 ,    818.35 ,    818.45 ], dtype=float32)

    In [7]: 


    In [11]: np.where( np.abs( a.node - b.node ) > 1e-3 )
    Out[11]: 
    (array([  222,   223,   229,   230,   236, ..., 15651, 15652, 15653, 15654, 15655]),
     array([3, 3, 3, 3, 3, ..., 2, 2, 2, 2, 2]),
     array([1, 1, 1, 1, 1, ..., 2, 2, 2, 2, 2]))



