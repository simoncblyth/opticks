opticks-t-CSG-GEOM-interference
==================================

Some CSG test is writing into GEOM folder unexpectedly, 
causing non-repeatable test fails.
So make the GEOM directory readonly and look for permission error.::

   gxt
   ./G4CXTest_raindrop.sh    # create the geometry
   chmod -R ugo-w $HOME/.opticks/GEOM/RaindropRockAirWater/   # make readonly 
   c
   om-test


::

    The following tests FAILED:
         25 - CSGTest.CSGMakerTest (Child aborted)
         28 - CSGTest.CSGSimtraceRerunTest (INTERRUPT)
         29 - CSGTest.CSGSimtraceSampleTest (INTERRUPT)
         30 - CSGTest.CSGCopyTest (Child aborted)
         36 - CSGTest.CSGIntersectComparisonTest (INTERRUPT)

    The following tests FAILED:
         28 - CSGTest.CSGSimtraceRerunTest (INTERRUPT)
         29 - CSGTest.CSGSimtraceSampleTest (INTERRUPT)
         30 - CSGTest.CSGCopyTest (Child aborted)
         36 - CSGTest.CSGIntersectComparisonTest (INTERRUPT)

    The following tests FAILED:
         29 - CSGTest.CSGSimtraceSampleTest (INTERRUPT)
         30 - CSGTest.CSGCopyTest (Child aborted)
         36 - CSGTest.CSGIntersectComparisonTest (INTERRUPT)
    Errors while running CTest

    The following tests FAILED:
         30 - CSGTest.CSGCopyTest (Child aborted)
         36 - CSGTest.CSGIntersectComparisonTest (INTERRUPT)


And then there was one
-------------------------

::

    30/42 Test #30: CSGTest.CSGCopyTest ...................................Child aborted***Exception:   0.71 sec
    2023-11-03 22:06:48.074 INFO  [4141409] [main@13]  mode [K]
    2023-11-03 22:06:48.747 INFO  [4141409] [main@21]  ELV         t   3 : 111
    CSGFoundry::descELV elv.num_bits 3 num_include 3 num_exclude 0
    INCLUDE:3

    p:  0:midx:  0:mn:drop_solid
    p:  1:midx:  1:mn:medium_solid
    p:  2:midx:  2:mn:container_solid
    EXCLUDE:0


    2023-11-03 22:06:48.752 INFO  [4141409] [CSGFoundry::CompareVec@489] plan size_match FAIL 4 vs 0
    2023-11-03 22:06:48.753 INFO  [4141409] [CSGFoundry::CompareVec@494] inst sizeof(T) 64 data_match FAIL 
    2023-11-03 22:06:48.753 INFO  [4141409] [CSGFoundry::CompareVec@498] inst sizeof(T) 64 byte_match FAIL 
    2023-11-03 22:06:48.753 FATAL [4141409] [CSGFoundry::CompareVec@501]  mismatch FAIL for inst
     mismatch FAIL for inst
    2023-11-03 22:06:48.753 FATAL [4141409] [CSGFoundry::Compare@442]  mismatch FAIL 
    2023-11-03 22:06:48.753 INFO  [4141409] [main@27]  src 0x7fd51fc14670 dst 0x7fd51fc184a0 cf 3
    Assertion failed: (cf == 0), function main, file /Users/blyth/opticks/CSG/tests/CSGCopyTest.cc, line 35.


::

    In [6]: ai = a.inst.view(np.int32)
    In [7]: bi = b.inst.view(np.int32)

    In [8]: ai
    Out[8]:
    array([[[1065353216,          0,          0,          0],
            [         0, 1065353216,          0,          0],
            [         0,          0, 1065353216,          0],
            [         0,          0,          0,         -1]]], dtype=int32)

    In [9]: bi
    Out[9]:
    array([[[1065353216,          0,          0,          0],
            [         0, 1065353216,          0,          0],
            [         0,          0, 1065353216,          1],
            [         0,          0,          0,         -1]]], dtype=int32)

    In [10]: np.where( ai != bi )
    Out[10]: (array([0]), array([2]), array([3]))

::

    383     QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier_1, int sensor_index )
    384     {
    385         assert( sensor_identifier_1 >= 0 );
    386 
    387         q0.i.w = ins_idx ;             // formerly unsigned and "+ 1"
    388         q1.i.w = gas_idx ;
    389         q2.i.w = sensor_identifier_1 ;   // now +1 with 0 meaning not-a-sensor 
    390         q3.i.w = sensor_index ;
    391     }

Discrepancy in inst "sensor_identifier_1" 




