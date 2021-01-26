G4OKTest_fail_from_nhit_nhiy_mismatch
======================================


::

    2021-01-25 23:31:28.398 INFO  [4947927] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2021-01-25 23:31:28.414 INFO  [4947927] [OEvent::downloadHits@443]  nhit 17 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2021-01-25 23:31:28.417 INFO  [4947927] [OEvent::downloadHiys@476]  nhiy 47 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    Assertion failed: (nhit == nhiy), function download, file /Users/blyth/opticks/optixrap/OEvent.cc, line 511.
    Abort trap: 6
    epsilon:opticks blyth$ 




