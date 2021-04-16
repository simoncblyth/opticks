GParts_ordering_difference_on_different_machine
==================================================

LOOKS LIKE THE 1s ARE IN REVERSED ORDER ON O vs G : SOURCE OF THIS PROBLEM NEEDS TO BE FOUND + FIXED 


Issue
--------

::

    OpSnapTest --savegparts    
    # any Opticks executable can do this (necessary as GParts are now postcache so this does not belong in geocache)
    # the parts are saved into $TMP/GParts

    epsilon:ana blyth$ GParts.py 
    Solid 0 : /tmp/blyth/opticks/GParts/0 : primbuf (3084, 4) partbuf (17346, 4, 4) tranbuf (7917, 3, 4, 4) idxbuf (3084, 4) 
    Solid 1 : /tmp/blyth/opticks/GParts/1 : primbuf (5, 4) partbuf (7, 4, 4) tranbuf (5, 3, 4, 4) idxbuf (5, 4) 
    Solid 2 : /tmp/blyth/opticks/GParts/2 : primbuf (6, 4) partbuf (30, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 
    Solid 3 : /tmp/blyth/opticks/GParts/3 : primbuf (6, 4) partbuf (54, 4, 4) tranbuf (29, 3, 4, 4) idxbuf (6, 4) 
    Solid 4 : /tmp/blyth/opticks/GParts/4 : primbuf (6, 4) partbuf (28, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 

    Solid 5 : /tmp/blyth/opticks/GParts/5 : primbuf (1, 4) partbuf (3, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 6 : /tmp/blyth/opticks/GParts/6 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (9, 3, 4, 4) idxbuf (1, 4) 
    Solid 7 : /tmp/blyth/opticks/GParts/7 : primbuf (1, 4) partbuf (1, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 8 : /tmp/blyth/opticks/GParts/8 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (11, 3, 4, 4) idxbuf (1, 4) 

    Solid 9 : /tmp/blyth/opticks/GParts/9 : primbuf (130, 4) partbuf (130, 4, 4) tranbuf (130, 3, 4, 4) idxbuf (130, 4) 
    epsilon:ana blyth$ 

Huh observed an ordering difference, on different node::

    O[blyth@localhost opticks]$ python3 ./ana/GParts.py 
    Solid 0 : /tmp/blyth/opticks/GParts/0 : primbuf (3084, 4) partbuf (17346, 4, 4) tranbuf (7917, 3, 4, 4) idxbuf (3084, 4) 
    Solid 1 : /tmp/blyth/opticks/GParts/1 : primbuf (5, 4) partbuf (7, 4, 4) tranbuf (5, 3, 4, 4) idxbuf (5, 4) 
    Solid 2 : /tmp/blyth/opticks/GParts/2 : primbuf (6, 4) partbuf (30, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 
    Solid 3 : /tmp/blyth/opticks/GParts/3 : primbuf (6, 4) partbuf (54, 4, 4) tranbuf (29, 3, 4, 4) idxbuf (6, 4) 
    Solid 4 : /tmp/blyth/opticks/GParts/4 : primbuf (6, 4) partbuf (28, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 

    Solid 5 : /tmp/blyth/opticks/GParts/5 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (11, 3, 4, 4) idxbuf (1, 4) 
    Solid 6 : /tmp/blyth/opticks/GParts/6 : primbuf (1, 4) partbuf (1, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 7 : /tmp/blyth/opticks/GParts/7 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (9, 3, 4, 4) idxbuf (1, 4) 
    Solid 8 : /tmp/blyth/opticks/GParts/8 : primbuf (1, 4) partbuf (3, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
       

    Solid 9 : /tmp/blyth/opticks/GParts/9 : primbuf (130, 4) partbuf (130, 4, 4) tranbuf (130, 3, 4, 4) idxbuf (130, 4) 
    O[blyth@localhost opticks]$ 


Fix with new method GInstancer::sortRepeatCandidates
-------------------------------------------------------

::

    2021-04-16 15:10:12.726 INFO  [22347699] [GInstancer::sortRepeatCandidates@518]  num_repcan 9
    2021-04-16 15:10:12.726 INFO  [22347699] [GInstancer::dumpDigests@530] before sort
     i          0 pdig         f66e1795f939ffb74853712f78d03da6 ndig      25600 first 0x16414d610 first.nidx     176632
     i          1 pdig         9fb0c0b03b943768979663a035ada6f0 ndig      12612 first 0x123cc1a40 first.nidx      70960
     i          2 pdig         9e5c3b0d4f9e3d7ccdeb629041042737 ndig       5000 first 0x11efe2780 first.nidx      70966
     i          3 pdig         27173da50c17bbbee1fa19c0c5191bae ndig       2400 first 0x192f8b660 first.nidx     304636
     i          4 pdig         812b4ba287f5ee0bc9d43bbf5bbe87fb ndig        590 first 0x11ee746f0 first.nidx      69668
     i          5 pdig         f4b9ec30ad9f68f89b29639786cb62ef ndig        590 first 0x11ef32460 first.nidx      69078
     i          6 pdig         98dce83da57b0395e163467c9dae521b ndig        590 first 0x11ebd3db0 first.nidx      68488
     i          7 pdig         26657d5ff9020d2abefe558796b99584 ndig        590 first 0x11ee9c9c0 first.nidx      70258
     i          8 pdig         36986ef0e48b2696d09c93846c6b69d0 ndig        504 first 0x116f33c20 first.nidx         10
    2021-04-16 15:10:13.427 INFO  [22347699] [GInstancer::dumpDigests@530] after sort
     i          0 pdig         f66e1795f939ffb74853712f78d03da6 ndig      25600 first 0x16414d610 first.nidx     176632
     i          1 pdig         9fb0c0b03b943768979663a035ada6f0 ndig      12612 first 0x123cc1a40 first.nidx      70960
     i          2 pdig         9e5c3b0d4f9e3d7ccdeb629041042737 ndig       5000 first 0x11efe2780 first.nidx      70966
     i          3 pdig         27173da50c17bbbee1fa19c0c5191bae ndig       2400 first 0x192f8b660 first.nidx     304636
     i          4 pdig         98dce83da57b0395e163467c9dae521b ndig        590 first 0x11ebd3db0 first.nidx      68488
     i          5 pdig         f4b9ec30ad9f68f89b29639786cb62ef ndig        590 first 0x11ef32460 first.nidx      69078
     i          6 pdig         812b4ba287f5ee0bc9d43bbf5bbe87fb ndig        590 first 0x11ee746f0 first.nidx      69668
     i          7 pdig         26657d5ff9020d2abefe558796b99584 ndig        590 first 0x11ee9c9c0 first.nidx      70258
     i          8 pdig         36986ef0e48b2696d09c93846c6b69d0 ndig        504 first 0x116f33c20 first.nidx         10
    2021-04-16 15:10:13.577 INFO  [22347699] [GInstancer::findRepeatCandidates@346]  nall 131 repeat_min 400 vertex_min 0 num_repcan 9
    2021-04-16 15:10:13.577 INFO  [22347699] [GInstancer::findRepeatCandidates@356]  num_all 131 num_repcan 9 dmax 30
     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 



