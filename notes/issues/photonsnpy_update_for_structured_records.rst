PhotonsNPY Update for structured records ?
============================================

For example run::

    ggv-pmt-test

Noisy::

    [2016-Jun-07 18:07:53.820611]:info: RecordsNPY::dumpRecord ij 0,8
    PhotonsNPY::dumpPhotonRecord (i,j)        0 post (       79.68      32.45     300.00         0.10) polw (    1.01  -0.74   0.70    98.75) flag.x/m1 185:                        ? flag.y/m2 41:                        ? iflag.z [ 21] NAN_ABORT  
    [2016-Jun-07 18:07:53.820685]:info: RecordsNPY::dumpRecord ij 0,9
    PhotonsNPY::dumpPhotonRecord (i,j)        0 post (       79.68      32.45     300.00         0.10) polw (    1.01  -0.74   0.70    98.75) flag.x/m1 185:                        ? flag.y/m2 41:                        ? iflag.z [ 21] NAN_ABORT  
    [2016-Jun-07 18:07:53.820763]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 0 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.820891]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 1 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821012]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 2 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821156]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 3 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821283]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 4 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821408]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 5 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821562]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 6 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821715]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 7 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.821870]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 8 photon_id 0 flag 185,41,21,14 etype 1
    [2016-Jun-07 18:07:53.822024]:fatal: RecordsNPY::getSequenceString bitpos out of range 185 bitmax 40 record 9 photon_id 0 flag 185,41,21,14 etype 1
    pho        0 (       79.84      32.51      91.68         1.13) (   -0.38   0.93  -0.00   380.00) ERRERRERRERRERRERRERRERRERRERR NA NA NA NA NA NA NA NA NA NA  SURFACE_ABSORB BOUNDARY_TRANSMIT TORCH  
    ERRERRERRERRERRERRERRERRERRERR
    NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT NAN_ABORT 

                 ce vec4      39.918     16.256    150.000    150.000 
                ldd vec4     334.170    312.093      0.000     -0.100 
    App::indexEvtOld dpho 100
    PhotonsNPY::dumpPhotonRecord (i,j)      100 post (       33.51      21.34     300.00         0.10) polw (   -0.40  -0.89  -0.79    86.82) flag.x/m1 66:                        ? flag.y/m2 53:                        ? iflag.z [ 94] BOUNDARY_TRANSMIT  
    [2016-Jun-07 18:07:53.822290]:info: RecordsNPY::dumpRecord ij 100,1
    PhotonsNPY::dumpPhotonRecord (i,j)      100 post (       33.51      21.34     300.00         0.10) polw (   -0.40  -0.89  -0.79    86.82) flag.x/m1 66:                        ? flag.y/m2 53:                        ? iflag.z [ 94] BOUNDARY_TRANSMIT  


