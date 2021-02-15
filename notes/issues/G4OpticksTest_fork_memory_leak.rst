G4OpticksTest_fork_memory_leak
=================================

Added simple per G4Opticks::propagateOptical call profile collection.

* VM steadily increasing at around 4MB per call to propagate. 


Plotting the leak
------------------- 

::

    epsilon:~ blyth$ cd /tmp/blyth/opticks/G4Opticks/tests/

    epsilon:tests blyth$ scp P:/tmp/simon/opticks/G4Opticks/tests/G4OpticksProfilePlot.npy .

    epsilon:tests blyth$ np.py G4OpticksProfilePlot.npy
    a :                                     G4OpticksProfilePlot.npy :            (1000, 4) : 1c0676926c9acdb982556aa220b126fe : 20210215-1225 

    epsilon:tests blyth$ ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py
    [[71888.7   21316.36      0.        0.   ]
     [71889.41  21279.992     0.        0.   ]
     [71890.07  21247.656     0.        0.   ]
     ...
     [72650.75  24906.477     0.        0.   ]
     [72651.55  24900.863     0.        0.   ]
     [72652.22  24830.473     0.        0.   ]]
     delta:   3514.11 slope0:      3.51 
    line fit:  slope       3.62    intercept   21284.92 

    In [1]:  
     


Check the measurement using deliberate extra leak::

    export G4OPTICKSTEST_PROFILE_LEAK_MB=10   # deliberate leak to check measurement

    2021-02-15 23:14:37.203 FATAL [403972] [G4Opticks::dumpSkipGencode@351] OPTICKS_SKIP_GENCODE m_skip_gencode_count 0
    2021-02-15 23:14:37.203 INFO  [403972] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-15 23:14:37.203 INFO  [403972] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 10 m_profile_leak_mb 10     t0 83670.9 t1 83677.2 dt 6.28906 dt/(num_stamp-1) 0.698785     v0 (MB) 21328.7 v1 (MB) 21431.1 dv 102.361 dv/(num_stamp-1) 11.3735


    export G4OPTICKSTEST_PROFILE_LEAK_MB=20   # deliberate leak to check measurement

    2021-02-15 23:17:17.347 FATAL [408250] [G4Opticks::dumpSkipGencode@351] OPTICKS_SKIP_GENCODE m_skip_gencode_count 0
    2021-02-15 23:17:17.347 INFO  [408250] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-15 23:17:17.347 INFO  [408250] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 10 m_profile_leak_mb 20     t0 83831 t1 83837.3 dt 6.28125 dt/(num_stamp-1) 0.697917     v0 (MB) 21338.5 v1 (MB) 21527.5 dv 189 dv/(num_stamp-1) 21


    2021-02-15 23:19:21.107 INFO  [411659] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 10 m_profile_leak_mb 0     t0 83954.7 t1 83961.1 dt 6.42188 dt/(num_stamp-1) 0.713542     v0 (MB) 21316.4 v1 (MB) 21329.1 dv 12.7734 dv/(num_stamp-1) 1.41927


    2021-02-15 23:33:09.817 INFO  [413858] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-15 23:33:09.817 INFO  [413858] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 1000 m_profile_leak_mb 0     t0 84030.2 t1 84789.8 dt 759.57 dt/(num_stamp-1) 0.760331     v0 (MB) 21316.8 v1 (MB) 24825.6 dv 3508.79 dv/(num_stamp-1) 3.5123



* looks like a leak of 1.4~3.5 MB per propagate



