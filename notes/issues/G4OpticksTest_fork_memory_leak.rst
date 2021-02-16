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


::

    2021-02-16 01:39:00.035 FATAL [158904] [G4Opticks::dumpSkipGencode@351] OPTICKS_SKIP_GENCODE m_skip_gencode_count 0
    2021-02-16 01:39:00.036 INFO  [158904] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-16 01:39:00.036 INFO  [158904] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 1000 m_profile_leak_mb 0     t0 5119.42 t1 5939.96 dt 820.541 dt/(num_stamp-1) 0.821362     v0 (MB) 21317.1 v1 (MB) 24825.6 dv 3508.57 dv/(num_stamp-1) 3.51208


Adding the reset of m_hits and m_hiys seems to make no difference::

     499 void G4Opticks::reset()
     500 {
     501     resetCollectors();
     502 
     503     m_hits->reset();   // the cloned hits (and hiys) are owned by G4Opticks, so they must be reset here  
     504 #ifdef WITH_WAY_BUFFER
     505     m_hiys->reset();
     506 #endif
     507 
     508 }


::

     157 template <typename T>
     158 void NPY<T>::deallocate()
     159 {
     160     setHasData(false);
     161     m_data.clear();
     162     setBasePtr(NULL);
     163     setNumItems( 0 );
     164 }
     165 
     166 template <typename T>
     167 void NPY<T>::reset()
     168 {
     169     deallocate();
     170 }



Notice high genstep counts in G4OpticksTest
----------------------------------------------

::

    EventAction::EndOfEventAction eventid 0 num_gensteps 3271 num_photons 4536823 num_hits 36180
    EventAction::EndOfEventAction eventid 1 num_gensteps 3270 num_photons 4470236 num_hits 35264
    EventAction::EndOfEventAction eventid 2 num_gensteps 3057 num_photons 4092944 num_hits 32331
    EventAction::EndOfEventAction eventid 3 num_gensteps 3453 num_photons 4657689 num_hits 37099
    EventAction::EndOfEventAction eventid 4 num_gensteps 3459 num_photons 4751552 num_hits 37818
    EventAction::EndOfEventAction eventid 5 num_gensteps 3362 num_photons 4568483 num_hits 35884
    EventAction::EndOfEventAction eventid 6 num_gensteps 3111 num_photons 4248472 num_hits 33761
    EventAction::EndOfEventAction eventid 7 num_gensteps 3143 num_photons 4307171 num_hits 34277
    EventAction::EndOfEventAction eventid 8 num_gensteps 3702 num_photons 4944439 num_hits 39437
    EventAction::EndOfEventAction eventid 9 num_gensteps 3479 num_photons 4700233 num_hits 37371


Possibly a leak from NPY::add which has to do dynamic resizing rather a lot with such large genstep counts.

But NPY6Test.cc shows no leak, so long as reset is called of course.



CGenstepCollectorLeakTest
----------------------------

::

    epsilon:cfg4 blyth$ CGenstepCollectorLeakTest 
    2021-02-15 21:11:59.512 ERROR [10112770] [CGenstepCollector::CGenstepCollector@64]  lookup is not complete : will not be able to collect real gensteps, only machinery ones 
    2021-02-15 21:11:59.840 INFO  [10112770] [NPY<float>::dump@2298] NPY::dump (10,4) 

    (  0)   76319.516    4580.335       0.000       0.000 
    (  1)   76319.547    4589.772       0.000       0.000 
    (  2)   76319.570    4589.772       0.000       0.000 
    (  3)   76319.609    4602.355       0.000       0.000 
    (  4)   76319.609    4602.355       0.000       0.000 
    (  5)   76319.648    4602.355       0.000       0.000 
    (  6)   76319.664    4602.355       0.000       0.000 
    (  7)   76319.828    4677.853       0.000       0.000 
    (  8)   76319.836    4677.853       0.000       0.000 
    (  9)   76319.844    4677.853       0.000       0.000 
    2021-02-15 21:11:59.841 INFO  [10112770] [OpticksProfile::Report@521]  num_stamp 10 profile_leak_mb 0     t0 76319.5 t1 76319.8 dt 0.328125 dt/(num_stamp-1) 0.0364583     v0 (MB) 4580.33 v1 (MB) 4677.85 dv 97.5181 dv/(num_stamp-1) 10.8353
    epsilon:cfg4 blyth$ 
    epsilon:cfg4 blyth$ 




G4OKTest not leaking at all
------------------------------

::

    [blyth@localhost g4ok]$ G4OKTest 100
    ...
    2021-02-16 01:44:52.508 FATAL [201160] [G4Opticks::dumpSkipGencode@351] OPTICKS_SKIP_GENCODE m_skip_gencode_count 0
    2021-02-16 01:44:52.508 INFO  [201160] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-16 01:44:52.508 INFO  [201160] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 100 m_profile_leak_mb 0     t0 6282.86 t1 6292.51 dt 9.64453 dt/(num_stamp-1) 0.0974195     v0 (MB) 20009 v1 (MB) 20009 dv 0 dv/(num_stamp-1) 0
    [blyth@localhost g4ok]$ 


    [blyth@localhost g4ok]$ G4OKTEST_PROFILE_LEAK_MB=10 G4OKTest 100   ## checking that the measument works
    ...
    2021-02-16 01:47:00.135 FATAL [204436] [G4Opticks::dumpSkipGencode@351] OPTICKS_SKIP_GENCODE m_skip_gencode_count 0
    2021-02-16 01:47:00.135 INFO  [204436] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-16 01:47:00.135 INFO  [204436] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 100 m_profile_leak_mb 10     t0 6411.61 t1 6420.13 dt 8.52686 dt/(num_stamp-1) 0.0861299     v0 (MB) 20017.8 v1 (MB) 20984.6 dv 966.801 dv/(num_stamp-1) 9.76566
    [blyth@localhost g4ok]$ 


Try upping the photon sizes with G4OKTest but getting a negative leak!

::

    2021-02-16 03:57:01.426 FATAL [418142] [G4Opticks::dumpSkipGencode@351] OPTICKS_SKIP_GENCODE m_skip_gencode_count 0
    2021-02-16 03:57:01.427 INFO  [418142] [G4Opticks::finalizeProfile@392] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-16 03:57:01.427 INFO  [418142] [G4Opticks::finalizeProfile@393] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 10 m_profile_leak_mb 0     t0 14169.9 t1 14221.4 dt 51.4902 dt/(num_stamp-1) 5.72114     v0 (MB) 21504.1 v1 (MB) 21117 dv -387.08 dv/(num_stamp-1) -43.0089
    [blyth@localhost tests]$ 




