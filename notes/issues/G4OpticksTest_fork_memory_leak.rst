G4OpticksTest_fork_memory_leak
=================================

Added simple per G4Opticks::propagateOptical call profile collection.

* VM steadily increasing at around 4MB per call to propagate. 


.. contents:: Table of Contents


Possible Reduction of Resources with G4Opticks::setGenstepsReservation
-----------------------------------------------------------------------

::

    /**
    G4Opticks::setGenstepReservation
    ----------------------------------

    Setting the genstep reservation is optional. 
    Doing so may reduce the resource usage when collecting 
    large numbers of gensteps. For maximum effect the *max_gensteps_expected* 
    value should be larger than the maximum expected number of gensteps 
    collected prior to a reset bringing that down to zero. 
    Values less than the actual maxium do not cause a problem.

    **/

    void G4Opticks::setGenstepReservation(int max_gensteps_expected)
    {
    m_genstep_collector->setReservation(max_gensteps_expected); 
    }

    int  G4Opticks::getGenstepReservation() const 
    {
    return m_genstep_collector->getReservation() ;  
    }



Overview
-----------

**is this a leak, or is it fragmentation : ie using up clumps of memory that will be reclaimed by OS later**

G4OpticksTest appears to leak when observing SProc::VirtualMemoryUsageMB
but G4OKTest does not.  They are both doing much the same thing, the primary difference
is that G4OpticksTest does lots of dynamic genstep array growing up to 3000 or so items. 
G4OKTest uses few torch gensteps with lots of photons each.

Investigations reveal that reserving ahead does seem to reduce resource usage.



cfg4/tests/CGenstepCollectorLeakTest.cc
    simple application of CGenstepCollector

cfg4/tests/CGenstepCollectorLeak2Test.cc
    eliminates the collector, just uses the array and adopts NPX 

npy/NPX.hpp
    stripped down variant of NPY for memory issue testing and 
    trying out alternative m_data approaches
    
npy/tests/NPY6Test.cc
    dynamic growing of NPX, looking into base pointer shifts and capacity changes

sysrap/tests/reallocTest.cc
    take a look at a lower level 


Reliability of /proc/self/status and mac equivalent ?
---------------------------------------------------------

* https://stackoverflow.com/questions/42260960/inconsistent-change-in-vmrss-or-vmsize-when-looking-at-proc-self-status

* https://unix.stackexchange.com/questions/35129/need-explanation-on-resident-set-size-virtual-size

RSS seems less flakey a candle than VM on macOS::

    reserve  items 4000 itemvals 24 vals 96000 cap0 0 cap1 96000
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 96000
        8 : 4383.27 : 2.379 : 3479,6,4
    reserve  items 4000 itemvals 24 vals 96000 cap0 0 cap1 96000
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 96000
        9 : 4384.32 : 2.392 : 3500,6,4
     i   0 numsteps 3271
     i   1 numsteps 3270
     i   2 numsteps 3057
     i   3 numsteps 3453
     i   4 numsteps 3459
     i   5 numsteps 3362
     i   6 numsteps 3111
     i   7 numsteps 3702
     i   8 numsteps 3479
     i   9 numsteps 3500
     reservation 4000 nevt 10 vm0 4382.88 vm1 4384.32 dvm 1.43799 dvm_nevt 0.143799 rss0 1.957 rss1 2.392 drss 0.435 drss_nevt 0.0435
    epsilon:npy blyth$ RESERVATION=4000 NPY6Test 


    reservation 4000 nevt 10 vm0 4382.88 vm1 4383.27 dvm 0.38916 dvm_nevt 0.038916 rss0 1.957 rss1 2.379 drss 0.422 drss_nevt 0.0422
    epsilon:npy blyth$ RESERVATION=4000 NPY6Test 

    reserve  items 3500 itemvals 24 vals 84000 cap0 0 cap1 84000
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 84000
        9 : 4385.27 : 3.375 : 3500,6,4
     i   0 numsteps 3271
     i   1 numsteps 3270
     i   2 numsteps 3057
     i   3 numsteps 3453
     i   4 numsteps 3459
     i   5 numsteps 3362
     i   6 numsteps 3111
     i   7 numsteps 3702
     i   8 numsteps 3479
     i   9 numsteps 3500
     reservation -1 nevt 10 vm0 4382.88 vm1 4385.27 dvm 2.39209 dvm_nevt 0.239209 rss0 1.957 rss1 3.375 drss 1.418 drss_nevt 0.1418
     epsilon:npy blyth$ RESERVATION=-1 NPY6Test 


Resource wise the best thing to do is to set a fixed max size to the array.
Presumably as that prevents realloc calls.
Adjusting that using pre-knowledge of the number of steps event by event does not help, actually does harm.


NPY6Test 
-----------

::

    epsilon:npy blyth$ NPYBase=INFO RESERVATION=4096 NPY6Test 
    PLOG::EnvLevel adjusting loglevel by envvar   key NPYBase level INFO fallback DEBUG
     nevt 10 reservation 4096
    2021-02-16 14:33:20.564 INFO  [10766527] [NPYBase::setReservation@209] items 4096
     i   0 numsteps 3271
     i   1 numsteps 3270
     i   2 numsteps 3057
     i   3 numsteps 3453
     i   4 numsteps 3459
     i   5 numsteps 3362
     i   6 numsteps 3111
     i   7 numsteps 3702
     i   8 numsteps 3479
     i   9 numsteps 3500
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 24
    NPY<T>::grow base_ptr shift 1,6,4 m_data->size() 48 m_data->capacity() 48
    NPY<T>::grow base_ptr shift 2,6,4 m_data->size() 72 m_data->capacity() 96
    NPY<T>::grow base_ptr shift 4,6,4 m_data->size() 120 m_data->capacity() 192



Looks like resource use is approx halved when reserve ahead
-----------------------------------------------------------------

::

    RESERVATION=4000 CGenstepCollectorLeak2Test 
    RESERVATION=0 CGenstepCollectorLeak2Test 


G4OpticksTest_fork
--------------------

::

    (base) [simon@localhost G4OpticksTest_fork]$ ./run.sh 
    === ./check.sh : environment check PASSED : rc 0
    G4OpticksTest /home/simon/G4OpticksTest_fork/gdml/G4Opticks_50000.gdml macros/muon_noIO_10.mac
    ...

    2021-02-16 21:01:12.970 INFO  [157434] [G4Opticks::finalizeProfile@392] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-16 21:01:12.970 INFO  [157434] [G4Opticks::finalizeProfile@393] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
    2021-02-16 21:01:12.971 INFO  [157434] [OpticksProfile::Report@526]  num_stamp 10 profile_leak_mb 0 v0,v1 VmSize(MB) r0,r1 RSS(MB) 
     t0  75665.9 t1  75672.9 dt  7.03906 dt/(num_stamp-1) 0.782118
     v0    21317 v1    21300 dv -17.0352 dv/(num_stamp-1)  -1.8928
     r0  1745.91 r1  1761.68 dr   15.764 dr/(num_stamp-1)  1.75156


    ###] RunAction::EndOfRunAction G4Opticks.Finalize


    TimeTotal> 17.704 17.590
    (base) [simon@localhost G4OpticksTest_fork]$ 



CGenstepCollectorLeakTest : exercises a lot of dynamic genstep growing 
-------------------------------------------------------------------------

::

    epsilon:opticks blyth$ CGenstepCollectorLeakTest
     evt 0 num_steps 10000 gs 10000,6,4
     evt 1 num_steps 50000 gs 50000,6,4
     evt 2 num_steps 60000 gs 60000,6,4
     evt 3 num_steps 80000 gs 80000,6,4
     evt 4 num_steps 10000 gs 10000,6,4
     evt 5 num_steps 100000 gs 100000,6,4
     evt 6 num_steps 30000 gs 30000,6,4
     evt 7 num_steps 300000 gs 300000,6,4
     evt 8 num_steps 20000 gs 20000,6,4
     evt 9 num_steps 10000 gs 10000,6,4
     mock_numevt 10 v0 4604.66 v1 4713.52 dv 108.86 dvp 10.886


CGenstepCollectorLeak2Test : simplifies CGenstepCollectorLeakTest and uses NPX 
---------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ CGenstepCollectorLeak2Test
     ...
     evt 0 num_steps 10000 gs 10000,6,4
     evt 1 num_steps 50000 gs 50000,6,4
     evt 2 num_steps 60000 gs 60000,6,4
     evt 3 num_steps 80000 gs 80000,6,4
     evt 4 num_steps 10000 gs 10000,6,4
     evt 5 num_steps 100000 gs 100000,6,4
     evt 6 num_steps 30000 gs 30000,6,4
     evt 7 num_steps 300000 gs 300000,6,4
     evt 8 num_steps 20000 gs 20000,6,4
     evt 9 num_steps 10000 gs 10000,6,4
     mock_numevt 10 v0 4585.78 v1 4686.25 dv 100.471 dvp 10.0471
    epsilon:opticks blyth$ 


NPY6Test : NPX capacity reservation tests
-------------------------------------------

::

    .  98 : 4538.92 : 20000,6,4
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 24
    NPY<T>::grow base_ptr shift 1,6,4 m_data->size() 48 m_data->capacity() 48
    NPY<T>::grow base_ptr shift 2,6,4 m_data->size() 72 m_data->capacity() 96
    NPY<T>::grow base_ptr shift 4,6,4 m_data->size() 120 m_data->capacity() 192
    NPY<T>::grow base_ptr shift 8,6,4 m_data->size() 216 m_data->capacity() 384
    NPY<T>::grow base_ptr shift 16,6,4 m_data->size() 408 m_data->capacity() 768
    NPY<T>::grow base_ptr shift 32,6,4 m_data->size() 792 m_data->capacity() 1536
    NPY<T>::grow base_ptr shift 64,6,4 m_data->size() 1560 m_data->capacity() 3072
    NPY<T>::grow base_ptr shift 128,6,4 m_data->size() 3096 m_data->capacity() 6144
    NPY<T>::grow base_ptr shift 256,6,4 m_data->size() 6168 m_data->capacity() 12288
    NPY<T>::grow base_ptr shift 512,6,4 m_data->size() 12312 m_data->capacity() 24576
    NPY<T>::grow base_ptr shift 1024,6,4 m_data->size() 24600 m_data->capacity() 49152
    NPY<T>::grow base_ptr shift 2048,6,4 m_data->size() 49176 m_data->capacity() 98304
    NPY<T>::grow base_ptr shift 4096,6,4 m_data->size() 98328 m_data->capacity() 196608
    NPY<T>::grow base_ptr shift 8192,6,4 m_data->size() 196632 m_data->capacity() 393216
       99 : 4548.36 : 10000,6,4
     reservation 0 nevt 100 vm0 4382.88 vm1 4548.36 dvm 165.483 dvm_nevt 1.65483
    epsilon:npy blyth$ 


Reserving capacity ahead does seem to reduce resource usage::

    epsilon:npy blyth$ NEVT=1 RESERVATION=10000 NPY6Test 
     nevt 1 reservation 10000
     reserve  items 10000 itemvals 24 vals 240000 cap0 0 cap1 240000
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 240000
        0 : 4383.85 : 10000,6,4
     reservation 10000 nevt 1 vm0 4383.85 vm1 4383.85 dvm 0 dvm_nevt 0
    epsilon:npy blyth$ 



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



NPY6Test reserving ahead helps, but seems flakey 
---------------------------------------------------

::

    epsilon:npy blyth$ RESERVATION=4000 NPY6Test 
    ...

     reserve  items 4000 itemvals 24 vals 96000 cap0 0 cap1 96000
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 96000
        8 : 4383.27 : 3479,6,4
     reserve  items 4000 itemvals 24 vals 96000 cap0 0 cap1 96000
    NPY<T>::grow base_ptr shift 0,6,4 m_data->size() 24 m_data->capacity() 96000
        9 : 4383.27 : 3500,6,4
     i   0 numsteps 3271
     i   1 numsteps 3270
     i   2 numsteps 3057
     i   3 numsteps 3453
     i   4 numsteps 3459
     i   5 numsteps 3362
     i   6 numsteps 3111
     i   7 numsteps 3702
     i   8 numsteps 3479
     i   9 numsteps 3500
     reservation 4000 nevt 10 vm0 4382.88 vm1 4383.27 dvm 0.38916 dvm_nevt 0.038916
    epsilon:npy blyth$ 
    epsilon:npy blyth$ 



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




