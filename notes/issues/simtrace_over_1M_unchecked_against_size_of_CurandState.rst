simtrace_over_1M_unchecked_against_size_of_CurandState
========================================================


::

    N[blyth@localhost opticks]$ MOI=Hama:0:1000 ~/opticks/g4cx/gxt.sh run 
                       BASH_SOURCE : /home/blyth/opticks/g4cx/../bin/GEOM_.sh 
                               gp_ : J004_GDMLPath 
                                gp :  
                               cg_ : J004_CFBaseFromGEOM 
                                cg : /home/blyth/.opticks/GEOM/J004 
                       TMP_GEOMDIR : /tmp/blyth/opticks/GEOM/J004 
                           GEOMDIR : /home/blyth/.opticks/GEOM/J004 
                       BASH_SOURCE : /home/blyth/opticks/g4cx/../bin/GEOM_.sh 

    === cehigh : GEOM J004 MOI Hama:0:1000
    === cehigh_PMT
    CEHIGH_0=-8:8:0:0:-6:-4:1000:4
    === /home/blyth/opticks/g4cx/gxt.sh : run G4CXSimtraceTest log G4CXSimtraceTest.log
    stran.h : Tran::checkIsIdentity FAIL :  caller FromPair epsilon 1e-06 mxdif_from_identity 12075.9
    stran.h Tran::FromPair checkIsIdentity FAIL 
    //CSGOptiX7.cu : simtrace idx 0 genstep_id 0 evt->num_simtrace 1212000 
    2022-10-12 18:51:32.472 INFO  [419174] [SEvt::save@1568]  dir /home/blyth/.opticks/GEOM/J004/G4CXSimtraceTest/Hama:0:1000
    N[blyth@localhost opticks]$ 
    N[blyth@localhost opticks]$ 


Note that even with only one CEHIGH block, the num_simtrace exceeds the default curandState. 

* This should trigger an assert. 

Capture backtrace from checkIsIdenity issue::

    N[blyth@localhost opticks]$ env | grep SIGINT
    stran_checkIsIdentity_SIGINT=1

    N[blyth@localhost opticks]$ MOI=Hama:0:1000 ~/opticks/g4cx/gxt.sh dbg 
    === cehigh : GEOM J004 MOI Hama:0:1000
    === cehigh_PMT
    CEHIGH_0=-8:8:0:0:-6:-4:1000:4
    gdb -ex r --args G4CXSimtraceTest -ex r
    Wed Oct 12 19:01:37 CST 2022

    stran.h : Tran::checkIsIdentity FAIL :  caller FromPair epsilon 1e-06 mxdif_from_identity 12075.9

    Program received signal SIGINT, Interrupt.
    0x00007fffecd0b4fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007fffecd0b4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007fffed64c502 in Tran<double>::checkIsIdentity (this=0x4027090, mat=105 'i', caller=0x7fffed72397c "FromPair", epsilon=9.9999999999999995e-07)
        at /data/blyth/junotop/opticks/sysrap/stran.h:638
    #2  0x00007fffed64b7f0 in Tran<double>::FromPair (t=0x2dd6cc0, v=0x2dd6d00, epsilon=9.9999999999999995e-07) at /data/blyth/junotop/opticks/sysrap/stran.h:712
    #3  0x00007fffed65ef4a in SFrameGenstep::MakeCenterExtentGensteps (fr=...) at /data/blyth/junotop/opticks/sysrap/SFrameGenstep.cc:160
    #4  0x00007fffed676682 in SEvt::setFrame (this=0x2dd6bf0, fr=...) at /data/blyth/junotop/opticks/sysrap/SEvt.cc:269
    #5  0x00007ffff7b8fcfb in G4CXOpticks::simtrace (this=0x7fffffff57a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:391
    #6  0x0000000000408d52 in main (argc=3, argv=0x7fffffff5908) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimtraceTest.cc:27
    (gdb) 



