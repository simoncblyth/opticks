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


How to fix ?
----------------

* start by pulling SCurandState out of QCurandState so can do the CurandState checks/config from SEvt 





