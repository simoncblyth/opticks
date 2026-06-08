attempt_to_rerun_with_old_persisted_geometry_fails
===================================================


stree serialization changes prevent use of "J_2024aug27" CSGFoundry GEOM with current code
---------------------------------------------------------------------------------------------

::

    [lo] A[blyth@localhost ~]$ cxs_min.sh 
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh - Internal GEOM setup detected
    BASH_SOURCE                    : /data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    SDIR                           : /data1/blyth/local/opticks_Debug/bin 
    SDIR_NOTE                      : Think of SDIR as sibling-dir not source-dir as this script is used from release bin dir too 
    defarg                         : run_report_info 
    arg                            : run_report_info 
    allarg                         : info_env_fold_run_dbg_meta_report_grab_grep_gevt_du_pdb1_pdb0_AB_ana_pvcap_pvpub_mpcap_mppub 
    bin                            : CSGOptiXSMTest 
    script                         : /data1/blyth/local/opticks_Debug/bin/cxs_min.py 
    script_AB                      : /data1/blyth/local/opticks_Debug/bin/cxs_min_AB.py 
    script_lite                    : /data1/blyth/local/opticks_Debug/bin/cxs_min_lite.py 
    script_hlm                     : /data1/blyth/local/opticks_Debug/bin/cxs_min_hlm.py 
    GEOM                           : J_2024aug27 
    ...

    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2026-06-08 15:51:46.810  810023430 : [/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    stree::ImportArray array is null, label[prim_nidx.npy]
    stree::ImportArray array is null, label[nidx_prim.npy]
    stree::ImportNames array is null, label[prname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    stree::ImportNames array is null, label[soname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    CSGOptiXSMTest: /home/blyth/opticks/sysrap/SPMT.h:619: void SPMT::init_total(): Assertion `pmtTotal->shape.size() == 1 && pmtTotal->shape[0] == SPMT_Total::FIELDS' failed.
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh: line 826: 674385 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh run error
    [lo] A[blyth@localhost ~]$ 


Try to convert old GDML file with current code using ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
-----------------------------------------------------------------------------------------------------

Geometry folders standardly include origin GDML::

    0 lrwxrwxrwx.  1 blyth blyth    51 Aug 29  2024 J_2024aug27 -> /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27
    A[blyth@localhost GEOM]$ ls J_2024aug27/
    CSGFoundry  origin.gdml  origin_gdxml_report.txt  origin_raw.gdml
    A[blyth@localhost GEOM]$ ls -alst J_2024aug27/
    total 41004
        1 drwxrwxr-x. 4 cvmfs cvmfs       33 Dec 15  2024 ..
        1 drwxrwxr-x. 3 cvmfs cvmfs      117 Sep 11  2024 .
        1 drwxr-xr-x. 3 cvmfs cvmfs      238 Sep 10  2024 CSGFoundry
    20501 -rw-rw-r--. 1 cvmfs cvmfs 20992027 Sep 10  2024 origin.gdml
        1 -rw-rw-r--. 1 cvmfs cvmfs      197 Sep 10  2024 origin_gdxml_report.txt
    20501 -rw-rw-r--. 1 cvmfs cvmfs 20992947 Sep 10  2024 origin_raw.gdml
    A[blyth@localhost GEOM]$ 
     


Using gxt

1. set ~/.opticks/GEOM/GEOM.sh  to export GEOM=J_2024aug27_recreated
2. create directory for the recreated geometry and copy GDML into it::

   mkdir ~/.opticks/GEOM/J_2024aug27_recreated
   cp ~/.opticks/GEOM/J_2024aug27/origin.gdml ~/.opticks/GEOM/J_2024aug27_recreated/


3. invoke the conversion after first using "info" to check GEOM is "J_2024aug27_recreated"::

   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh info
   ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh info_run



FAILS : no surprise did not do GDML conversion in long time
---------------------------------------------------------------


::

    ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh info_dbg


    Reading symbols from G4CXOpticks_setGeometry_Test...
    Starting program: /data1/blyth/local/opticks_Debug/lib/G4CXOpticks_setGeometry_Test 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [Detaching after vfork from child process 676438]
    2026-06-08 16:16:21.221 INFO  [676432] [main@16] [SetGeometry
    [New Thread 0x7fffe81ff000 (LWP 676439)]
    2026-06-08 16:16:21.324 FATAL [676432] [G4CXOpticks::setGeometry@233]  failed to setGeometry 
    G4CXOpticks_setGeometry_Test: /home/blyth/opticks/g4cx/G4CXOpticks.cc:234: void G4CXOpticks::setGeometry(): Assertion `0' failed.

    Thread 1 "G4CXOpticks_set" received signal SIGABRT, Aborted.
    (gdb) bt
    #0  0x00007ffff148bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff143eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff1428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff142875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff1437886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff7ececd5 in G4CXOpticks::setGeometry (this=0x491360) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:234
    #6  0x00007ffff7ecda22 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:62
    #7  0x00000000004038c9 in main (argc=1, argv=0x7fffffffb8c8) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) f 6
    #6  0x00007ffff7ecda22 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:62
    62	    gx->setGeometry();
    (gdb) f 5
    #5  0x00007ffff7ececd5 in G4CXOpticks::setGeometry (this=0x491360) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:234
    234	        assert(0);
    (gdb) 



Looks like missing gdml route::

    219 void G4CXOpticks::setGeometry()
    220 {
    221     if(spath::has_CFBaseFromGEOM())
    222     {
    223         LOG(LEVEL) << "[ CSGFoundry::Load " ;
    224         CSGFoundry* cf = CSGFoundry::Load() ;
    225         LOG(LEVEL) << "] CSGFoundry::Load " ;
    226 
    227         LOG(LEVEL) << "[ setGeometry(cf)  " ;
    228         setGeometry(cf);
    229         LOG(LEVEL) << "] setGeometry(cf)  " ;
    230     }
    231     else
    232     {
    233         LOG(fatal) << " failed to setGeometry " ;
    234         assert(0);
    235     }
    236 }


::

    Resolve_GDMLPathFromGEOM()
    {   
       local origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml 
       if [ -f "$origin" ]; then 
            export ${GEOM}_GDMLPathFromGEOM=$origin
            echo $BASH_SOURCE : FOUND origin $origin
       else 
            echo $BASH_SOURCE : NOT-FOUND origin $origin
       fi  
    }


Restored that functionality by fallback to doing geometry conversion from gdml::

    222 void G4CXOpticks::setGeometry()
    223 {
    224     if(spath::has_CFBaseFromGEOM())
    225     {
    226         LOG(LEVEL) << "[ CSGFoundry::Load " ;
    227         CSGFoundry* cf = CSGFoundry::Load() ;
    228         LOG(LEVEL) << "] CSGFoundry::Load " ;
    229 
    230         LOG(LEVEL) << "[ setGeometry(cf)  " ;
    231         setGeometry(cf);
    232         LOG(LEVEL) << "] setGeometry(cf)  " ;
    233     }
    234     else if(spath::has_GDMLPathFromGEOM())
    235     {
    236         const char* gdmlpath = spath::GDMLPathFromGEOM();
    237         LOG(LEVEL) << " has_GDMLPathFromGEOM gdmlpath[" << ( gdmlpath ? gdmlpath : "-" ) << "]" ;
    238         setGeometry(gdmlpath);
    239     }
    240     else
    241     {
    242         LOG(fatal) << " failed to setGeometry " ;
    243         assert(0);
    244     }
    245 }
    246 




