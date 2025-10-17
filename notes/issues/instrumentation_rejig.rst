instrumentation_rejig.rst
==================================

For production running, its not practical to ordinarily store arrays.
So need to add SEvt count meta_data to run_meta.txt ?
* HMM but do not like SProf stamps mixed in with run_meta.txt, so 
  added optional metadata field to the SProf.hh



Former CSGOptiX::SimulateMain
-------------------------------

At tail did::

    SProf::Write("run_meta.txt", true ); // append:true

But thats has several problems:

1. a nasty kludge mixing SProf.hh and NP.hh serialization.
2. writing only on clean exit is terrible for crash debug. 
3. thats cxs_min.sh only, not general to OJ running 



RUN_META
----------

::

     089 /**
      90 SEvt::Init_RUN_META
      91 ---------------------
      92 
      93 As this is a static it happens just as libSysRap is loaded,
      94 very soon after starting the executable.
      95 
      96 **/
      97 
      98 
      99 NP* SEvt::Init_RUN_META() // static
     100 {
     101     NP* run_meta = NP::Make<float>(1);
     102     run_meta->set_meta<std::string>("SEvt__Init_RUN_META", sprof::Now() );
     103     return run_meta ;
     104 }
     105 
     106 NP* SEvt::RUN_META = Init_RUN_META() ;
     107 





CHECK SProf::Write
---------------------

SProf::Add to vectors in memory 

::

    SProf::Write("run_meta.txt", true );

::

    (ok) A[blyth@localhost CSGOptiX]$ BP=SProf::Write cxs_min.sh

    (gdb) bt
    #0  SProf::Write (path=0x7ffff7f25beb "run_meta.txt", append=true) at /data1/blyth/local/opticks_Debug/include/SysRap/SProf.hh:145
    #1  0x00007ffff7e32576 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:170
    #2  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb008) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 




QSim::simulate
----------------

::

     521     int tot_ht = sev->getNumHit() ;  // NB from fold, so requires hits array gathering to be configured to get non-zero


