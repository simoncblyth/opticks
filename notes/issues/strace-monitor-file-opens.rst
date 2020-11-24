strace-monitor-file-opens
============================



Commandline to enable strace monitoring
-----------------------------------------

::

    o.sh --g4oktest --strace


Still some remain
--------------------

But none of the remaining are per-event so should be OK like this.

::

    2020-11-25 03:07:37.074 DEBUG [415365] [G4OKTest::propagate@297] ]
    === o-main : runline PWD /home/blyth RC 0 Wed Nov 25 03:07:37 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/G4OKTest --g4oktest --strace
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     G4OKTest.log                                                                     :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/sensorData.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/angularEfficiency.npy                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/snap/snap.ppm                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    /home/blyth/local/opticks/bin/o.sh : RC : 0
    [blyth@localhost ~]$ 


::

    2020-11-25 02:55:12.241 INFO  [395872] [OpTracer::snap@149] )
    2020-11-25 02:55:12.241 DEBUG [395872] [G4OKTest::propagate@297] ]
    === o-main : runline PWD /home/blyth/opticks/optixrap RC 0 Wed Nov 25 02:55:12 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/G4OKTest --g4oktest --strace
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     G4OKTest.log                                                                     :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/sensorData.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/angularEfficiency.npy                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/snap/snap.ppm                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/0/parameters.json                   :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    /home/blyth/local/opticks/bin/o.sh : RC : 0
    [blyth@localhost optixrap]$ 


::

    2020-11-25 02:46:17.524 INFO  [381203] [OpTracer::snap@149] )
    2020-11-25 02:46:17.524 DEBUG [381203] [G4OKTest::propagate@297] ]
    === o-main : runline PWD /home/blyth/opticks RC 0 Wed Nov 25 02:46:17 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/G4OKTest --g4oktest --strace
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     G4OKTest.log                                                                     :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/sensorData.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/angularEfficiency.npy                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/snap/snap.ppm                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /home/blyth/local/opticks/results/G4OKTest/R0_cvd_/20201125_024604/OTracerTimes.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/0/parameters.json                   :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    /home/blyth/local/opticks/bin/o.sh : RC : 0
    [blyth@localhost opticks]$ 


    strace -e open /home/blyth/local/opticks/lib/G4OKTest 
    



::

    o.sh --g4oktest --strace

    2020-11-25 00:55:55.311 DEBUG [206873] [G4OKTest::propagate@297] ]
    === o-main : runline PWD /home/blyth/opticks/g4ok RC 0 Wed Nov 25 00:55:55 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/G4OKTest --g4oktest --strace
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     G4OKTest.log                                                                     :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/sensorData.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/angularEfficiency.npy                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/blyth/opticks/g4ok/tests/G4OKTest/snap.ppm                                  :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /home/blyth/local/opticks/results/G4OKTest/R0_cvd_/20201125_005541/OTracerTimes.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/0/parameters.json                   :  O_WRONLY|O_CREAT|O_TRUNC :  0666 


    /home/blyth/local/opticks/bin/o.sh : RC : 0
    [blyth@localhost g4ok]$ 


    ::

    G4Opticks=INFO G4OKTest=INFO strace  -e open /home/blyth/local/opticks/lib/G4OKTest

    ...
    2020-11-25 00:42:10.681 INFO  [184824] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@249]  target_lvname /dd/Geometry/AD/lvADE0xc2a78c00x3ef9140 nidxs.size() 2 nidx 3153
    2020-11-25 00:42:10.682 ERROR [184824] [G4OKTest::collectGensteps@267]  eventID 0 num_genstep_photons 5000
    open("/tmp/blyth/opticks/evt/g4live/torch/1/gs.json", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 54
    open("/tmp/blyth/opticks/evt/g4live/torch/1/gs.npy", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 54
    open("/proc/self/status", O_RDONLY)     = 54


Skip saving gensteps in production::

     862 int G4Opticks::propagateOpticalPhotons(G4int eventID)
     863 {
     864     LOG(LEVEL) << "[[" ;
     865     assert( m_genstep_collector );
     866     m_gensteps = m_genstep_collector->getGensteps();
     867     m_gensteps->setArrayContentVersion(G4VERSION_NUMBER);
     868     m_gensteps->setArrayContentIndex(eventID);
     869 
     870     unsigned num_gensteps = m_gensteps->getNumItems();
     871     LOG(LEVEL) << " num_gensteps "  << num_gensteps ;
     872     if( num_gensteps == 0 )
     873     {
     874         LOG(fatal) << "SKIP as no gensteps have been collected " ;
     875         return 0 ;
     876     }
     877 
     878 
     879     unsigned tagoffset = eventID ;  // tags are 1-based : so this will normally be the Geant4 eventID + 1
     880     
     881     if(!m_ok->isProduction()) // --production
     882     {
     883         const char* gspath = m_ok->getDirectGenstepPath(tagoffset);
     884         LOG(LEVEL) << "[ saving gensteps to " << gspath ; 
     885         m_gensteps->save(gspath);  
     886         LOG(LEVEL) << "] saving gensteps to " << gspath ;
     887     }   






Check with new defaults for embedded commandline in G4Opticks
-----------------------------------------------------------------

::

    o.sh --g4oktest --strace

    2020-11-25 00:12:34.053 INFO  [139870] [G4Opticks::InitOpticks@193] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --production --nosave 
    ...

    2020-11-25 00:12:47.181 INFO  [139870] [OpTracer::snap@148] )
    === o-main : runline PWD /home/blyth/opticks/g4ok RC 0 Wed Nov 25 00:12:47 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/G4OKTest --g4oktest --strace
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     G4OKTest.log                                                                     :          O_WRONLY|O_CREAT :  0644 

     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
           judged OK

     /tmp/blyth/opticks/G4OKTest/SensorLib/sensorData.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/SensorLib/angularEfficiency.npy                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
           explicitly done from that test and positioned appropriately, so OK 

     /tmp/blyth/opticks/evt/g4live/torch/1/gs.json                                    :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
           genstep saving from where ? 

     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/-1/ht.npy                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/-1/so.json                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/-1/so.npy                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
           Geant4 source photons are empty anyhow  : so skip these in production

     /tmp/blyth/opticks/g4ok/tests/G4OKTest/snap.ppm                                  :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
           TODO: should use standard dir for G4OKTest 
 
     /home/blyth/local/opticks/results/G4OKTest/R0_cvd_/20201125_001234/OTracerTimes.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/G4OKTest/evt/g4live/torch/0/parameters.json                   :  O_WRONLY|O_CREAT|O_TRUNC :  0666 



    /home/blyth/local/opticks/bin/o.sh : RC : 0
    [blyth@localhost g4ok]$ 



Interleaved::

   strace -e open /home/blyth/local/opticks/lib/G4OKTest --g4oktest --strace


The -1 are Geant4 which are empty anyhow so skip them in production::

     862 int G4Opticks::propagateOpticalPhotons(G4int eventID)
     863 {
     ...
     902     if(m_gpu_propagate)
     903     {
     904         m_opmgr->setGensteps(m_gensteps);
     905 
     906         m_opmgr->propagate();     // GPU simulation is done in here 
     907 
     908         OpticksEvent* event = m_opmgr->getEvent();
     909         m_hits = event->getHitData()->clone() ;
     910         m_num_hits = m_hits->getNumItems() ;
     911 
     912         m_hits_wrapper->setPhotons( m_hits );
     913 
     914         
     915         if(!m_ok->isProduction())
     916         {
     917             // minimal g4 side instrumentation in "1st executable" 
     918             // do after propagate, so the event will have been created already
     919             m_g4hit = m_g4hit_collector->getPhoton();
     920             m_g4evt = m_opmgr->getG4Event();
     921             m_g4evt->saveHitData( m_g4hit ) ; // pass thru to the dir, owned by m_g4hit_collector ?
     922             m_g4evt->saveSourceData( m_genphotons ) ;
     923         }
     924         






Nov 2020 : OKTest strace check
--------------------------------

::

    [blyth@localhost ok]$ o.sh --oktest --strace --production --nosave
    ...
    2020-11-24 18:10:27.892 INFO  [29724] [Opticks::dumpRC@247]  rc 0 rcmsg : -
    === o-main : runline PWD /home/blyth/opticks/ok RC 0 Tue Nov 24 18:10:28 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/OKTest --oktest --strace --production --nosave
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     OKTest.log                                                                       :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/Time.ini                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/DeltaTime.ini                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/VM.ini                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/DeltaVM.ini                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfile.npy                    :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileLabels.npy              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileAcc.npy                 :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileAccLabels.npy           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileLis.npy                 :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileLisLabels.npy           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/0/parameters.json                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    /home/blyth/local/opticks/bin/o.sh : RC : 0


Running with logging intermingled::

    strace -e open /home/blyth/local/opticks/lib/OKTest --oktest --strace --production --nosave 

Shows that all those creates are happening together, coming from Opticks::postpropagate. 
So avoid that by skipping in production::

     596 void Opticks::postpropagate()
     597 {
     598    if(isProduction()) return ;  // --production
     600    saveProfile();
     ...
     620    saveParameters();
     623 }


After that are down to six O_CREAT::

    === o-main : runline PWD /home/blyth/opticks RC 0 Tue Nov 24 19:57:14 CST 2020
    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/OKTest --oktest --strace --production --nosave
    /home/blyth/local/opticks/bin/strace.py -f O_CREAT
    strace.py -f O_CREAT
     OKTest.log                                                                       :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 



/home/blyth/.opticks/runcache/CDevice.bin
    written from OContext::initDevices CDevice::Visible the save happens at OContext instanciation
    when CUDA_VISIBLE_DEVICES envvar is not defined  

/var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
/var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
/var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
/var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 

     the cache is created at every run because it is by default deleted at termination by OContext::cleanUpCache
     that used to be necessary due to the default path not including the username but could now
     be skipped as the default cache path is controlled to be within user directory 
 

Because these creates only happen at startup, not per event, I judge it OK even in production running.




FIXED ISSUE : strace running shows log being written into unexpected location beside the binary /home/blyth/local/opticks/lib/OKG4Test.log
--------------------------------------------------------------------------------------------------------------------------------------------


* many logs found in that directory 
* need to avoid this as would cause permission failure in shared installation
* FIXED using SProc::ExecutableName() in PLOG.cc instead of argv[0]
* Also while looking into PLOG setup note that the RollingFileAppender is not enabled, due
  to a default zero argument : tried setting these to 500,000 bytes and 3 files

::

    [blyth@localhost tmp]$ cd /tmp ; strace -o /tmp/strace.log -e open $(which OKG4Test) --help >/dev/null ; strace.py
    strace.py
     /home/blyth/local/opticks/lib/OKG4Test.log                                       :          O_WRONLY|O_CREAT :  0644 

    [blyth@localhost tmp]$ cd /tmp ; strace -o /tmp/strace.log -e open OKG4Test --help >/dev/null ; strace.py
    strace.py
     OKG4Test.log                                                                     :          O_WRONLY|O_CREAT :  0644 

::

    068 const char* PLOG::_logpath_parse(int argc, char** argv)
     69 {
     70     assert( argc < MAXARGC && " argc sanity check fail ");
     71     //  Construct logfile path based on executable name argv[0] with .log appended 
     72     std::string lp(argc > 0 ? argv[0] : "default") ;
     73     lp += ".log" ;
     74     return strdup(lp.c_str());
     75 }
     76




strace technique
-----------------------



Using "--strace" argumment to old op.sh script::

    822    elif [ "${OPTICKS_DBG}" == "2" ]; then
    823       runline="strace -o /tmp/strace.log -e open ${OPTICKS_BINARY} ${OPTICKS_ARGS}"
    824    else


sets up strace monitoring of all file opens by the binary eg OKG4Test, creating a log of 2000 lines::

    [blyth@localhost bin]$ wc /tmp/strace.log 
      2004  11302 251061 /tmp/strace.log

    [blyth@localhost bin]$ head -10 /tmp/strace.log
    open("/home/blyth/local/opticks/lib/../lib/tls/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib/tls/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/tls/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/tls/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/libOKG4.so", O_RDONLY|O_CLOEXEC) = 3
    open("/home/blyth/local/opticks/lib/../lib/libOK.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/libOK.so", O_RDONLY|O_CLOEXEC) = 3



Use strace.py script to parse, filter and report. For example showing creates::

    calhost bin]$ strace.py -f CREAT
    strace.py -f CREAT
     /home/blyth/local/opticks/lib/OKG4Test.log"                                      :          O_WRONLY|O_CREAT :  0644 
     tboolean-box/GItemList/GMaterialLib.txt"                                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     tboolean-box/GItemList/GSurfaceLib.txt"                                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ why these relative, all other absolute ?

     /var/tmp/OptixCache/cache.db"                                                    :            O_RDWR|O_CREAT :  0666 
     /var/tmp/OptixCache/cache.db"                                                    : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/OptixCache/cache.db-journal"                                            :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-wal"                                                :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-shm"                                                :            O_RDWR|O_CREAT :  0664 

     /tmp/blyth/location/seq.npy"                                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/his.npy"                                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/mat.npy"                                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^ debug dumping from okc.Indexer 

     /tmp/blyth/location/cg4/primary.npy"                                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^^ debug dumping from CG4  
     

     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ht.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/gs.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ox.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ph.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ...  skipped expected ...
     /tmp/tboolean-box/evt/tboolean-box/torch/1/report.txt"                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190602_200126/t_absolute.ini"       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190602_200126/t_delta.ini"          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190602_200126/report.txt"           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/blyth/opticks/evt/tboolean-box/torch/Time.ini"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/DeltaTime.ini"                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/VM.ini"                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/DeltaVM.ini"                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/Opticks.npy"                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^ OpticksProfile::save metadata going to wrong place    





Relative write::

    334 void GGeoTest::importCSG(std::vector<GVolume*>& volumes)
    ...
    439     // see notes/issues/material-names-wrong-python-side.rst
    440     LOG(info) << "Save mlib/slib names "
    441               << " numTree : " << numTree
    442               << " csgpath : " << m_csgpath
    443               ;
    444 
    445     if( numTree > 0 )
    446     {
    447         m_mlib->saveNames(m_csgpath);
    448         m_slib->saveNames(m_csgpath);
    449     }
    450 
    451 
    452     LOG(info) << "]" ;
    453 }


::

    [blyth@localhost opticks]$ opticks-f \$TMP | grep seq.npy 
    ./optickscore/Indexer.cc:    m_seq->save("$TMP/seq.npy");  

    105 template <typename T>
    106 void Indexer<T>::save()
    107 {
    108     m_seq->save("$TMP/seq.npy");
    109     m_his->save("$TMP/his.npy");
    110     m_mat->save("$TMP/mat.npy");
    111 }


CG4.cc::

    344     pr->save("$TMP/cg4/primary.npy");   // debugging primary position issue 


::

    1735     m_profile->setDir(getEventFold());  // from Opticks::configure (from m_spec (OpticksEventSpec)

    [blyth@localhost optickscore]$ OpticksEventSpecTest
    2019-06-02 21:16:24.784 INFO  [362461] [OpticksEventSpec::Summary@148] s0 (no cat) typ typ tag tag itag 0 det det cat (null) dir /tmp/blyth/opticks/evt/det/typ/tag
    2019-06-02 21:16:24.784 INFO  [362461] [OpticksEventSpec::Summary@148] s1 (with cat) typ typ tag tag itag 0 det det cat cat dir /tmp/blyth/opticks/evt/cat/typ/tag










