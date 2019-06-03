profiling-machinery-review
===============================

ana/metadata.py 
-----------------

::

    [blyth@localhost ana]$ grep metadata *.py 
    ab.py:        self.print_("ab.a.metadata:%s" % self.a.metadata)
    ab.py:        self.print_("ab.a.metadata:%s" % self.a.metadata)
    ab.py:        self.print_("ab.a.metadata.csgmeta0:%s" % self.a.metadata.csgmeta0 )
    ab.py:        amd = ",".join(self.a.metadata.csgbnd)
    ab.py:        bmd = ",".join(self.b.metadata.csgbnd)
    ab.py:        acsgp = self.a.metadata.TestCSGPath
    ab.py:        bcsgp = self.b.metadata.TestCSGPath
    ab.py:        #aNote = "A:%s" % self.a.metadata.Note
    ab.py:        ##bNote = "B:%s" % self.b.metadata.Note
    ab.py:        :return point: recarray for holding point level metadata
    cfg4_speedplot.py:from opticks.ana.metadata import Metadata, Catdir
    cfg4_speedplot.py:        log.warning("no metadata skipping")
    evt.py:from opticks.ana.metadata import Metadata
    evt.py:        ok = self.init_metadata()
    evt.py:        testcsgpath = self.metadata.TestCSGPath
    evt.py:    def init_metadata(self):
    evt.py:        log.debug("init_metadata")
    evt.py:        metadata = Metadata(self.tagdir)
    evt.py:        log.info("loaded metadata from %s " % self.tagdir)
    evt.py:        log.info("metadata %s " % repr(metadata))
    evt.py:        self.metadata = metadata  
    evt.py:        fdom.desc = "(metadata) 3*float4 domains of position, time, wavelength (used for compression)"
    evt.py:        self.desc['idom'] = "(metadata) %s " % pdict_(self.idomd)
    evt.py:             return "%s %s %s %s (%s)" % (self.label, self.stamp, pdict_(self.idomd), self.path, self.metadata.Note)
    metadata.py:Access the metadata json files written by Opticks runs, 
    metadata.py:TODO: extract the good stuff from here as migrate from metadata.py to meta.py
    metadata.py:    timestamp folders contain just metadata for prior runs not full evt::
    metadata.py:    Reads in metadata from dated folders corresponding to runs of::
    metadata.py:        log.info("times metadata for tag %s " % tag + "\n".join(map(str,mds)))
    metadata.py:            log.warning("no metadata found")
    metadata.py:            log.warning("skipped metadata with inconsistent photon count n %s nc %s " % (n, nc)) 
    metadata.py:def test_metadata():
    meta.py:Attempt at more general metadata handling than metadata.py.
    nload.py:    event metadata
    nodelib.py:            # This is handled C++ side with gltftarget (config) and NScene targetnode GLTF asset metadata
    tevt.py:     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
    tevt.py:     idom :            (1, 1, 4) : (metadata) int domain 
    tmeta.py:tmeta.py : Load a single events metadata
    tmeta.py:    [2016-08-19 15:51:42,378] p23598 {/Users/blyth/opticks/ana/tmeta.py:21} INFO - loaded metadata from /tmp/blyth/opticks/evt/dayabay/torch/1 :                       /tmp/blyth/opticks/evt/dayabay/torch/1 571d76cd06acc1e992c211d6833dd0ff a32520a5215239cf54ee03d61ed154f6  100000     4.2878 CFG4_MODE  
    tmeta.py:from opticks.ana.metadata import Metadata
    tmeta.py:    log.info("loaded metadata from %s : %s " % (mdir, repr(md)))
    [blyth@localhost ana]$ 




geocache-bench geocache-bench360 bench.py : based on simple meta.py
--------------------------------------------------------------------------------------


bench.py lists run metadata from dated folders found beneath $OPTICKS_RESULTS_PREFIX/results::

   bench.py --name 360


    ---  GROUPCOMMAND : geocache-bench360 --xanalytic  GEOFUNC : geocache-j1808-v4 
     OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench360 --runstamp 1558873537 --runlabel R1_TITAN_RTX --xanalytic
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
                    20190526_202537  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.227      1.000      0.294           1.340    : /home/blyth/local/opticks/results/geocache-bench360/R1_TITAN_RTX/20190526_202537  
            R0_TITAN_V_AND_TITAN_RTX      0.400      1.758      0.516           2.188    : /home/blyth/local/opticks/results/geocache-bench360/R0_TITAN_V_AND_TITAN_RTX/20190526_202537  
                          R1_TITAN_V      0.519      2.282      0.671           1.060    : /home/blyth/local/opticks/results/geocache-bench360/R1_TITAN_V/20190526_202537  
                          R0_TITAN_V      0.657      2.888      0.849           1.341    : /home/blyth/local/opticks/results/geocache-bench360/R0_TITAN_V/20190526_202537  
                        R0_TITAN_RTX      0.774      3.403      1.000           1.203    : /home/blyth/local/opticks/results/geocache-bench360/R0_TITAN_RTX/20190526_202537  



::

    [blyth@localhost opticks]$ cd /home/blyth/local/opticks/results/geocache-bench360/R0_TITAN_RTX/20190526_202537
    [blyth@localhost 20190526_202537]$ l
    total 8
    -rw-rw-r--. 1 blyth blyth 277 May 26 20:26 OTracerTimes.ini
    -rw-rw-r--. 1 blyth blyth 968 May 26 20:26 parameters.json

    [blyth@localhost 20190526_202537]$ cat OTracerTimes.ini
    validate000=0.06056400000670692
    compile000=6.0000020312145352e-06
    prelaunch000=1.2031310000020312
    launch000=0.7807020000036573
    launch001=0.77108399999997346
    launch002=0.77083200000197394
    launch003=0.77083899999706773
    launch004=0.77544599999964703
    launchAVG=0.77378060000046389


    [blyth@localhost 20190526_202537]$ jsn.py parameters.json 
    {u'--envkey': 1,
     u'CMDLINE': u' OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 1 --rtx 0 --runfolder geocache-bench360 --runstamp 1558873537 --runlabel R0_TITAN_RTX --xanalytic',
     u'COMMENT': u'reproduce-rtx-inversion-skipping-just-lv-22-maskVirtual',
     u'COMPUTE_CAPABILITY': u'70',
     u'GEOFUNC': u'geocache-j1808-v4',
     u'GROUPCOMMAND': u'geocache-bench360 --xanalytic',
     u'HOME': u'/home/blyth/opticks',
     u'KEY': u'OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce',
     u'OptiXVersion': 60000,
     u'RESULTS_PREFIX': u'/home/blyth/local/opticks',
     u'XERCESC_INCLUDE_DIR': u'/usr/include',
     u'XERCESC_LIBRARY': u'/usr/lib64/libxerces-c-3.1.so',
     u'idpath': u'/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1',
     u'stacksize': 2180}




::

    [blyth@localhost ana]$ bench.py -h
    usage: 
    bench.py
    ============

    Presents launchAVG times and prelaunch times for groups of Opticks runs
    with filtering based on commandline arguments of the runs and the digest 
    of the geocache used.

    ::

        bench.py --include xanalytic --digest f6cc352e44243f8fa536ab483ad390ce
        bench.py --include xanalytic --digest f6
            selecting analytic results for a particular geometry 

        bench.py --include xanalytic --digest 52e --since May22_1030
            selecting analytic results for a particular geometry after some time 

        bench.py --digest 52 --since 6pm

        bench.py --name geocache-bench360
             fullname of the results dir

        bench.py --name 360
             also works with just a tail string, so long as it selects 
             one of the results dirs 

        bench.py --name 360 --runlabel R1
              select runs with runlabel starting R1

           [-h] [--resultsdir RESULTSDIR] [--name NAME] [--digest DIGEST]
           [--since SINCE] [--include [INCLUDE [INCLUDE ...]]]
           [--exclude [EXCLUDE [EXCLUDE ...]]] [--runlabel RUNLABEL]
           [--xrunlabel XRUNLABEL] [--metric METRIC] [--other OTHER] [--nodirs]
           [--splay] [--nosort]




Uses::

    OpticksResource::initRunResultsDir
    OpticksResource::getRunResultsDir

::

    168 void OTracer::report(const char* msg)
    169 {
    170     LOG(info)<< msg ;
    171     if(m_trace_count == 0 ) return ;
    172 
    173     std::cout
    174           << " trace_count     " << std::setw(10) << m_trace_count
    175           << " trace_prep      " << std::setw(10) << m_trace_prep   << " avg " << std::setw(10) << m_trace_prep/m_trace_count  << std::endl
    176           << " trace_time      " << std::setw(10) << m_trace_time   << " avg " << std::setw(10) << m_trace_time/m_trace_count  << std::endl
    177           << std::endl
    178            ;
    179 
    180     m_trace_times->addAverage("launch");
    181     m_trace_times->dump("OTracer::report");
    182 
    183     const char* runresultsdir = m_ocontext->getRunResultsDir();
    184     LOG(info) << "save to " << runresultsdir ;
    185     m_trace_times->save(runresultsdir);
    186 }


::

    104 void OpTracer::snap()   // --snapconfig="steps=5,eyestartz=0,eyestopz=0"
    105 {
    106     LOG(info) << "(" << m_snap_config->desc();

    ...    skip the snapping loop

    159     m_otracer->report("OpTracer::snap");   // saves for runresultsdir
    160     //m_ok->dumpMeta("OpTracer::snap");
    161 
    162     m_ok->saveParameters();
    163 
    164     LOG(info) << ")" ;
    165 }






tboolean.sh strace open logging shows lots of metadata, who writes what
------------------------------------------------------------------------------

::

    [blyth@localhost opticks]$ strace.py -f O_CREAT
    strace.py -f O_CREAT
     /home/blyth/local/opticks/lib/OKG4Test.log                                       :          O_WRONLY|O_CREAT :  0644 
     tboolean-box/GItemList/GMaterialLib.txt                                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     tboolean-box/GItemList/GSurfaceLib.txt                                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /var/tmp/OptixCache/cache.db                                                     :            O_RDWR|O_CREAT :  0666 
     /var/tmp/OptixCache/cache.db                                                     : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/OptixCache/cache.db-journal                                             :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-wal                                                 :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-shm                                                 :            O_RDWR|O_CREAT :  0664 
     /tmp/blyth/location/seq.npy                                                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/his.npy                                                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/mat.npy                                                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/cg4/primary.npy                                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ht.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/gs.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ox.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/so.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/rx.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ph.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ps.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/rs.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/fdom.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/idom.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/tboolean-box/evt/tboolean-box/torch/-1/History_SequenceSource.json          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/History_SequenceLocal.json           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/Material_SequenceSource.json         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/Material_SequenceLocal.json          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/tboolean-box/evt/tboolean-box/torch/-1/parameters.json                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/t_absolute.ini                       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/t_delta.ini                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/report.txt                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/tboolean-box/evt/tboolean-box/torch/-1/20190603_133044/parameters.json      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/20190603_133044/t_absolute.ini       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/20190603_133044/t_delta.ini          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/20190603_133044/report.txt           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     [blyth@localhost -1]$ diff report.txt 20190603_133044/report.txt 
     [blyth@localhost -1]$ diff t_delta.ini 20190603_133044/t_delta.ini 
     [blyth@localhost -1]$ diff t_absolute.ini 20190603_133044/t_absolute.ini 
     [blyth@localhost -1]$ diff parameters.json 20190603_133044/parameters.json      



     /tmp/tboolean-box/evt/tboolean-box/torch/1/ht.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/gs.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/ox.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/so.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/rx.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/ph.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/ps.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/rs.npy                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/fdom.npy                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/idom.npy                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/tboolean-box/evt/tboolean-box/torch/1/History_SequenceSource.json           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/History_SequenceLocal.json            :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/Material_SequenceSource.json          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/Material_SequenceLocal.json           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/Boundary_IndexSource.json             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/Boundary_IndexLocal.json              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 


     /tmp/tboolean-box/evt/tboolean-box/torch/1/parameters.json                       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/t_absolute.ini                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/t_delta.ini                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/report.txt                            :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190603_133044/parameters.json       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190603_133044/t_absolute.ini        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190603_133044/t_delta.ini           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190603_133044/report.txt            :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ these from m_timer  OpticksEvent::saveReport 


     /tmp/tboolean-box/evt/tboolean-box/torch/Time.ini                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/DeltaTime.ini                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/VM.ini                                  :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/DeltaVM.ini                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/Opticks.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^^^^^^^^ these from m_ok.m_profile OK_PROFILE Opticks::saveProfile ^^^^^^^^^^^^^^^^^^^^^^^^

    [blyth@localhost opticks]$ 





m_timer BTimeKeeper
------------------------

boostrap/BTimes.hh
    vector of string double pairs 

boostrap/BTimeKeeper.cc
      m_timer instances in Opticks, OpticksEvent, OScene    

 
 

OK_PROFILE m_ok.m_profile OpticksProfile 
--------------------------------------------------

Opticks.hh::

     59 #define OK_PROFILE(s) \
     60     { \
     61        if(m_ok)\
     62        {\
     63           m_ok->profile((s)) ;\
     64        }\
     65     }
     66 

VM and Time stamps are collected from all over the place into m_profile::

    [blyth@localhost optickscore]$ opticks-f OK_PROFILE
    ./cfg4/CG4.cc:    OK_PROFILE("CG4::CG4");
    ./cfg4/CG4.cc:    OK_PROFILE("_CG4::propagate");
    ./cfg4/CG4.cc:    OK_PROFILE("CG4::propagate");
    ./extg4/X4PhysicalVolume.cc:    OK_PROFILE("_X4PhysicalVolume::convertMaterials");
    ./extg4/X4PhysicalVolume.cc:    OK_PROFILE("X4PhysicalVolume::convertMaterials");
    ./extg4/X4PhysicalVolume.cc:    OK_PROFILE("_X4PhysicalVolume::convertSolids");
    ./extg4/X4PhysicalVolume.cc:    OK_PROFILE("X4PhysicalVolume::convertSolids");
    ./extg4/X4PhysicalVolume.cc:    OK_PROFILE("_X4PhysicalVolume::convertStructure");
    ./extg4/X4PhysicalVolume.cc:    OK_PROFILE("X4PhysicalVolume::convertStructure");
    ./ok/OKPropagator.cc:    OK_PROFILE("OKPropagator::propagate.BEG");
    ./ok/OKPropagator.cc:    OK_PROFILE("OKPropagator::propagate.MID");
    ./ok/OKPropagator.cc:    OK_PROFILE("OKPropagator::propagate.END");
    ./opticksgeo/OpticksGeometry.cc:// TODO: move to OK_PROFILE 
    ./optickscore/Opticks.hh:#define OK_PROFILE(s) \
    ./optickscore/Opticks.hh:       Opticks*             m_ok ;   // for OK_PROFILE 
    ./optickscore/Opticks.cc:    OK_PROFILE("Opticks::Opticks");
    ./optickscore/OpticksEvent.cc:    OK_PROFILE("_OpticksEvent::collectPhotonHitsCPU");
    ./optickscore/OpticksEvent.cc:    OK_PROFILE("OpticksEvent::collectPhotonHitsCPU");
    ./optickscore/OpticksEvent.cc:    OK_PROFILE("_OpticksEvent::indexPhotonsCPU");
    ./optickscore/OpticksEvent.cc:    OK_PROFILE("OpticksEvent::indexPhotonsCPU");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::OpticksRun");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::createEvent.BEG");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::createEvent.END");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::resetEvent.BEG");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::resetEvent.END");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::saveEvent.BEG");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::saveEvent.END");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::anaEvent.BEG");
    ./optickscore/OpticksRun.cc:    OK_PROFILE("OpticksRun::anaEvent.END");
    ./optixrap/OPropagator.cc:    OK_PROFILE("_OPropagator::prelaunch");
    ./optixrap/OPropagator.cc:    OK_PROFILE("OPropagator::prelaunch");
    ./optixrap/OPropagator.cc:    OK_PROFILE("_OPropagator::launch");
    ./optixrap/OPropagator.cc:    OK_PROFILE("OPropagator::launch");
    ./optixrap/OEvent.cc:    OK_PROFILE("_OEvent::upload");
    ./optixrap/OEvent.cc:    OK_PROFILE("OEvent::upload");
    ./optixrap/OEvent.cc:    OK_PROFILE("_OEvent::download");
    ./optixrap/OEvent.cc:    OK_PROFILE("OEvent::download");
    ./optixrap/OEvent.cc:    OK_PROFILE("_OEvent::downloadHits");
    ./optixrap/OEvent.cc:    OK_PROFILE("OEvent::downloadHits");
    ./okop/OpIndexer.cc:    OK_PROFILE("_OpIndexer::indexSequence");
    ./okop/OpIndexer.cc:    OK_PROFILE("OpIndexer::indexSequence");
    ./okop/OpPropagator.cc:    OK_PROFILE("OpPropagator::propagate.BEG");
    ./okop/OpPropagator.cc:    OK_PROFILE("OpPropagator::propagate.MID");
    ./okop/OpPropagator.cc:    OK_PROFILE("OpPropagator::propagate.END");
    ./okop/OpSeeder.cc:    OK_PROFILE("_OpSeeder::seedPhotonsFromGenstepsViaOptiX");
    ./okop/OpSeeder.cc:    OK_PROFILE("OpSeeder::seedPhotonsFromGenstepsViaOptiX");
    [blyth@localhost opticks]$ 

Note the splitting into ini sections when dots are used in profile labels (unintended?) NOW FIXED::

    [blyth@localhost optickscore]$ cat /tmp/tboolean-box/evt/tboolean-box/torch/DeltaTime.ini
    OpticksRun::OpticksRun_0=19839.74609375
    Opticks::Opticks_0=0.001953125
    CG4::CG4_0=0.85546875
    [OpticksRun::createEvent]
    BEG_0=3.7734375
    END_0=0.001953125
    _CG4::propagate_0=0.05078125
    CG4::propagate_0=10.91796875
    _OpticksEvent::indexPhotonsCPU_0=0
    OpticksEvent::indexPhotonsCPU_0=0.087890625
    _OpticksEvent::collectPhotonHitsCPU_0=0
    OpticksEvent::collectPhotonHitsCPU_0=0.009765625
    [OKPropagator::propagate]
    BEG_0=0.005859375
    MID_0=0.001953125
    END_0=0
    _OEvent::upload_0=0
    OEvent::upload_0=0.01171875
    _OpSeeder::seedPhotonsFromGenstepsViaOptiX_0=0
    OpSeeder::seedPhotonsFromGenstepsViaOptiX_0=0.017578125
    _OPropagator::prelaunch_0=0
    OPropagator::prelaunch_0=2.134765625
    _OPropagator::launch_0=0
    OPropagator::launch_0=0.01171875
    _OpIndexer::indexSequence_0=0
    OpIndexer::indexSequence_0=0.025390625
    _OEvent::download_0=0
    OEvent::download_0=0.037109375
    _OEvent::downloadHits_0=0
    OEvent::downloadHits_0=0.001953125
    [OpticksRun::saveEvent]
    BEG_0=0
    END_0=0.15625
    [OpticksRun::anaEvent]
    BEG_0=0.01171875
    END_0=1.41015625
    [OpticksRun::resetEvent]
    BEG_0=0.001953125
    END_0=0
    [blyth@localhost optickscore]$ 

::

    0245 Opticks::Opticks(int argc, char** argv, const char* argforced )
     246     :
     247     m_log(new SLog("Opticks::Opticks","",debug)),
     248     m_ok(this),
     249     m_sargs(new SArgs(argc, argv, argforced)),
     250     m_argc(m_sargs->argc),
     251     m_argv(m_sargs->argv),
     252     m_dumpenv(m_sargs->hasArg("--dumpenv")),
     253     m_envkey(m_sargs->hasArg("--envkey") ? BOpticksKey::SetKey(NULL) : false),  // see tests/OpticksEventDumpTest.cc makes sensitive to OPTICKS_KEY
     254     m_production(m_sargs->hasArg("--production")),
     255     m_profile(new OpticksProfile("Opticks",m_sargs->hasArg("--stamp"))),
     256     m_materialprefix(NULL),


    0349 template <typename T>
     350 void Opticks::profile(T label)
     351 {
     352     m_profile->stamp<T>(label, m_tagoffset);
     353    // m_tagoffset is set by Opticks::makeEvent
     354 }
     355 void Opticks::dumpProfile(const char* msg, const char* startswith, const char* spacewith, double tcut)
     356 {
     357    m_profile->dump(msg, startswith, spacewith, tcut);
     358 }
     359 void Opticks::saveProfile()
     360 {
     361    m_profile->save();
     362 }

    1962 void Opticks::postgeometry()
    1963 {
    1964     configureDomains();
    1965 
    1966     defineEventSpec();  // <-- configure was too soon for test geometry that adjusts evtbase, so try here 
    1967     m_profile->setDir(getEventFold());
    1968 }


::

     17 OpticksProfile::OpticksProfile(const char* name, bool stamp_out)
     18    :
     19    m_dir(NULL),
     20    m_name(BStr::concat(NULL,name,".npy")),
     21    m_columns("Time,DeltaTime,VM,DeltaVM"),
     22    m_tt(new BTimesTable(m_columns)),
     23    m_npy(NPY<float>::make(0,1,m_tt->getNumColumns())),
     24 
     25    m_t0(0),




TIMER : m_timer looks to be from an earlier epoch being replaced by m_profile
-----------------------------------------------------------------------------------

* but i like the dated folder copies : not yet in OK_PROFILE : where done ?  NOW ADDED


okop/OpIndexer_.cu::

     33 #define TIMER(s) \
     34     { \
     35        if(m_ok)\
     36        {\
     37           BTimeKeeper& t = *(m_ok->getTimer()) ;\
     38           t((s)) ;\
     39        }\
     40     }
     41 

::

    122 
    123     TIMER("_seqhisMakeLookup");
    124     seqhis.make_lookup();
    125     TIMER("seqhisMakeLookup");
    126     seqhis.apply_lookup<unsigned char>(tp_his);
    127     TIMER("seqhisApplyLookup");
    128 
    129     if(verbose) dumpHis(tphosel, seqhis) ;
    130 
    131     TIMER("_seqmatMakeLookup");
    132     seqmat.make_lookup();
    133     TIMER("seqmatMakeLookup");
    134     seqmat.apply_lookup<unsigned char>(tp_mat);
    135     TIMER("seqmatApplyLookup");
    136 

::

    [blyth@localhost okop]$ grep getTimer *.*
    OpIndexer.cc:          BTimeKeeper& t = *(m_ok->getTimer()) ;\
    OpIndexer_.cu:          BTimeKeeper& t = *(m_ok->getTimer()) ;\
    OpMgr.cc:          BTimeKeeper& t = *(m_ok->getTimer()) ;\
    OpPropagator.cc:          BTimeKeeper& t = *(m_hub->getTimer()) ;\
    OpZeroer.cc:          BTimeKeeper& t = *(m_ok->getTimer()) ;\
    [blyth@localhost okop]$ 




::

    [blyth@localhost opticks]$ cat /tmp/tboolean-box/evt/tboolean-box/torch/1/t_delta.ini
    _seqhisMakeLookup=13.252762000000075
    seqhisMakeLookup=0.011238999999477528
    seqhisApplyLookup=0.00018099999942933209
    _seqmatMakeLookup=2.0000006770715117e-06
    seqmatMakeLookup=0.0068749999991268851
    seqmatApplyLookup=0.00018500000078347512
    indexSequenceCompute=0.0019599999977799598
    indexBoundaries=0.0016450000002805609
    _save=0.11958200000299257
    save=0.073017999999137828
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ cat /tmp/tboolean-box/evt/tboolean-box/torch/1/20190603_133044/t_delta.ini
    _seqhisMakeLookup=13.252762000000075
    seqhisMakeLookup=0.011238999999477528
    seqhisApplyLookup=0.00018099999942933209
    _seqmatMakeLookup=2.0000006770715117e-06
    seqmatMakeLookup=0.0068749999991268851
    seqmatApplyLookup=0.00018500000078347512
    indexSequenceCompute=0.0019599999977799598
    indexBoundaries=0.0016450000002805609
    _save=0.11958200000299257
    save=0.073017999999137828
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ diff /tmp/tboolean-box/evt/tboolean-box/torch/1/t_delta.ini /tmp/tboolean-box/evt/tboolean-box/torch/1/20190603_133044/t_delta.ini
    [blyth@localhost opticks]$ 


    [blyth@localhost okop]$ cat /tmp/tboolean-box/evt/tboolean-box/torch/-1/20190603_133044/t_delta.ini 
    _save=13.315819999999803
    save=0.077373000000079628


TIMER reportage 
---------------------

* adopted the same pattern with m_profile


::

    1755 void OpticksEvent::makeReport(bool verbose)
    1756 {
    1757     LOG(info) << "tagdir " << getTagDir()  ;
    1758 
    1759     if(verbose)
    1760     m_parameters->dump();
    1761 
    1762     m_timer->stop();
    1763 
    1764     m_ttable = m_timer->makeTable();
    1765     if(verbose)
    1766     m_ttable->dump("OpticksEvent::makeReport");
    1767 
    1768     // TODO: add some context lines in the report  eg 
    1769     //       OS uname, NODE_TAG, hostname, OptiX version, CUDA version, G4 Version etc..
    1770 
    1771     m_report->add(m_versions->getLines());
    1772     m_report->add(m_parameters->getLines());
    1773     m_report->add(m_ttable->getLines());
    1774 }
    1775 
    1776 
    1777 void OpticksEvent::saveReport()
    1778 {
    1779     std::string tagdir = getTagDir();
    1780     saveReport(tagdir.c_str());
    1781 
    1782     std::string anno = getTimeStamp() ;
    1783     std::string tagdir_ts = getTagDir(anno.c_str());
    1784     saveReport(tagdir_ts.c_str());
    1785 }


    1837 void OpticksEvent::saveReport(const char* dir)
    1838 {
    1839     if(!m_ttable || !m_report) return ;
    1840     LOG(debug) << "OpticksEvent::saveReport to " << dir  ;
    1841 
    1842     m_ttable->save(dir);
    // BTimesTable*

    1843     m_report->save(dir);
    1844 }



