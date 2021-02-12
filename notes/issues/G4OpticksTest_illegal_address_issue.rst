G4OpticksTest_illegal_address_issue
=======================================


Report from Hans
------------------

Running produces crashes because out of bound memory is accessed on some of he threads.::

    export OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="--rngmax 100"
    cuda-memcheck --leak-check full ./G4OpticksTest /data2/wenzel/gputest5/G4OpticksTest-install/bin/G4Opticks_50000.gdml muon_noIO.mac  >& check3.txt

I tried to run producing less photons per MeV but changing the gdml file from the 
usual G4Opticks_50000.gdml to G4Opticks_5000.gdml (difference in numbers is just the photon yield) now fails
see the file run5000::

    export OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="--rngmax 100 --gdmlpath /data2/wenzel/gputest5/G4OpticksTest-install/bin/G4Opticks_5000.gdml"
    ./G4OpticksTest /data2/wenzel/gputest5/G4OpticksTest-install/bin/G4Opticks_5000.gdml muon_IO.mac  > run5000.txt


( the output files are at:
https://fermicloud-my.sharepoint.com/:f:/g/personal/wenzel_services_fnal_gov/EnuVx0NpChRJqPAN89DKs-EBbuK0h2NX01n7G8G18BE8uQ?e=c5OvO3 )


Response
----------

I investigated the “illegal address” problem, adding the capability 
to skip gensteps at collection based on a list of gencodes passed in 
by envvar. But after all that I found that the problem doesn’t depend 
on the gensteps.

However I observe that switching off WITH_DEBUG_BUFFER in optickscore/OpticksSwitches.h
makes the issue go away.  The problem is some kind of flakey fail : the fail location 
varies with the kernel used.  The “—trivial” option changes to a very simple kernel. 

All this low level stuff is going to soon be reimplemented for OptiX 7 so no point 
delving deeper.

My testing was in my fork of G4OpticksTest using Geant4 1062 with 
the private->public change.    

The run.sh in the fork also shows how to run with different GDML files:

* https://github.com/simoncblyth/G4OpticksTest/blob/master/run.sh

My notes are in 

* https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/G4OpticksTest_illegal_address_issue.rst    



Possible fix : switch off WITH_DEBUG_BUFFER
-----------------------------------------------

::

    diff --git a/optickscore/OpticksSwitches.h b/optickscore/OpticksSwitches.h
    index 553dfd5..20bfc9f 100644
    --- a/optickscore/OpticksSwitches.h
    +++ b/optickscore/OpticksSwitches.h
    @@ -50,6 +50,11 @@ NB when searching for switches in python include the space at the end, eg::
     
     #define WITH_ANGULAR 1
     
    +// have observed flaky fails when WITH_DEBUG_BUFFER is enabled
    +// that is probably related to the PerRayData_propagate.h float3 
    +// see notes/issues/G4OpticksTest_illegal_address_issue.rst
    +// CONCLUSION : DO NOT LEAVE WITH_DEBUG_BUFFER enabled 
    +// use only for quick checks
     //#define WITH_DEBUG_BUFFER 1
     


Immediate impression
------------------------

* yuck access to fermicloud-my.sharepoint.com  needs a microsoft login and then you cannot even copy/paste from it 
* gdmlpath argument on the embedded commandline ? That probably aint going to work 
* probably there is an OPTICKS_KEY in the environment 


::

    ###[ RunAction::BeginOfRunAction G4Opticks.setGeometry


    2021-02-11 12:16:19.841 INFO  [556783] [G4Opticks::G4Opticks@305] ctor : DISABLE FPE detection : as it breaks OptiX launches
    2021-02-11 12:16:19.841 INFO  [556783] [BOpticksKey::SetKey@77]  spec G4OpticksTest.X4PhysicalVolume.World_PV.f7c1fdc5b730d7e828aa5e90857cb31d
    2021-02-11 12:16:19.842 INFO  [556783] [BOpticksResource::initViaKey@779] 
                 BOpticksKey  : KEYSOURCE
          spec (OPTICKS_KEY)  : G4OpticksTest.X4PhysicalVolume.World_PV.f7c1fdc5b730d7e828aa5e90857cb31d
                     exename  : G4OpticksTest
             current_exename  : G4OpticksTest
                       class  : X4PhysicalVolume
                     volname  : World_PV
                      digest  : f7c1fdc5b730d7e828aa5e90857cb31d
                      idname  : G4OpticksTest_World_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2021-02-11 12:16:19.842 INFO  [556783] [G4Opticks::InitOpticks@186] 
    # BOpticksKey::export_ 
    export OPTICKS_KEY=G4OpticksTest.X4PhysicalVolume.World_PV.f7c1fdc5b730d7e828aa5e90857cb31d

    2021-02-11 12:16:19.842 INFO  [556783] [G4Opticks::InitOpticks@206] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --production --nosave  --rngmax 100 --gdmlpath /data2/wenzel/gputest5/G4OpticksTest-install/bin/G4Opticks_5000.gdml
    2021-02-11 12:16:19.842 INFO  [556783] [Opticks::init@430] COMPUTE_MODE compute_requested  hostname aichi
    2021-02-11 12:16:19.842 INFO  [556783] [Opticks::init@439]  mandatory keyed access to geometry, opticksaux 
    2021-02-11 12:16:19.842 INFO  [556783] [Opticks::init@458] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER 
    2021-02-11 12:16:19.844 INFO  [556783] [Opticks::loadOriginCacheMeta@1882]  cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f7c1fdc5b730d7e828aa5e90857cb31d/1/cachemeta.json
    2021-02-11 12:16:19.844 INFO  [556783] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    2021-02-11 12:16:19.844 FATAL [556783] [Opticks::ExtractCacheMetaGDMLPath@2049]  FAILED TO EXTRACT ORIGIN GDMLPATH FROM METADATA argline 
     argline -
    2021-02-11 12:16:19.844 INFO  [556783] [Opticks::loadOriginCacheMeta@1893] ExtractCacheMetaGDMLPath 
    2021-02-11 12:16:19.844 FATAL [556783] [Opticks::loadOriginCacheMeta@1899] cachemetapath /home/wenzel/.opticks/geocache/G4OpticksTest_World_PV_g4live/g4ok_gltf/f7c1fdc5b730d7e828aa5e90857cb31d/1/cachemeta.json
    2021-02-11 12:16:19.844 FATAL [556783] [Opticks::loadOriginCacheMeta@1900] argline that creates cachemetapath must include "--gdmlpath /path/to/geometry.gdml" 
    G4OpticksTest: /data2/wenzel/gputest5/opticks/optickscore/Opticks.cc:1902: void Opticks::loadOriginCacheMeta(): Assertion `m_origin_gdmlpath' failed.




Try to reproduce
------------------

::

    [blyth@localhost G4OpticksTest]$ git remote -v
    origin	https://github.com/hanswenzel/G4OpticksTest (fetch)
    origin	https://github.com/hanswenzel/G4OpticksTest (push)
    [blyth@localhost G4OpticksTest]$ 
    [blyth@localhost G4OpticksTest]$ git pull

Yuck still a mess::

    [blyth@localhost G4OpticksTest]$ l
    total 84
    drwxrwxr-x. 2 blyth blyth  4096 Feb 12 17:58 src
    -rw-rw-r--. 1 blyth blyth  2640 Feb 12 17:58 readHits.cc
    drwxrwxr-x. 2 blyth blyth   170 Feb 12 17:58 macros
    drwxrwxr-x. 2 blyth blyth  4096 Feb 12 17:58 include
    -rw-rw-r--. 1 blyth blyth  3770 Feb 12 17:58 CMakeLists.txt
    -rw-rw-r--. 1 blyth blyth  2877 Feb 12 17:58 G4OpticksTest.cc
    -rw-rw-r--. 1 blyth blyth  1826 Feb 12 17:58 README.md
    -rw-rw-r--. 1 blyth blyth  2980 Feb 12 17:58 go.sh
    drwxrwxr-x. 2 blyth blyth    27 Dec 16 21:57 xml
    drwxrwxr-x. 2 blyth blyth    50 Dec 16 21:57 scripts
    -rw-rw-r--. 1 blyth blyth   417 Dec 16 21:57 set_env_hanshome.sh
    -rw-rw-r--. 1 blyth blyth  1781 Dec 16 21:57 set_env_lq.sh
    -rw-rw-r--. 1 blyth blyth   426 Dec 16 21:57 set_env.sh
    -rw-rw-r--. 1 blyth blyth  3489 Dec 16 21:57 setup_opticks.sh
    -rwxrwxr-x. 1 blyth blyth  1067 Dec 16 21:57 om.sh
    drwxrwxr-x. 2 blyth blyth    26 Dec 16 21:57 logs
    drwxrwxr-x. 2 blyth blyth   263 Dec 16 21:57 gdml
    -rw-rw-r--. 1 blyth blyth  1083 Dec 16 21:57 go-release.sh
    -rw-rw-r--. 1 blyth blyth  2980 Dec 16 21:57 go.sh_save
    -rw-rw-r--. 1 blyth blyth 13213 Dec 16 21:57 ckm.bash
    -rw-rw-r--. 1 blyth blyth   273 Dec 16 21:57 cm.txt
    -rw-rw-r--. 1 blyth blyth  6806 Dec 16 21:57 G4OpticksTest.rst
    [blyth@localhost G4OpticksTest]$ 

So use my fork::

    [blyth@localhost ~]$ git clone git@github.com:simoncblyth/G4OpticksTest.git G4OpticksTest_fork
    Cloning into 'G4OpticksTest_fork'...
    remote: Enumerating objects: 373, done.
    remote: Counting objects: 100% (373/373), done.
    remote: Compressing objects: 100% (275/275), done.
    remote: Total 613 (delta 259), reused 192 (delta 94), pack-reused 240
    Receiving objects: 100% (613/613), 1.34 MiB | 1.14 MiB/s, done.
    Resolving deltas: 100% (398/398), done.
    Checking connectivity... done

    [blyth@localhost ~]$ ./build.sh  # fails because this account is with Geant 1042

Remove them from blyth account::

    [blyth@localhost ~]$ rm -rf G4OpticksTest
    [blyth@localhost ~]$ rm -rf G4OpticksTest_fork


Use simon account with Geant4 1062, and my fork::

    [simon@localhost ~]$ git clone git@github.com:simoncblyth/G4OpticksTest.git G4OpticksTest_fork
    Cloning into 'G4OpticksTest_fork'...
    Enter passphrase for key '/home/simon/.ssh/id_rsa': 
    remote: Enumerating objects: 373, done.
    remote: Counting objects: 100% (373/373), done.
    remote: Compressing objects: 100% (275/275), done.
    remote: Total 613 (delta 259), reused 192 (delta 94), pack-reused 240
    Receiving objects: 100% (613/613), 1.34 MiB | 848.00 KiB/s, done.
    Resolving deltas: 100% (398/398), done.
    [simon@localhost ~]$ 


Recall "simon" has its own build of externals but is symbolically linked 
to use the same opticks source as "blyth".::

    [simon@localhost local]$ cd opticks_externals/
    [simon@localhost opticks_externals]$ l
    total 0
    drwxrwxr-x. 4 simon simon 32 Dec 19 01:20 boost
    drwxrwxr-x. 4 simon simon 79 Dec 19 01:18 boost.build
    drwxrwxr-x. 5 simon simon 43 Dec 19 01:32 clhep
    drwxrwxr-x. 3 simon simon 46 Dec 19 01:25 clhep.build
    drwxrwxr-x. 6 simon simon 58 Dec 19 02:34 g4_1062
    drwxrwxr-x. 4 simon simon 97 Dec 19 01:37 g4_1062.build
    drwxrwxr-x. 5 simon simon 43 Dec 19 01:37 xercesc
    drwxrwxr-x. 3 simon simon 57 Dec 19 01:32 xercesc.build
    [simon@localhost opticks_externals]$ pwd
    /home/simon/local/opticks_externals
    [simon@localhost opticks_externals]$ 


G4OpticksTest_fork build.sh needs a modified Geant4 10.6 and public GetAverageNumberOfPhotons
---------------------------------------------------------------------------------------------------

::

    [ 65%] Building CXX object CMakeFiles/G4OpticksTest.dir/src/RadiatorSD.cc.o
    /home/simon/G4OpticksTest_fork/src/RadiatorSD.cc: In member function ‘virtual G4bool RadiatorSD::ProcessHits(G4Step*, G4TouchableHistory*)’:
    /home/simon/G4OpticksTest_fork/src/RadiatorSD.cc:163:113: error: ‘G4double G4Cerenkov::GetAverageNumberOfPhotons(G4double, G4double, const G4Material*, G4MaterialPropertyVector*) const’ is private within this context
                             MeanNumberOfPhotons1 = proc-> GetAverageNumberOfPhotons(charge, beta1, aMaterial, Rindex);
                                                                                                                     ^
    In file included from /home/simon/G4OpticksTest_fork/src/RadiatorSD.cc:24:
    /home/simon/local/opticks_externals/g4_1062/include/Geant4/G4Cerenkov.hh:200:12: note: declared private here
       G4double GetAverageNumberOfPhotons(const G4double charge,
                ^~~~~~~~~~~~~~~~~~~~~~~~~
    /home/simon/G4OpticksTest_fork/src/RadiatorSD.cc:164:113: error: ‘G4double G4Cerenkov::GetAverageNumberOfPhotons(G4double, G4double, const G4Material*, G4MaterialPropertyVector*) const’ is private within this context
                             MeanNumberOfPhotons2 = proc-> GetAverageNumberOfPhotons(charge, beta2, aMaterial, Rindex);
                                                                                                                     ^
    In file included from /home/simon/G4OpticksTest_fork/src/RadiatorSD.cc:24:
    /home/simon/local/opticks_externals/g4_1062/include/Geant4/G4Cerenkov.hh:200:12: note: declared private here
       G4double GetAverageNumberOfPhotons(const G4double charge,
                ^~~~~~~~~~~~~~~~~~~~~~~~~
    make[2]: *** [CMakeFiles/G4OpticksTest.dir/src/RadiatorSD.cc.o] Error 1
    make[1]: *** [CMakeFiles/G4OpticksTest.dir/all] Error 2
    make: *** [all] Error 2


Rebuild Geant4 1062 with the private to public change::

    simon@localhost opticks_externals]$ vi source/processes/electromagnetic/xrays/include/G4Cerenkov.hh
    [simon@localhost opticks_externals]$ g4-cls G4Cerenkov
    /home/simon/local/opticks_externals/g4_1062.build/geant4.10.06.p02
    vi -R source/processes/electromagnetic/xrays/include/G4Cerenkov.hh source/processes/electromagnetic/xrays/src/G4Cerenkov.cc
    2 files to edit
    [simon@localhost opticks_externals]$ g4-vi
    [simon@localhost opticks_externals]$ g4-cd
    [simon@localhost geant4.10.06.p02]$ vi source/processes/electromagnetic/xrays/include/G4Cerenkov.hh
    [simon@localhost geant4.10.06.p02]$ g4-build
    Fri Feb 12 18:36:42 CST 2021
    [  0%] Built target G4ENSDFSTATE


::

    OEvent::downloadHits@467:  nhit 36180 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    G4OpticksTest: /home/simon/opticks/optixrap/OEvent.cc:691: unsigned int OEvent::downloadHiysCompute(OpticksEvent*): Assertion `cway.size % 4 == 0' failed.
    ./run.sh: line 19: 77390 Aborted                 (core dumped) G4OpticksTest /home/simon/G4OpticksTest_fork/gdml/G4Opticks_50000.gdml macros/muon_noIO.mac
    [simon@localhost G4OpticksTest_fork]$ 

::

    OEvent::downloadHits@467:  nhit 36180 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::downloadHiysCompute@693:  unexpected cway.size (should be multiple of 4)  9073646
    G4OpticksTest: /home/simon/opticks/optixrap/OEvent.cc:694: unsigned int OEvent::downloadHiysCompute(OpticksEvent*): Assertion `expected' failed.
    ./run.sh: line 19: 96038 Aborted                 (core dumped) G4OpticksTest /home/simon/G4OpticksTest_fork/gdml/G4Opticks_50000.gdml macros/muon_noIO.mac
    [simon@localhost G4OpticksTest_fork]$ 

::

     49 CBufSpec OBufBase::bufspec()
     50 {
     51    return CBufSpec( getDevicePtr(), getSize(), getNumBytes()) ;
     52 }
     53 


::


    2021-02-12 19:29:22.052 INFO  [168218] [OEvent::downloadHiysCompute@693] into hiy array :0,2,4
    2021-02-12 19:29:22.052 FATAL [168218] [OEvent::downloadHiysCompute@699]  unexpected cway.size (should be multiple of 4)  9073646
    OEvent::downloadHiysCompute unexpected cway.size : dev_ptr 0x7f29ae0a4010 size 9073646 num_bytes 145178336 hexdump 0 
    G4OpticksTest: /home/simon/opticks/optixrap/OEvent.cc:702: unsigned int OEvent::downloadHiysCompute(OpticksEvent*): Assertion `expected' failed.
    ./run.sh: line 22: 168218 Aborted                 (core dumped) G4OpticksTest /home/simon/G4OpticksTest_fork/gdml/G4Opticks_50000.gdml macros/muon_noIO.mac
    [simon@localhost G4OpticksTest_fork]$ echo $(( 145178336/9073646 ))
    16
    [simon@localhost G4OpticksTest_fork]$ echo $(( 9073646/2 ))
    4536823
    [simon@localhost G4OpticksTest_fork]$ echo $(( 4536823*2 ))
    9073646
    [simon@localhost G4OpticksTest_fork]$ 


* the assert is wrong the way buffers should have a CBufSpec size of 2*num_photon because it takes 2*float4 


::

     953 void OpticksEvent::createSpec()
     954 {
     955     // invoked by Opticks::makeEvent   or OpticksEvent::load
     956     unsigned int maxrec = getMaxRec();
     957     bool compute = isCompute();
     958 
     959     m_genstep_spec = GenstepSpec(compute);
     960     m_seed_spec    = SeedSpec(compute);
     961     m_source_spec  = SourceSpec(compute);
     962 
     963     m_hit_spec      = new NPYSpec(hit_       , 0,4,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(hit_, compute));
     964     m_hiy_spec      = new NPYSpec(hiy_       , 0,2,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(hiy_, compute));
     965     m_photon_spec   = new NPYSpec(photon_   ,  0,4,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(photon_, compute)) ;
     966     m_debug_spec    = new NPYSpec(debug_    ,  0,1,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(debug_, compute)) ;
     967     m_way_spec      = new NPYSpec(way_      ,  0,2,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(way_, compute)) ;
     968     m_record_spec   = new NPYSpec(record_   ,  0,maxrec,2,4,0, NPYBase::SHORT     ,  OpticksBufferSpec::Get(record_, compute)) ;
     969     //   SHORT -> RT_FORMAT_SHORT4 and size set to  num_quads = num_photons*maxrec*2  
     970 
     971     m_sequence_spec = new NPYSpec(sequence_ ,  0,1,2,0,0,      NPYBase::ULONGLONG ,  OpticksBufferSpec::Get(sequence_, compute)) ;
     972     //    ULONGLONG -> RT_FORMAT_USER  and size set to ni*nj*nk = num_photons*1*2
     973 
     974     m_nopstep_spec = new NPYSpec(nopstep_   ,  0,4,4,0,0,      NPYBase::FLOAT     , OpticksBufferSpec::Get(nopstep_, compute) ) ;
     975     m_phosel_spec   = new NPYSpec(phosel_   ,  0,1,4,0,0,      NPYBase::UCHAR     , OpticksBufferSpec::Get(phosel_, compute) ) ;
     976     m_recsel_spec   = new NPYSpec(recsel_   ,  0,maxrec,1,4,0, NPYBase::UCHAR     , OpticksBufferSpec::Get(recsel_, compute) ) ;
     977 
     978     m_fdom_spec    = new NPYSpec(fdom_      ,  3,1,4,0,0,      NPYBase::FLOAT     ,  "" ) ;
     979     m_idom_spec    = new NPYSpec(idom_      ,  1,1,4,0,0,      NPYBase::INT       ,  "" ) ;
     980 
     981 }


Without WAY_BUFFER::

    2021-02-12 19:56:46.532 INFO  [217746] [OEvent::download@551] ]
    2021-02-12 19:56:46.532 FATAL [217746] [OpPropagator::propagate@84] evtId(2) DONE nhit: 32331
    2021-02-12 19:56:46.534 FATAL [217746] [G4Opticks::propagateOpticalPhotons@981]  no-WAY_BUFFER 
    EventAction::EndOfEventAction num_hits 32331   m_num_hits: 32331 hits 0x5379480
    Event:   3
    2021-02-12 19:56:46.605 FATAL [217746] [OpPropagator::propagate@73] evtId(3) OK COMPUTE PRODUCTION
    2021-02-12 19:56:46.605 INFO  [217746] [OEvent::upload@388] [ id 3
    2021-02-12 19:56:46.605 INFO  [217746] [OEvent::setEvent@54]  this (OEvent*) 0x45235d0 evt (OpticksEvent*) 0x506d240
    2021-02-12 19:56:46.605 INFO  [217746] [OEvent::resizeBuffers@327]  genstep 3453,6,4 nopstep 0,4,4 photon 4657689,4,4 debug 4657689,1,4 way 4657689,2,4 source NULL record 4657689,10,2,4 phosel 4657689,1,4 recsel 4657689,10,1,4 sequence 4657689,1,2 seed 4657689,1,1 hit 0,4,4
    2021-02-12 19:56:46.632 INFO  [217746] [OEvent::uploadGensteps@424] (COMPUTE) id 3 3453,6,4 -> 4657689
    2021-02-12 19:56:46.632 INFO  [217746] [OEvent::upload@407] ] id 3
    2021-02-12 19:56:46.632 INFO  [217746] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@174] SEEDING TO SEED BUF  
    2021-02-12 19:56:46.632 INFO  [217746] [OEvent::markDirty@254] 
    2021-02-12 19:56:46.632 INFO  [217746] [OPropagator::launch@268] LAUNCH NOW   printLaunchIndex ( -1 -1 -1) -
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    ./run.sh: line 22: 217746 Aborted                 (core dumped) G4OpticksTest /home/simon/G4OpticksTest_fork/gdml/G4Opticks_50000.gdml macros/muon_noIO.mac
    [simon@localhost G4OpticksTest_fork]$ 
    [simon@localhost G4OpticksTest_fork]$ 


With WAY_BUFFER::

    2021-02-12 20:04:55.970 INFO  [237670] [GPho::wayConsistencyCheck@152]  mismatch_flags 0 mismatch_index 0
    EventAction::EndOfEventAction num_hits 32331   m_num_hits: 32331 hits 0x5ed3500
    Event:   3
    2021-02-12 20:04:56.037 FATAL [237670] [OpPropagator::propagate@73] evtId(3) OK COMPUTE PRODUCTION
    2021-02-12 20:04:56.037 INFO  [237670] [OEvent::upload@388] [ id 3
    2021-02-12 20:04:56.037 INFO  [237670] [OEvent::setEvent@54]  this (OEvent*) 0x3ff22c0 evt (OpticksEvent*) 0x5414680
    2021-02-12 20:04:56.037 INFO  [237670] [OEvent::resizeBuffers@327]  genstep 3453,6,4 nopstep 0,4,4 photon 4657689,4,4 debug 4657689,1,4 way 4657689,2,4 source NULL record 4657689,10,2,4 phosel 4657689,1,4 recsel 4657689,10,1,4 sequence 4657689,1,2 seed 4657689,1,1 hit 0,4,4
    2021-02-12 20:04:56.075 INFO  [237670] [OEvent::uploadGensteps@424] (COMPUTE) id 3 3453,6,4 -> 4657689
    2021-02-12 20:04:56.075 INFO  [237670] [OEvent::upload@407] ] id 3
    2021-02-12 20:04:56.075 INFO  [237670] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@174] SEEDING TO SEED BUF  
    2021-02-12 20:04:56.076 INFO  [237670] [OEvent::markDirty@254] 
    2021-02-12 20:04:56.076 INFO  [237670] [OPropagator::launch@268] LAUNCH NOW   printLaunchIndex ( -1 -1 -1) -
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    ./run.sh: line 22: 237670 Aborted                 (core dumped) G4OpticksTest /home/simon/G4OpticksTest_fork/gdml/G4Opticks_50000.gdml macros/muon_noIO.mac
    [simon@localhost G4OpticksTest_fork]$ 
    [simon@localhost G4OpticksTest_fork]$ 


Switching to the trivial kernel gets through all the events::

    export OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="--rngmax 10 --trivial"

But switching to dev to save the gensteps for perusal with the trivial kernel still fails, which is bizarre::


    2021-02-12 20:30:56.175 INFO  [277672] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2021-02-12 20:30:56.442 INFO  [277672] [OEvent::download@529] [
    2021-02-12 20:30:56.442 INFO  [277672] [OEvent::download@569] [ id 3

    Program received signal SIGSEGV, Segmentation fault.
    0x00007fffe1ee7476 in __memcpy_ssse3_back () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libidn-1.28-4.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe1ee7476 in __memcpy_ssse3_back () from /lib64/libc.so.6
    #1  0x00007fffea23373c in NPY<float>::read (this=0x4053510, src=0x7ffcca5d1010) at /home/simon/opticks/npy/NPY.cpp:188
    #2  0x00007fffeb6cfd58 in OContext::download<float> (buffer=..., npy=0x4053510) at /home/simon/opticks/optixrap/OContext.cc:994
    #3  0x00007fffeb6e8a0e in OEvent::download (this=0x2c60f30, evt=0x4035530, mask=412) at /home/simon/opticks/optixrap/OEvent.cc:608
    #4  0x00007fffeb6e7ec7 in OEvent::download (this=0x2c60f30) at /home/simon/opticks/optixrap/OEvent.cc:531
    #5  0x00007fffeba4d6ac in OpEngine::downloadEvent (this=0x1a2a630) at /home/simon/opticks/okop/OpEngine.cc:242
    #6  0x00007fffeba50b65 in OpPropagator::downloadEvent (this=0x1a2a720) at /home/simon/opticks/okop/OpPropagator.cc:101
    #7  0x00007fffeba50818 in OpPropagator::propagate (this=0x1a2a720) at /home/simon/opticks/okop/OpPropagator.cc:82
    #8  0x00007fffeba4e7d1 in OpMgr::propagate (this=0x1a1f390) at /home/simon/opticks/okop/OpMgr.cc:138
    #9  0x00007ffff7bcc3a0 in G4Opticks::propagateOpticalPhotons (this=0x8caef0, eventID=3) at /home/simon/opticks/g4ok/G4Opticks.cc:969
    #10 0x000000000041adf8 in EventAction::EndOfEventAction (this=0xaaf0c0, event=0x2c10be0) at /home/simon/G4OpticksTest_fork/src/EventAction.cc:86
    #11 0x00007ffff3e08d0f in G4EventManager::DoProcessing (this=0x8895a0, anEvent=0x2c10be0)


::

    (gdb) f 2
    #2  0x00007fffeb6cfd58 in OContext::download<float> (buffer=..., npy=0x4053510) at /home/simon/opticks/optixrap/OContext.cc:994
    994	        npy->read( ptr );
    (gdb) f 3
    #3  0x00007fffeb6e8a0e in OEvent::download (this=0x2c60f30, evt=0x4035530, mask=412) at /home/simon/opticks/optixrap/OEvent.cc:608
    608	        OContext::download<float>( m_debug_buffer, dg );
    (gdb) 
    (gdb) p ptr
    $1 = (void *) 0x7ffcca5d1010
    (gdb) p npy->getShapeString()
    Too few arguments in function call.
    (gdb) p npy->getShapeString(0)
    $2 = "4657689,1,4"
    (gdb) 



Investigate the writing empty warning by planting assert in NPYBase::

    (gdb) bt
    #0  0x00007fffe1dc7387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe1dc8a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe1dc01a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe1dc0252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffea21c32e in NPYBase::write_ (this=0x2e724c0, dst=0x2d866b0) at /home/simon/opticks/npy/NPYBase.cpp:298
    #5  0x00007fffeb6f9c03 in OCtx::upload_buffer (this=0x1badab0, arr=0x2e724c0, buffer_ptr=0x2e72ec0, item=-1) at /home/simon/opticks/optixrap/OCtx.cc:350
    #6  0x00007fffeb6f91f6 in OCtx::create_buffer (this=0x1badab0, arr=0x2e724c0, key=0x7fffeb770ce4 "OSensorLib_texid", type=73 'I', flag=32 ' ', item=-1, transpose=true)
        at /home/simon/opticks/optixrap/OCtx.cc:297
    #7  0x00007fffeb702773 in OSensorLib::makeSensorAngularEfficiencyTexture (this=0x2e725e0) at /home/simon/opticks/optixrap/OSensorLib.cc:124
    #8  0x00007fffeb702111 in OSensorLib::convert (this=0x2e725e0) at /home/simon/opticks/optixrap/OSensorLib.cc:88
    #9  0x00007fffeb6e1f27 in OScene::uploadSensorLib (this=0x1a2abe0, sensorlib=0x1a1f210) at /home/simon/opticks/optixrap/OScene.cc:199
    #10 0x00007fffeba4c73d in OpEngine::uploadSensorLib (this=0x1a2a5d0, sensorlib=0x1a1f210) at /home/simon/opticks/okop/OpEngine.cc:123
    #11 0x00007fffeba4cf53 in OpEngine::close (this=0x1a2a5d0) at /home/simon/opticks/okop/OpEngine.cc:178
    #12 0x00007fffeba4d306 in OpEngine::propagate (this=0x1a2a5d0) at /home/simon/opticks/okop/OpEngine.cc:202
    #13 0x00007fffeba50b34 in OpPropagator::propagate (this=0x1a2a6c0) at /home/simon/opticks/okop/OpPropagator.cc:77
    #14 0x00007fffeba4ea0f in OpMgr::propagate (this=0x1a1f330) at /home/simon/opticks/okop/OpMgr.cc:139
    #15 0x00007ffff7bcc442 in G4Opticks::propagateOpticalPhotons (this=0x8caeb0, eventID=0) at /home/simon/opticks/g4ok/G4Opticks.cc:970




Observe that using trivial WITH_DEBUG_BUFFER fails immediately. Switch it off.


Take a look at the saved gensteps from evt 2(1-based) of the dev trivial run::

    [simon@localhost ~]$ cd /tmp/simon/opticks/source/evt/g4live/natural/2
    [simon@localhost 2]$ l *.npy
    total 1380304
    -rw-rw-r--. 1 simon simon        80 Feb 12 22:03 dg.npy
    -rw-rw-r--. 1 simon simon       128 Feb 12 22:03 fdom.npy
    -rw-rw-r--. 1 simon simon    314000 Feb 12 22:03 gs.npy
    -rw-rw-r--. 1 simon simon    286224 Feb 12 22:03 ht.npy
    -rw-rw-r--. 1 simon simon    143152 Feb 12 22:03 hy.npy
    -rw-rw-r--. 1 simon simon        96 Feb 12 22:03 idom.npy
    -rw-rw-r--. 1 simon simon       144 Feb 12 22:03 OpticksProfileAccLabels.npy
    -rw-rw-r--. 1 simon simon        96 Feb 12 22:03 OpticksProfileAcc.npy
    -rw-rw-r--. 1 simon simon        80 Feb 12 22:03 OpticksProfileLabels.npy
    -rw-rw-r--. 1 simon simon       144 Feb 12 22:03 OpticksProfileLisLabels.npy
    -rw-rw-r--. 1 simon simon        88 Feb 12 22:03 OpticksProfileLis.npy
    -rw-rw-r--. 1 simon simon        80 Feb 12 22:03 OpticksProfile.npy
    -rw-rw-r--. 1 simon simon 286095184 Feb 12 22:03 ox.npy
    -rw-rw-r--. 1 simon simon  71523856 Feb 12 22:03 ph.npy
    -rw-rw-r--. 1 simon simon  17881024 Feb 12 22:03 ps.npy
    -rw-rw-r--. 1 simon simon 178809536 Feb 12 22:03 rs.npy
    -rw-rw-r--. 1 simon simon 715237856 Feb 12 22:03 rx.npy
    -rw-rw-r--. 1 simon simon 143047632 Feb 12 22:03 wy.npy
    [simon@localhost 2]$ date
    Fri Feb 12 22:08:56 CST 2021
    [simon@localhost 2]$ ipython
    Python 2.7.5 (default, Apr  2 2020, 13:16:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 3.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: import numpy as np

    In [2]: gs = np.load("gs.npy")

    In [4]: gs.shape
    Out[4]: (3270, 6, 4)

    In [5]: gs[0]
    Out[5]: 
    array([[  2.80259693e-45,   1.40129846e-45,   9.80908925e-45,
              2.79839303e-42],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   3.43414158e-01,
              3.43414187e-01],
           [             nan,   1.00000000e+00,   1.00000000e+00,
              2.98419617e+02],
           [  1.40129846e-45,   7.50000000e-01,   7.00000000e+00,
              1.40000000e+03],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00]], dtype=float32)

    In [6]: np.set_printoptions(suppress=True)

    In [7]: gs[0]
    Out[7]: 
    array([[    0.        ,     0.        ,     0.        ,     0.        ],
           [    0.        ,     0.        ,     0.        ,     0.        ],
           [    0.        ,     0.        ,     0.34341416,     0.34341419],
           [           nan,     1.        ,     1.        ,   298.4196167 ],
           [    0.        ,     0.75      ,     7.        ,  1400.        ],
           [    0.        ,     0.        ,     0.        ,     0.        ]], dtype=float32)

    In [8]: gs[0].view(np.uint32)
    Out[8]: 
    array([[         2,          1,          7,       1997],
           [         0,          0,          0,          0],
           [         0,          0, 1051710459, 1051710460],
           [4294967283, 1065353216, 1065353216, 1133852086],
           [         1, 1061158912, 1088421888, 1152319488],
           [         0,          0,          0,          0]], dtype=uint32)

    In [9]: gs[0].view(np.int32)
    Out[9]: 
    array([[         2,          1,          7,       1997],
           [         0,          0,          0,          0],
           [         0,          0, 1051710459, 1051710460],
           [       -13, 1065353216, 1065353216, 1133852086],
           [         1, 1061158912, 1088421888, 1152319488],
           [         0,          0,          0,          0]], dtype=int32)

    In [10]: 


::

     19 enum
     20 {
     21     OpticksGenstep_INVALID                  = 0,
     22     OpticksGenstep_G4Cerenkov_1042          = 1,
     23     OpticksGenstep_G4Scintillation_1042     = 2,
     24     OpticksGenstep_DsG4Cerenkov_r3971       = 3,
     25     OpticksGenstep_DsG4Scintillation_r3971  = 4,
     26     OpticksGenstep_TORCH                    = 5,
     27     OpticksGenstep_FABRICATED               = 6,
     28     OpticksGenstep_EMITSOURCE               = 7,
     29     OpticksGenstep_NATURAL                  = 8,
     30     OpticksGenstep_MACHINERY                = 9,
     31     OpticksGenstep_G4GUN                    = 10,
     32     OpticksGenstep_PRIMARYSOURCE            = 11,
     33     OpticksGenstep_GENSTEPSOURCE            = 12,
     34     OpticksGenstep_NumType                  = 13
     35 };

    In [11]: gs[:,0,0].view(np.uint32)
    Out[11]: array([2, 1, 2, ..., 2, 2, 2], dtype=uint32)


::

    In [1]: import numpy as np

    In [2]: gs = np.load("gs.npy")

    In [3]: gs[:,0,0].view(np.uint32)
    Out[3]: array([2, 1, 2, ..., 2, 2, 2], dtype=uint32)

    In [4]: np.unique(gs[:,0,0].view(np.uint32), return_counts=True)
    Out[4]: (array([1, 2], dtype=uint32), array([1604, 1666]))


::

    (base) [simon@localhost 2]$ ~/opticks/ana/gs.py $PWD/gs.npy 
    [2021-02-12 22:28:55,808] p9659 {/home/simon/opticks/ana/gs.py:66} INFO - check_counts
    num_gensteps : 3270 
    num_photons  : 4470236 
     (2)G4Scintillation_1042      : ngs: 1666  npho:4313831 
     (1)G4Cerenkov_1042           : ngs: 1604  npho:156405 
     (0)TOTALS                    : ngs: 3270  npho:4470236 
    [2021-02-12 22:28:55,809] p9659 {/home/simon/opticks/ana/gs.py:89} INFO - check_pdgcode
    [[18446744073709551603                 2930]
     [                  11                  340]]
     18446744073709551603 : INVALID CODE : 2930 
          11 :         e- : 340 
    [2021-02-12 22:28:55,809] p9659 {/home/simon/opticks/ana/gs.py:98} INFO - check_ranges
     tr     0.0000     1.6765 
     xr    -7.9757    12.7474 
     yr    -2.3427     1.3077 
     zr     0.0000   499.9060 
    (base) [simon@localhost 2]$ 



::

    (base) [simon@localhost 2]$ ~/opticks/ana/gs.py /tmp/simon/opticks/source/evt/g4live/natural/?/gs.npy
    [2021-02-12 23:05:39,117] p64721 {/home/simon/opticks/ana/gs.py:48} INFO -  path /tmp/simon/opticks/source/evt/g4live/natural/2/gs.npy shape (3270, 6, 4) 
    [2021-02-12 23:05:39,117] p64721 {/home/simon/opticks/ana/gs.py:68} INFO - check_counts
    num_gensteps : 3270 
    num_photons  : 4470236 
     (2)G4Scintillation_1042      : ngs: 1666  npho:4313831 
     (1)G4Cerenkov_1042           : ngs: 1604  npho:156405 
     (0)TOTALS                    : ngs: 3270  npho:4470236 
    [2021-02-12 23:05:39,120] p64721 {/home/simon/opticks/ana/gs.py:92} INFO - check_pdgcode
         -13 :        mu+ : 2930 
          11 :         e- : 340 
    [2021-02-12 23:05:39,121] p64721 {/home/simon/opticks/ana/gs.py:106} INFO - check_ranges
     tr     0.0000     1.6765 
     xr    -7.9757    12.7474 
     yr    -2.3427     1.3077 
     zr     0.0000   499.9060 
    [2021-02-12 23:05:39,123] p64721 {/home/simon/opticks/ana/gs.py:48} INFO -  path /tmp/simon/opticks/source/evt/g4live/natural/3/gs.npy shape (3057, 6, 4) 
    [2021-02-12 23:05:39,123] p64721 {/home/simon/opticks/ana/gs.py:68} INFO - check_counts
    num_gensteps : 3057 
    num_photons  : 4092944 
     (2)G4Scintillation_1042      : ngs: 1539  npho:3943220 
     (1)G4Cerenkov_1042           : ngs: 1518  npho:149724 
     (0)TOTALS                    : ngs: 3057  npho:4092944 
    [2021-02-12 23:05:39,124] p64721 {/home/simon/opticks/ana/gs.py:92} INFO - check_pdgcode
         -13 :        mu+ : 2920 
          11 :         e- : 137 
    [2021-02-12 23:05:39,125] p64721 {/home/simon/opticks/ana/gs.py:106} INFO - check_ranges
     tr     0.0000     1.6782 
     xr    -2.5158     0.4160 
     yr    -5.1557     3.3601 
     zr     0.0000   499.9178 
    [2021-02-12 23:05:39,127] p64721 {/home/simon/opticks/ana/gs.py:48} INFO -  path /tmp/simon/opticks/source/evt/g4live/natural/4/gs.npy shape (3453, 6, 4) 
    [2021-02-12 23:05:39,127] p64721 {/home/simon/opticks/ana/gs.py:68} INFO - check_counts
    num_gensteps : 3453 
    num_photons  : 4657689 
     (2)G4Scintillation_1042      : ngs: 1746  npho:4489952 
     (1)G4Cerenkov_1042           : ngs: 1707  npho:167737 
     (0)TOTALS                    : ngs: 3453  npho:4657689 
    [2021-02-12 23:05:39,128] p64721 {/home/simon/opticks/ana/gs.py:92} INFO - check_pdgcode
         -13 :        mu+ : 2918 
          11 :         e- : 535 
    [2021-02-12 23:05:39,128] p64721 {/home/simon/opticks/ana/gs.py:106} INFO - check_ranges
     tr     0.0000     2.1488 
     xr   -20.5331    70.5315 
     yr  -110.8419   142.0468 
     zr     0.0000   499.7828 
    [2021-02-12 23:05:39,130] p64721 {/home/simon/opticks/ana/gs.py:48} INFO -  path /tmp/simon/opticks/source/evt/g4live/natural/5/gs.npy shape (3459, 6, 4) 
    [2021-02-12 23:05:39,130] p64721 {/home/simon/opticks/ana/gs.py:68} INFO - check_counts
    num_gensteps : 3459 
    num_photons  : 4751552 
     (2)G4Scintillation_1042      : ngs: 1762  npho:4587457 
     (1)G4Cerenkov_1042           : ngs: 1697  npho:164095 
     (0)TOTALS                    : ngs: 3459  npho:4751552 
    [2021-02-12 23:05:39,131] p64721 {/home/simon/opticks/ana/gs.py:92} INFO - check_pdgcode
         -13 :        mu+ : 2932 
          11 :         e- : 527 
    [2021-02-12 23:05:39,131] p64721 {/home/simon/opticks/ana/gs.py:106} INFO - check_ranges
     tr     0.0000     1.6762 
     xr   -15.5851     6.8342 
     yr   -13.4175     3.9259 
     zr     0.0000   499.9221 
    (base) [simon@localhost 2]$ 




Hmm this suggests an obvious debug approach. Switch off collection of cerenkov and then scintillation gensteps.

