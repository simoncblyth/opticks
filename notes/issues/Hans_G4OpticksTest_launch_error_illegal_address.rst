Hans_G4OpticksTest_launch_error_illegal_address
================================================


::

    020-12-15 20:47:51.103 FATAL [2442339] [OpPropagator::propagate@84] evtId(2) DONE nhit: 32331
    EventAction::EndOfEventAction num_hits 32331   m_num_hits: 32331 hits 0x555558224c70
    Event:   3
    2020-12-15 20:47:51.125 FATAL [2442339] [OpPropagator::propagate@73] evtId(3) OK COMPUTE PRODUCTION
    2020-12-15 20:47:51.126 INFO  [2442339] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@174] SEEDING TO SEED BUF  
    2020-12-15 20:47:51.127 INFO  [2442339] [OPropagator::launch@266] LAUNCH NOW   printLaunchIndex ( -1 -1 -1) -
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, stream.get() ) returned (700): Illegal address, file: <internal>, line: 0)

    Thread 1 "G4OpticksTest" received signal SIGABRT, Aborted.
    __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
    50 ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) backtrace
    #0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
    #1  0x00007fffeced0859 in __GI_abort () at abort.c:79
    #2  0x00007fffed42d951 in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #3  0x00007fffed43947c in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #4  0x00007fffed4394e7 in std::terminate() ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #5  0x00007fffed439799 in __cxa_throw ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007fffee50b43e in optix::ContextObj::checkError (this=0x555556dd3370,
        code=RT_ERROR_UNKNOWN)
        at /home/wenzel/NVIDIA-OptiX-SDK-6.5.0-linux64/include/optixu/optixpp_namespace.h:2219
    #7  0x00007fffee51e3d2 in optix::ContextObj::launch (this=0x555556dd3370,
        entry_point_index=0, image_width=4657689, image_height=1)
        at /home/wenzel/NVIDIA-OptiX-SDK-6.5.0-linux64/include/optixu/optixpp_namespace.h:3006
    #8  0x00007fffee51c758 in OContext::launch_ (this=0x555556e60020, entry=0,
        width=4657689, height=1)
        at /data2/wenzel/gputest2/opticks/optixrap/OContext.cc:855
    #9  0x00007fffee51c465 in OContext::launch (this=0x555556e60020, lmode=16,
        entry=0, width=4657689, height=1, times=0x5555580824f0)
        at /data2/wenzel/gputest2/opticks/optixrap/OContext.cc:817
    #10 0x00007fffee536ed3 in OPropagator::launch (this=0x5555575ede80)
    --Type <RET> for more, q to quit, c to continue without paging--
        at /data2/wenzel/gputest2/opticks/optixrap/OPropagator.cc:269
    #11 0x00007fffee709546 in OpEngine::propagate (this=0x555556bf2610)
        at /data2/wenzel/gputest2/opticks/okop/OpEngine.cc:211
    #12 0x00007fffee70cc9e in OpPropagator::propagate (this=0x555556bf3110)
        at /data2/wenzel/gputest2/opticks/okop/OpPropagator.cc:77
    #13 0x00007fffee70ab10 in OpMgr::propagate (this=0x555556c00c40)
        at /data2/wenzel/gputest2/opticks/okop/OpMgr.cc:138
    #14 0x00007ffff7fb8503 in G4Opticks::propagateOpticalPhotons (
        this=0x5555567f44c0, eventID=3)
        at /data2/wenzel/gputest2/opticks/g4ok/G4Opticks.cc:920
    #15 0x00005555555780af in EventAction::EndOfEventAction (this=0x555555a7a010,
        event=0x555556c79080)





Try to reproduce this on precision.
-------------------------------------

::

    [blyth@localhost ~]$ git clone https://github.com/hanswenzel/G4OpticksTest
    [blyth@localhost ~]$ cd G4OpticksTest
    [blyth@localhost G4OpticksTest]$ chmod ugo+x go.sh 
    [blyth@localhost G4OpticksTest]$ ./go.sh 

    ...

    In file included from /home/blyth/G4OpticksTest/src/RadiatorSD.cc:24:0:
    /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/include/Geant4/G4Cerenkov.hh: In member function ‘virtual G4bool RadiatorSD::ProcessHits(G4Step*, G4TouchableHistory*)’:
    /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/include/Geant4/G4Cerenkov.hh:201:12: error: ‘G4double G4Cerenkov::GetAverageNumberOfPhotons(G4double, G4double, const G4Material*, G4MaterialPropertyVector*) const’ is private
       G4double GetAverageNumberOfPhotons(const G4double charge,
                ^


Avoided the private method::

    -                        MeanNumberOfPhotons1 = proc-> GetAverageNumberOfPhotons(charge, beta1, aMaterial, Rindex);
    -                        MeanNumberOfPhotons2 = proc-> GetAverageNumberOfPhotons(charge, beta2, aMaterial, Rindex);
    +                        MeanNumberOfPhotons1 = 100.0 ; // proc-> GetAverageNumberOfPhotons(charge, beta1, aMaterial, Rindex);
    +                        MeanNumberOfPhotons2 = 100.0 ; // proc-> GetAverageNumberOfPhotons(charge, beta2, aMaterial, Rindex);



But still a problem with old Geant4 1042::

    ...

    [ 66%] Built target G4OpticksTest
    [ 67%] Linking CXX shared library libG4OpticksTestClassesDict.so
    /usr/bin/ld: anHCAllocator_G4MT_TLS_: TLS definition in /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so section .tbss mismatches non-TLS reference in CMakeFiles/G4OpticksTestClassesDict.dir/src/PhotonSD.cc.o
    /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so: error adding symbols: Bad value
    collect2: error: ld returned 1 exit status
    make[2]: *** [libG4OpticksTestClassesDict.so] Error 1
    make[1]: *** [CMakeFiles/G4OpticksTestClassesDict.dir/all] Error 2
    make: *** [all] Error 2




Clear up some space then::

    [blyth@localhost ~]$ g4-;g4--1062


Nope::

    [ 13%] Building CXX object source/geometry/CMakeFiles/G4geometry.dir/magneticfield/src/G4TrialsCounter.cc.o
    -- [download 13% complete]
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/geometry/magneticfield/src/G4FieldManagerStore.cc: In static member function ‘static void G4FieldManagerStore::DeRegister(G4FieldManager*)’:
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/geometry/magneticfield/src/G4FieldManagerStore.cc:119:31: error: no matching function for call to ‘G4FieldManagerStore::erase(__gnu_cxx::__normal_iterator<G4FieldManager* const*, std::vector<G4FieldManager*> >&)’
             GetInstance()->erase(i);
                                   ^

* https://geant4-forum.web.cern.ch/t/error-when-making-geant4/1774

gcosmo::

    You must use a more recent gcc compiler to build Geant4 10.6.
    The minimum required is gcc-4.9.3 and you’re using gcc-4.8.2…i

See env-;centos- for notes on yum installation of devtoolset-9 which comes with gcc 9.3.1 


     

