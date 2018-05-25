[FIXED] G4NuclideTable-exception-missing-G4-data
==================================================

Fix
-----

The envvars are set but corresponding data is missing.
Fixed by re-installing::

    g4-;g4-bcd
    make install



Check the cause
------------------

::

    (lldb) r
    Process 38529 launched: '/usr/local/opticks-cmake-overhaul/lib/CRandomEngineTest' (x86_64)
    2018-05-25 15:39:22.121 INFO  [8626714] [main@72] CRandomEngineTest
    2018-05-25 15:39:22.122 INFO  [8626714] [main@76]  pindex 0
      0 : CRandomEngineTest

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70001
          issued by : G4NuclideTable
    ENSDFSTATE.dat is not found.
    *** Fatal Exception *** core dump ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


    *** G4Exception: Aborting execution ***
    Process 38529 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (CRandomEngineTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff561cc080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff55f5d1ae libsystem_c.dylib`abort + 127
        frame #3: 0x000000010328b32c libG4global.dylib`G4Exception(originOfException="G4NuclideTable", exceptionCode=<unavailable>, severity=FatalException, description="ENSDFSTATE.dat is not found.") at G4Exception.cc:100 [opt]
        frame #4: 0x0000000102d6fdc9 libG4particles.dylib`G4NuclideTable::GenerateNuclide(this=0x0000000102dffbe8) at G4NuclideTable.cc:183 [opt]
        frame #5: 0x0000000102d6faec libG4particles.dylib`G4NuclideTable::G4NuclideTable(this=0x0000000102dffbe8) at G4NuclideTable.cc:73 [opt]
        frame #6: 0x0000000102d6f97c libG4particles.dylib`G4NuclideTable::GetInstance() [inlined] G4NuclideTable::G4NuclideTable(this=<unavailable>) at G4NuclideTable.cc:68 [opt]
        frame #7: 0x0000000102d6f970 libG4particles.dylib`G4NuclideTable::GetInstance() at G4NuclideTable.cc:56 [opt]
        frame #8: 0x0000000102d5c1dd libG4particles.dylib`G4IonTable::G4IonTable() [inlined] G4NuclideTable::GetNuclideTable() at G4NuclideTable.hh:73 [opt]
        frame #9: 0x0000000102d5c1d8 libG4particles.dylib`G4IonTable::G4IonTable() [inlined] G4IonTable::PrepareNuclideTable(this=<unavailable>) at G4IonTable.cc:1563 [opt]
        frame #10: 0x0000000102d5c1d8 libG4particles.dylib`G4IonTable::G4IonTable() [inlined] G4IonTable::G4IonTable(this=0x000000010c5aca50) at G4IonTable.cc:140 [opt]
        frame #11: 0x0000000102d5c149 libG4particles.dylib`G4IonTable::G4IonTable(this=0x000000010c5aca50) at G4IonTable.cc:121 [opt]
        frame #12: 0x0000000102d797c4 libG4particles.dylib`G4ParticleTable::G4ParticleTable(this=0x0000000102dffcb8) at G4ParticleTable.cc:145 [opt]
        frame #13: 0x0000000102d7938c libG4particles.dylib`G4ParticleTable::GetParticleTable() [inlined] G4ParticleTable::G4ParticleTable(this=<unavailable>) at G4ParticleTable.cc:116 [opt]
        frame #14: 0x0000000102d79380 libG4particles.dylib`G4ParticleTable::GetParticleTable() at G4ParticleTable.cc:98 [opt]
        frame #15: 0x0000000100deb010 libG4run.dylib`G4RunManagerKernel::G4RunManagerKernel(this=0x000000010c5afc60) at G4RunManagerKernel.cc:99 [opt]
        frame #16: 0x0000000100ddd83b libG4run.dylib`G4RunManager::G4RunManager(this=0x000000010c5afaf0) at G4RunManager.cc:105 [opt]
        frame #17: 0x000000010011ac6b libCFG4.dylib`CPhysics::CPhysics(this=0x000000010c5afac0, g4=0x000000010c5af910) at CPhysics.cc:19
        frame #18: 0x000000010011ad6d libCFG4.dylib`CPhysics::CPhysics(this=0x000000010c5afac0, g4=0x000000010c5af910) at CPhysics.cc:25
        frame #19: 0x00000001001de365 libCFG4.dylib`CG4::CG4(this=0x000000010c5af910, hub=0x00007ffeefbfe430) at CG4.cc:164
        frame #20: 0x00000001001decad libCFG4.dylib`CG4::CG4(this=0x000000010c5af910, hub=0x00007ffeefbfe430) at CG4.cc:185
        frame #21: 0x000000010000f3fb CRandomEngineTest`main(argc=1, argv=0x00007ffeefbfe9a8) at CRandomEngineTest.cc:83
        frame #22: 0x00007fff55eb1015 libdyld.dylib`start + 1
        frame #23: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) 

::

    (lldb) f 4
    frame #4: 0x0000000102d6fdc9 libG4particles.dylib`G4NuclideTable::GenerateNuclide(this=0x0000000102dffbe8) at G4NuclideTable.cc:183 [opt]
       180 	      ifs.open( filename.c_str() );
       181 	
       182 	      if ( !ifs.good() ) {
    -> 183 	         G4Exception("G4NuclideTable", "PART70001",
       184 	                     FatalException, "ENSDFSTATE.dat is not found.");
       185 	         return;
       186 	      }
    (lldb) 


::

    160 ///////////////////////////////////////////////////////////////////////////////
    161 void G4NuclideTable::GenerateNuclide()
    162 {
    163 
    164    if ( threshold_of_half_life < minimum_threshold_of_half_life ) {
    165 
    166       // Need to update full list
    167 
    168       char* path = getenv("G4ENSDFSTATEDATA");
    169 
    170       if ( !path ) {
    171          G4Exception("G4NuclideTable", "PART70000",
    172                      FatalException, "G4ENSDFSTATEDATA environment variable must be set");
    173          return;
    174       }
    175 
    176       std::ifstream ifs;
    177       G4String filename(path);
    178       filename += "/ENSDFSTATE.dat";
    179 
    180       ifs.open( filename.c_str() );
    181 
    182       if ( !ifs.good() ) {
    183          G4Exception("G4NuclideTable", "PART70001",
    184                      FatalException, "ENSDFSTATE.dat is not found.");
    185          return;
    186       }
    187 

::

    (lldb) p getenv("G4ENSDFSTATEDATA")
    error: 'getenv' has unknown return type; cast the call to its declared return type
    (lldb) p (char*)getenv("G4ENSDFSTATEDATA")
    (char *) $0 = 0x0000000107a12d11 "/usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ENSDFSTATE1.2.1"
    (lldb) 

    epsilon:issues blyth$ ll /usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ENSDFSTATE1.2.1
    ls: /usr/local/opticks/externals/share/Geant4-10.2.1/data/G4ENSDFSTATE1.2.1: No such file or directory



Hmm did bcm stomp on the G4 data ?::

    epsilon:issues blyth$ ll /usr/local/opticks/externals/share/
    total 0
    drwxr-xr-x  25 blyth  staff  800 May 24 15:29 ..
    drwxr-xr-x   3 blyth  staff   96 May 24 15:29 .
    drwxr-xr-x   3 blyth  staff   96 May 24 15:29 bcm
    epsilon:issues blyth$ 


Using the G4 Makefile generated by CMake redo the install : recovers the data
--------------------------------------------------------------------------------

::

    epsilon:issues blyth$ g4-bcd
    make help 
    make install   

    epsilon:geant4_10_02_p01.Debug.build blyth$ ll /usr/local/opticks/externals/share/
    total 0
    drwxr-xr-x  25 blyth  staff  800 May 24 15:29 ..
    drwxr-xr-x   3 blyth  staff   96 May 24 15:29 bcm
    drwxr-xr-x   4 blyth  staff  128 May 25 15:54 .
    drwxr-xr-x   7 blyth  staff  224 May 25 15:55 Geant4-10.2.1
    epsilon:geant4_10_02_p01.Debug.build blyth$ 
    epsilon:geant4_10_02_p01.Debug.build blyth$ 


It survives a bcm install, must have been accidental removal::

    epsilon:geant4_10_02_p01.Debug.build blyth$ bcm-
    epsilon:geant4_10_02_p01.Debug.build blyth$ bcm--
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/opticks-cmake-overhaul/externals/bcm/bcm.build
    Install the project...
    -- Install configuration: ""
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMTest.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMInstallTargets.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMSetupVersion.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMIgnorePackage.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMFuture.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/version.hpp
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMPkgConfig.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMProperties.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMDeploy.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMExport.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMConfig.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul/externals/./share/bcm/cmake/BCMToSnakeCase.cmake
    epsilon:geant4_10_02_p01.Debug.build blyth$ 
    epsilon:geant4_10_02_p01.Debug.build blyth$ ll /usr/local/opticks/externals/share/
    total 0
    drwxr-xr-x  25 blyth  staff  800 May 24 15:29 ..
    drwxr-xr-x   3 blyth  staff   96 May 24 15:29 bcm
    drwxr-xr-x   4 blyth  staff  128 May 25 15:54 .
    drwxr-xr-x   7 blyth  staff  224 May 25 15:55 Geant4-10.2.1
    epsilon:geant4_10_02_p01.Debug.build blyth$ 



