IDPath_still_in_the_wild
===========================


::

    epsilon:tests blyth$ lldb__ U4PMTFastSimGeomTest
    /Applications/Xcode/Xcode_10_1.app/Contents/Developer/usr/bin/lldb -f U4PMTFastSimGeomTest --
    (lldb) target create "/usr/local/opticks/lib/U4PMTFastSimGeomTest"
    Current executable set to '/usr/local/opticks/lib/U4PMTFastSimGeomTest' (x86_64).
    (lldb) r
    Process 16783 launched: '/usr/local/opticks/lib/U4PMTFastSimGeomTest' (x86_64)
    *U4VolumeMaker::PVP_@210:  not-WITH_PMTSIM name [BoxOfScintillator]
    SDir::List FAILED TO OPEN DIR /tmp/blyth/opticks/GScintillatorLib/LS_ori
    Process 16783 stopped
    Target 0: (U4PMTFastSimGeomTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGINT
      * frame #0: 0x00007fff70412b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff705dd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff703206fe libsystem_c.dylib`raise + 26
        frame #3: 0x000000010016e516 libU4.dylib`SDir::List(names=size=0, path="/tmp/blyth/opticks/GScintillatorLib/LS_ori", ext=".npy") at SDir.h:38
        frame #4: 0x000000010016d73c libU4.dylib`U4Material::MakeMaterialPropertiesTable(matdir_="$IDPath/GScintillatorLib/LS_ori") at U4Material.cc:431
        frame #5: 0x0000000100172f76 libU4.dylib`U4Material::MakeMaterial(name="LS", matdir="$IDPath/GScintillatorLib/LS_ori") at U4Material.cc:674
        frame #6: 0x0000000100165ffb libU4.dylib`U4Material::MakeScintillator() at U4Material.cc:695
        frame #7: 0x0000000100165fa2 libU4.dylib`U4Material::Get_(name="SCINTILLATOR") at U4Material.cc:107
        frame #8: 0x0000000100165f36 libU4.dylib`U4Material::Get(name="SCINTILLATOR") at U4Material.cc:100
        frame #9: 0x000000010019bab4 libU4.dylib`U4VolumeMaker::Box_(halfside=1000, mat="SCINTILLATOR", prefix="BoxOfScintillator") at U4VolumeMaker.cc:831
        frame #10: 0x000000010019e0ae libU4.dylib`U4VolumeMaker::Box(halfside=1000, mat="SCINTILLATOR", prefix="BoxOfScintillator", mother_lv=0x0000000000000000) at U4VolumeMaker.cc:685
        frame #11: 0x000000010019e107 libU4.dylib`U4VolumeMaker::BoxOfScintillator(halfside=1000, prefix="BoxOfScintillator", mother_lv=0x0000000000000000) at U4VolumeMaker.cc:680
        frame #12: 0x0000000100198892 libU4.dylib`U4VolumeMaker::BoxOfScintillator(halfside=1000) at U4VolumeMaker.cc:676
        frame #13: 0x0000000100196d5a libU4.dylib`U4VolumeMaker::PVS_(name="BoxOfScintillator") at U4VolumeMaker.cc:259
        frame #14: 0x0000000100195fb4 libU4.dylib`U4VolumeMaker::PV(name="BoxOfScintillator") at U4VolumeMaker.cc:104
        frame #15: 0x0000000100195e00 libU4.dylib`U4VolumeMaker::PV() at U4VolumeMaker.cc:96
        frame #16: 0x000000010000c0bd U4PMTFastSimGeomTest`main(argc=1, argv=0x00007ffeefbfe8e0) at U4PMTFastSimGeomTest.cc:8
        frame #17: 0x00007fff702c2015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x000000010016d73c libU4.dylib`U4Material::MakeMaterialPropertiesTable(matdir_="$IDPath/GScintillatorLib/LS_ori") at U4Material.cc:431
       428 	{
       429 	    const char* matdir = SPath::Resolve(matdir_, NOOP); 
       430 	    std::vector<std::string> names ; 
    -> 431 	    SDir::List(names, matdir, ".npy" ); 
       432 	
       433 	    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
       434 	
    (lldb) 


