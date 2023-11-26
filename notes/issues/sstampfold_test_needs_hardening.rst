FIXED : sstampfold_test_needs_hardening
==========================================


Crashes when run from directory with no NPFold::

    cd /usr/local/opticks/build/sysrap
    sstampfold_test



::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff64d67b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff64f32080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff64cc31ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff64c8b1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000100053489 sstampfold_test`NP* NP::LoadFromTxtFile<double>(spec_or_path="@/usr/local/opticks/build/sysrap/CMakeCache.txt") at NP.hh:5663
        frame #5: 0x000000010005319a sstampfold_test`NP* NP::LoadFromTxtFile<double>(base="@/usr/local/opticks/build/sysrap", relp="CMakeCache.txt") at NP.hh:5634
        frame #6: 0x0000000100052faa sstampfold_test`NPFold::load_array(this=0x0000000101201350, _base="@/usr/local/opticks/build/sysrap", relp="CMakeCache.txt") at NPFold.h:1486
        frame #7: 0x000000010004dd67 sstampfold_test`NPFold::load_dir(this=0x0000000101201350, _base="@/usr/local/opticks/build/sysrap") at NPFold.h:1606
        frame #8: 0x000000010004cbbc sstampfold_test`NPFold::load(this=0x0000000101201350, _base="@/usr/local/opticks/build/sysrap") at NPFold.h:1674
        frame #9: 0x000000010004af8d sstampfold_test`NPFold::LoadNoData_(base_="/usr/local/opticks/build/sysrap") at NPFold.h:422
        frame #10: 0x0000000100027f4c sstampfold_test`NPFold::LoadNoData(base_="/usr/local/opticks/build/sysrap") at NPFold.h:463
        frame #11: 0x0000000100027864 sstampfold_test`main(argc=1, argv=0x00007ffeefbfe978) at sstampfold_test.cc:65
        frame #12: 0x00007fff64c17015 libdyld.dylib`start + 1
        frame #13: 0x00007fff64c17015 libdyld.dylib`start + 1
    (lldb) ^D

