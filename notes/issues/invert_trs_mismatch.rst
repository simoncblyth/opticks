Invert TRS Mismatch
======================

Need improved matrix comparison, with separate handling of translation and rotation portions.

::

    simon:opticks blyth$ tboolean-;tboolean-0-deserialize 
    (lldb) target create "NCSGDeserializeTest"
    Current executable set to 'NCSGDeserializeTest' (x86_64).
    (lldb) settings set -- target.run-args  "/tmp/blyth/opticks/tboolean-0-"
    (lldb) r
    Process 37780 launched: '/usr/local/opticks/lib/NCSGDeserializeTest' (x86_64)
    2017-05-02 21:10:09.568 INFO  [3740425] [main@37]  argc 2 argv[0] NCSGDeserializeTest VERBOSITY 0
    2017-05-02 21:10:09.577 INFO  [3740425] [NTxt::read@63] NTxt::read  path /tmp/blyth/opticks/tboolean-0-/csg.txt lines 12231
    2017-05-02 21:10:09.577 INFO  [3740425] [NCSG::Deserialize@715] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tboolean-0- txtpath /tmp/blyth/opticks/tboolean-0-/csg.txt nbnd 12231
    2017-05-02 21:10:09.577 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/12230
    2017-05-02 21:10:09.578 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/12229
    2017-05-02 21:10:09.578 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/12228
    2017-05-02 21:10:09.579 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/12227
    ...
    2017-05-02 21:10:09.818 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/11418
    2017-05-02 21:10:09.818 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/11417
    2017-05-02 21:10:09.819 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/11416
    2017-05-02 21:10:09.819 INFO  [3740425] [NCSG::load@208] NCSG::load /tmp/blyth/opticks/tboolean-0-/11415
    nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  diff 0.00012207 diff2 0.00012207 diffFractional 2
           trs  0.354   0.354  -0.866   0.000 
                0.612   0.612   0.500   0.000 
                0.707  -0.707   0.000   0.000 
              6342.316 -2384.416 3909.600   1.000 

        isirit  0.354   0.612   0.707   0.000 
                0.354   0.612  -0.707   0.000 
               -0.866   0.500   0.000   0.000 
              1986.484 -4378.509 -6170.732   1.000 

        i_trs   0.354   0.612   0.707  -0.000 
                0.354   0.612  -0.707   0.000 
               -0.866   0.500   0.000  -0.000 
              1986.484 -4378.509 -6170.732   1.000 

    [  0.353553:  0.353553:         0:         1][  0.612372:  0.612372:         0:         1][  0.707107:  0.707107:         0:         1][         0:        -0:         0:       nan]
    [  0.353553:  0.353553:         0:         1][  0.612372:  0.612372:         0:         1][ -0.707107: -0.707107:         0:         1][         0:         0:         0:       nan]
    [ -0.866025: -0.866025:         0:         1][       0.5:       0.5:5.96046e-08:         1][2.16489e-17:         0:2.16489e-17:       inf][         0:        -0:         0:       nan]
    [   1986.48:   1986.48:0.00012207:         1][  -4378.51:  -4378.51:         0:         1][  -6170.73:  -6170.73:         0:         1][         1:         1:         0:         1]
    Assertion failed: (match), function invert_trs, file /Users/blyth/opticks/opticksnpy/NGLMExt.cpp, line 154.
    Process 37780 stopped
    * thread #1: tid = 0x391309, 0x00007fff96061866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff96061866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff96061866:  jae    0x7fff96061870            ; __pthread_kill + 20
       0x7fff96061868:  movq   %rax, %rdi
       0x7fff9606186b:  jmp    0x7fff9605e175            ; cerror_nocancel
       0x7fff96061870:  retq   
    (lldb) bt
    * thread #1: tid = 0x391309, 0x00007fff96061866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff96061866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8d6fe35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff9444eb1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff944189bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001008bd2db libNPY.dylib`nglmext::invert_trs(trs=0x00007fff5fbfd6b0)0> const&) + 2667 at NGLMExt.cpp:154
        frame #5: 0x00000001006cdcbe libNPY.dylib`NPY<float>::make_triple_transforms(src=0x00000001034ee720) + 238 at NPY.cpp:952
        frame #6: 0x000000010089047a libNPY.dylib`NCSG::loadTransforms(this=0x00000001034ee2d0) + 1450 at NCSG.cpp:169
        frame #7: 0x0000000100890f3e libNPY.dylib`NCSG::load(this=0x00000001034ee2d0) + 510 at NCSG.cpp:213
        frame #8: 0x00000001008952a1 libNPY.dylib`NCSG::Deserialize(basedir=0x00007fff5fbff124, trees=0x00007fff5fbfeb30, verbosity=0) + 2353 at NCSG.cpp:734
        frame #9: 0x0000000100004d4b NCSGDeserializeTest`test_Deserialize(basedir=0x00007fff5fbff124, verbosity=0) + 171 at NCSGDeserializeTest.cc:27
        frame #10: 0x0000000100005194 NCSGDeserializeTest`main(argc=2, argv=0x00007fff5fbfee00) + 916 at NCSGDeserializeTest.cc:42
        frame #11: 0x00007fff914d45fd libdyld.dylib`start + 1
    (lldb) f 6
    frame #6: 0x000000010089047a libNPY.dylib`NCSG::loadTransforms(this=0x00000001034ee2d0) + 1450 at NCSG.cpp:169
       166                
       167      assert(NTRAN == 2 || NTRAN == 3);
       168  
    -> 169      NPY<float>* transforms = NTRAN == 2 ? NPY<float>::make_paired_transforms(src) : NPY<float>::make_triple_transforms(src) ;
       170      assert(transforms->hasShape(ni,NTRAN,4,4));
       171  
       172      m_transforms = transforms ; 
    (lldb) f 5
    frame #5: 0x00000001006cdcbe libNPY.dylib`NPY<float>::make_triple_transforms(src=0x00000001034ee720) + 238 at NPY.cpp:952
       949       for(unsigned i=0 ; i < ni ; i++)
       950       {
       951           glm::mat4 t = src->getMat4(i);  
    -> 952           glm::mat4 v = nglmext::invert_trs( t );
       953           glm::mat4 q = glm::transpose( v ) ;
       954  
       955           dst->setMat4(t, i, 0 );
    (lldb) 





