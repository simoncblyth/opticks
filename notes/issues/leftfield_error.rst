Leftfield Error Following BFile change attempt
================================================


Argh, back again in::

    tboolean-;tboolean-hybrid
    tboolean-;tboolean-hyctrl

Argh flakey, its gone away::

    tboolean-;tboolean-prism --debugger --NPY debug


Observe that removing the torch option prevents the issue.


Guard Malloc : points to GGeo instanciation
-------------------------------------------------

* GGeo is definitely in need to a clean up, far too much of a mixed bag scope...

Hmm suspect that doing the below ifdef-ing within a "dirty" header is asking for trouble...
instead just remove the ifdef and intialize NULL...  Do ifdef in implementation, not header.

::

    377 #ifdef WITH_YoctoGL
    378     private:
    379         // glTF route                      
    380         NScene*                            m_nscene ; 
    381         GScene*                            m_gscene ;
    382 #endif




::

    simon:gmalloctest blyth$ tboolean-;tboolean-hybrid --debugger
    288 -rwxr-xr-x  1 blyth  staff  143804 Jun 15 13:26 /usr/local/opticks/lib/OKTest
    proceeding : lldb /usr/local/opticks/lib/OKTest -- --debugger --animtimemax 10 --timemax 10 --geocenter --eye 0,0,1 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tboolean-hybrid--_name=tboolean-hybrid--_mode=PyCsgInBox --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --tag 1 --cat boolean --save
    (lldb) target create "/usr/local/opticks/lib/OKTest"
    Current executable set to '/usr/local/opticks/lib/OKTest' (x86_64).
    (lldb) settings set -- target.run-args  "--debugger" "--animtimemax" "10" "--timemax" "10" "--geocenter" "--eye" "0,0,1" "--dbganalytic" "--test" "--testconfig" "analytic=1_csgpath=/tmp/blyth/opticks/tboolean-hybrid--_name=tboolean-hybrid--_mode=PyCsgInBox" "--torch" "--torchconfig" "type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500" "--tag" "1" "--cat" "boolean" "--save"

    (lldb) env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib

    (lldb) r
    GuardMalloc[sh-30772]: Allocations will be placed on 16 byte boundaries.
    GuardMalloc[sh-30772]:  - Some buffer overruns may not be noticed.
    GuardMalloc[sh-30772]:  - Applications using vector instructions (e.g., SSE) should work.
    GuardMalloc[sh-30772]: version 27
    GuardMalloc[arch-30772]: Allocations will be placed on 16 byte boundaries.
    GuardMalloc[arch-30772]:  - Some buffer overruns may not be noticed.
    GuardMalloc[arch-30772]:  - Applications using vector instructions (e.g., SSE) should work.
    GuardMalloc[arch-30772]: version 27
    Process 30772 launched: '/usr/local/opticks/lib/OKTest' (x86_64)
    GuardMalloc[OKTest-30772]: Allocations will be placed on 16 byte boundaries.
    GuardMalloc[OKTest-30772]:  - Some buffer overruns may not be noticed.
    GuardMalloc[OKTest-30772]:  - Applications using vector instructions (e.g., SSE) should work.
    GuardMalloc[OKTest-30772]: version 27
    2017-06-15 14:18:31.525 INFO  [7624535] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    Process 30772 stopped
    * thread #1: tid = 0x745757, 0x0000000102065581 libGGeo.dylib`GGeo::GGeo(this=0x0000000109b0bd80, opticks=0x00000001075b5ed0) + 3617 at GGeo.cc:112, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x109b0c000)
        frame #0: 0x0000000102065581 libGGeo.dylib`GGeo::GGeo(this=0x0000000109b0bd80, opticks=0x00000001075b5ed0) + 3617 at GGeo.cc:112
       109     m_join_cfg(NULL),
       110     m_loader_verbosity(0),
       111     m_mesh_verbosity(0)
    -> 112  {
       113     init(); 
       114  }
       115  
    (lldb) 


::

    In [1]: this = 0x0000000109b0bd80

    In [2]: address = 0x109b0c000

    In [3]: address
    Out[3]: 4457545728

    In [4]: address - this
    Out[4]: 640


::

    (lldb) f 0
    frame #0: 0x0000000102065581 libGGeo.dylib`GGeo::GGeo(this=0x0000000109b0bd80, opticks=0x00000001075b5ed0) + 3617 at GGeo.cc:112
       109     m_join_cfg(NULL),
       110     m_loader_verbosity(0),
       111     m_mesh_verbosity(0)
    -> 112  {
       113     init(); 
       114  }
       115  
    (lldb) p m_mesh_verbosity 
    error: libDualContouringSample.dylib debug map object file '/usr/local/opticks/externals/DualContouringSample/dualcontouringsample.build/CMakeFiles/DualContouringSample.dir/octree.cpp.o' has changed (actual time is 0x5940de3b, debug map time is 0x5940ddac) since this executable was linked, file will be ignored
    error: libYoctoGL.dylib debug map object file '/usr/local/opticks/externals/yoctogl/yocto-gl.build/CMakeFiles/YoctoGL.dir/yocto/yocto_img.cpp.o' has changed (actual time is 0x5940df85, debug map time is 0x5940c67d) since this executable was linked, file will be ignored
    error: Couldn't apply expression side effects : Couldn't dematerialize a result variable: couldn't read its memory
    (lldb) p &m_mesh_verbosity 
    (unsigned int *) $1 = 0x0000000109b0c004
    (lldb) p this
    (GGeo *) $2 = 0x0000000109b0bd80
    (lldb) p &m_mesh_verbosity - this 
    error: 'unsigned int *' and 'GGeo *' are not pointers to compatible types
    error: 1 errors parsing expression
    (lldb) p (char*)&m_mesh_verbosity - (char*)this
    (long) $3 = 644
    (lldb) p (char*)&m_loader_verbosity - (char*)this
    (long) $4 = 640
    (lldb) p (char*)&m_join_cfg - (char*)this
    (long) $5 = 616
    (lldb) 




::

    373     private:
    374         std::map<unsigned int, GSolid*>    m_solidmap ;
    375         Index_t                            m_index ;
    376         unsigned int                       m_sensitive_count ;
    377         bool                               m_volnames ;
    378         GMaterial*                         m_cathode ;
    379         const char*                        m_join_cfg ;
    380         GJoinImpFunctionPtr                m_join_imp ;
    381         GLoaderImpFunctionPtr              m_loader_imp ;
    382         unsigned int                       m_loader_verbosity ;
    383         unsigned int                       m_mesh_verbosity ;
    384 
    385 };



::

    (lldb) p 0x109b0c000 - (char*)0x0000000109b0bd80
    error: libDualContouringSample.dylib debug map object file '/usr/local/opticks/externals/DualContouringSample/dualcontouringsample.build/CMakeFiles/DualContouringSample.dir/octree.cpp.o' has changed (actual time is 0x5940de3b, debug map time is 0x5940ddac) since this executable was linked, file will be ignored
    error: libYoctoGL.dylib debug map object file '/usr/local/opticks/externals/yoctogl/yocto-gl.build/CMakeFiles/YoctoGL.dir/yocto/yocto_img.cpp.o' has changed (actual time is 0x5940df85, debug map time is 0x5940c67d) since this executable was linked, file will be ignored
    error: invalid operands to binary expression ('long' and 'char *')
    error: 1 errors parsing expression
    (lldb) p (char*)0x109b0c000 - (char*)0x0000000109b0bd80
    (long) $0 = 640
    (lldb) 






Test Fails
-------------


Test fails seem unrelated to the flakiness. 
They are mostly too low level ?
Does that indicate the problem is up at very high integration level ?


TODO: get the high level ana-t tests working again


::


    97% tests passed, 6 tests failed out of 234

    Total Test time (real) = 116.68 sec

    The following tests FAILED:
        208 - OptiXRapTest.OEventTest (OTHER_FAULT)
        209 - OptiXRapTest.OInterpolationTest (Failed)
        213 - OKOPTest.OpSeederTest (OTHER_FAULT)
        220 - OKTest.VizTest (OTHER_FAULT)

        222 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        223 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output
    simon:opticks blyth$ 





::

    simon:Modules blyth$ tboolean-;tboolean-hybrid
    288 -rwxr-xr-x  1 blyth  staff  146000 Jun 14 16:32 /usr/local/opticks/lib/OKTest
    proceeding : 
          /usr/local/opticks/lib/OKTest
               --animtimemax 10 
               --timemax 10 
               --geocenter 
               --eye 0,0,1 
               --dbganalytic
                --test 
                --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tboolean-hybrid--_name=tboolean-hybrid--_mode=PyCsgInBox
                --torch 
                --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --tag 1 --cat boolean --save
    2017-06-14 16:37:51.836 INFO  [7408864] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    OKTest(81748,0x7fff75379310) malloc: *** error for object 0x7fb840c365b0: incorrect checksum for freed object - object was probably modified after being freed.
    *** set a breakpoint in malloc_error_break to debug
    /Users/blyth/opticks/bin/op.sh: line 580: 81748 Abort trap: 6           /usr/local/opticks/lib/OKTest --animtimemax 10 --timemax 10 --geocenter --eye 0,0,1 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tboolean-hybrid--_name=tboolean-hybrid--_mode=PyCsgInBox --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --tag 1 --cat boolean --save
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:Modules blyth$ 




Turning up verbosity makes the issue go away::

    tboolean-;tboolean-hybrid --NPY trace --BRAP --trace 







Flakeys give good motivation for getting all the ctests going again... 

::

    98% tests passed, 5 tests failed out of 234

    Total Test time (real) = 109.53 sec

    The following tests FAILED:
        208 - OptiXRapTest.OEventTest (OTHER_FAULT)        
        213 - OKOPTest.OpSeederTest (OTHER_FAULT)
        220 - OKTest.VizTest (OTHER_FAULT)
        222 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        223 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output
    simon:ggeo blyth$ 


First 2 from same cause::


    simon:opticks blyth$ OpSeederTest 
    2017-06-13 20:56:36.612 INFO  [7180182] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2017-06-13 20:56:36.993 FATAL [7180182] [GenstepNPY::addStep@73] GenstepNPY::addStep target MUST be set for non-dummy frameGenstepNPY  frameIndex 0 frameTargetted 0 frameTransform 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    Assertion failed: (target_acquired), function addStep, file /Users/blyth/opticks/opticksnpy/GenstepNPY.cpp, line 77.
    Abort trap: 6
    simon:opticks blyth$ 

    (lldb) target create "OpSeederTest"
    Current executable set to 'OpSeederTest' (x86_64).
    (lldb) r
    Process 78834 launched: '/usr/local/opticks/lib/OpSeederTest' (x86_64)
    2017-06-13 20:58:06.890 INFO  [7180972] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
        0 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        1 ce             gfloat4      0.005     -0.003    -18.252    146.252  bb bb min    -98.995    -99.003   -164.504  max     99.005     98.997    128.000 
        2 ce             gfloat4      0.005     -0.004     91.998     98.143  bb bb min    -98.138    -98.147     55.996  max     98.148     98.139    128.000 
        3 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        4 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
        0 ni[nf/nv/nidx/pidx] (720,362,3199,3155)  id[nidx,midx,bidx,sidx]  (3199, 47, 27,  0) 
        1 ni[nf/nv/nidx/pidx] (672,338,3200,3199)  id[nidx,midx,bidx,sidx]  (3200, 46, 28,  0) 
        2 ni[nf/nv/nidx/pidx] (960,482,3201,3200)  id[nidx,midx,bidx,sidx]  (3201, 43, 29,  3) 
        3 ni[nf/nv/nidx/pidx] (480,242,3202,3200)  id[nidx,midx,bidx,sidx]  (3202, 44, 30,  0) 
        4 ni[nf/nv/nidx/pidx] ( 96, 50,3203,3200)  id[nidx,midx,bidx,sidx]  (3203, 45, 30,  0) 
    2017-06-13 20:58:07.269 FATAL [7180972] [GenstepNPY::addStep@73] GenstepNPY::addStep target MUST be set for non-dummy frameGenstepNPY  frameIndex 0 frameTargetted 0 frameTransform 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    Assertion failed: (target_acquired), function addStep, file /Users/blyth/opticks/opticksnpy/GenstepNPY.cpp, line 77.
    Process 78834 stopped
    * thread #1: tid = 0x6d92ac, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) bt
    * thread #1: tid = 0x6d92ac, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8d3cf9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000100835cd9 libNPY.dylib`GenstepNPY::addStep(this=0x000000010591fc60, verbose=false) + 473 at GenstepNPY.cpp:77
        frame #5: 0x000000010083576f libNPY.dylib`FabStepNPY::init(this=0x000000010591fc60) + 111 at FabStepNPY.cpp:20
        frame #6: 0x00000001008356d6 libNPY.dylib`FabStepNPY::FabStepNPY(this=0x000000010591fc60, genstep_type=32768, num_step=10, num_photons_per_step=10) + 70 at FabStepNPY.cpp:10
        frame #7: 0x00000001008357b7 libNPY.dylib`FabStepNPY::FabStepNPY(this=0x000000010591fc60, genstep_type=32768, num_step=10, num_photons_per_step=10) + 39 at FabStepNPY.cpp:11
        frame #8: 0x0000000101103119 libOpticksGeometry.dylib`OpticksGen::makeFabstep(this=0x000000010591fbe0) + 73 at OpticksGen.cc:173
        frame #9: 0x0000000101102d72 libOpticksGeometry.dylib`OpticksGen::initInputGensteps(this=0x000000010591fbe0) + 690 at OpticksGen.cc:74
        frame #10: 0x0000000101102a85 libOpticksGeometry.dylib`OpticksGen::init(this=0x000000010591fbe0) + 21 at OpticksGen.cc:37
        frame #11: 0x0000000101102a63 libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x000000010591fbe0, hub=0x00007fff5fbfec28) + 131 at OpticksGen.cc:32
        frame #12: 0x0000000101102aad libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x000000010591fbe0, hub=0x00007fff5fbfec28) + 29 at OpticksGen.cc:33
        frame #13: 0x0000000101100026 libOpticksGeometry.dylib`OpticksHub::init(this=0x00007fff5fbfec28) + 118 at OpticksHub.cc:96
        frame #14: 0x00000001010fff00 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfec28, ok=0x00007fff5fbfec98) + 416 at OpticksHub.cc:81
        frame #15: 0x00000001011000dd libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfec28, ok=0x00007fff5fbfec98) + 29 at OpticksHub.cc:83
        frame #16: 0x0000000100005fff OpSeederTest`main(argc=1, argv=0x00007fff5fbfee58) + 799 at OpSeederTest.cc:52
        frame #17: 0x00007fff8a48b5fd libdyld.dylib`start + 1
        frame #18: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) 





Full Build : without optionals
--------------------------------

::

    /Users/blyth/opticks/optickscore/tests/OpticksBufferSpecTest.cc:20:48: error: use of undeclared identifier 'CFG4_G4VERSION_NUMBER'
        LOG(info) << "CFG4_G4VERSION_NUMBER : " << CFG4_G4VERSION_NUMBER ;
                        






Initial Indication of corruption in NSensorList 
-----------------------------------------------------

* BUT that position is probably random 

Changed from using the boost tokenizer to boost split but 
thats just moved the error elsewhere, so its a corruption issue.


::

    (lldb) r
    Process 46255 launched: '/usr/local/opticks/lib/OKTest' (x86_64)
    2017-06-13 18:51:22.847 INFO  [7087690] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    OKTest(46255,0x7fff75379310) malloc: *** error for object 0x105d14ed0: incorrect checksum for freed object - object was probably modified after being freed.
    *** set a breakpoint in malloc_error_break to debug
    Process 46255 stopped
    * thread #1: tid = 0x6c264a, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) bt
    * thread #1: tid = 0x6c264a, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff86e35690 libsystem_malloc.dylib`szone_error + 587
        frame #4: 0x00007fff86e33595 libsystem_malloc.dylib`szone_free_definite_size + 3011
        frame #5: 0x00000001007e7cc5 libNPY.dylib`boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >::~token_iterator(this=0x00007fff5fbfb370) + 37 at token_iterator.hpp:30
        frame #6: 0x00000001007e2295 libNPY.dylib`boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >::~token_iterator(this=0x00007fff5fbfb370) + 21 at token_iterator.hpp:30
        frame #7: 0x00000001007e1828 libNPY.dylib`std::__1::enable_if<(__is_forward_iterator<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >::value) && (is_constructible<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::iterator_traits<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >::reference>::value), void>::type std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >::assign<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >(boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >) [inlined] std::__1::iterator_traits<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >::difference_type std::__1::distance<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >(__first=token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<const char *>, std::__1::basic_string<char> > at 0x0000000000000000, __last=token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<const char *>, std::__1::basic_string<char> > at 0x0000000000000000) + 360 at iterator:503
        frame #8: 0x00000001007e1726 libNPY.dylib`std::__1::enable_if<(this=0x00007fff5fbfbb48, __first=<unavailable>, __last=<unavailable>) && (is_constructible<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::iterator_traits<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >::reference>::value), void>::type std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >::assign<boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >(boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, boost::token_iterator<boost::char_separator<char, std::__1::char_traits<char> >, std::__1::__wrap_iter<char const*>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >) + 102 at vector:1342
        frame #9: 0x00000001007df9a7 libNPY.dylib`NSensorList::read(this=0x0000000105d13ba0, path=0x0000000105d11ed0) + 2503 at NSensorList.cpp:138
        frame #10: 0x00000001007deeb6 libNPY.dylib`NSensorList::load(this=0x0000000105d13ba0, idpath_=0x0000000105d0ca70, ext=0x000000010208f9cc) + 4758 at NSensorList.cpp:113
        frame #11: 0x0000000102057e6e libGGeo.dylib`GGeo::init(this=0x0000000105d14890) + 1662 at GGeo.cc:418
        frame #12: 0x00000001020576f4 libGGeo.dylib`GGeo::GGeo(this=0x0000000105d14890, opticks=0x0000000105b21c10) + 3636 at GGeo.cc:113
        frame #13: 0x000000010205884d libGGeo.dylib`GGeo::GGeo(this=0x0000000105d14890, opticks=0x0000000105b21c10) + 29 at GGeo.cc:114
        frame #14: 0x00000001021921cd libOpticksGeometry.dylib`OpticksGeometry::init(this=0x0000000105d13b60) + 509 at OpticksGeometry.cc:90
        frame #15: 0x0000000102191fc6 libOpticksGeometry.dylib`OpticksGeometry::OpticksGeometry(this=0x0000000105d13b60, hub=0x0000000105d0c7d0) + 118 at OpticksGeometry.cc:68
        frame #16: 0x000000010219226d libOpticksGeometry.dylib`OpticksGeometry::OpticksGeometry(this=0x0000000105d13b60, hub=0x0000000105d0c7d0) + 29 at OpticksGeometry.cc:69
        frame #17: 0x0000000102196ef9 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x0000000105d0c7d0) + 377 at OpticksHub.cc:241
        frame #18: 0x00000001021960ad libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105d0c7d0) + 77 at OpticksHub.cc:94
        frame #19: 0x0000000102195fb0 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0c7d0, ok=0x0000000105b21c10) + 416 at OpticksHub.cc:81
        frame #20: 0x000000010219618d libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0c7d0, ok=0x0000000105b21c10) + 29 at OpticksHub.cc:83
        frame #21: 0x0000000103b051e6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe668, argc=23, argv=0x00007fff5fbfe740, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #22: 0x0000000103b0564b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe668, argc=23, argv=0x00007fff5fbfe740, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #23: 0x000000010000a93d OKTest`main(argc=23, argv=0x00007fff5fbfe740) + 1373 at OKTest.cc:60
        frame #24: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) f 9
    frame #9: 0x00000001007df9a7 libNPY.dylib`NSensorList::read(this=0x0000000105d13ba0, path=0x0000000105d11ed0) + 2503 at NSensorList.cpp:138
       135          if(line[0] == '#') continue ; 
       136  
       137          Tok_t tok(line, delim) ;
    -> 138          elem.assign(tok.begin(), tok.end());
       139          NSensor* sensor = createSensor(elem);
       140          if(sensor) add(sensor);
       141  
    (lldb) f 10

    (lldb) f 11
    frame #11: 0x0000000102057e6e libGGeo.dylib`GGeo::init(this=0x0000000105d14890) + 1662 at GGeo.cc:418
       415   
       416     m_sensor_list = new NSensorList();
       417  
    -> 418     m_sensor_list->load( idpath, "idmap");
       419  
       420  
       421     LOG(debug) << "GGeo::init loadSensorList " << m_sensor_list->description() ; 
    (lldb) p idpath
    (const char *) $0 = 0x0000000105d0ca70 "/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae"
    (lldb) 




Guard Malloc doesnt get along with OptiX
------------------------------------------

::

    2017-06-15 14:55:51.119 INFO  [7642183] [SLog::operator@15] OpticksViz::OpticksViz DONE
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Memory allocation failed
    Process 35852 stopped
    * thread #1: tid = 0x749c47, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) bt
    * thread #1: tid = 0x749c47, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8ccc5f31 libc++abi.dylib`abort_message + 257
        frame #4: 0x00007fff8cceb93a libc++abi.dylib`default_terminate_handler() + 240
        frame #5: 0x00007fff8d023322 libobjc.A.dylib`_objc_terminate() + 124
        frame #6: 0x00007fff8cce91d1 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff8cce8c5b libc++abi.dylib`__cxa_throw + 124
        frame #8: 0x0000000103498f33 libOptiXRap.dylib`optix::ContextObj::create() + 115 at optixpp_namespace.h:1872
        frame #9: 0x0000000103498df4 libOptiXRap.dylib`optix::Handle<optix::ContextObj>::create() + 20 at optixpp_namespace.h:166
        frame #10: 0x0000000103497782 libOptiXRap.dylib`OScene::init(this=0x0000123448dfdfa0) + 1602 at OScene.cc:97
        frame #11: 0x00000001034970d9 libOptiXRap.dylib`OScene::OScene(this=0x0000123448dfdfa0, hub=0x000000010820ef90) + 281 at OScene.cc:79
        frame #12: 0x0000000103498c6d libOptiXRap.dylib`OScene::OScene(this=0x0000123448dfdfa0, hub=0x000000010820ef90) + 29 at OScene.cc:81
        frame #13: 0x0000000103a24bf6 libOKOP.dylib`OpEngine::OpEngine(this=0x0000123448df7fa0, hub=0x000000010820ef90) + 182 at OpEngine.cc:43
        frame #14: 0x0000000103a24f1d libOKOP.dylib`OpEngine::OpEngine(this=0x0000123448df7fa0, hub=0x000000010820ef90) + 29 at OpEngine.cc:55
        frame #15: 0x0000000103b13a44 libOK.dylib`OKPropagator::OKPropagator(this=0x0000123448df1fc0, hub=0x000000010820ef90, idx=0x000000016bc7cfe0, viz=0x000000016bc7ef50) + 196 at OKPropagator.cc:44
        frame #16: 0x0000000103b13bbd libOK.dylib`OKPropagator::OKPropagator(this=0x0000123448df1fc0, hub=0x000000010820ef90, idx=0x000000016bc7cfe0, viz=0x000000016bc7ef50) + 45 at OKPropagator.cc:52
        frame #17: 0x0000000103b13377 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe5f8, argc=21, argv=0x00007fff5fbfe6d0, argforced=0x0000000000000000) + 663 at OKMgr.cc:43
        frame #18: 0x0000000103b1364b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe5f8, argc=21, argv=0x00007fff5fbfe6d0, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #19: 0x000000010000adad OKTest`main(argc=21, argv=0x00007fff5fbfe6d0) + 1373 at OKTest.cc:58
        frame #20: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) f 8
    frame #8: 0x0000000103498f33 libOptiXRap.dylib`optix::ContextObj::create() + 115 at optixpp_namespace.h:1872
       1869   {
       1870     RTcontext c;
       1871     if( RTresult code = rtContextCreate(&c) )
    -> 1872       throw Exception::makeException( code, 0 );
       1873 
       1874     return Context::take(c);
       1875   }
    (lldb) 


