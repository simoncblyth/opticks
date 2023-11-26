OutDir_asserts_causing_render_test_fails
============================================


::

    Thread 1 "CSGOptiXRenderT" received signal SIGABRT, Aborted.

    #9  0x00007ffff71c9449 in spath::_Join<char const*, char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:437
    #10 0x00007ffff71c8eef in spath::Resolve<char const*, char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:421
    #11 0x00007ffff7255653 in SEventConfig::OutDir () at /home/blyth/junotop/opticks/sysrap/SEventConfig.cc:508
    #12 0x00007ffff7e9fe65 in CSGOptiX::render (this=0x11326860, stem_=0x0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:1001
    #13 0x000000000040d075 in main (argc=1, argv=0x7fffffff1e38)
        at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:198
    (gdb) 




    (gdb) f 13
    #13 0x000000000040d075 in main (argc=1, argv=0x7fffffff1e38)
        at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:198
    198	            t.cx->render(); 
    (gdb) f 12
    #12 0x00007ffff7e9fe65 in CSGOptiX::render (this=0x11326860, stem_=0x0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:1001
    1001	    const char* outdir = SEventConfig::OutDir();
    (gdb) p outdir
    $1 = 0x467ad0 "\001"
    (gdb) f 11
    #11 0x00007ffff7255653 in SEventConfig::OutDir () at /home/blyth/junotop/opticks/sysrap/SEventConfig.cc:508
    508	    const char* dir = spath::Resolve( OutFold(), OutName() ); 
    (gdb) f 10
    #10 0x00007ffff71c8eef in spath::Resolve<char const*, char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:421
    421	    std::string spec = _Join(std::forward<Args>(args)... ); 
    (gdb) f 9
    #9  0x00007ffff71c9449 in spath::_Join<char const*, char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:437
    437	    std::vector<std::string> args = {args_...};
    (gdb) f 8
    #8  0x00007ffff71ae142 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> > (this=0x7fffffff1060, __s=0x0, __a=...)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/include/c++/11.2.0/bits/basic_string.h:539
    539		_M_construct(__s, __end, random_access_iterator_tag());
    (gdb) 



::

    #9  0x00007ffff3daa449 in spath::_Join<char const*, char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:437
    #10 0x00007ffff3da9eef in spath::Resolve<char const*, char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:421
    #11 0x00007ffff3e36653 in SEventConfig::OutDir () at /home/blyth/junotop/opticks/sysrap/SEventConfig.cc:508
    #12 0x00007ffff49dfe65 in CSGOptiX::render (this=0x1c6343f0, stem_=0x0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:1001
    #13 0x00007ffff7f0d03e in G4CXOpticks::render (this=0xa74d40) at /home/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:486
    #14 0x0000000000407708 in main (argc=1, argv=0x7fffffff1e48) at /home/blyth/junotop/opticks/g4cx/tests/G4CXRenderTest.cc:27
    (gdb) 


::

    (gdb) f 12
    #12 0x00007ffff49dfe65 in CSGOptiX::render (this=0x1c6343f0, stem_=0x0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:1001
    1001	    const char* outdir = SEventConfig::OutDir();
    (gdb) f 11
    #11 0x00007ffff3e35799 in SEventConfig::OutDir () at /home/blyth/junotop/opticks/sysrap/SEventConfig.cc:516
    516	    const char* dir = spath::Resolve( outfold, outname ); 
    (gdb) p outfold
    $1 = 0x7ffff3f481ba "$DefaultOutputDir"
    (gdb) p outname
    $2 = 0x0
    (gdb) 




