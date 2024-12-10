check_gdml_geometry_conversion
================================


Create directory for the geometry, here named "gabor_pfrich_min", following conventional layout::

    cd ~/.opticks/GEOM
    mkdir -p gabor_pfrich_min
    cp ~/Downloads/pfrich_min.gdml gabor_pfrich_min/origin.gdml


Change ~/.optick/GEOM/GEOM.sh script to export the geometry name "gabor_pfrich_min"::

    vi ~/.opticks/GEOM/GEOM.sh  # GEOM bash script does this 


Try conversion::

    P[blyth@localhost ~]$ ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
    /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh : GEOM gabor_pfrich_min : no geomscript
                       BASH_SOURCE : /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh 
                               arg : info_run_ana 
                              SDIR :  
                              GEOM : gabor_pfrich_min 
                              FOLD : /data/blyth/opticks/G4CXOpticks_setGeometry_Test/gabor_pfrich_min 
                               bin : G4CXOpticks_setGeometry_Test 
                        geomscript :  
                            script : G4CXOpticks_setGeometry_Test.py 
                            origin : /home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml 
    ./GXTestRunner.sh : FOUND origin /home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml
                    HOME : /home/blyth
                     PWD : /data/blyth/junotop/opticks/g4cx/tests
                    GEOM : gabor_pfrich_min
    gabor_pfrich_min_GDMLPathFromGEOM : /home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXOpticks_setGeometry_Test
                    ARGS : 
    2024-12-10 10:25:16.485 INFO  [275100] [main@16] [SetGeometry
    U4VolumeMaker::PV name gabor_pfrich_min
    U4VolumeMaker::PVG_ name gabor_pfrich_min gdmlpath /home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml sub - exists 1
    G4GDML: Reading '/home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml' done!
    U4GDML::read                   yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    2024-12-10 10:25:16.551 INFO  [275100] [U4Tree::Create@216] [new U4Tree
    2024-12-10 10:25:16.579 INFO  [275100] [U4Tree::init@273] -initRayleigh
    2024-12-10 10:25:16.579 INFO  [275100] [U4Tree::init@275] -initMaterials
    2024-12-10 10:25:16.584 INFO  [275100] [U4Tree::init@277] -initMaterials_NoRINDEX
    2024-12-10 10:25:16.584 INFO  [275100] [U4Tree::init@280] -initScint
    2024-12-10 10:25:16.585 INFO  [275100] [U4Tree::init@283] -initSurfaces
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 3
    G4CXOpticks_setGeometry_Test: /data/blyth/opticks_Debug/include/SysRap/NPFold.h:818: void NPFold::add_subfold(const char*, NPFold*): Assertion `unique_f' failed.
    ./GXTestRunner.sh: line 42: 275100 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXOpticks_setGeometry_Test
    /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh : run error
    P[blyth@localhost ~]$ 


Do that under debugger to get the backtrace::

    P[blyth@localhost ~]$ ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh dbg
    /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh : GEOM gabor_pfrich_min : no geomscript
    gdb -ex r --args G4CXOpticks_setGeometry_Test
    Tue Dec 10 10:34:02 CST 2024
    GNU gdb (GDB) 12.1
    Copyright (C) 2022 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-pc-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <https://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from G4CXOpticks_setGeometry_Test...
    Starting program: /data/blyth/opticks_Debug/lib/G4CXOpticks_setGeometry_Test 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2024-12-10 10:34:06.064 INFO  [288396] [main@16] [SetGeometry
    U4VolumeMaker::PV name gabor_pfrich_min
    U4VolumeMaker::PVG_ name gabor_pfrich_min gdmlpath /home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml sub - exists 1
    G4GDML: Reading '/home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/.opticks/GEOM/gabor_pfrich_min/origin.gdml' done!
    U4GDML::read                   yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    2024-12-10 10:34:06.111 INFO  [288396] [U4Tree::Create@216] [new U4Tree
    2024-12-10 10:34:06.142 INFO  [288396] [U4Tree::init@273] -initRayleigh
    2024-12-10 10:34:06.142 INFO  [288396] [U4Tree::init@275] -initMaterials
    2024-12-10 10:34:06.147 INFO  [288396] [U4Tree::init@277] -initMaterials_NoRINDEX
    2024-12-10 10:34:06.147 INFO  [288396] [U4Tree::init@280] -initScint
    2024-12-10 10:34:06.147 INFO  [288396] [U4Tree::init@283] -initSurfaces
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 3
    G4CXOpticks_setGeometry_Test: /data/blyth/opticks_Debug/include/SysRap/NPFold.h:818: void NPFold::add_subfold(const char*, NPFold*): Assertion `unique_f' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff24ab387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff24ab387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff24aca78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff24a41a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff24a4252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7eaad78 in NPFold::add_subfold (this=0x76f5d0, f=0x6127d8 "MirrorPyramid", fo=0x7712c0) at /data/blyth/opticks_Debug/include/SysRap/NPFold.h:818
    #5  0x00007ffff7eccf75 in U4Surface::MakeFold (surfaces=std::vector of length 66, capacity 128 = {...}) at /data/blyth/opticks_Debug/include/U4/U4Surface.h:372
    #6  0x00007ffff7ed8c12 in U4Tree::initSurfaces (this=0x506510) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:494
    #7  0x00007ffff7ed770e in U4Tree::init (this=0x506510) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:284
    #8  0x00007ffff7ed72c3 in U4Tree::U4Tree (this=0x506510, st_=0x494b10, top_=0x4fb8d0, sid_=0x0) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:260
    #9  0x00007ffff7ed6814 in U4Tree::Create (st=0x494b10, top=0x4fb8d0, sid=0x0) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:217
    #10 0x00007ffff7e891ec in G4CXOpticks::setGeometry (this=0x494730, world=0x4fb8d0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:311
    #11 0x00007ffff7e8853c in G4CXOpticks::setGeometry (this=0x494730) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:233
    #12 0x00007ffff7e874e0 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:52
    #13 0x00000000004038d9 in main (argc=1, argv=0x7fffffff4778) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) 


Added envvar control to avoid the assert in g4cx/tests/G4CXOpticks_setGeometry_Test.sh::

    105 if [ "$GEOM" == "gabor_pfrich_min" ]; then
    106    echo $BASH_SOURCE : GEOM $GEOM : DEBUGGING : ALLOW DUPLICATE FOLDER KEYS 
    107    export NPFold__add_subfold_ALLOW_DUPLICATE_KEY=1
    108 fi

Gets further, then hit another surface related assert::

    P[blyth@localhost ~]$ ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh dbg
    ...
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 56[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 57[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 58[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 59[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 60[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 61[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 62[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 63[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 64[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    NPFold::add_subfold ERROR repeated subfold key f[MirrorPyramid] ff.size 65[NPFold__add_subfold_ALLOW_DUPLICATE_KEY] 1
    2024-12-10 10:55:34.856 INFO  [341046] [U4Tree::init@286] -initSolids
    [U4Tree::initSolids
    ]U4Tree::initSolids
    2024-12-10 10:55:34.858 INFO  [341046] [U4Tree::init@288] -initNodes
    2024-12-10 10:55:34.859 INFO  [341046] [U4Tree::init@290] -initSurfaces_Serialize
    2024-12-10 10:55:34.875 INFO  [341046] [U4Tree::init@293] -initStandard
    G4CXOpticks_setGeometry_Test: /data/blyth/opticks_Debug/include/SysRap/sstandard.h:353: static NP* sstandard::make_optical(const std::vector<int4>&, const std::vector<std::__cxx11::basic_string<char> >&, const NPFold*): Assertion `sn' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff24aa387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff24aa387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff24aba78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff24a31a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff24a3252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7ec0ee5 in sstandard::make_optical (vbd=std::vector of length 68, capacity 128 = {...}, suname=std::vector of length 8, capacity 8 = {...}, surface=0x76f5d0)
        at /data/blyth/opticks_Debug/include/SysRap/sstandard.h:353
    #5  0x00007ffff7ec0c05 in sstandard::deferred_init (this=0x494ee0, vbd=std::vector of length 68, capacity 128 = {...}, bdname=std::vector of length 68, capacity 128 = {...}, 
        suname=std::vector of length 8, capacity 8 = {...}, surface=0x76f5d0) at /data/blyth/opticks_Debug/include/SysRap/sstandard.h:193
    #6  0x00007ffff7ecb0fa in stree::initStandard (this=0x494b10) at /data/blyth/opticks_Debug/include/SysRap/stree.h:5083
    #7  0x00007ffff7ed9d5f in U4Tree::initStandard (this=0x506510) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:875
    #8  0x00007ffff7ed7ac0 in U4Tree::init (this=0x506510) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:294
    #9  0x00007ffff7ed7379 in U4Tree::U4Tree (this=0x506510, st_=0x494b10, top_=0x4fb8d0, sid_=0x0) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:260
    #10 0x00007ffff7ed68ca in U4Tree::Create (st=0x494b10, top=0x4fb8d0, sid=0x0) at /data/blyth/opticks_Debug/include/U4/U4Tree.h:217
    #11 0x00007ffff7e891ec in G4CXOpticks::setGeometry (this=0x494730, world=0x4fb8d0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:311
    #12 0x00007ffff7e8853c in G4CXOpticks::setGeometry (this=0x494730) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:233
    #13 0x00007ffff7e874e0 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:52
    #14 0x00000000004038d9 in main (argc=1, argv=0x7fffffff4748) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) 


    (gdb) f 4
    #4  0x00007ffff7ec0ee5 in sstandard::make_optical (vbd=std::vector of length 68, capacity 128 = {...}, suname=std::vector of length 8, capacity 8 = {...}, surface=0x76f5d0)
        at /data/blyth/opticks_Debug/include/SysRap/sstandard.h:353
    353                 if(idx > -1 ) assert(sn) ;  
    (gdb) f 5
    #5  0x00007ffff7ec0c05 in sstandard::deferred_init (this=0x494ee0, vbd=std::vector of length 68, capacity 128 = {...}, bdname=std::vector of length 68, capacity 128 = {...}, 
        suname=std::vector of length 8, capacity 8 = {...}, surface=0x76f5d0) at /data/blyth/opticks_Debug/include/SysRap/sstandard.h:193
    193     optical = make_optical(vbd, suname, surface) ; 
    (gdb) f 4
    #4  0x00007ffff7ec0ee5 in sstandard::make_optical (vbd=std::vector of length 68, capacity 128 = {...}, suname=std::vector of length 8, capacity 8 = {...}, surface=0x76f5d0)
        at /data/blyth/opticks_Debug/include/SysRap/sstandard.h:353
    353                 if(idx > -1 ) assert(sn) ;  
    (gdb) list 
    348                 op_v[op_index+3] = 0 ; 
    349             }
    350             else if(is_sur)
    351             {
    352                 const char* sn = snam::get(suname, idx) ; 
    353                 if(idx > -1 ) assert(sn) ;  
    354                 // all surf should have name, do not always have surf
    355 
    356                 NPFold* surf = sn ? surface->get_subfold(sn) : nullptr ;
    357                 bool is_implicit = sn && strncmp(sn, IMPLICIT_PREFIX, strlen(IMPLICIT_PREFIX) ) == 0 ; 
    (gdb) 



After added some handling for uniquing surface names, the convert and load
works and can visualize with::

   ~/o/cx.sh 




