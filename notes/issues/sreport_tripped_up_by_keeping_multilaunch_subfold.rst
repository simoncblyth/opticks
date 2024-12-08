FIXED : sreport_tripped_up_by_keeping_multilaunch_subfold
===============================================================

FIXED::

    P[blyth@localhost np]$ git commit -m "add maxdepth control to avoid NPFold::find_subfold_with_prefix needed for opticks/sysrap/tests/sreport.cc when multi-launch subfold are kept" 



While checking get the subfold::

    TEST=ref10_multilaunch ~/o/cxs_min.sh


    NPFold::clear_subfold[]0xefa8dc0
    2024-12-08 14:10:17.260  260953670 : ]/home/blyth/o/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1 is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges    :(6, 5, )
    -sreport_Creator::init.4 fold_valid Y
    [NPFold::substamp find_subfold_with_prefix //A num_sub 11 sub0  subfold 10 ff 10 kk 2 aa 2 num_stamp0 13 skip NO 
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid
    /home/blyth/o/cxs_min.sh: line 479: 65898 Aborted                 (core dumped) sreport
    /home/blyth/o/cxs_min.sh sreport error
    P[blyth@localhost opticks]$ 



Thats the report sub-command::

    P[blyth@localhost opticks]$ TEST=ref10_multilaunch ~/o/cxs_min.sh report
    /home/blyth/o/cxs_min.sh : FOUND A_CFBaseFromGEOM /home/blyth/.opticks/GEOM/J_2024nov27 containing CSGFoundry/prim.npy
    [sreport.main  argv0 sreport dirp /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1 is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges    :(6, 5, )
    -sreport_Creator::init.4 fold_valid Y
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid
    /home/blyth/o/cxs_min.sh: line 480: 73611 Aborted                 (core dumped) sreport
    /home/blyth/o/cxs_min.sh sreport error
    P[blyth@localhost opticks]$ 



Invoke sreport from the logdir::

    P[blyth@localhost opticks]$ cd /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1
    P[blyth@localhost ALL1]$ l
    total 16
    4 -rw-rw-r--.  1 blyth blyth  705 Dec  8 14:10 CSGOptiX__Ctx.log
    4 -rw-rw-r--.  1 blyth blyth 3860 Dec  8 14:10 run_meta.txt
    4 -rw-rw-r--.  1 blyth blyth  132 Dec  8 14:10 run.npy
    0 drwxr-xr-x. 12 blyth blyth  272 Dec  8 14:10 A000
    4 -rw-r--r--.  1 blyth blyth  887 Dec  8 14:10 CSGOptiXSMTest.log
    0 drwxrwxr-x.  3 blyth blyth  104 Dec  8 14:10 .
    0 drwxrwxr-x.  5 blyth blyth   66 Dec  4 10:34 ..
    P[blyth@localhost ALL1]$ sreport
    [sreport.main  argv0 sreport dirp /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1 is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges    :(6, 5, )
    -sreport_Creator::init.4 fold_valid Y
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid
    Aborted (core dumped)


gdb::

    P[blyth@localhost ALL1]$ gdb sreport
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
    Reading symbols from sreport...
    (gdb) r
    Starting program: /data/blyth/opticks_Debug/lib/sreport 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [sreport.main  argv0 /data/blyth/opticks_Debug/lib/sreport dirp /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1 is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges    :(6, 5, )
    -sreport_Creator::init.4 fold_valid Y
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff64cc387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff64cc387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff64cda78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff6c0b89a in __gnu_cxx::__verbose_terminate_handler () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff6c1736a in __cxxabiv1::__terminate (handler=<optimized out>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff6c173d5 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff6c17669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff6d6c188 <typeinfo for std::logic_error>, dest=0x7ffff6c2bfe0 <std::logic_error::~logic_error()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff6c0e121 in std::__throw_logic_error (__s=0x43cc18 "basic_string::_M_construct null not valid")
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/src/c++11/functexcept.cc:70
    #7  0x0000000000425315 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*> (this=0x7fffffff3d40, __beg=0x0, 
        __end=0x1 <error: Cannot access memory at address 0x1>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/include/c++/11.2.0/bits/basic_string.tcc:212
    #8  0x0000000000422074 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> > (this=0x7fffffff3d40, __s=0x0, __a=...)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/include/c++/11.2.0/bits/basic_string.h:539
    #9  0x000000000040e825 in NP::get_meta_string_[abi:cxx11](char const*, char const*) (metadata=0x0, key=0x4664a0 "NumPhotonCollected") at /home/blyth/opticks/sysrap/NP.hh:4670
    #10 0x000000000040ea82 in NP::get_meta_string (meta="", key=0x4664a0 "NumPhotonCollected") at /home/blyth/opticks/sysrap/NP.hh:4709
    #11 0x00000000004199f2 in NPFold::get_meta_string[abi:cxx11](char const*) const (this=0x4698e0, key=0x4664a0 "NumPhotonCollected") at /home/blyth/opticks/sysrap/NPFold.h:2132
    #12 0x000000000041f53e in NPFold::SubCommonKV (okey=std::vector of length 0, capacity 0, ckey=std::vector of length 0, capacity 0, cval=std::vector of length 0, capacity 0, 
        subs=std::vector of length 11, capacity 16 = {...}) at /home/blyth/opticks/sysrap/NPFold.h:3350
    #13 0x000000000041c78b in NPFold::substamp (this=0x465150, prefix=0x7fffffff42a0 "//A", keyname=0x43bed7 "substamp") at /home/blyth/opticks/sysrap/NPFold.h:2940
    #14 0x000000000041e642 in NPFold::subfold_summary<char const*, char const*> (this=0x465150, method=0x43bed7 "substamp") at /home/blyth/opticks/sysrap/NPFold.h:3169
    #15 0x0000000000421056 in sreport_Creator::init (this=0x7fffffff4770) at /home/blyth/opticks/sysrap/tests/sreport.cc:328
    #16 0x0000000000420c9c in sreport_Creator::sreport_Creator (this=0x7fffffff4770, dirp_=0x7fffffffb3c0 "/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1")
        at /home/blyth/opticks/sysrap/tests/sreport.cc:308
    #17 0x0000000000405bd2 in main (argc=1, argv=0x7fffffff4948) at /home/blyth/opticks/sysrap/tests/sreport.cc:430
    (gdb) 



::

    312 inline void sreport_Creator::init()
    313 {
    314     std::cout << "[sreport_Creator::init" << std::endl ;
    315 
    316     report->runprof = run ? run->makeMetaKVProfileArray("Index") : nullptr ;
    317     std::cout << "-sreport_Creator::init.1:runprof   :" << ( report->runprof ? report->runprof->sstr() : "-" ) << std::endl ;
    318 
    319     report->run     = run ? run->copy() : nullptr ;
    320     std::cout << "-sreport_Creator::init.2.run       :" << ( report->run ? report->run->sstr() : "-" ) << std::endl ;
    321 
    322     report->ranges = run ? run->makeMetaKVS_ranges( sreport::RANGES ) : nullptr ;
    323     std::cout << "-sreport_Creator::init.3.ranges    :" << ( report->ranges ?  report->ranges->sstr() : "-" ) <<  std::endl ;
    324 
    325 
    326     std::cout << "-sreport_Creator::init.4 fold_valid " << ( fold_valid ? "Y" : "N" ) << std::endl ;
    327 
    328     report->substamp   = fold_valid ? fold->subfold_summary("substamp",   ASEL, BSEL) : nullptr ;
    329     std::cout << "-sreport_Creator::init.4.substamp   :[" << ( report->substamp ? report->substamp->stats() : "-" ) << "]\n" ;
    330 
    331     report->subprofile = fold_valid ? fold->subfold_summary("subprofile", ASEL, BSEL) : nullptr ;
    332     std::cout << "-sreport_Creator::init.5.subprofile :[" << ( report->subprofile ? report->subprofile->stats() : "-" )  << "]\n" ;
    333 





::

    (gdb) f 15
    #15 0x0000000000421056 in sreport_Creator::init (this=0x7fffffff4770) at /home/blyth/opticks/sysrap/tests/sreport.cc:328
    328     report->substamp   = fold_valid ? fold->subfold_summary("substamp",   ASEL, BSEL) : nullptr ; 
    (gdb) 




Added test to reproduce that over in np::

    P[blyth@localhost tests]$ TEST=substamp ~/np/tests/NPFold_LoadNoData_test.sh
    /data/blyth/opticks/NPFold_LoadNoData_test
    NPFold::LoadNoData("/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1")
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid
    /home/blyth/np/tests/NPFold_LoadNoData_test.sh: line 41: 170250 Aborted                 (core dumped) $bin
    /home/blyth/np/tests/NPFold_LoadNoData_test.sh : run error


    P[blyth@localhost tests]$ TEST=substamp ~/np/tests/NPFold_LoadNoData_test.sh dbg
    gdb -ex r --args /data/blyth/opticks/NPFold_LoadNoData_test
    Sun Dec  8 15:19:02 CST 2024
    GNU gdb (GDB) 12.1
    ...
    Starting program: /data/blyth/opticks/NPFold_LoadNoData_test 
    /data/blyth/opticks/NPFold_LoadNoData_test
    NPFold::LoadNoData("/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1")
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff782f387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff782f387 in raise () from /lib64/libc.so.6
    #7  0x00007ffff7d0e8ef in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*> (this=0x7fffffff3ba0, __beg=0x0, 
        __end=0x1 <error: Cannot access memory at address 0x1>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0.build/x86_64-pc-linux-gnu/libstdc++-v3/include/bits/basic_string.tcc:212
    #8  0x000000000040cca7 in NP::get_meta_string_[abi:cxx11](char const*, char const*) (metadata=0x0, key=0x45a4b0 "NumPhotonCollected") at /home/blyth/np/tests/../NP.hh:4670
    #9  0x000000000040cf04 in NP::get_meta_string (meta="", key=0x45a4b0 "NumPhotonCollected") at /home/blyth/np/tests/../NP.hh:4709
    #10 0x0000000000416e94 in NPFold::get_meta_string[abi:cxx11](char const*) const (this=0x45d7a0, key=0x45a4b0 "NumPhotonCollected") at /home/blyth/np/tests/../NPFold.h:2132
    #11 0x000000000041c47a in NPFold::SubCommonKV (okey=std::vector of length 0, capacity 0, ckey=std::vector of length 0, capacity 0, cval=std::vector of length 0, capacity 0, 
        subs=std::vector of length 11, capacity 16 = {...}) at /home/blyth/np/tests/../NPFold.h:3350
    #12 0x00000000004196f1 in NPFold::substamp (this=0x459d10, prefix=0x7fffffff4100 "//A", keyname=0x433bb7 "substamp") at /home/blyth/np/tests/../NPFold.h:2940
    #13 0x000000000041b588 in NPFold::subfold_summary<char const*, char const*> (this=0x459d10, method=0x433bb7 "substamp") at /home/blyth/np/tests/../NPFold.h:3169
    #14 0x0000000000405016 in NPFold_LoadNoData_test::substamp (this=0x7fffffff4450) at /home/blyth/np/tests/NPFold_LoadNoData_test.cc:89
    #15 0x0000000000404c6e in NPFold_LoadNoData_test::test (this=0x7fffffff4450) at /home/blyth/np/tests/NPFold_LoadNoData_test.cc:66
    #16 0x0000000000404ab5 in NPFold_LoadNoData_test::main (argc=1, argv=0x7fffffff4568) at /home/blyth/np/tests/NPFold_LoadNoData_test.cc:45
    #17 0x0000000000405061 in main (argc=1, argv=0x7fffffff4568) at /home/blyth/np/tests/NPFold_LoadNoData_test.cc:98
    (gdb) 


Need depth restriction on sub finding::

    P[blyth@localhost tests]$ TEST=substamp ~/np/tests/NPFold_LoadNoData_test.sh 
             BASH_SOURCE : /home/blyth/np/tests/NPFold_LoadNoData_test.sh
                    name : NPFold_LoadNoData_test
                    SDIR : /home/blyth/np/tests
                     PWD : /home/blyth/np/tests
                     bin : /data/blyth/opticks/NPFold_LoadNoData_test
                  defarg : info_build_run
                     arg : info_build_run
                    JDIR : /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1
                    test : substamp
                    TEST : substamp
    /data/blyth/opticks/NPFold_LoadNoData_test
    NPFold::LoadNoData("/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1")
    [NPFold::substamp find_subfold_with_prefix //A num_sub 11 sub0  subfold 10 ff 10 kk 2 aa 2 num_stamp0 13 skip NO 
    [NPFold::DescFoldAndPaths
     sub YES  p  //A000
     sub YES  p  //A000/f000
     sub YES  p  //A000/f001
     sub YES  p  //A000/f002
     sub YES  p  //A000/f003
     sub YES  p  //A000/f004
     sub YES  p  //A000/f005
     sub YES  p  //A000/f006
     sub YES  p  //A000/f007
     sub YES  p  //A000/f008
     sub YES  p  //A000/f009
    ]NPFold::DescFoldAndPaths
    NPFold::get_meta_string meta_empty YES key NumPhotonCollected treepath /A000/f000
    NPFold::SubCommonKV MISSING KEY  num_sub 11 num_ukey 25 k NumPhotonCollected v -
    /home/blyth/np/tests/NPFold_LoadNoData_test.sh : run error
    P[blyth@localhost tests]$ 



    P[blyth@localhost np]$ git s
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   NPFold.h
        modified:   NPU.hh
        modified:   tests/NPFold_LoadNoData_test.cc
        modified:   tests/NPFold_LoadNoData_test.sh

    no changes added to commit (use "git add" and/or "git commit -a")
    P[blyth@localhost np]$ git add .
    P[blyth@localhost np]$ git commit -m "add maxdepth control to avoid NPFold::find_subfold_with_prefix needed for opticks/sysrap/tests/sreport.cc when multi-launch subfold are kept" 
    [master 7e04957] add maxdepth control to avoid NPFold::find_subfold_with_prefix needed for opticks/sysrap/tests/sreport.cc when multi-launch subfold are kept
     4 files changed, 229 insertions(+), 54 deletions(-)
    P[blyth@localhost np]$ git push 
    Enumerating objects: 13, done.
    Counting objects: 100% (13/13), done.
    Delta compression using up to 48 threads
    Compressing objects: 100% (7/7), done.
    Writing objects: 100% (7/7), 2.77 KiB | 404.00 KiB/s, done.
    Total 7 (delta 4), reused 0 (delta 0), pack-reused 0
    remote: Resolving deltas: 100% (4/4), completed with 4 local objects.
    To github.com:simoncblyth/np.git
       0082744..7e04957  master -> master
    P[blyth@localhost np]$ 




