PLOG-test-fails-with-gcc5
==========================


PLOG Review
-------------

* :doc:`../../sysrap/PLOG_review`




What is special about the 4 failing tests ?
----------------------------------------------

Graping at straws all 4 failing tests "LOG(info) << argv[0]" 
between PLOG_ and PKG_LOG__ this is kinda dirty using logging 
prior to the logging being fully setup.


::

    The following tests FAILED:
        135 - OpticksCoreTest.OpticksTest (OTHER_FAULT)
        244 - cfg4Test.CGDMLDetectorTest (OTHER_FAULT)
        245 - cfg4Test.CGeometryTest (OTHER_FAULT)
        251 - cfg4Test.CInterpolationTest (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output



OpticksTest::

     70 int main(int argc, char** argv)
     71 {
     72     PLOG_(argc,argv);
     73     LOG(info) << argv[0] ;
     74 
     75     SYSRAP_LOG__ ;
     76     BRAP_LOG__ ;
     77     NPY_LOG__ ;
     78     OKCORE_LOG__ ;
     79 
     80     Opticks ok(argc, argv);
     81     ok.configure();
     82 
     83     ok.Summary();
     84 
     85     LOG(info) << "OpticksTest::main aft configure" ;
     86 
     87     test_MaterialSequence();
     88     test_getDAEPath(&ok);
     89     test_getGDMLPath(&ok);
     90     test_getMaterialMap(&ok);
     91 
     92     return 0 ;
     93 }


CGDMLDetectorTest::

     31 int main(int argc, char** argv)
     32 {
     33     PLOG_(argc, argv);
     34 
     35     LOG(info) << argv[0] ;
     36 
     37     CFG4_LOG__ ;
     38     GGEO_LOG__ ;
     39 
     40     Opticks ok(argc, argv);
     ..








2017-10-19 Axel (Linux gcc5) reports test fails
-------------------------------------------------


Four of the fails coming from PLOG::


    98% tests passed, 6 tests failed out of 254

    Total Test time (real) = 151.86 sec

    The following tests FAILED:
        134 - OpticksCoreTest.OpticksTest (SEGFAULT)          # PLOG issues
        243 - cfg4Test.CGDMLDetectorTest (OTHER_FAULT)
        244 - cfg4Test.CGeometryTest (OTHER_FAULT)
        250 - cfg4Test.CInterpolationTest (OTHER_FAULT)

        242 - cfg4Test.CTestDetectorTest (OTHER_FAULT)        # old style PmtInBox geometry needs overhaul


        226 - OptiXRapTest.OInterpolationTest (Failed)        # missing GBndLib.py 

        ## reproduced this one, python analysis scripts using TMP need to use opticks_main 
        ## otherwise requires TMP envvar to be defined  


    Errors while running CTest
    opticks-t- : use -V to show output


gdb OpticksTest 
-----------------

::

    Reading symbols from OpticksTest...done.
    (gdb) r
    Starting program: /usr/local/opticks/lib/OpticksTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2017-10-18 15:47:20.652 INFO  [3839] [main@77] /usr/local/opticks/lib/OpticksTest
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_S_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff663b428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    54	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  0x00007ffff663b428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    #1  0x00007ffff663d02a in __GI_abort () at abort.c:89
    #2  0x00007ffff6c7584d in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #3  0x00007ffff6c736b6 in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #4  0x00007ffff6c73701 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #5  0x00007ffff6c73919 in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007ffff6c9c14f in std::__throw_logic_error(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x0000000000405c3d in std::string::_S_construct<char const*> (__beg=0x0, __end=0xffffffffffffffff <error: Cannot access memory at address 0xffffffffffffffff>, __a=...)
        at /usr/include/c++/4.8/bits/basic_string.tcc:133
    #8  0x00007ffff6cb6d16 in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #9  0x00007ffff75ae5ce in PLOG::_prefixlevel_parse (argc=1508334440, argv=0x7fffffff028c, fallback=0x7ffff75b3bbe "INFO", prefix=0x406c8c "SYSRAP") at /home/gpu/opticks/sysrap/PLOG.cc:82
                                                       ^^^^^^^^^^^^^^^^^^^ CRAZY argc
    #10 0x00007ffff75ae8c3 in PLOG::prefixlevel_parse (this=0x7fffffffd710, fallback=0x7ffff75b3bbe "INFO", prefix=0x406c8c "SYSRAP") at /home/gpu/opticks/sysrap/PLOG.cc:131
    #11 0x00007ffff75ae890 in PLOG::prefixlevel_parse (this=0x7fffffffd710, _fallback=plog::info, prefix=0x406c8c "SYSRAP") at /home/gpu/opticks/sysrap/PLOG.cc:127
    #12 0x0000000000403a54 in main (argc=1, argv=0x7fffffffd9a8) at /home/gpu/opticks/optickscore/tests/OpticksTest.cc:80
    (gdb) 



Thoughts
----------

* something must be stomping on argc argv, because the same arg loop code is done in the main parse 
  and that causes no error 


::

     74 int main(int argc, char** argv)
     75 {
     76     PLOG_(argc,argv);
     77     LOG(info) << argv[0] ;
     78 
     79 
     80     SYSRAP_LOG__ ;        ## <<<<
     81     BRAP_LOG__ ;
     82     NPY_LOG__ ;
     83     OKCORE_LOG__ ;
     84     
     85 
     86     Opticks ok(argc, argv);
     87 
     88     ok.Summary();
     89 


::

    060 int PLOG::_prefixlevel_parse(int argc, char** argv, const char* fallback, const char* prefix)
     61 {
     62     // Parse commandline to find project logging level  
     63     // looking for a single project prefix, eg 
     64     // with the below commandline and prefix of sysrap
     65     // the level "error" should be set.
     66     //
     67     // When no level is found the fallback level is used.
     68     //
     69     //    --okcore info --sysrap error --brap trace --npy trace
     70     //  
     71     // Both prefix and the arguments are lowercased before comparison.
     72     //
     73 
     74     std::string pfx(prefix);
     75     std::transform(pfx.begin(), pfx.end(), pfx.begin(), ::tolower);
     76     std::string apfx("--");
     77     apfx += pfx ;
     78 
     79     std::string ll(fallback) ;
     80     for(int i=1 ; i < argc ; ++i )
     81     {
     82         std::string arg(argv[i]);                    
     ..         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     83         std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
     84         //std::cerr << arg << std::endl ; 
     85 
     86         if(arg.compare(apfx) == 0 && i + 1 < argc ) ll.assign(argv[i+1]) ;
     87     }
     88 
     89     std::transform(ll.begin(), ll.end(), ll.begin(), ::toupper);
     90 
     91     const char* llc = ll.c_str();
     92     plog::Severity severity = strcmp(llc, "TRACE")==0 ? plog::severityFromString("VERB") : plog::severityFromString(llc) ;
     93     int level = static_cast<int>(severity);
     94 
     95     //_dump("PLOG::prefix_parse", argc, argv );
     96 
     97     return level ;
     98 }
    ...
    124 int PLOG::prefixlevel_parse(plog::Severity _fallback, const char* prefix)
    125 {
    126     const char* fallback = _name(_fallback);
    127     return prefixlevel_parse(fallback, prefix) ;  
    ..               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    128 }
    129 int PLOG::prefixlevel_parse(const char* fallback, const char* prefix)
    130 {
    131     int ll =  _prefixlevel_parse(argc, argv, fallback, prefix);    ### <<<
    ..               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    132 
    133 #ifdef DBG
    134     std::cerr << "PLOG::prefixlevel_parse"
    135               << " fallback " << fallback
    136               << " prefix " << prefix
    137               << " level " << ll
    138               << " name " << _name(ll)
    139               << std::endl ;
    140 #endif
    141 
    142     return ll ;
    143 }





something is stomping on PLOG argc with gcc5 ?
---------------------------------------------------

::

    simon:issues blyth$ grep argc= PLOG-test-fails-with-gcc5.rst
        #9  0x00007ffff75ae5ce in PLOG::_prefixlevel_parse (argc=1508334440, argv=0x7fffffff028c, fallback=0x7ffff75b3bbe "INFO", prefix=0x406c8c "SYSRAP") at /home/gpu/opticks/sysrap/PLOG.cc:82
        #9  0x00007ffff7bcc5ce in PLOG::_prefixlevel_parse (argc=1508334973, argv=0x7fffffff03bb, fallback=0x7ffff7bd1bbe "INFO", prefix=0x40938d "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:82
        #9  0x00007ffff7bcc5ce in PLOG::_prefixlevel_parse (argc=1508334904, argv=0x7fffffff012c, fallback=0x7ffff7bd1bbe "INFO", prefix=0x4076fc "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:82
        #9  0x00007ffff7bcc5ce in PLOG::_prefixlevel_parse (argc=1508335052, argv=0x7fffffff00bf, fallback=0x7ffff7bd1bbe "INFO", prefix=0x4087c8 "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:82
    simon:issues blyth$ 


gdb/lldb watch points
------------------------

* https://sourceware.org/gdb/onlinedocs/gdb/Set-Watchpoints.html
* https://lldb.llvm.org/lldb-gdb.html

Set a watchpoint on a variable when it is written to.::

    (gdb) watch global_var

    (lldb) watchpoint set variable global_var
    (lldb) wa s v global_var

Set a watchpoint on a memory location when it is written into. The size of the
region to watch for defaults to the pointer size if no '-x byte_size' is
specified. This command takes raw input, evaluated as an expression returning
an unsigned integer pointing to the start of the region, after the '--' option
terminator.

::

    (gdb) watch -location g_char_ptr

    (lldb) watchpoint set expression -- my_ptr
    (lldb) wa s e -- my_ptr



watching for a stomper with lldb
------------------------------------

::

    simon:sysrap blyth$ lldb OpticksTest 
    (lldb) target create "OpticksTest"
    Current executable set to 'OpticksTest' (x86_64).
    (lldb) b PLOG::PLOG
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) r
    Process 65014 launched: '/usr/local/opticks/lib/OpticksTest' (x86_64)
    2 locations added to breakpoint 1
    Process 65014 stopped
    * thread #1: tid = 0x45fc8, 0x000000010088afef libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfed88, argc_=1, argv_=0x00007fff5fbfede0, fallback=0x000000010001034a, prefix=0x0000000000000000) + 31 at PLOG.cc:181, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
        frame #0: 0x000000010088afef libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfed88, argc_=1, argv_=0x00007fff5fbfede0, fallback=0x000000010001034a, prefix=0x0000000000000000) + 31 at PLOG.cc:181
       178      :
       179        argc(argc_),
       180        argv(argv_),
    -> 181        level(info),
       182        logpath(_logpath_parse(argc_, argv_)),
       183        logmax(3)
       184  {

    (lldb) br del 1 
    1 breakpoints deleted; 0 breakpoint locations disabled.

    (lldb) watchpoint set variable this->argc
    Watchpoint created: Watchpoint 1: addr = 0x7fff5fbfed88 size = 4 state = enabled type = w
        watchpoint spec = 'this->argc'
        new value: 0


    (lldb) c
    Process 65014 resuming
    Process 65014 stopped
    * thread #1: tid = 0x45fc8, 0x000000010088af54 libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfed88, argc_=1, argv_=0x00007fff5fbfede0, fallback=0x000000010001034a, prefix=0x0000000000000000) + 36 at PLOG.cc:178, queue = 'com.apple.main-thread', stop reason = watchpoint 1
        frame #0: 0x000000010088af54 libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfed88, argc_=1, argv_=0x00007fff5fbfede0, fallback=0x000000010001034a, prefix=0x0000000000000000) + 36 at PLOG.cc:178
       175  
       176  
       177  PLOG::PLOG(int argc_, char** argv_, const char* fallback, const char* prefix)
    -> 178      :
       179        argc(argc_),
       180        argv(argv_),
       181        level(info),

    Watchpoint 1 hit:
    old value: 0
    new value: 1
    (lldb) c
    Process 65014 resuming
    2017-10-19 13:47:40.097 INFO  [286664] [main@77] OpticksTest
    2017-10-19 13:47:40.099 INFO  [286664] [OpticksQuery::dump@79] OpticksQuery::init queryType range query_string range:3153:12221 query_name NULL query_index 0 query_depth 0 no_selection 0 nrange 2 : 3153 : 12221
    2017-10-19 13:47:40.099 INFO  [286664] [Opticks::init@325] Opticks::init DONE OpticksResource::desc digest 96ff965744a2f6b78c24e33c80d3a4cd age.tot_seconds 4417277 age.tot_minutes 73621.281 age.tot_hours 1227.021 age.tot_days     51.126
    2017-10-19 13:47:40.099 INFO  [286664] [Opticks::Summary@851] Opticks::Summary sourceCode 4096 sourceType torch mode INTEROP_MODE
    ... elided output ...
    Process 65014 stopped
    * thread #1: tid = 0x45fc8, 0x00007fff899646fa libsystem_c.dylib`__cxa_finalize + 10, queue = 'com.apple.main-thread', stop reason = watchpoint 1
        frame #0: 0x00007fff899646fa libsystem_c.dylib`__cxa_finalize + 10
    libsystem_c.dylib`__cxa_finalize + 10:
    -> 0x7fff899646fa:  pushq  %r12
       0x7fff899646fc:  pushq  %rbx
       0x7fff899646fd:  subq   $0x18, %rsp
       0x7fff89964701:  movq   %rdi, -0x30(%rbp)

    Watchpoint 1 hit:        ## this just from cleanup at exit
    old value: 1
    new value: 0
    (lldb) bt
    * thread #1: tid = 0x45fc8, 0x00007fff899646fa libsystem_c.dylib`__cxa_finalize + 10, queue = 'com.apple.main-thread', stop reason = watchpoint 1
      * frame #0: 0x00007fff899646fa libsystem_c.dylib`__cxa_finalize + 10
        frame #1: 0x00007fff89964a4c libsystem_c.dylib`exit + 22
        frame #2: 0x00007fff869e9604 libdyld.dylib`start + 8
    (lldb) exit
    Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] 
    simon:sysrap blyth$ 



watching for a stomper with gdb
------------------------------------

Expect very similar to above, unfortunately the Linux install I have access to 
has gdb 7.2::

   gdb OpticksTest 

   (gdb) b PLOG::PLOG
   (gdb) r
   (gdb) del 1 
   (gdb) watch -location this->argc        ## this requires gdb 7.3+          
   (gdb) c











gdb CGDMLDetectorTest
------------------------

::

    Reading symbols from CGDMLDetectorTest...done.
    (gdb) r
    Starting program: /usr/local/opticks/lib/CGDMLDetectorTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2017-10-18 15:56:13.955 INFO  [3996] [main@35] /usr/local/opticks/lib/CGDMLDetectorTest
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_S_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff5f9c428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    54	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  0x00007ffff5f9c428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    #1  0x00007ffff5f9e02a in __GI_abort () at abort.c:89
    #2  0x00007ffff65d684d in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #3  0x00007ffff65d46b6 in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #4  0x00007ffff65d4701 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #5  0x00007ffff65d4919 in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007ffff65fd14f in std::__throw_logic_error(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x0000000000407f79 in std::string::_S_construct<char const*> (__beg=0x0, __end=0xffffffffffffffff <error: Cannot access memory at address 0xffffffffffffffff>, __a=...)
        at /usr/include/c++/4.8/bits/basic_string.tcc:133
    #8  0x00007ffff6617d16 in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #9  0x00007ffff7bcc5ce in PLOG::_prefixlevel_parse (argc=1508334973, argv=0x7fffffff03bb, fallback=0x7ffff7bd1bbe "INFO", prefix=0x40938d "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:82
    #10 0x00007ffff7bcc8c3 in PLOG::prefixlevel_parse (this=0x7fffffffd700, fallback=0x7ffff7bd1bbe "INFO", prefix=0x40938d "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:131
    #11 0x00007ffff7bcc890 in PLOG::prefixlevel_parse (this=0x7fffffffd700, _fallback=plog::info, prefix=0x40938d "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:127
    #12 0x00000000004053da in main (argc=1, argv=0x7fffffffd998) at /home/gpu/opticks/cfg4/tests/CGDMLDetectorTest.cc:37



::

     31 int main(int argc, char** argv)
     32 {
     33     PLOG_(argc, argv);
     34 
     35     LOG(info) << argv[0] ;
     36 
     37     CFG4_LOG__ ;           ### <<<<
     38     GGEO_LOG__ ;
     39 
     40     Opticks ok(argc, argv);
     41 
     42     OpticksHub hub(&ok);



gdb CGeometryTest
--------------------

::

    Reading symbols from CGeometryTest...done.
    (gdb) r
    Starting program: /usr/local/opticks/lib/CGeometryTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2017-10-18 15:55:04.300 INFO  [3982] [main@37] /usr/local/opticks/lib/CGeometryTest
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_S_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff65ba428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    54	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  0x00007ffff65ba428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    #1  0x00007ffff65bc02a in __GI_abort () at abort.c:89
    #2  0x00007ffff6bf484d in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #3  0x00007ffff6bf26b6 in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #4  0x00007ffff6bf2701 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #5  0x00007ffff6bf2919 in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007ffff6c1b14f in std::__throw_logic_error(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x00000000004065e5 in std::string::_S_construct<char const*> (__beg=0x0, __end=0xffffffffffffffff <error: Cannot access memory at address 0xffffffffffffffff>, __a=...)
        at /usr/include/c++/4.8/bits/basic_string.tcc:133
    #8  0x00007ffff6c35d16 in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #9  0x00007ffff7bcc5ce in PLOG::_prefixlevel_parse (argc=1508334904, argv=0x7fffffff012c, fallback=0x7ffff7bd1bbe "INFO", prefix=0x4076fc "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:82
    #10 0x00007ffff7bcc8c3 in PLOG::prefixlevel_parse (this=0x7fffffffd710, fallback=0x7ffff7bd1bbe "INFO", prefix=0x4076fc "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:131
    #11 0x00007ffff7bcc890 in PLOG::prefixlevel_parse (this=0x7fffffffd710, _fallback=plog::info, prefix=0x4076fc "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:127
    #12 0x0000000000403fe8 in main (argc=1, argv=0x7fffffffd9a8) at /home/gpu/opticks/cfg4/tests/CGeometryTest.cc:39



::

     27 
     28 #include "GGEO_LOG.hh"
     29 #include "CFG4_LOG.hh"
     30 #include "PLOG.hh"
     31 
     32 
     33 int main(int argc, char** argv)
     34 {
     35     PLOG_(argc, argv);
     36 
     37     LOG(info) << argv[0] ;
     38 
     39     CFG4_LOG__ ;       ### <<<<
     40     GGEO_LOG__ ;
     41 
     42     Opticks ok(argc, argv);
     43     OpticksHub hub(&ok) ;




gdb CInterpolationTest
------------------------

::

    Reading symbols from CInterpolationTest...done.
    (gdb) r
    Starting program: /usr/local/opticks/lib/CInterpolationTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2017-10-18 15:57:32.191 INFO  [4049] [main@53] /usr/local/opticks/lib/CInterpolationTest
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_S_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff5f9c428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    54	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  0x00007ffff5f9c428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    #1  0x00007ffff5f9e02a in __GI_abort () at abort.c:89
    #2  0x00007ffff65d684d in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #3  0x00007ffff65d46b6 in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #4  0x00007ffff65d4701 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #5  0x00007ffff65d4919 in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007ffff65fd14f in std::__throw_logic_error(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x000000000040754f in std::string::_S_construct<char const*> (__beg=0x0, __end=0xffffffffffffffff <error: Cannot access memory at address 0xffffffffffffffff>, __a=...)
        at /usr/include/c++/4.8/bits/basic_string.tcc:133
    #8  0x00007ffff6617d16 in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #9  0x00007ffff7bcc5ce in PLOG::_prefixlevel_parse (argc=1508335052, argv=0x7fffffff00bf, fallback=0x7ffff7bd1bbe "INFO", prefix=0x4087c8 "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:82
    #10 0x00007ffff7bcc8c3 in PLOG::prefixlevel_parse (this=0x7fffffffd6f0, fallback=0x7ffff7bd1bbe "INFO", prefix=0x4087c8 "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:131
    #11 0x00007ffff7bcc890 in PLOG::prefixlevel_parse (this=0x7fffffffd6f0, _fallback=plog::info, prefix=0x4087c8 "CFG4") at /home/gpu/opticks/sysrap/PLOG.cc:127
    #12 0x000000000040434c in main (argc=1, argv=0x7fffffffd998) at /home/gpu/opticks/cfg4/tests/CInterpolationTest.cc:55



::

     47 
     48 
     49 int main(int argc, char** argv)
     50 {
     51     PLOG_(argc, argv);
     52 
     53     LOG(info) << argv[0] ;
     54 
     55     CFG4_LOG__ ;
     56     GGEO_LOG__ ;
     57 
     58     Opticks ok(argc, argv);
     59     OpticksHub hub(&ok) ;
     60 
     61     CG4 g4(&hub);





Watching PLOG::instance with lldb
-------------------------------------


::

    simon:opticks blyth$ lldb OpticksTest 
    (lldb) target create "OpticksTest"
    Current executable set to 'OpticksTest' (x86_64).
    (lldb) watchpoint set variable PLOG::instance
    error: invalid frame                           ## need to build the stack frames first 
    (lldb) b PLOG::PLOG
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) r
    Process 54869 launched: '/usr/local/opticks/lib/OpticksTest' (x86_64)
    2 locations added to breakpoint 1
    Process 54869 stopped
    * thread #1: tid = 0x2e53d, 0x000000010088abaf libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfedb8, argc_=1, argv_=0x00007fff5fbfee10, fallback=0x000000010001034a, prefix=0x0000000000000000) + 31 at PLOG.cc:209, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
        frame #0: 0x000000010088abaf libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfedb8, argc_=1, argv_=0x00007fff5fbfee10, fallback=0x000000010001034a, prefix=0x0000000000000000) + 31 at PLOG.cc:209
       206               << std::endl
       207               ;
       208  
    -> 209  }
       210  
       211  
       212  
    (lldb) watchpoint set variable PLOG::instance
    Watchpoint created: Watchpoint 1: addr = 0x10089ca10 size = 8 state = enabled type = w
        declare @ '/Users/blyth/opticks/sysrap/PLOG.hh:143'
        watchpoint spec = 'PLOG::instance'
        new value: 0x0000000000000000
    (lldb) c
    Process 54869 resuming
    Process 54869 stopped
    * thread #1: tid = 0x2e53d, 0x000000010088aa3f libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfedb8, argc_=1, argv_=0x00007fff5fbfee10, fallback=0x000000010001034a, prefix=0x0000000000000000) + 31 at PLOG.cc:197, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x000000010088aa3f libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfedb8, argc_=1, argv_=0x00007fff5fbfee10, fallback=0x000000010001034a, prefix=0x0000000000000000) + 31 at PLOG.cc:197
       194        level(info),
       195        logpath(_logpath_parse(argc_, argv_)),
       196        logmax(3)
    -> 197  {
       198     level = prefix == NULL ?  parse(fallback) : prefixlevel_parse(fallback, prefix ) ;    
       199  
       200     assert( instance == NULL && "ONLY EXPECTING A SINGLE PLOG INSTANCE" );
    (lldb) c
    Process 54869 resuming
    SAr _argc 1 (  OpticksTest ) 
    Process 54869 stopped
    * thread #1: tid = 0x2e53d, 0x000000010088ab2f libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfedb8, argc_=1, argv_=0x00007fff5fbfee10, fallback=0x000000010001034a, prefix=0x0000000000000000) + 271 at PLOG.cc:203, queue = 'com.apple.main-thread', stop reason = watchpoint 1
        frame #0: 0x000000010088ab2f libSysRap.dylib`PLOG::PLOG(this=0x00007fff5fbfedb8, argc_=1, argv_=0x00007fff5fbfee10, fallback=0x000000010001034a, prefix=0x0000000000000000) + 271 at PLOG.cc:203
       200     assert( instance == NULL && "ONLY EXPECTING A SINGLE PLOG INSTANCE" );
       201     instance = this ; 
       202  
    -> 203     std::cerr << "PLOG::PLOG " 
       204               << " instance " << instance 
       205               << " this " << this 
       206               << std::endl

    Watchpoint 1 hit:
    old value: 0x0000000000000000
    new value: 0x00007fff5fbfedb8
    (lldb) c
    Process 54869 resuming
    PLOG::PLOG  instance 0x7fff5fbfedb8 this 0x7fff5fbfedb8
    2017-10-23 12:08:27.029 INFO  [189757] [main@73] OpticksTest
    2017-10-23 12:08:27.031 INFO  [189757] [OpticksQuery::dump@79] OpticksQuery::init queryType range query_string range:3153:12221 query_name NULL query_index 0 query_depth 0 no_selection 0 nrange 2 : 3153 : 12221
    2017-10-23 12:08:27.032 INFO  [189757] [Opticks::init@327] Opticks::init DONE OpticksResource::desc digest 96ff965744a2f6b78c24e33c80d3a4cd age.tot_seconds 4756924 age.tot_minutes 79282.070 age.tot_hours 1321.368 age.tot_days     55.057
    2017-10-23 12:08:27.032 INFO  [189757] [Opticks::dumpArgs@768] Opticks::configure argc 1
      0 : OpticksTest
    2017-10-23 12:08:27.032 INFO  [189757] [Opticks::configure@836] Opticks::configure  m_size 2880,1704,2,0 m_position 200,200,0,0 prefdir $HOME/.opticks/dayabay/State
    2017-10-23 12:08:27.032 INFO  [189757] [Opticks::configure@857] Opticks::configure DONE  verbosity 0
    2017-10-23 12:08:27.032 INFO  [189757] [Opticks::Summary@884] Opticks::Summary sourceCode 4096 sourceType torch mode INTEROP_MODE
    Opticks::Summary
    install_prefix    : /usr/local/opticks
    opticksdata_dir   : /usr/local/opticks/opticksdata
    resource_dir      : /usr/local/opticks/opticksdata/resource
    valid    : valid
    envprefix: OPTICKS_
    geokey   : OPTICKSDATA_DAEPATH_DYB
    daepath  : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    gdmlpath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    gltfpath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    metapath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini
    query    : range:3153:12221
    ctrl     : volnames
    digest   : 96ff965744a2f6b78c24e33c80d3a4cd
    idpath   : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    idpath_tmp NULL
    idfold   : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
    idname   : DayaBay_VGDX_20140414-1300
    idbase   : /usr/local/opticks/opticksdata/export
    detector : dayabay
    detector_name : DayaBay
    detector_base : /usr/local/opticks/opticksdata/export/DayaBay
    material_map  : /usr/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json
    getPmtPath(0) : /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    meshfix  : iav,oav
    example_matnames  : GdDopedLS,Acrylic,LiquidScintillator,MineralOil,Bialkali
    sensor_surface    : lvPmtHemiCathodeSensorSurface
    default_medium    : MineralOil
    ------ from /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini -------- 
                          AnalyticPMTMedium 2017-10-23 12:08:27.034 INFO  [189757] [*Opticks::getAnalyticPMTMedium@608]  cmed  cmed.empty 1 dmed MineralOil dmed.empty 0
    MineralOil
    2017-10-23 12:08:27.034 INFO  [189757] [Opticks::Summary@898] Opticks::SummaryDONE
    2017-10-23 12:08:27.035 INFO  [189757] [main@85] OpticksTest::main aft configure
    2017-10-23 12:08:27.035 INFO  [189757] [Opticks::MaterialSequence@1363] Opticks::MaterialSequence seqmat 123456789abcdef
    2017-10-23 12:08:27.035 INFO  [189757] [*Opticks::Material@1355] Opticks::Material populating global G_MATERIAL_NAMES 
    2017-10-23 12:08:27.035 FATAL [189757] [NTxt::read@65] NTxt::read failed to open /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GMaterialLib
    2017-10-23 12:08:27.035 INFO  [189757] [test_MaterialSequence@23] OpticksTest::main seqmat 123456789abcdef MaterialSequence NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL NULL 
    2017-10-23 12:08:27.035 INFO  [189757] [test_path@38] getDAEPath path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae npath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae exists 1
    2017-10-23 12:08:27.035 INFO  [189757] [test_path@38] getGDMLPath path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml npath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml exists 1
    2017-10-23 12:08:27.035 INFO  [189757] [test_path@38] getMaterialMap path /usr/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json npath /usr/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json exists 1
    Process 54869 exited with status = 0 (0x00000000) 
    (lldb) 




