Ubuntu16.04.6-gcc5.4.0-npy-NNodeDumpTest-SEGV
=================================================

Succeed to reproduce what looks like the issue reported by Elias, 
by installing virtualbox + Ubuntu 16.04.6 and doing a partial Opticks install.

Note that with virtualbox + Ubuntu 18.04.2 the issue does not occur.  See vbx- 

Table of Ubuntu release dates: https://wiki.ubuntu.com/Releases



::

    97% tests passed, 3 tests failed out of 117

    Total Test time (real) =   1.93 sec

    The following tests FAILED:
         67 - NPYTest.NNodeDumpTest (SEGFAULT)
         82 - NPYTest.NLoadTest (Child aborted)
         92 - NPYTest.NCSGRoundTripTest (SEGFAULT)
    Errors while running CTest
    Thu Apr  4 21:19:59 CST 2019


::

    (gdb) r
    Starting program: /usr/local/opticks/lib/NNodeDumpTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2019-04-04 21:20:41.886 INFO  [3247] [test_dump@15] 
     sample idx : 0

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7972274 in NNodeDump2::dump_label (this=0x621710, pfx=0x7ffff7ac01c2 "du") at /home/blyth/opticks/npy/NNodeDump2.cpp:38
    38           << std::setw(nnode::desc_indent) << m_node->desc() 
    (gdb) list
    33  
    34  void NNodeDump2::dump_label(const char* pfx) const 
    35  {
    36      std::cout 
    37           << std::setw(3) << (  pfx ? pfx : "-" ) << " " 
    38           << std::setw(nnode::desc_indent) << m_node->desc() 
    39           ; 
    40  }
    41  
    42  void NNodeDump2::dump_base() const 
    (gdb) p m_node
    $1 = (const nnode *) 0x7fffffffe0c0
    (gdb) p m_node->desc()
    Cannot access memory at address 0x40
    (gdb) p m_node
    $2 = (const nnode *) 0x7fffffffe0c0
    (gdb) bt
    #0  0x00007ffff7972274 in NNodeDump2::dump_label (this=0x621710, pfx=0x7ffff7ac01c2 "du") at /home/blyth/opticks/npy/NNodeDump2.cpp:38
    #1  0x00007ffff797237b in NNodeDump2::dump_base (this=0x621710) at /home/blyth/opticks/npy/NNodeDump2.cpp:44
    #2  0x00007ffff7972105 in NNodeDump2::dump (this=0x621710) at /home/blyth/opticks/npy/NNodeDump2.cpp:22
    #3  0x00007ffff795de5d in nnode::dump (this=0x621730, msg=0x0) at /home/blyth/opticks/npy/NNode.cpp:1360
    #4  0x000000000040492c in test_dump (nodes=std::vector of length 12, capacity 16 = {...}, idx=0) at /home/blyth/opticks/npy/tests/NNodeDumpTest.cc:16
    #5  0x00000000004049a8 in test_dump (nodes=std::vector of length 12, capacity 16 = {...}) at /home/blyth/opticks/npy/tests/NNodeDumpTest.cc:21
    #6  0x0000000000404b6d in main (argc=1, argv=0x7fffffffe388) at /home/blyth/opticks/npy/tests/NNodeDumpTest.cc:38
    (gdb) f 3
    #3  0x00007ffff795de5d in nnode::dump (this=0x621730, msg=0x0) at /home/blyth/opticks/npy/NNode.cpp:1360
    1360        _dump->dump();
    (gdb) 


