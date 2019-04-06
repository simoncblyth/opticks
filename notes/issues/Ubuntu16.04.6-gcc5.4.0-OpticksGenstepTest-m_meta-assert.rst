Ubuntu16.04.6-gcc5.4.0-OpticksGenstepTest-m_meta-assert  : FIXED
====================================================================

Continued Ubuntu-16 install beyond NPY inside virtualbox::

    cd ~/opticks/optickscore
    om-install
    om-test    # many fails from lack of resources

    opticksdata-;opticksdata--
    ..
    2173 files updated, 0 files merged, 0 files removed, 0 files unresolved                                                                                                                                                             
    === opticksdata-export-ini : writing OPTICKS_DAEPATH_ environment to /usr/local/opticks/opticksdata/config/opticksdata.ini
    OPTICKSDATA_DAEPATH_DFAR=/usr/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    OPTICKSDATA_DAEPATH_DLIN=/usr/local/opticks/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    OPTICKSDATA_DAEPATH_DPIB=/usr/local/opticks/opticksdata/export/dpib/cfg4.dae
    OPTICKSDATA_DAEPATH_DYB=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    OPTICKSDATA_DAEPATH_J1707=/usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
    OPTICKSDATA_DAEPATH_J1808=/usr/local/opticks/opticksdata/export/juno1808/g4_00.dae
    OPTICKSDATA_DAEPATH_JPMT=/usr/local/opticks/opticksdata/export/juno/test3.dae
    OPTICKSDATA_DAEPATH_LXE=/usr/local/opticks/opticksdata/export/LXe/g4_00.dae

    ## Notice that getting the opticksdata writes /usr/local/opticks/opticksdata/config/opticksdata.ini
    ## which points at the input dae files
    ##
    ## This is the old export/import way of handling geometry with the inputs coming from opticksdata, 
    ## not the new direct approach. 


After getting opticksdata are down one FAIL : OpticksGenstepTest. Matching issue reported by Elias.:: 

    cd ~/opticks/optickscore  # okc-c
    om-test
    ...
    26/26 Test #26: OpticksCoreTest.OpticksGenstepTest .........Child aborted***Exception:   0.18 sec

    96% tests passed, 1 tests failed out of 26

    Total Test time (real) =   0.44 sec

    The following tests FAILED:
         26 - OpticksCoreTest.OpticksGenstepTest (Child aborted)
    Errors while running CTest
    Sat Apr  6 10:51:49 CST 2019



    blyth@blyth-VirtualBox:~/opticks/optickscore$ OpticksGenstepTest
    OpticksGenstepTest: /home/blyth/opticks/npy/NPYBase.cpp:217: T NPYBase::getMeta(const char*, const char*) const [with T = int]: Assertion \`m_meta\' failed.
    Aborted (core dumped)

    blyth@blyth-VirtualBox:~/opticks/optickscore$ gdb $(which OpticksGenstepTest)
    GNU gdb (Ubuntu 7.11.1-0ubuntu1~16.5) 7.11.1
    ...
    OpticksGenstepTest: /home/blyth/opticks/npy/NPYBase.cpp:217: T NPYBase::getMeta(const char*, const char*) const [with T = int]: Assertion \`m_meta\' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff64aa428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    54  ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  0x00007ffff64aa428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    #1  0x00007ffff64ac02a in __GI_abort () at abort.c:89
    #2  0x00007ffff64a2bd7 in __assert_fail_base (fmt=<optimized out>, assertion=assertion@entry=0x7ffff771280c "m_meta", file=file@entry=0x7ffff7712580 "/home/blyth/opticks/npy/NPYBase.cpp", line=line@entry=217, 
        function=function@entry=0x7ffff7712b60 <int NPYBase::getMeta<int>(char const*, char const*) const::__PRETTY_FUNCTION__> "T NPYBase::getMeta(const char*, const char*) const [with T = int]") at assert.c:92
    #3  0x00007ffff64a2c82 in __GI___assert_fail (assertion=0x7ffff771280c "m_meta", file=0x7ffff7712580 "/home/blyth/opticks/npy/NPYBase.cpp", line=217, 
        function=0x7ffff7712b60 <int NPYBase::getMeta<int>(char const*, char const*) const::__PRETTY_FUNCTION__> "T NPYBase::getMeta(const char*, const char*) const [with T = int]") at assert.c:101
    #4  0x00007ffff74e1f29 in NPYBase::getMeta<int> (this=0x61fc50, key=0x7ffff77125e7 "ArrayContentVersion", fallback=0x7ffff77125cc "0") at /home/blyth/opticks/npy/NPYBase.cpp:217
    #5  0x00007ffff74df43c in NPYBase::getArrayContentVersion (this=0x61fc50) at /home/blyth/opticks/npy/NPYBase.cpp:228
    #6  0x00007ffff7b204d1 in OpticksGenstep::getContentVersion (this=0x61fa20) at /home/blyth/opticks/optickscore/OpticksGenstep.cc:30
    #7  0x00007ffff7b2068d in OpticksGenstep::desc[abi:cxx11]() const (this=0x61fa20) at /home/blyth/opticks/optickscore/OpticksGenstep.cc:46
    #8  0x00007ffff7b21576 in OpticksGenstep::dump (this=0x61fa20, modulo=1000, margin=10, msg=0x407156 "OpticksGenstep::dump") at /home/blyth/opticks/optickscore/OpticksGenstep.cc:164
    #9  0x0000000000403ec8 in main (argc=1, argv=0x7fffffffe088) at /home/blyth/opticks/optickscore/tests/OpticksGenstepTest.cc:21
    (gdb) 




Oops OpticksGenstepTest has a hardcoded path, it happens to be correct in the virtualbox + on macOS though::

     01 // TEST=OpticksGenstepTest om-t
      2 
      3 #include "OPTICKS_LOG.hh"
      4 #include "NPY.hpp"
      5 #include "OpticksGenstep.hh"
      6 
      7 int main(int argc, char** argv)
      8 {
      9     OPTICKS_LOG(argc, argv);
     10 
     11     const char* def = "/usr/local/opticks/opticksdata/gensteps/dayabay/natural/1.npy" ;
     12     const char* path = argc > 1 ? argv[1] : def ;
     13 
     14     NPY<float>* np = NPY<float>::load(path) ;
     15     if(np == NULL) return 0 ;
     16 
     17     OpticksGenstep* gs = new OpticksGenstep(np) ;
     18 
     19     unsigned modulo = 1000 ;
     20     unsigned margin = 10 ;
     21     gs->dump( modulo, margin ) ;
     22 
     23     return 0 ;
     24 }


Added a DATADIR internal BResource key to avoid the hardcoded path.


