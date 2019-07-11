double free or corruption
============================


ISSUE : double free or corruption at cleanup : FIXED BY FULL BUILD 
-------------------------------------------------------------------------

::

    2019-07-11 11:46:29.875 INFO  [128995] [CG4::cleanup@403] [
    2019-07-11 11:46:29.875 INFO  [128995] [CG4::cleanup@405] ]
    *** Error in `/home/blyth/local/opticks/lib/OKG4Test': double free or corruption (out): 0x000000000a1b9230 ***
    ======= Backtrace: =========
    /lib64/libc.so.6(+0x81489)[0x7effa5be4489]
    /lib64/libnvoptix.so.1(+0x6e1514)[0x7eff81309514]
    /lib64/libnvoptix.so.1(+0x6e1a76)[0x7eff81309a76]
    /lib64/libnvoptix.so.1(+0x57f790)[0x7eff811a7790]
    /lib64/libc.so.6(+0x39b69)[0x7effa5b9cb69]
    /lib64/libc.so.6(+0x39bb7)[0x7effa5b9cbb7]
    /lib64/libc.so.6(__libc_start_main+0xfc)[0x7effa5b853dc]
    /home/blyth/local/opticks/lib/OKG4Test[0x403709]
    ======= Memory map: ========
    00400000-0040a000 r-xp 00000000 fd:02 138492934                          /home/blyth/local/opticks/lib/OKG4Test
    00609000-0060a000 r--p 00009000 fd:02 138492934                          /home/blyth/local/opticks/lib/OKG4Test
    0060a000-0060b000 rw-p 0000a000 fd:02 138492934                          /home/blyth/local/opticks/lib/OKG4Test
    01b65000-13b17000 rw-p 00000000 00:00 0                                  [heap]
    200000000-200200000 rw-s 00000000 00:05 34651                            /dev/nvidiactl



Occured after header changes across 4 subprojs
--------------------------------------------------

::

    [blyth@localhost okop]$ o
    M notes/issues/large-vm-for-cuda-process.rst
    M notes/issues/plugging-cfg4-leaks.rst
    M oglrap/OpticksViz.cc
    M oglrap/OpticksViz.hh
    M ok/OKPropagator.cc
    M ok/OKPropagator.hh
    M okg4/OKG4Mgr.cc
    M okg4/OKG4Mgr.hh
    M okop/OpEngine.cc
    M okop/OpEngine.hh
    ? notes/issues/double-free.rst
    [blyth@localhost opticks]$ 



* in normal development updating one or two subprojs and
  running the individual subproj updaters eg okop-- ok-- works fine 

* BUT after making changes across many sub-projects, expecially 
  to headers this "double free or corruption" issue sometimes happens,
  manifesting usually at cleanup

* solution this time was a simple full build, a clean
  build would have been the next thing to try 

::

    o
    om--


