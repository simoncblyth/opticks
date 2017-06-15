Malloc Debugging
====================

* https://developer.apple.com/library/content/documentation/Performance/Conceptual/ManagingMemory/Articles/MallocDebug.html




how malloc works
------------------

* https://www.cocoawithlove.com/2010/05/look-at-how-malloc-works-on-mac.html


MallocStackLogging 
          to record all stacks. Tools like leaks can then be applied
MallocStackLoggingNoCompact 
         to record all stacks. Needed for malloc_history



malloc_history
---------------

::


    malloc_history(1)         BSD General Commands Manual        malloc_history(1)

    NAME
         malloc_history -- Show the malloc allocations that the process has performed

    SYNOPSIS
         malloc_history pid [-highWaterMark] address [address ...]
         malloc_history pid -allBySize [-highWaterMark] [address ...]
         malloc_history pid -allByCount [-highWaterMark] [address ...]
         malloc_history pid -allEvents [-highWaterMark]
         malloc_history pid -callTree [-highWaterMark] [-showContent] [-invert] [-ignoreThreads] [-collapseRecursion] [-chargeSystemLibraries] [address ...]

    DESCRIPTION
         malloc_history inspects a given process and lists the malloc allocations performed by it.  The target process may be specified by pid or by full or partial name.
         malloc_history relies on information provided by the standard malloc library when malloc stack logging has been enabled for the target process.  See below for further
         information.



libgmalloc
-----------

::

    simon:gmalloctest blyth$ lldb /tmp/gmalloctest
    (lldb) target create "/tmp/gmalloctest"
    Current executable set to '/tmp/gmalloctest' (x86_64).
    (lldb) env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib
    (lldb) r
    GuardMalloc[sh-30107]: Allocations will be placed on 16 byte boundaries.
    GuardMalloc[sh-30107]:  - Some buffer overruns may not be noticed.
    GuardMalloc[sh-30107]:  - Applications using vector instructions (e.g., SSE) should work.
    GuardMalloc[sh-30107]: version 27
    GuardMalloc[arch-30107]: Allocations will be placed on 16 byte boundaries.
    GuardMalloc[arch-30107]:  - Some buffer overruns may not be noticed.
    GuardMalloc[arch-30107]:  - Applications using vector instructions (e.g., SSE) should work.
    GuardMalloc[arch-30107]: version 27
    Process 30107 launched: '/tmp/gmalloctest' (x86_64)
    GuardMalloc[gmalloctest-30107]: Allocations will be placed on 16 byte boundaries.
    GuardMalloc[gmalloctest-30107]:  - Some buffer overruns may not be noticed.
    GuardMalloc[gmalloctest-30107]:  - Applications using vector instructions (e.g., SSE) should work.
    GuardMalloc[gmalloctest-30107]: version 27
    Process 30107 stopped
    * thread #1: tid = 0x744e23, 0x0000000100000efc gmalloctest`main(argc=1, argv=0x00007fff5fbfee48) + 76 at gmalloctest.c:13, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10034a000)
        frame #0: 0x0000000100000efc gmalloctest`main(argc=1, argv=0x00007fff5fbfee48) + 76 at gmalloctest.c:13
       10       unsigned i;
       11   
       12       for (i = 0; i < 200; i++) {
    -> 13          buffer[i] = i;
       14       }
       15   
       16       for (i = 0; i < 200; i++) {
    (lldb) p i
    (unsigned int) $0 = 100
    (lldb) ^D
    simon:gmalloctest blyth$ 





Unfortunately, the normal build of magazine_malloc.c in Mac OS X has the
limitation that it won't apply guard pages to "small" or "tiny" allocations. To
apply guard pages to all data, you'll need to use the libgmalloc library. Do
this by setting the following environment variable:

export DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib For more information,
see the libgmalloc Manual Page.


::


    libgmalloc(3)            BSD Library Functions Manual            libgmalloc(3)

    NAME
         libgmalloc -- (Guard Malloc), an aggressive debugging malloc library

    DESCRIPTION
         libgmalloc is a debugging malloc library that can track down insidious bugs in your code or library.  If your application crashes when using libgmalloc, then you've found a
         bug.

         libgmalloc is used in place of the standard system malloc, and uses the virtual memory system to identify memory access bugs.  Each malloc allocation is placed on its own
         virtual memory page (or pages).  By default, the returned address for the allocation is positioned such that the end of the allocated buffer is at the end of the last page,
         and the next page after that is kept unallocated.  Thus, accesses beyond the end of the buffer cause a bad access error immediately.  When memory is freed, libgmalloc deal-
         locates its virtual memory, so reads or writes to the freed buffer cause a bad access error.  Bugs which had been difficult to isolate become immediately obvious, and
         you'll know exactly which code is causing the problem.

         Guard Malloc is thread-safe and works for all uses of malloc(), Objective-C's alloc method, C++'s new operator, and other functions which result in allocation in the malloc
         heap.

         As of Mac OS X 10.5, libgmalloc aligns the start of allocated buffers on 16-byte boundaries by default, to allow proper use of vector instructions (e.g., SSE).  (The use of
         vector instructions is common, including in some Mac OS X system libraries.  The regular system malloc also uses 16-byte alignment.)  Because of this 16-byte alignment, up
         to 15 bytes at the end of an allocated block may be excess at the end of the page, and libgmalloc will not detect buffer overruns into that area by default.  This default
         alignment can be changed with environment variables.

         libgmalloc is available in /usr/lib/libgmalloc.dylib.  To use it, set this environment variable:

               set DYLD_INSERT_LIBRARIES to /usr/lib/libgmalloc.dylib

         Note:  it is no longer necessary to set DYLD_FORCE_FLAT_NAMESPACE.

         This tells dyld to use Guard Malloc instead of the standard version of malloc.  Run the program, and wait for the crash indicating the bad access.  When the program
         crashes, examine it in the debugger to identify the cause.

         As of Mac OS X 10.6, libgmalloc can be used with the standard malloc stack logging by setting the MallocStackLogging environment variable.  The malloc_history(1) command
         can then be used to show backtraces of all malloc and free events made when using libgmalloc.

    USING libgmalloc WITH THE XCODE DEBUGGER OR LLDB
         Because the goal of libgmalloc is to "encourage" your application to crash if memory access errors occur, it is best to run your application under a debugger such as the
         Xcode IDE's debugger, or lldb at the command line.

         To use Guard Malloc with the Xcode debugger, choose Edit Scheme... from the Scheme popup.  Click on the Diagnostics tab then turn on the Enable Guard Malloc check box.
         Then when launching the target application, Xcode automatically sets the DYLD_INSERT_LIBRARIES environment variable properly.  Xcode retains that setting with that exe-
         cutable.  To set any of the additional environment variables described below, click on the Arguments tab in the Scheme editor and add them in the Environment Variables sec-
         tion.

         If you're using lldb from the command line, use lldb's "settings set target.env-vars VAR=VALUE" command to set the environment variables.  Or simply use the "env VAR=VALUE"
         command alias.




leaks
-------

::

    leaks(1)                  BSD General Commands Manual                 leaks(1)

    NAME
         leaks -- Search a process's memory for unreferenced malloc buffers

    SYNOPSIS
         leaks pid | partial-executable-name [-nocontext] [-nostacks] [-exclude symbol] [-trace address]

    DESCRIPTION
         leaks identifies leaked memory -- memory that the application has allocated, but has been lost and cannot be freed.  Specifically, leaks examines a specified process's mem-
         ory for values that may be pointers to malloc-allocated buffers.  Any buffer reachable from a pointer in writable global memory (e.g., __DATA segments), a register, or on
         the stack is assumed to be memory in use.  Any buffer reachable from a pointer in a reachable malloc-allocated buffer is also assumed to be in use.  The buffers which are
         not reachable are leaks; the buffers could never be freed because no pointer exists in memory to the buffer, and thus free() could never be called for these buffers.  Such
         buffers waste memory; removing them can reduce swapping and memory usage.  Leaks are particularly dangerous for long-running programs, for eventually the leaks could fill
         memory and cause the application to crash.



Detecting Heap Corruption
----------------------------


To enable heap checking, assign values to the MallocCheckHeapStart and
MallocCheckHeapEach environment variables. You must set both of these variables
to enable heap checking. The MallocCheckHeapStart variable tells the malloc
library how many malloc calls to process before initiating the first heap
check. Set the second to the number of malloc calls to process between heap
checks.

The MallocCheckHeapStart variable is useful when the heap corruption occurs at
a predictable time. Once it hits the appropriate start point, the malloc
library starts logging allocation messages to the Terminal window. You can
watch the number of allocations and use that information to determine
approximately where the heap is being corrupted. Adjust the values for
MallocCheckHeapStart and MallocCheckHeapEach as necessary to narrow down the
actual point of corruption.









::

man malloc

ENVIRONMENT
     The following environment variables change the behavior of the allocation-related functions.

     MallocLogFile <f>            Create/append messages to the given file path <f> instead of writing to the standard error.

     MallocGuardEdges             If set, add a guard page before and after each large block.

     MallocDoNotProtectPrelude    If set, do not add a guard page before large blocks, even if the MallocGuardEdges environment variable is set.

     MallocDoNotProtectPostlude   If set, do not add a guard page after large blocks, even if the MallocGuardEdges environment variable is set.

     MallocStackLogging           If set, record all stacks, so that tools like leaks can be used.

     MallocStackLoggingNoCompact  If set, record all stacks in a manner that is compatible with the malloc_history program.

     MallocStackLoggingDirectory  If set, records stack logs to the directory specified instead of saving them to the default location (/tmp).

     MallocScribble               If set, fill memory that has been allocated with 0xaa bytes.  This increases the likelihood that a program
                                  making assumptions about the contents of freshly allocated memory will fail.  Also if set, fill memory that
                                  has been deallocated with 0x55 bytes.  This increases the likelihood that a program will fail due to accessing
                                  memory that is no longer allocated.

     MallocCheckHeapStart <s>     If set, specifies the number of allocations <s> to wait before begining periodic heap checks every <n> as
                                  specified by MallocCheckHeapEach.  If MallocCheckHeapStart is set but MallocCheckHeapEach is not specified,
                                  the default check repetition is 1000.

     MallocCheckHeapEach <n>      If set, run a consistency check on the heap every <n> operations.  MallocCheckHeapEach is only meaningful if
                                  MallocCheckHeapStart is also set.

     MallocCheckHeapSleep <t>     Sets the number of seconds to sleep (waiting for a debugger to attach) when MallocCheckHeapStart is set and a
                                  heap corruption is detected.  The default is 100 seconds.  Setting this to zero means not to sleep at all.
                                  Setting this to a negative number means to sleep (for the positive number of seconds) only the very first time
                                  a heap corruption is detected.

     MallocCheckHeapAbort <b>     When MallocCheckHeapStart is set and this is set to a non-zero value, causes abort(3) to be called if a heap
                                  corruption is detected, instead of any sleeping.

     MallocErrorAbort             If set, causes abort(3) to be called if an error was encountered in malloc(3) or free(3) , such as a calling
                                  free(3) on a pointer previously freed.

     MallocCorruptionAbort        Similar to MallocErrorAbort but will not abort in out of memory conditions, making it more useful to catch
                                  only those errors which will cause memory corruption.  MallocCorruptionAbort is always set on 64-bit pro-
                                  cesses.



::

    simon:optickscore blyth$ tboolean-;tboolean-hybrid --GGEO debug
    288 -rwxr-xr-x  1 blyth  staff  143804 Jun 15 13:26 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest --GGEO debug --animtimemax 10 --timemax 10 --geocenter --eye 0,0,1 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tboolean-hybrid--_name=tboolean-hybrid--_mode=PyCsgInBox --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --tag 1 --cat boolean --save
    OKTest(29616,0x7fff75379310) malloc: protecting edges
    OKTest(29616,0x7fff75379310) malloc: stack logs being written into /tmp/stack-logs.29616.10acf3000.OKTest.8P58WC.index
    OKTest(29616,0x7fff75379310) malloc: recording malloc and VM allocation stacks to disk using standard recorder
    OKTest(29616,0x7fff75379310) malloc: stack logging compaction turned off; size of log files on disk can increase rapidly
    OKTest(29616,0x7fff75379310) malloc: enabling scribbling to detect mods to free blocks
    OKTest(29616,0x7fff75379310) malloc: checks heap after 1th operation and each 1 operations
    OKTest(29616,0x7fff75379310) malloc: will sleep for 100 seconds on heap corruption
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=10000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=20000
    2017-06-15 13:49:14.106 INFO  [7614443] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=30000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=40000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=50000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=60000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=70000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=80000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=90000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=100000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=110000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=120000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=130000
    OKTest(29616,0x7fff75379310) malloc: at szone_check counter=140000



