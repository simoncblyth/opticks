Malloc Debugging
====================


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

