SMeta-concurrence-header-gcc-compilation-issue
=================================================

::

    [jiajun@compute-0-5 opticks]$ nljson-
    [jiajun@compute-0-5 opticks]$ nljson--
    === nljson-pc : /home/jiajun/workspace/JUNO-offline/JUNO-SOFT/ExternalLibs/opticks/head/externals/lib/pkgconfig/NLJSON.pc
    [jiajun@compute-0-5 opticks]$ plog-
    [jiajun@compute-0-5 opticks]$ plog--
    === plog-get : url https://github.com/simoncblyth/plog.git
    === plog-get : plog already cloned
    === plog-pc : /home/jiajun/workspace/JUNO-offline/JUNO-SOFT/ExternalLibs/opticks/head/externals/lib/pkgconfig/PLog.pc
    And then,


    [jiajun@compute-0-5 sysrap]$ g++ SMeta.cc -I /home/jiajun/workspace/JUNO-offline/JUNO-SOFT/ExternalLibs/opticks/head/externals/plog/include/ -std=c++11
    In file included from /usr/include/c++/4.8.2/x86_64-redhat-linux/bits/gthr-default.h:35:0,
                     from /usr/include/c++/4.8.2/x86_64-redhat-linux/bits/gthr.h:148,
                     from /usr/include/c++/4.8.2/ext/atomicity.h:35,
                     from /usr/include/c++/4.8.2/bits/ios_base.h:39,
                     from /usr/include/c++/4.8.2/ios:42,
                     from /usr/include/c++/4.8.2/ostream:38,
                     from /usr/include/c++/4.8.2/iostream:39,
                     from SMeta.cc:1:
    /usr/include/c++/4.8.2/ext/concurrence.h:122:34: error: cannot convert ?.brace-enclosed initializer list>?.to ?.hort int?.in initialization
         __gthread_mutex_t _M_mutex = __GTHREAD_MUTEX_INIT;
                                      ^
    /usr/include/c++/4.8.2/ext/concurrence.h:177:44: error: cannot convert ?.brace-enclosed initializer list>?.to ?.hort int?.in initialization
         __gthread_recursive_mutex_t _M_mutex = __GTHREAD_RECURSIVE_MUTEX_INIT;
                                                ^
    [jiajun@compute-0-5 sysrap]$ pwd
    /home/jiajun/workspace/JUNO-offline/JUNO-SOFT/opticks/sysrap
    So, that does seem to be the problem, so could you please tell me what should I do next?

    Jiajun


Hi Jiajun, 

In my commits just now I have added three standalone go.sh scripts.
The scripts do everything, including generating simple sources 
and cloning plog. They use nothing from the rest of Opticks.  

opticks/examples/UseIOStreamStandalone/go.sh
    simple "hello world" use of iostream

opticks/examples/UsePlogStandalone/go.sh
    test compilation with my fork of plog 
    and with the latest plog by changing the url in the script 
    or by URL envvar. 

opticks/examples/UsePthreadStandalone/go.sh
    simple use of pthread sensitive to argumnent 0 or 1
 
    0: C usage with stdio.h/printf 
    1: C++ usage with iostream/std::cout 

Use can use these scripts to test your environment and check if there
is any difference between the old Plog that Opticks currently uses
and the latest Plog. 


* https://stackoverflow.com/questions/71153495/compile-gcc4-9-2-from-source-on-centos7

Googling reveals the above from someone with a similar error. 
They concluded the problem is arising from mixed pthread versions. 

Due to this I added UsePthreadStandaline. 
iostream includes pthread.h as does plog/Util.h 


Simon








/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/include/c++/8.3.0/ext/atomicity.h 

     29 #ifndef _GLIBCXX_ATOMICITY_H
     30 #define _GLIBCXX_ATOMICITY_H    1
     31 
     32 #pragma GCC system_header
     33 
     34 #include <bits/c++config.h>
     35 #include <bits/gthr.h>
     36 #include <bits/atomic_word.h>
     37 
     38 namespace __gnu_cxx _GLIBCXX_VISIBILITY(default)
     39 {
     40 _GLIBCXX_BEGIN_NAMESPACE_VERSION
        

    N[blyth@localhost ~]$ l /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/include/c++/8.3.0/bits/g*
    8 -rw-r--r--. 1 cvmfs cvmfs 7769 Apr 10  2020 /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/include/c++/8.3.0/bits/gslice_array.h
    6 -rw-r--r--. 1 cvmfs cvmfs 5518 Apr 10  2020 /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/include/c++/8.3.0/bits/gslice.h
    N[blyth@localhost ~]$ 

    N[blyth@localhost 8.3.0]$ find . -name gthr.h
    ./x86_64-pc-linux-gnu/bits/gthr.h



::

    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/include/c++/8.3.0/x86_64-pc-linux-gnu/bits/gthr.h
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/include/c++/8.3.0/x86_64-pc-linux-gnu/bits/gthr-default.h

    028 
     29 /* POSIX threads specific definitions.
     30    Easy, since the interface is just one-to-one mapping.  */
     31 
     32 #define __GTHREADS 1
     33 #define __GTHREADS_CXX0X 1
     34 
     35 #include <pthread.h>
     36 
     37 #if ((defined(_LIBOBJC) || defined(_LIBOBJC_WEAK)) \
     38      || !defined(_GTHREAD_USE_MUTEX_TIMEDLOCK))
     39 # include <unistd.h>
     40 # if defined(_POSIX_TIMEOUTS) && _POSIX_TIMEOUTS >= 0
     41 #  define _GTHREAD_USE_MUTEX_TIMEDLOCK 1
     42 # else
     43 #  define _GTHREAD_USE_MUTEX_TIMEDLOCK 0
     44 # endif
     45 #endif
     46 
     47 typedef pthread_t __gthread_t;
     48 typedef pthread_key_t __gthread_key_t;
     49 typedef pthread_once_t __gthread_once_t;
     50 typedef pthread_mutex_t __gthread_mutex_t;
     51 typedef pthread_mutex_t __gthread_recursive_mutex_t;
     52 typedef pthread_cond_t __gthread_cond_t;
     53 typedef struct timespec __gthread_time_t;
     54 
     55 /* POSIX like conditional variables are supported.  Please look at comments
     56    in gthr.h for details. */
     57 #define __GTHREAD_HAS_COND  1
     58 
     59 #define __GTHREAD_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
     60 #define __GTHREAD_MUTEX_INIT_FUNCTION __gthread_mutex_init_function
     61 #define __GTHREAD_ONCE_INIT PTHREAD_ONCE_INIT
     62 #if defined(PTHREAD_RECURSIVE_MUTEX_INITIALIZER)
     63 #define __GTHREAD_RECURSIVE_MUTEX_INIT PTHREAD_RECURSIVE_MUTEX_INITIALIZER
     64 #elif defined(PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP)
     65 #define __GTHREAD_RECURSIVE_MUTEX_INIT PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
     66 #else
     67 #define __GTHREAD_RECURSIVE_MUTEX_INIT_FUNCTION __gthread_recursive_mutex_init_function
     68 #endif
     69 #define __GTHREAD_COND_INIT PTHREAD_COND_INITIALIZER
     70 #define __GTHREAD_TIME_INIT {0,0}
     71 





Googling for "__gthread_mutex_t _M_mutex = __GTHREAD_MUTEX_INIT"

* https://forum.calculate-linux.org/t/solved-dev-qt-qtcore-5-5-1-fails-to-compile/6806






