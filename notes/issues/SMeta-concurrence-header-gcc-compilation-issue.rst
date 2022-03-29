SMeta-concurrence-header-gcc-compilation-issue
=================================================



Hi Simon,


Now, at your suggestion, I carried out the following operations:



[jiajun@compute-0-5 opticks]$ nljson-
[jiajun@compute-0-5 opticks]$ nljson--
=== nljson-pc : /home/jiajun/workspace/JUNO-offline/JUNO-SOFT/ExternalLibs/opticks/head/externals/lib/pkgconfig/NLJSON.pc
[jiajun@compute-0-5 opticks]$ plog-
[jiajun@compute-0-5 opticks]$ plog--
=== plog-get : url https://github.com/simoncblyth/plog.git
=== plog-get : plog already cloned
=== plog-pc : /home/jiajun/workspace/JUNO-offline/JUNO-SOFT/ExternalLibs/opticks/head/externals/lib/pkgconfig/PLog.pc
And then,

 

[jiajun@compute-0-5 sysrap]$ g++ SMeta.cc -std=c++11
In file included from SMeta.cc:7:0:
PLOG.hh:26:23: fatal error: /plog/Log.h: No such file or directory
 #include </plog/Log.h>
                       ^
compilation terminated.
[jiajun@compute-0-5 sysrap]$ vi PLOG.hh
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

Many thanks,


Best,

Jiajun





