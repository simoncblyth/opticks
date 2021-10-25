plog-appender-not-equal-this-assert
=======================================


::

    O[blyth@localhost CSGOptiX]$ ./cxs.sh
    mo .bashrc VIP_MODE:dev O : ordinary opticks dev ontop of juno externals CMTEXTRATAGS:opticks
    CSGOptiXSimulateTest
    CSGOptiXSimulateTest: /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Logger.h:22: plog::Logger<instance>& plog::Logger<instance>::addAppender(plog::IAppender*) [with int instance = 0]: Assertion `appender != this' failed.
    ./cxs.sh: line 115: 242668 Aborted                 (core dumped) $GDB CSGOptiXSimulateTest
    O[blyth@localhost CSGOptiX]$ 


    (gdb) bt
    #0  0x00007ffff3e18387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff3e19a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff3e111a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff3e11252 in __assert_fail () from /lib64/libc.so.6
    #4  0x000000000041616a in plog::Logger<0>::addAppender (this=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>, 
        appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>) at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Logger.h:22
    #5  0x0000000000415efc in plog::init<0> (maxSeverity=plog::info, appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Init.h:30
    #6  0x0000000000415afb in plog::init (maxSeverity=plog::info, appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Init.h:35
    #7  0x00007ffff77d7c00 in CSG_LOG::Initialize (level=4, app1=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>, app2=0x0) at /home/blyth/opticks/CSG/CSG_LOG.cc:29
    #8  0x0000000000415c27 in OPTICKS_LOG_::Initialize (instance=0x69b350, app1=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/OPTICKS_LOG.hh:217
    #9  0x0000000000411693 in main (argc=1, argv=0x7fffffff4378) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSimulateTest.cc:47
    (gdb) 


There was a prior typo where the below "CSG_GGEO_LOG::Initialize" was "CSG_LOG::Initialize" meaning that this was duplicated.

sysrap/OPTICKS_LOG.hh::

    147        static void Initialize(PLOG* instance, void* app1, void* /*app2*/ )
    148        {
    149            int max_level = instance->parse("info") ;
    150            // note : can decrease verbosity from the max_level in the subproj, but not increase
    151 
    152 #ifdef OPTICKS_SYSRAP
    153     SYSRAP_LOG::Initialize(instance->prefixlevel_parse( max_level, "SYSRAP"), app1, NULL );
    154 #endif
    ...
    216 #ifdef OPTICKS_CSG
    217     CSG_LOG::Initialize(instance->prefixlevel_parse( max_level, "CSG"), app1, NULL );
    218 #endif
    219 #ifdef OPTICKS_CSG_GGEO
    220     CSG_GGEO_LOG::Initialize(instance->prefixlevel_parse( max_level, "CSG_GGEO"), app1, NULL );
    221 #endif
    222 
    223 

Because this header is included most everywhere, this incorrect duplication in logging initialization 
has been compiled into objects all over the place. So, do a cleaninstall of the standard and non-standard Opticks 
pkgs to try to fix this::

    o
    om-cleaninstall

    epsilon:sysrap blyth$ om-alt
    CSG
    CSG_GGeo
    qudarap
    CSGOptiX
    GeoChain

    c  ; om
    cg ; om
    qu ; om
    cx ; ./b7
    cg ; om

    

CSG/CSG_LOG.cc::

     23 #include "CSG_LOG.hh"
     24 #include "PLOG_INIT.hh"
     25 #include "PLOG.hh"
     26 
     27 void CSG_LOG::Initialize(int level, void* app1, void* app2 )
     28 {
     29     PLOG_INIT(level, app1, app2);
     30 }
     31 void CSG_LOG::Check(const char* msg)
     32 {
     33     PLOG_CHECK(msg);
     34 }
     35 


sysrap/PLOG_INIT.hh::

     65 #define PLOG_INIT(level, app1, app2 ) \
     66 { \
     67     plog::IAppender* appender1 = static_cast<plog::IAppender*>(app1) ; \
     68     plog::IAppender* appender2 = static_cast<plog::IAppender*>(app2) ; \
     69     plog::Severity severity = static_cast<plog::Severity>(level) ; \
     70     plog::init( severity ,  appender1 ); \
     71     if(appender2) \
     72         plog::get()->addAppender(appender2) ; \
     73 } \
     74 



Nope not fixed::


    O[blyth@localhost CSGOptiX]$ ./cxsd.sh 
    mo .bashrc VIP_MODE:dev O : ordinary opticks dev ontop of juno externals CMTEXTRATAGS:opticks
    mo .bashrc VIP_MODE:dev O : ordinary opticks dev ontop of juno externals CMTEXTRATAGS:opticks
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    Copyright (C) 2013 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-redhat-linux-gnu".
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>...
    Reading symbols from /data/blyth/junotop/ExternalLibs/opticks/head/lib/CSGOptiXSimulateTest...done.
    (gdb) r
    Starting program: /data/blyth/junotop/ExternalLibs/opticks/head/lib/CSGOptiXSimulateTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    /data/blyth/junotop/ExternalLibs/opticks/head/lib/CSGOptiXSimulateTest
    CSGOptiXSimulateTest: /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Logger.h:22: plog::Logger<instance>& plog::Logger<instance>::addAppender(plog::IAppender*) [with int instance = 0]: Assertion `appender != this' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff3e18387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-44.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-44.el7.x86_64 openssl-libs-1.0.2k-21.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff3e18387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff3e19a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff3e111a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff3e11252 in __assert_fail () from /lib64/libc.so.6
    #4  0x000000000041616a in plog::Logger<0>::addAppender (this=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>, 
        appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>) at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Logger.h:22
    #5  0x0000000000415efc in plog::init<0> (maxSeverity=plog::info, appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Init.h:30
    #6  0x0000000000415afb in plog::init (maxSeverity=plog::info, appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Init.h:35
    #7  0x00007ffff77d7c00 in CSG_LOG::Initialize (level=4, app1=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>, app2=0x0) at /home/blyth/opticks/CSG/CSG_LOG.cc:29
    #8  0x0000000000415c27 in OPTICKS_LOG_::Initialize (instance=0x69b350, app1=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/OPTICKS_LOG.hh:217
    #9  0x0000000000411693 in main (argc=1, argv=0x7fffffff3e38) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSimulateTest.cc:47
    (gdb) f 9
    #9  0x0000000000411693 in main (argc=1, argv=0x7fffffff3e38) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSimulateTest.cc:47
    47	    OPTICKS_LOG(argc, argv); 
    (gdb) f 8
    #8  0x0000000000415c27 in OPTICKS_LOG_::Initialize (instance=0x69b350, app1=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/OPTICKS_LOG.hh:217
    217	    CSG_LOG::Initialize(instance->prefixlevel_parse( max_level, "CSG"), app1, NULL );
    (gdb) list
    212	#ifdef OPTICKS_G4OK
    213	    G4OK_LOG::Initialize(instance->prefixlevel_parse( max_level, "G4OK"), app1, NULL );
    214	#endif
    215	
    216	#ifdef OPTICKS_CSG
    217	    CSG_LOG::Initialize(instance->prefixlevel_parse( max_level, "CSG"), app1, NULL );
    218	#endif
    219	#ifdef OPTICKS_CSG_GGEO
    220	    CSG_GGEO_LOG::Initialize(instance->prefixlevel_parse( max_level, "CSG_GGEO"), app1, NULL );
    221	#endif
    (gdb) f 7
    #7  0x00007ffff77d7c00 in CSG_LOG::Initialize (level=4, app1=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>, app2=0x0) at /home/blyth/opticks/CSG/CSG_LOG.cc:29
    29	    PLOG_INIT(level, app1, app2);
    (gdb) list 
    24	#include "PLOG_INIT.hh"
    25	#include "PLOG.hh"
    26	       
    27	void CSG_LOG::Initialize(int level, void* app1, void* app2 )
    28	{
    29	    PLOG_INIT(level, app1, app2);
    30	}
    31	void CSG_LOG::Check(const char* msg)
    32	{
    33	    PLOG_CHECK(msg);
    (gdb) f 6
    #6  0x0000000000415afb in plog::init (maxSeverity=plog::info, appender=0x693ac0 <plog::Logger<0>& plog::init<0>(plog::Severity, plog::IAppender*)::logger>)
        at /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Init.h:35
    35	        return init<PLOG_DEFAULT_INSTANCE>(maxSeverity, appender);
    (gdb) list
    30	        return appender ? logger.addAppender(appender) : logger;
    31	    }
    32	
    33	    inline Logger<PLOG_DEFAULT_INSTANCE>& init(Severity maxSeverity = none, IAppender* appender = NULL)
    34	    {
    35	        return init<PLOG_DEFAULT_INSTANCE>(maxSeverity, appender);
    36	    }
    37	
    38	    //////////////////////////////////////////////////////////////////////////
    39	    // RollingFileAppender with any Formatter
    (gdb) 




Perhaps macro clash, "OPTICKS_CSG" might be suffering double use::

    epsilon:sysrap blyth$ opticks-f OPTICKS_CSG
    ./CSG/CMakeLists.txt:target_compile_definitions( ${name} PUBLIC OPTICKS_CSG )
    ./CSG/tests/CSGFoundryLoadTest.cc:#ifdef OPTICKS_CSG
    ./sysrap/OPTICKS_LOG.hh:#ifdef OPTICKS_CSG
    ./sysrap/OPTICKS_LOG.hh:#ifdef OPTICKS_CSG_GGEO
    ./sysrap/OPTICKS_LOG.hh:#ifdef OPTICKS_CSG
    ./sysrap/OPTICKS_LOG.hh:#ifdef OPTICKS_CSG_GGEO
    ./sysrap/OPTICKS_LOG.hh:#ifdef OPTICKS_CSG
    ./sysrap/OPTICKS_LOG.hh:#ifdef OPTICKS_CSG_GGEO
    ./CSG_GGeo/CMakeLists.txt:target_compile_definitions( ${name} PUBLIC OPTICKS_CSG_GGEO )
    ./npy/CMakeLists.txt:   target_compile_definitions(${name} PUBLIC OPTICKS_CSGBSP  )
    ./npy/NPYConfig.cpp:#ifdef OPTICKS_CSGBSP
    ./npy/NOpenMesh.cpp:#ifdef OPTICKS_CSGBSP
    ./npy/NOpenMesh.cpp:#ifdef OPTICKS_CSGBSP
    epsilon:opticks blyth$ 



CSG/tests/CSGFoundryLoadTest.cc the below is unhealthy duplication::

      5 #ifdef OPTICKS_CSG
      6 #include "CSG_LOG.hh"
      7 #endif
      8 
      9 #include "OPTICKS_LOG.hh"


::

    epsilon:CSG blyth$ CSGFoundryLoadTest
    Assertion failed: (appender != this), function addAppender, file /usr/local/opticks/externals/plog/include/plog/Logger.h, line 22.
    Abort trap: 6
    epsilon:CSG blyth$ 
    epsilon:CSG blyth$ 
    epsilon:CSG blyth$ CSGFoundryTest
    Assertion failed: (appender != this), function addAppender, file /usr/local/opticks/externals/plog/include/plog/Logger.h, line 22.
    Abort trap: 6
    epsilon:CSG blyth$ 

    epsilon:CSG blyth$ CSGNameTest 
    Assertion failed: (appender != this), function addAppender, file /usr/local/opticks/externals/plog/include/plog/Logger.h, line 22.
    Abort trap: 6
    epsilon:CSG blyth$ 


::

    epsilon:tests blyth$ cat CSGLogTest.cc 
    #include "OPTICKS_LOG.hh"

    int main(int argc, char** argv)
    { 
        OPTICKS_LOG(argc, argv); 
        LOG(info) ; 
        return 0 ; 
    }


