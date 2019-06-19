strace-monitor-file-opens
============================




FIXED ISSUE : strace running shows log being written into unexpected location beside the binary /home/blyth/local/opticks/lib/OKG4Test.log
--------------------------------------------------------------------------------------------------------------------------------------------


* many logs found in that directory 
* need to avoid this as would cause permission failure in shared installation
* FIXED using SProc::ExecutableName() in PLOG.cc instead of argv[0]
* Also while looking into PLOG setup note that the RollingFileAppender is not enabled, due
  to a default zero argument : tried setting these to 500,000 bytes and 3 files

::

    [blyth@localhost tmp]$ cd /tmp ; strace -o /tmp/strace.log -e open $(which OKG4Test) --help >/dev/null ; strace.py
    strace.py
     /home/blyth/local/opticks/lib/OKG4Test.log                                       :          O_WRONLY|O_CREAT :  0644 

    [blyth@localhost tmp]$ cd /tmp ; strace -o /tmp/strace.log -e open OKG4Test --help >/dev/null ; strace.py
    strace.py
     OKG4Test.log                                                                     :          O_WRONLY|O_CREAT :  0644 

::

    068 const char* PLOG::_logpath_parse(int argc, char** argv)
     69 {
     70     assert( argc < MAXARGC && " argc sanity check fail ");
     71     //  Construct logfile path based on executable name argv[0] with .log appended 
     72     std::string lp(argc > 0 ? argv[0] : "default") ;
     73     lp += ".log" ;
     74     return strdup(lp.c_str());
     75 }
     76




strace technique
-----------------------



Using "--strace" argumment to old op.sh script::

    822    elif [ "${OPTICKS_DBG}" == "2" ]; then
    823       runline="strace -o /tmp/strace.log -e open ${OPTICKS_BINARY} ${OPTICKS_ARGS}"
    824    else


sets up strace monitoring of all file opens by the binary eg OKG4Test, creating a log of 2000 lines::

    [blyth@localhost bin]$ wc /tmp/strace.log 
      2004  11302 251061 /tmp/strace.log

    [blyth@localhost bin]$ head -10 /tmp/strace.log
    open("/home/blyth/local/opticks/lib/../lib/tls/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib/tls/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/tls/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/tls/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/x86_64/libOKG4.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/libOKG4.so", O_RDONLY|O_CLOEXEC) = 3
    open("/home/blyth/local/opticks/lib/../lib/libOK.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
    open("/home/blyth/local/opticks/lib/../lib64/libOK.so", O_RDONLY|O_CLOEXEC) = 3



Use strace.py script to parse, filter and report. For example showing creates::

    calhost bin]$ strace.py -f CREAT
    strace.py -f CREAT
     /home/blyth/local/opticks/lib/OKG4Test.log"                                      :          O_WRONLY|O_CREAT :  0644 
     tboolean-box/GItemList/GMaterialLib.txt"                                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     tboolean-box/GItemList/GSurfaceLib.txt"                                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ why these relative, all other absolute ?

     /var/tmp/OptixCache/cache.db"                                                    :            O_RDWR|O_CREAT :  0666 
     /var/tmp/OptixCache/cache.db"                                                    : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/OptixCache/cache.db-journal"                                            :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-wal"                                                :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-shm"                                                :            O_RDWR|O_CREAT :  0664 

     /tmp/blyth/location/seq.npy"                                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/his.npy"                                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/mat.npy"                                                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^ debug dumping from okc.Indexer 

     /tmp/blyth/location/cg4/primary.npy"                                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^^^^ debug dumping from CG4  
     

     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ht.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/gs.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ox.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-1/ph.npy"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ...  skipped expected ...
     /tmp/tboolean-box/evt/tboolean-box/torch/1/report.txt"                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190602_200126/t_absolute.ini"       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190602_200126/t_delta.ini"          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/1/20190602_200126/report.txt"           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/blyth/opticks/evt/tboolean-box/torch/Time.ini"                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/DeltaTime.ini"                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/VM.ini"                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/DeltaVM.ini"                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/evt/tboolean-box/torch/Opticks.npy"                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ^^^^^^^^^^^^ OpticksProfile::save metadata going to wrong place    





Relative write::

    334 void GGeoTest::importCSG(std::vector<GVolume*>& volumes)
    ...
    439     // see notes/issues/material-names-wrong-python-side.rst
    440     LOG(info) << "Save mlib/slib names "
    441               << " numTree : " << numTree
    442               << " csgpath : " << m_csgpath
    443               ;
    444 
    445     if( numTree > 0 )
    446     {
    447         m_mlib->saveNames(m_csgpath);
    448         m_slib->saveNames(m_csgpath);
    449     }
    450 
    451 
    452     LOG(info) << "]" ;
    453 }


::

    [blyth@localhost opticks]$ opticks-f \$TMP | grep seq.npy 
    ./optickscore/Indexer.cc:    m_seq->save("$TMP/seq.npy");  

    105 template <typename T>
    106 void Indexer<T>::save()
    107 {
    108     m_seq->save("$TMP/seq.npy");
    109     m_his->save("$TMP/his.npy");
    110     m_mat->save("$TMP/mat.npy");
    111 }


CG4.cc::

    344     pr->save("$TMP/cg4/primary.npy");   // debugging primary position issue 


::

    1735     m_profile->setDir(getEventFold());  // from Opticks::configure (from m_spec (OpticksEventSpec)

    [blyth@localhost optickscore]$ OpticksEventSpecTest
    2019-06-02 21:16:24.784 INFO  [362461] [OpticksEventSpec::Summary@148] s0 (no cat) typ typ tag tag itag 0 det det cat (null) dir /tmp/blyth/opticks/evt/det/typ/tag
    2019-06-02 21:16:24.784 INFO  [362461] [OpticksEventSpec::Summary@148] s1 (with cat) typ typ tag tag itag 0 det det cat cat dir /tmp/blyth/opticks/evt/cat/typ/tag










