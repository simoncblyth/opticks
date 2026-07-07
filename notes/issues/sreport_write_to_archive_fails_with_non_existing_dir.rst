sreport_write_to_archive_fails_with_non_existing_dir.rst
===========================================================


Reproduce gitlab-ci failure on commandline:: 

    cd  /tmp/gitlab-runner/opticks/GEOM/J26_1_1_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan_first_sreport

    [lo] A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$ export SREPORT_ARCHIVE_DIR=/tmp/sreport_archive
    [lo] A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$ sreport

    

::

    [lo] A[blyth@localhost ALL1_Debug_Philox_medium_scan_first_sreport]$ gdb $(which sreport)

    ]sreport.desc_subcount
    ]sreport.desc
    ]sreport.main : LOADED REPORT 
    [sreport.main : save_into_archive [/tmp/sreport_archive]
    [sreport::save_into_archive {/tmp/sreport_archive}
    terminate called after throwing an instance of 'std::runtime_error'
      what():  Target directory does not exist or is not a directory.

    Program received signal SIGABRT, Aborted.
    0x00007ffff688bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  0x00007ffff688bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff683eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff6828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff6ca1b21 in __gnu_cxx::__verbose_terminate_handler() [clone .cold] () from /lib64/libstdc++.so.6
    #4  0x00007ffff6cad52c in __cxxabiv1::__terminate(void (*)()) () from /lib64/libstdc++.so.6
    #5  0x00007ffff6cad597 in std::terminate() () from /lib64/libstdc++.so.6
    #6  0x00007ffff6cad7f9 in __cxa_throw () from /lib64/libstdc++.so.6
    #7  0x0000000000429852 in sfilesystem::find_index_of_max_indexed_dirname (container_dir_=0x7fffffffce0d "/tmp/sreport_archive", prefix=0x452cd2 "sreport_") at /home/blyth/opticks/sysrap/sfilesystem.h:108
    #8  0x000000000042a21d in sreport::FindIndexOfMaxIndexedDirname (collection_dir=0x7fffffffce0d "/tmp/sreport_archive") at /home/blyth/opticks/sysrap/sreport.h:175
    #9  0x000000000042a151 in sreport::save_into_archive (this=0x4a11f0, archive_dir=0x7fffffffce0d "/tmp/sreport_archive") at /home/blyth/opticks/sysrap/sreport.h:165
    #10 0x000000000040745b in main (argc=1, argv=0x7fffffffb168) at /home/blyth/opticks/sysrap/tests/sreport.cc:145
    (gdb) 



    (gdb) f 9
    #9  0x000000000042a151 in sreport::save_into_archive (this=0x4a11f0, archive_dir=0x7fffffffce0d "/tmp/sreport_archive") at /home/blyth/opticks/sysrap/sreport.h:165
    165	    long long max_index = FindIndexOfMaxIndexedDirname(archive_dir);
    (gdb) f 8
    #8  0x000000000042a21d in sreport::FindIndexOfMaxIndexedDirname (collection_dir=0x7fffffffce0d "/tmp/sreport_archive") at /home/blyth/opticks/sysrap/sreport.h:175
    175	    return sfilesystem::find_index_of_max_indexed_dirname(collection_dir, COLLECTION_PREFIX);
    (gdb) p COLLECTION_PREFIX
    No symbol "COLLECTION_PREFIX" in current context.
    (gdb) f 7
    #7  0x0000000000429852 in sfilesystem::find_index_of_max_indexed_dirname (container_dir_=0x7fffffffce0d "/tmp/sreport_archive", prefix=0x452cd2 "sreport_") at /home/blyth/opticks/sysrap/sfilesystem.h:108
    108	        throw std::runtime_error("Target directory does not exist or is not a directory.");
    (gdb) list
    103	    namespace fs = std::filesystem;
    104	
    105	    fs::path container_dir = container_dir_ ;
    106	
    107	    if (!fs::exists(container_dir) || !fs::is_directory(container_dir)) {
    108	        throw std::runtime_error("Target directory does not exist or is not a directory.");
    109	    }
    110	
    111	    long long max_index = -1; // Returns -1 if no matching directories are found
    112	    for (const auto& entry : fs::directory_iterator(container_dir))
    (gdb) 



