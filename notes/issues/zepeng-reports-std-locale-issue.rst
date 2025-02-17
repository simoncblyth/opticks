zepeng-reports-std-locale-issue
===============================




2025/2/17 zepeng SSimTest::

    Here is the output from ‘bt’

    Program received signal SIGABRT, Aborted.
    __pthread_kill_implementation (no_tid=0, signo=6, threadid=140737352650752) at ./nptl/pthread_kill.c:44
    44	./nptl/pthread_kill.c: No such file or directory.
    (gdb) bt
    #0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=140737352650752) at ./nptl/pthread_kill.c:44
    #1  __pthread_kill_internal (signo=6, threadid=140737352650752) at ./nptl/pthread_kill.c:78
    #2  __GI___pthread_kill (threadid=140737352650752, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
    #3  0x00007ffff73c5476 in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
    #4  0x00007ffff73ab7f3 in __GI_abort () at ./stdlib/abort.c:79
    #5  0x00007ffff764eb9e in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007ffff765a20c in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x00007ffff765a277 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #8  0x00007ffff765a4d8 in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #9  0x00007ffff765151d in std::__throw_runtime_error(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #10 0x00007ffff767d258 in std::locale::facet::_S_create_c_locale(__locale_struct*&, char const*, __locale_struct*) ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #11 0x00007ffff766ed05 in std::locale::_Impl::_Impl(char const*, unsigned long) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #12 0x00007ffff767007f in std::locale::locale(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6


    5582 inline std::string NP::DescMetaKVS_kvs(const std::vector<std::string>& keys, const std::vector<std::string>& vals, const std::vector<int64_t>& tt )  // static
    5583 {
    5584     int num_keys = keys.size() ;
    5585     if(num_keys == 0) return "" ;
    5586 
    5587     // sort indices into increasing time order
    5588     // non-timestamped lines with placeholder timestamp zero will come first 
    5589     std::vector<int> ii(num_keys);
    5590     std::iota(ii.begin(), ii.end(), 0);  // init to 0,1,2,3,..., num_keys-1
    5591     auto order = [&tt](const size_t& a, const size_t &b) { return tt[a] < tt[b];}  ;
    5592     std::sort(ii.begin(), ii.end(), order );
    5593 
    5594     std::stringstream ss ;
    5595     ss.imbue(std::locale("")) ;  // commas for thousands
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^6

    5596 
    5597     int64_t t_first = 0 ;
    5598     int64_t t_second = 0 ;
    5599     int64_t t_prev  = 0 ;
    5600 
    5601     int t_count = 0 ;
    5602 
    5603     ss
    5604         << "[NP::DescMetaKVS_kvs "
    5605         << " keys.size " << keys.size()
    5606         << " vals.size " << vals.size()
    5607         << " tt.size "   << tt.size()
    5608         << " num_keys " << num_keys
    5609         << "\n"
    5610         ;



    #13 0x00007ffff7976830 in NP::DescMetaKVS_kvs (keys=std::vector of length 3, capacity 4 = {...},
        vals=std::vector of length 3, capacity 4 = {...}, tt=std::vector of length 3, capacity 4 = {...})
        at /sdf/group/exo/users/zpli/opticks/sysrap/NP.hh:5595
    #14 0x00007ffff7977bd3 in NP::DescMetaKVS (meta="entityType:G4Box\nsolidName:LModPBox\nlvid:0", juncture_=0x0, ranges_=0x0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NP.hh:5833
    #15 0x00007ffff798167a in NPFold::descMetaKVS[abi:cxx11]() const (this=0x555555616350)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2692
    #16 0x00007ffff798179c in NPFold::desc[abi:cxx11](int) const (this=0x555555616350, depth=3)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2754
    #17 0x00007ffff7981ab8 in NPFold::desc[abi:cxx11](int) const (this=0x555555615a90, depth=2)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2773
    #18 0x00007ffff7981ab8 in NPFold::desc[abi:cxx11](int) const (this=0x5555555e6960, depth=1)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2773
    #19 0x00007ffff7981ab8 in NPFold::desc[abi:cxx11](int) const (this=0x5555555e6510, depth=0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2773
    #20 0x00007ffff7981523 in NPFold::desc[abi:cxx11]() const (this=0x5555555e6510)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2684
    #21 0x00007ffff79a4ac5 in SSim::desc[abi:cxx11]() const (this=0x5555555e5a00)
        at /sdf/group/exo/users/zpli/opticks/sysrap/SSim.cc:226
    #22 0x00005555555761b6 in SSimTest::Load () at /sdf/group/exo/users/zpli/opticks/sysrap/tests/SSimTest.cc:51
    --Type <RET> for more, q to quit, c to continue without paging--
    #23 0x0000555555577eae in SSimTest::Main () at /sdf/group/exo/users/zpli/opticks/sysrap/tests/SSimTest.cc:203
    #24 0x00005555555781cd in main (argc=1, argv=0x7fffffffa8a8) at /sdf/group/exo/users/zpli/opticks/sysrap/tests/SSimTest.cc:225

    Start 108: SysRapTest.SBndTest
    108/108 Test #108: SysRapTest.SBndTest ......................................   Passed    1.40 sec
    99% tests passed, 1 tests failed out of 108


QSimTest::

    The other three have the same error message. It looks like the test still checks the geometry against JUNO requirement?


     6/21 Test  #6: QUDARapTest.QSimTest .....................***Failed    2.30 sec
                    HOME : /sdf/home/z/zpli
                     PWD : /sdf/group/exo/users/zpli/opticks/install/build/qudarap/tests
                    GEOM : V1J009
             BASH_SOURCE : /sdf/group/exo/users/zpli/opticks/install/bin/QTestRunner.sh
              EXECUTABLE : QSimTest
                    ARGS :
    2025-02-14 00:26:10.725 INFO  [1312453] [main@776] [ TEST hemisphere_s_polarized
    2025-02-14 00:26:12.507 INFO  [1312453] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000 SEventConfig::MaxCurand 1000000000
    2025-02-14 00:26:12.512 ERROR [1312453] [QSim::UploadComponents@160]  icdf null, snam::ICDF icdf.npy
    SPrd::init_evec THE GEOMETRY DOES NOT HAVE ALL THE BOUNDARY INDICES
    [
        Acrylic///LS
        Water///Acrylic
        Water///Pyrex
        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    See More
            Start 108: SysRapTest.SBndTest
    108/108 Test #108: SysRapTest.SBndTest ......................................   Passed    1.40 sec
    99% tests passed, 1 tests failed out of 108
    The other three have the same error message. It looks like the test still checks the geometry against JUNO requirement?
     6/21 Test  #6: QUDARapTest.QSimTest .....................***Failed    2.30 sec
                    HOME : /sdf/home/z/zpli
                     PWD : /sdf/group/exo/users/zpli/opticks/install/build/qudarap/tests
                    GEOM : V1J009
             BASH_SOURCE : /sdf/group/exo/users/zpli/opticks/install/bin/QTestRunner.sh
              EXECUTABLE : QSimTest
                    ARGS :
    2025-02-14 00:26:10.725 INFO  [1312453] [main@776] [ TEST hemisphere_s_polarized
    2025-02-14 00:26:12.507 INFO  [1312453] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000 SEventConfig::MaxCurand 1000000000
    2025-02-14 00:26:12.512 ERROR [1312453] [QSim::UploadComponents@160]  icdf null, snam::ICDF icdf.npy
    SPrd::init_evec THE GEOMETRY DOES NOT HAVE ALL THE BOUNDARY INDICES
    [
        Acrylic///LS
        Water///Acrylic
        Water///Pyrex
        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum

    This error looks the same as the above one?

    #0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=140737352650752) at ./nptl/pthread_kill.c:44
    #1  __pthread_kill_internal (signo=6, threadid=140737352650752) at ./nptl/pthread_kill.c:78
    #2  __GI___pthread_kill (threadid=140737352650752, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
    #3  0x00007ffff6ba7476 in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
    #4  0x00007ffff6b8d7f3 in __GI_abort () at ./stdlib/abort.c:79
    #5  0x00007ffff6e30b9e in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #6  0x00007ffff6e3c20c in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x00007ffff6e3c277 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #8  0x00007ffff6e3c4d8 in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #9  0x00007ffff6e3351d in std::__throw_runtime_error(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #10 0x00007ffff6e5f258 in std::locale::facet::_S_create_c_locale(__locale_struct*&, char const*, __locale_struct*) ()
       from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #11 0x00007ffff6e50d05 in std::locale::_Impl::_Impl(char const*, unsigned long) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #12 0x00007ffff6e5207f in std::locale::locale(char const*) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    #13 0x00007ffff7158830 in NP::DescMetaKVS_kvs (keys=std::vector of length 3, capacity 4 = {...},
        vals=std::vector of length 3, capacity 4 = {...}, tt=std::vector of length 3, capacity 4 = {...})
        at /sdf/group/exo/users/zpli/opticks/sysrap/NP.hh:5595
    #14 0x00007ffff7159bd3 in NP::DescMetaKVS (meta="entityType:G4Box\nsolidName:LModPBox\nlvid:0", juncture_=0x0, ranges_=0x0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NP.hh:5833
    #15 0x00007ffff716367a in NPFold::descMetaKVS[abi:cxx11]() const (this=0x5555556ee6e0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2692
    #16 0x00007ffff716379c in NPFold::desc[abi:cxx11](int) const (this=0x5555556ee6e0, depth=3)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2754
    #17 0x00007ffff7163ab8 in NPFold::desc[abi:cxx11](int) const (this=0x5555556ede20, depth=2)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2773
    #18 0x00007ffff7163ab8 in NPFold::desc[abi:cxx11](int) const (this=0x5555556beb10, depth=1)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2773
    #19 0x00007ffff7163ab8 in NPFold::desc[abi:cxx11](int) const (this=0x5555556bdfa0, depth=0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2773
    #20 0x00007ffff7163523 in NPFold::desc[abi:cxx11]() const (this=0x5555556bdfa0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/NPFold.h:2684
    #21 0x00007ffff7186ac5 in SSim::desc[abi:cxx11]() const (this=0x5555556bd4f0)
        at /sdf/group/exo/users/zpli/opticks/sysrap/SSim.cc:226
    #22 0x00005555555773c4 in main (argc=1, argv=0x7fffffffa8a8) at /sdf/group/exo/users/zpli/opticks/qudarap/tests/QSimTest.cc:792






