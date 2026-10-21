cxs_min_ser_sreport_segv
========================

* SEGV probably due to SProf.txt overwrite in the scanning
* nevertheless - the NP machinery should be hardened to not crash

* ABANDONED THE APPROACH OF SEPARATE INVOKATIONS - AS TOO MUCH REPORTING COMPLICATIONS
* INSTEAD ADDED MULTI-SER CONFIG - TO AVOID REPORT COMPLICATIONS AS NO OVERWRITING


::

    cxs_min.sh deport


    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    [sreport_Creator::init_runprof_run_ranges_from_SProf
    -sreport_Creator::init.SProf:runprof   :(2, 3, ) {runprof: grep Index SProf.txt, with full SProf.txt in runprof_meta.txt}
    --sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2[
    [NP::MakeMetaKVS_ranges2 num_keys:24 num_anno:24
    ]NP::Resolve_ranges
     num_keys :24 num_ranges :9 num_specs :9
    [NP::MakeMetaKVS_ranges2_table num_specs 9 num_keys 24 num_anno 24
     (specs correspond to ranges - so will be smaller than keys which should be same as anno) 
    .NP::MakeMetaKVS_ranges2_table kpp.size 9
                SEvt__Init_RUN_META ==>           CSGFoundry__Load_HEAD                10,251                               ## init
              CSGFoundry__Load_HEAD ==>           CSGFoundry__Load_TAIL             1,828,906                               ## load_geom
              CSGOptiX__Create_HEAD ==>           CSGOptiX__Create_TAIL               313,144                               ## upload_geom
           A004_QSim__simulate_HEAD ==>        A004_QSim__simulate_LBEG                    36                               ## slice_genstep
           A004_QSim__simulate_PRUP ==>        A004_QSim__simulate_PREL                35,525                               ## upload genstep slice
           A004_QSim__simulate_PREL ==>        A004_QSim__simulate_POST            32,521,109                               ## simulate slice
           A004_QSim__simulate_POST ==>        A004_QSim__simulate_DOWN               357,194                               ## download slice
           A004_QSim__simulate_LEND ==>        A004_QSim__simulate_PCAT                    24                               ## concat slices
           A004_QSim__simulate_BRES ==>        A004_QSim__simulate_TAIL               571,342                               ## save arrays
                                                                 TOTAL:            35,637,531
    ]NP::MakeMetaKVS_ranges2_table num_keys:24
    ]NP::MakeMetaKVS_ranges2 num_specs:9 rr (9, 5, )
    [NP::MakeMetaKVS_ranges2
     ranges2 (9, 5, )]NP::MakeMetaKVS_ranges2

    --sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2]
    -sreport_Creator::init_SProf.ranges2   :(9, 5, ) {ranges provides detailed timing info extracted from SProf.txt}
    -sreport_Creator::init_SProf.run       :(1, ) {run is dummy array just for holding metadata} 
    ]sreport_Creator::init_runprof_run_ranges_from_SProf
    [sreport_Creator::init_evsmry_from_ranges
    -sreport_Creator::init_evsmry_from_ranges.Summarize_ranges_to_evsmry[
    [NP::Summarize_ranges_to_evsmry
     -1 {SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init} {init} {10251} [Anno ] 
     -1 {CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom} {load_geom} {1828906} [Anno ] 
     -1 {CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom} {upload_geom} {313144} [Anno ] 
      4 {A004_QSim__simulate_HEAD:A004_QSim__simulate_LBEG:slice_genstep} {slice_genstep} {36} [Anno ] 
      4 {A004_QSim__simulate_PRUP:A004_QSim__simulate_PREL:upload genstep slice} {upload genstep slice} {35525} [Anno ] 
      4 {A004_QSim__simulate_PREL:A004_QSim__simulate_POST:simulate slice} {simulate slice} {32521109} [Anno ] 
      4 {A004_QSim__simulate_POST:A004_QSim__simulate_DOWN:download slice} {download slice} {357194} [Anno ] 
      4 {A004_QSim__simulate_LEND:A004_QSim__simulate_PCAT:concat slices # slice=1,max_slot_M=257} {concat slices} {24} [Anno max_slot_M:257 slice:1 ] 
      4 {A004_QSim__simulate_BRES:A004_QSim__simulate_TAIL:save arrays # numGenstepCollected=10,numPhotonCollected=100000000,numHit=21345733} {save arrays} {571342} [Anno numGenstepCollected:10 numHit:21345733 numPhotonCollected:100000000 ] 
    ]NP::Summarize_ranges_to_evsmry evsmry (1, 9, )
    -sreport_Creator::init_evsmry_from_ranges.Summarize_ranges_to_evsmry]
    ]sreport_Creator::init_evsmry_from_ranges
    [sreport_Creator::init_substamp
    -sreport_Creator::init_substamp fold_valid Y
    -sreport_Creator::init_substamp ((NPFold)report.substamp).stats [ subfold 1 ff 1 kk 0 aa 0]
    ]sreport_Creator::init_substamp
    [sreport_Creator::init_subprofile
    -sreport_Creator::init_subprofile :[ subfold 1 ff 1 kk 0 aa 0]
    ]sreport_Creator::init_subprofile
    [sreport_Creator::init_submeta
    -sreport_Creator::init_submeta.WITH_SUBMETA
    -sreport_Creator::init_submeta :[ subfold 0 ff 0 kk 2 aa 2]
    -sreport_Creator::init_submeta_NumPhotonCollected :[ subfold 0 ff 0 kk 2 aa 2]
    ]sreport_Creator::init_submeta
    [sreport_Creator::init_subcount
    -sreport_Creator::init_subcount :[ subfold 0 ff 0 kk 2 aa 2]
    ]sreport_Creator::init_subcount
    ]sreport_Creator::init
    ]sreport_Creator::sreport_Creator
    ]sreport.main : creator 
    [sreport.main : creator.desc 
    [sreport_Creator.desc
    [sreport_Creator.desc_fold
    fold = NPFold::LoadNoData("/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt")
    fold YES
    fold_valid YES
    ]sreport_Creator.desc_fold
    ]sreport_Creator.desc
    ]sreport.main : creator.desc 
    [sreport.main : report.desc 

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff6990e95 in __memmove_avx512_unaligned_erms () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  0x00007ffff6990e95 in __memmove_avx512_unaligned_erms () from /lib64/libc.so.6
    #1  0x00007ffff6d61c62 in std::char_traits<char>::copy (__s1=<optimized out>, __s2=<optimized out>, __n=131071)
        at /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/download/gcc-15.1.0.build/x86_64-pc-linux-gnu/libstdc++-v3/include/bits/char_traits.h:429
    #2  std::basic_streambuf<char, std::char_traits<char> >::xsputn (this=0x7fffffff9d98, __s=<optimized out>, __n=100000002)
        at /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/download/gcc-15.1.0.build/x86_64-pc-linux-gnu/libstdc++-v3/include/bits/streambuf.tcc:95
    #3  0x00007ffff6d52384 in std::basic_streambuf<char, std::char_traits<char> >::sputn (this=<optimized out>, __s=0x4a593a "x\247<^\006", __n=100000002)
        at /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/download/gcc-15.1.0.build/x86_64-pc-linux-gnu/libstdc++-v3/include/streambuf:459
    #4  std::__ostream_write<char, std::char_traits<char> > (__out=..., __s=0x4a593a "x\247<^\006", __n=100000002)
        at /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/download/gcc-15.1.0.build/x86_64-pc-linux-gnu/libstdc++-v3/include/bits/ostream_insert.h:58
    #5  std::__ostream_insert<char, std::char_traits<char> > (__out=..., __s=0x4a593a "x\247<^\006", __n=100000002)
        at /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/download/gcc-15.1.0.build/x86_64-pc-linux-gnu/libstdc++-v3/include/bits/ostream_insert.h:109
    #6  0x000000000043bebb in NP::descTable_<long> (this=0x4a5aa0, wid=17, column_labels=0x4a5a80, row_labels=0x4a5af0, tag=0x454071 "ranges", do_timestamp_delta=false) at /home/blyth/opticks/sysrap/NP.hh:3261
    #7  0x00000000004336a3 in NP::descTable<long> (this=0x4a5aa0, wid=17, tag=0x454071 "ranges", do_timestamp_delta=false) at /home/blyth/opticks/sysrap/NP.hh:3033
    #8  0x000000000042a618 in sreport::desc_ranges[abi:cxx11]() const (this=0x49e710) at /home/blyth/opticks/sysrap/sreport.h:374
    #9  0x0000000000429a59 in sreport::desc[abi:cxx11]() const (this=0x49e710) at /home/blyth/opticks/sysrap/sreport.h:282
    #10 0x0000000000405e3c in main (argc=1, argv=0x7fffffffa938) at /home/blyth/opticks/sysrap/tests/sreport.cc:111
    (gdb) f 6
    #6  0x000000000043bebb in NP::descTable_<long> (this=0x4a5aa0, wid=17, column_labels=0x4a5a80, row_labels=0x4a5af0, tag=0x454071 "ranges", do_timestamp_delta=false) at /home/blyth/opticks/sysrap/NP.hh:3261
    3261	                    << (*column_labels)[j]
    (gdb) f 10
    #10 0x0000000000405e3c in main (argc=1, argv=0x7fffffffa938) at /home/blyth/opticks/sysrap/tests/sreport.cc:111
    111	        std::cout << report->desc() ;
    (gdb) f 9
    #9  0x0000000000429a59 in sreport::desc[abi:cxx11]() const (this=0x49e710) at /home/blyth/opticks/sysrap/sreport.h:282
    282	       << ( is_config("ranges")   ? desc_ranges()  : "" )
    (gdb) f 


First sign of corruption with the ranges->labels::

    [sreport.main : report.desc 
    [sreport::desc_ranges DEBUG
     ranges : (9, 5, )
     ranges.names.size() 9
     ranges.labels.size() 5
    ]sreport::desc_ranges DEBUG
    [ranges.names
    SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init
    CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom
    CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom
    A004_QSim__simulate_HEAD:A004_QSim__simulate_LBEG:slice_genstep
    A004_QSim__simulate_PRUP:A004_QSim__simulate_PREL:upload genstep slice
    A004_QSim__simulate_PREL:A004_QSim__simulate_POST:simulate slice
    A004_QSim__simulate_POST:A004_QSim__simulate_DOWN:download slice
    A004_QSim__simulate_LEND:A004_QSim__simulate_PCAT:concat slices # slice=1,max_slot_M=257
    A004_QSim__simulate_BRES:A004_QSim__simulate_TAIL:save arrays # numGenstepCollected=10,numPhotonCollected=100000000,numHit=21345733
    ]ranges.names
    [ranges.labels
    x�<^%�J�;����Ɣ�<^�iJab�hJ�iJia�iJib�<^
        Q�iJa.npyran_order'jJb.npy, }y<^733ce!concat slices!save arrays!<i8!0iJ�iJ�iJ1XHxYHxYH�kJ�kJ�kJ�jJ�H��HЁH�jJ0kJPkJ�9J���cJpjJ-!	!<i8_007����f�Ԝ�_ranges2 num_keys:24 num_anno:24
    ]NP::Resolve_ranges





HUH report and deport doing different::

    [lo] A[blyth@localhost tests]$ 
    [lo] A[blyth@localhost tests]$ cxs_min.sh report
    /data1/blyth/local/opticks_Debug_g411/bin/cxs_min.sh - Internal GEOM setup detected
    /data1/blyth/local/opticks_Debug_g411/bin/cxs_min.sh - report - RUNNING FROM LOGDIR /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt - PWD /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt
    [sreport.main CONFIG [] argv0 sreport dirp /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt is_executable_sibling_path NO  is_indexed_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    [sreport_Creator::init_runprof_run_ranges_from_SProf
    -sreport_Creator::init.SProf:runprof   :(2, 3, ) {runprof: grep Index SProf.txt, with full SProf.txt in runprof_meta.txt}
    --sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2[
    [NP::MakeMetaKVS_ranges2 num_keys:24 num_anno:24
    ]NP::Resolve_ranges
     num_keys :24 num_ranges :9 num_specs :9
    [NP::MakeMetaKVS_ranges2_table num_specs 9 num_keys 24 num_anno 24
     (specs correspond to ranges - so will be smaller than keys which should be same as anno) 
    .NP::MakeMetaKVS_ranges2_table kpp.size 9







    lo] A[blyth@localhost tests]$ cxs_min.sh deport
    /data1/blyth/local/opticks_Debug_g411/bin/cxs_min.sh - Internal GEOM setup detected
    /data1/blyth/local/opticks_Debug_g411/bin/cxs_min.sh - deport - RUNNING FROM LOGDIR /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt - PWD /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt
    gdb -ex r --args sreport
    Tue Oct 20 02:13:58 PM CST 2026
    GNU gdb (AlmaLinux) 14.2-4.1.el9_6
    Copyright (C) 2023 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    Reading symbols from sreport...
    Starting program: /data1/blyth/local/opticks_Debug_g411/lib/sreport 
    warning: File "/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/lib64/libstdc++.so.6.0.34-gdb.py" auto-loading has been declined by your `auto-load safe-path' set to "$debugdir:$datadir/auto-load:/usr/lib/golang/src/pkg/runtime/runtime-gdb.py:/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/contrib/gcc/11.4.1/lib64/libstdc++.so.6.0.29-gdb.py".
    To enable execution of this file add
        add-auto-load-safe-path /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/contrib/gcc/15.1.0/lib64/libstdc++.so.6.0.34-gdb.py
    line to your configuration file "/home/blyth/.gdbinit".
    To completely disable this security protection add
        set auto-load safe-path /
    line to your configuration file "/home/blyth/.gdbinit".
    For more information about this security protection see the
    "Auto-loading safe path" section in the GDB manual.  E.g., run from the shell:
        info "(gdb)Auto-loading safe path"
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [Detaching after vfork from child process 388132]
    [sreport.main CONFIG [creator] argv0 /data1/blyth/local/opticks_Debug_g411/lib/sreport dirp /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt is_executable_sibling_path NO  is_indexed_path NO 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    [sreport_Creator::init_runprof_run_ranges_from_SProf
    -sreport_Creator::init.SProf:runprof   :(2, 3, ) {runprof: grep Index SProf.txt, with full SProf.txt in runprof_meta.txt}
    --sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2[
    [NP::MakeMetaKVS_ranges2 num_keys:24 num_anno:24
    ]NP::Resolve_ranges
     num_keys :24 num_ranges :9 num_specs :9
    [NP::MakeMetaKVS_ranges2_table num_specs 9 num_keys 24 num_anno 24
     (specs correspond to ranges - so will be smaller than keys which should be same as anno) 
    .NP::MakeMetaKVS_ranges2_table kpp.size 9
                SEvt__Init_RUN_META ==>           CSGFoundry__Load_HEAD                 9,585                               ## init
              CSGFoundry__Load_HEAD ==>           CSGFoundry__Load_TAIL             1,811,799                               ## load_geom
              CSGOptiX__Create_HEAD ==>           CSGOptiX__Create_TAIL               329,449                               ## upload_geom
           A000_QSim__simulate_HEAD ==>        A000_QSim__simulate_LBEG                    37                               ## slice_genstep
           A000_QSim__simulate_PRUP ==>        A000_QSim__simulate_PREL                35,661                               ## upload genstep slice
           A000_QSim__simulate_PREL ==>        A000_QSim__simulate_POST            31,139,098                               ## simulate slice
           A000_QSim__simulate_POST ==>        A000_QSim__simulate_DOWN               492,277                               ## download slice
           A000_QSim__simulate_LEND ==>        A000_QSim__simulate_PCAT                    24                               ## concat slices
           A000_QSim__simulate_BRES ==>        A000_QSim__simulate_TAIL            10,047,302                               ## save arrays
                                                                 TOTAL:            43,865,232
    ]NP::MakeMetaKVS_ranges2_table num_keys:24
    ]NP::MakeMetaKVS_ranges2 num_specs:9 rr (9, 5, )
    [NP::MakeMetaKVS_ranges2
     ranges2 (9, 5, )]NP::MakeMetaKVS_ranges2

    --sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2]
    -sreport_Creator::init_SProf.ranges2   :(9, 5, ) {ranges provides detailed timing info extracted from SProf.txt}
    -sreport_Creator::init_SProf.run       :(1, ) {run is dummy array just for holding metadata} 



