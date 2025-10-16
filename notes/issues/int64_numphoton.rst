int64_numphoton
================



3 billion launch split into 12 slices of 250M each, hit max_curand limit of 1 billion::

    2025-10-14 16:18:31.464 INFO  [1462587] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest f94d93c709d76d3f6c8cc0ad6c25e61a dynamic f94d93c709d76d3f6c8cc0ad6c25e61a
    2025-10-14 16:18:31.465 INFO  [1462587] [QSim::simulate@441]  eventID      0 igs (120, 6, 4, ) tot_ph_0 3000000000 tot_ph_0/M 3000 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 12 TotalPhoton 3000000000 TotalPhoton/M 3000.000000
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      10,         0, 250000000}250.000000
       1 : sslice {      10,      20, 250000000, 250000000}250.000000
       2 : sslice {      20,      30, 500000000, 250000000}250.000000
       3 : sslice {      30,      40, 750000000, 250000000}250.000000
       4 : sslice {      40,      50,1000000000, 250000000}250.000000
       5 : sslice {      50,      60,1250000000, 250000000}250.000000
       6 : sslice {      60,      70,1500000000, 250000000}250.000000
       7 : sslice {      70,      80,1750000000, 250000000}250.000000
       8 : sslice {      80,      90,2000000000, 250000000}250.000000
       9 : sslice {      90,     100,2250000000, 250000000}250.000000
      10 : sslice {     100,     110,2500000000, 250000000}250.000000
      11 : sslice {     110,     120,2750000000, 250000000}250.000000
                      start    stop     offset      count    count/M 
     num_slice 12
    2025-10-14 16:19:07.193 INFO  [1462587] [QSim::simulate@489]  eventID 0 xxl YES i    0 dt   35.685755 slice    0 : sslice {       0,      10,         0, 250000000}250.000000
    2025-10-14 16:19:46.484 INFO  [1462587] [QSim::simulate@489]  eventID 0 xxl YES i    1 dt   36.285198 slice    1 : sslice {      10,      20, 250000000, 250000000}250.000000
    2025-10-14 16:20:26.166 INFO  [1462587] [QSim::simulate@489]  eventID 0 xxl YES i    2 dt   36.385741 slice    2 : sslice {      20,      30, 500000000, 250000000}250.000000
    2025-10-14 16:21:05.720 INFO  [1462587] [QSim::simulate@489]  eventID 0 xxl YES i    3 dt   36.385928 slice    3 : sslice {      30,      40, 750000000, 250000000}250.000000
    2025-10-14 16:21:09.084 FATAL [1462587] [QEvent::setGenstepUpload_NP@236]  gss.desc sslice {      40,      50,1000000000, 250000000}250.000000
     gss->ph_offset 1000000000
     gss->ph_count 250000000
     gss->ph_offset + gss->ph_count 1250000000(last_rng_state_idx) must be <= max_curand for valid rng_state access
     evt->max_curand 1000000000
     evt->num_curand 0
     evt->max_slot 262000000

    CSGOptiXSMTest: /home/blyth/opticks/qudarap/QEvent.cc:247: int QEvent::setGenstepUpload_NP(const NP*, const sslice*): Assertion `in_range' failed.
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh: line 685: 1462587 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh run error
    (ok) A[blyth@localhost opticks]$ 




After avoid more int truncations::

    2025-10-14 16:38:00.086 INFO  [1469353] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest f94d93c709d76d3f6c8cc0ad6c25e61a dynamic f94d93c709d76d3f6c8cc0ad6c25e61a
    2025-10-14 16:38:00.088 INFO  [1469353] [QSim::simulate@441]  eventID      0 igs (120, 6, 4, ) tot_ph_0 3000000000 tot_ph_0/M 3000 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 12 TotalPhoton 3000000000 TotalPhoton/M 3000.000000
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      10,         0, 250000000}250.000000
       1 : sslice {      10,      20, 250000000, 250000000}250.000000
       2 : sslice {      20,      30, 500000000, 250000000}250.000000
       3 : sslice {      30,      40, 750000000, 250000000}250.000000
       4 : sslice {      40,      50,1000000000, 250000000}250.000000
       5 : sslice {      50,      60,1250000000, 250000000}250.000000
       6 : sslice {      60,      70,1500000000, 250000000}250.000000
       7 : sslice {      70,      80,1750000000, 250000000}250.000000
       8 : sslice {      80,      90,2000000000, 250000000}250.000000
       9 : sslice {      90,     100,2250000000, 250000000}250.000000
      10 : sslice {     100,     110,2500000000, 250000000}250.000000
      11 : sslice {     110,     120,2750000000, 250000000}250.000000
                      start    stop     offset      count    count/M 
     num_slice 12
    2025-10-14 16:38:35.111 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    0 dt   34.981788 slice    0 : sslice {       0,      10,         0, 250000000}250.000000
    2025-10-14 16:39:13.957 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    1 dt   35.852770 slice    1 : sslice {      10,      20, 250000000, 250000000}250.000000
    2025-10-14 16:39:53.199 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    2 dt   36.194476 slice    2 : sslice {      20,      30, 500000000, 250000000}250.000000
    2025-10-14 16:40:32.578 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    3 dt   36.421776 slice    3 : sslice {      30,      40, 750000000, 250000000}250.000000
    2025-10-14 16:41:12.038 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    4 dt   36.446676 slice    4 : sslice {      40,      50,1000000000, 250000000}250.000000
    2025-10-14 16:41:51.920 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    5 dt   36.429783 slice    5 : sslice {      50,      60,1250000000, 250000000}250.000000
    2025-10-14 16:42:31.642 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    6 dt   36.424107 slice    6 : sslice {      60,      70,1500000000, 250000000}250.000000
    2025-10-14 16:43:11.516 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    7 dt   36.452490 slice    7 : sslice {      70,      80,1750000000, 250000000}250.000000
    2025-10-14 16:43:51.456 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    8 dt   36.405278 slice    8 : sslice {      80,      90,2000000000, 250000000}250.000000
    2025-10-14 16:44:31.269 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i    9 dt   36.424378 slice    9 : sslice {      90,     100,2250000000, 250000000}250.000000
    2025-10-14 16:45:11.140 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i   10 dt   36.392739 slice   10 : sslice {     100,     110,2500000000, 250000000}250.000000
    2025-10-14 16:45:50.920 INFO  [1469353] [QSim::simulate@489]  eventID 0 xxl YES i   11 dt   36.410605 slice   11 : sslice {     110,     120,2750000000, 250000000}250.000000
    2025-10-14 16:46:37.259 INFO  [1469353] [QSim::simulate@523]  eventID 0 tot_dt  434.836866 tot_ph 3000000000 tot_ph/M 3000.000000 tot_ht  598640516 tot_ht/M 598.640503 tot_ht/tot_ph   0.199547 reset_ YES
    2025-10-14 16:46:37.260 INFO  [1469353] [SEvt::save@4370] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000 [genstep,hit]
    2025-10-14 16:48:30.948 INFO  [1469353] [QSim::simulate@541] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 630.862122 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 474.384796 (multilaunch loop begin to end) 
     (PCAT - LEND)/M  42.785988 (topfold concat and clear subfold) 
     (TAIL - BRES)/M 113.688904 (QSim::reset which saves hits) 
     tot_idt/M       434.837128 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          434.836866 int(tot_dt*M)    434836866 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M        39.413071 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-10-14 16:48:31.155  155454325 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges2   :(42, 5, )
    -sreport_Creator::init.4 fold_valid Y
    -sreport_Creator::init.4.substamp   :[ subfold 1 ff 1 kk 0 aa 0]
    -sreport_Creator::init.5.subprofile :[ subfold 1 ff 1 kk 0 aa 0]
    -sreport_Creator::init.6.WITH_SUBMETA
    -sreport_Creator::init.7.submeta :[ subfold 0 ff 0 kk 2 aa 2]
    -sreport_Creator::init.8.submeta_NumPhotonCollected :[ subfold 0 ff 0 kk 2 aa 2]
    -sreport_Creator::init.9.subcount :[ subfold 0 ff 0 kk 2 aa 2]
    ]sreport_Creator::init
    ]sreport_Creator::sreport_Creator
    ]sreport.main : creator 
    [sreport.main : creator.desc 
    [sreport_Creator.desc
    [sreport_Creator.desc_fold
    fold = NPFold::LoadNoData("/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt")
    fold YES
    fold_valid YES
    ]sreport_Creator.desc_fold
    ]sreport_Creator.desc
    ]sreport.main : creator.desc 



36G of hits::

    (ok) A[blyth@localhost opticks]$ du -h /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000/
    36G	/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000/



Checking 36G of hits::

    In [46]: %cpaste
    Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
    :low, high = max(0, i-step), i
    :while low < high:
    :    mid = (low + high) // 2
    :    if np.all(h[mid] == 0):
    :        high = mid
    :    else:
    :        low = mid + 1
    :print(f"First all-zero index: {low}")
    :--
    First all-zero index: 115203376

    In [47]: h[115203376]
    Out[47]: 
    memmap([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]], dtype=float32)

    In [48]: h[115203376-1]
    Out[48]: 
    memmap([[  -18361.959     ,    -6265.1875    ,       15.253477  ,       99.16047   ],
            [      -0.9325637 ,       -0.29345718,       -0.21025589,        0.        ],
            [       0.34890065,       -0.5831001 ,       -0.7336639 ,      464.87576   ],
            [       0.        ,        0.        , -4243004.5       ,        0.        ]], dtype=float32)

    In [49]: 


The file is not truncated::

    In [55]: ls -l hit.npy
    -rw-r--r--. 1 blyth blyth 38312993168 Oct 14 16:48 hit.npy

    In [56]: h.size*4
    Out[56]: 38312993024

    In [57]: h.size*4 + 128
    Out[57]: 38312993152

    In [58]: h.size*4 + 128 + 16
    Out[58]: 38312993168


    In [60]: 115203376/h.shape[0]
    Out[60]: 0.19244166226797788



GPU side truncation probably. Clocked photon_idx ?::

    355 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    356 {
    357     sevent* evt = params.evt ;
    358     if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon
    359 
    360     unsigned idx = launch_idx.x ;
    361     unsigned genstep_idx = evt->seed[idx] ;
    362     const quad6& gs = evt->genstep[genstep_idx] ;
    363     // genstep needs the raw index, from zero for each genstep slice sub-launch
    364 
    365     unsigned photon_idx = params.photon_slot_offset + idx ;
    366     // rng_state access and array recording needs the absolute photon_idx
    367     // for multi-launch and single-launch simulation to match.
    368     // The offset hides the technicality of the multi-launch from output.
    369 


    372 //#define OLD_WITHOUT_SKIPAHEAD 1
    373 #ifdef OLD_WITHOUT_SKIPAHEAD
    374     RNG rng = sim->rngstate[photon_idx] ;
    375 #else
    376     RNG rng ;
    377     sim->rng->init( rng, sim->evt->index, photon_idx );
    378 #endif



    434     evt->photon[idx] = ctx.p ;
    435     // not photon_idx, needs to go from zero for photons from a slice of genstep array
    436 }


Each launch index should easily fit in unsigned (4.29 B)::

    In [99]: 0xffffffff/1e9
    Out[99]: 4.294967295

    In [100]: 0x7fffffff/1e9
    Out[100]: 2.147483647






::

    In [73]: ix = h[:,3, 2].view(np.uint32) & 0x7fffffff

    In [74]: ix.min()
    Out[74]: np.uint32(0)

    In [75]: ix.max()
    Out[75]: np.uint32(1249999993)

    In [76]: ix[:10]
    Out[76]: array([ 5,  9, 16, 21, 27, 45, 47, 49, 83, 91], dtype=uint32)




    In [84]: w = np.where( ix[:last_non_zero] > ix[1:last_non_zero + 1] )[0] ; w
    Out[84]: array([61769603, 78992308, 96208825])


    In [93]: ix[w[0]-5:w[0]+5]/1e6
    Out[93]: array([852.516326, 852.516328, 852.516331, 852.516332, 852.516335, 852.516342, 516.241744, 516.241754, 516.241757, 516.241765])

    In [94]: ix[w[1]-5:w[1]+5]/1e6
    Out[94]: array([602.516322, 602.516324, 602.516325, 602.516343, 602.516349, 602.516351, 266.231331, 266.231335, 266.231339, 266.23134 ])

    In [95]: ix[w[2]-5:w[2]+5]/1e6
    Out[95]: array([352.516325, 352.516326, 352.516329, 352.516333, 352.516334, 352.516346,  16.219458,  16.219459,  16.21946 ,  16.219465])





Rerun after some int truncation fixes
----------------------------------------

::

    2025-10-15 09:34:46.690 INFO  [1543617] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest f94d93c709d76d3f6c8cc0ad6c25e61a dynamic f94d93c709d76d3f6c8cc0ad6c25e61a
    2025-10-15 09:34:46.691 INFO  [1543617] [QSim::simulate@441]  eventID      0 igs (120, 6, 4, ) tot_ph_0 3000000000 tot_ph_0/M 3000 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 12 TotalPhoton 3000000000 TotalPhoton/M 3000.000000
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      10,         0, 250000000}250.000000
       1 : sslice {      10,      20, 250000000, 250000000}250.000000
       2 : sslice {      20,      30, 500000000, 250000000}250.000000
       3 : sslice {      30,      40, 750000000, 250000000}250.000000
       4 : sslice {      40,      50,1000000000, 250000000}250.000000
       5 : sslice {      50,      60,1250000000, 250000000}250.000000
       6 : sslice {      60,      70,1500000000, 250000000}250.000000
       7 : sslice {      70,      80,1750000000, 250000000}250.000000
       8 : sslice {      80,      90,2000000000, 250000000}250.000000
       9 : sslice {      90,     100,2250000000, 250000000}250.000000
      10 : sslice {     100,     110,2500000000, 250000000}250.000000
      11 : sslice {     110,     120,2750000000, 250000000}250.000000
                      start    stop     offset      count    count/M 
     num_slice 12
    2025-10-15 09:35:21.819 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    0 dt   35.087175 slice    0 : sslice {       0,      10,         0, 250000000}250.000000
    2025-10-15 09:36:00.511 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    1 dt   35.645243 slice    1 : sslice {      10,      20, 250000000, 250000000}250.000000
    2025-10-15 09:36:39.587 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    2 dt   36.030300 slice    2 : sslice {      20,      30, 500000000, 250000000}250.000000
    2025-10-15 09:37:18.785 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    3 dt   36.172342 slice    3 : sslice {      30,      40, 750000000, 250000000}250.000000
    2025-10-15 09:37:58.053 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    4 dt   36.299530 slice    4 : sslice {      40,      50,1000000000, 250000000}250.000000
    2025-10-15 09:38:37.328 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    5 dt   36.346845 slice    5 : sslice {      50,      60,1250000000, 250000000}250.000000
    2025-10-15 09:39:16.672 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    6 dt   36.356918 slice    6 : sslice {      60,      70,1500000000, 250000000}250.000000
    2025-10-15 09:39:56.034 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    7 dt   36.348483 slice    7 : sslice {      70,      80,1750000000, 250000000}250.000000
    2025-10-15 09:40:35.425 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    8 dt   36.402305 slice    8 : sslice {      80,      90,2000000000, 250000000}250.000000
    2025-10-15 09:41:14.692 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i    9 dt   36.295515 slice    9 : sslice {      90,     100,2250000000, 250000000}250.000000
    2025-10-15 09:41:53.830 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i   10 dt   36.290130 slice   10 : sslice {     100,     110,2500000000, 250000000}250.000000
    2025-10-15 09:42:32.964 INFO  [1543617] [QSim::simulate@489]  eventID 0 xxl YES i   11 dt   36.281662 slice   11 : sslice {     110,     120,2750000000, 250000000}250.000000
    2025-10-15 09:43:11.308 INFO  [1543617] [QSim::simulate@523]  eventID 0 tot_dt  433.556447 tot_ph 3000000000 tot_ph/M 3000.000000 tot_ht  598640516 tot_ht/M 598.640503 tot_ht/tot_ph   0.199547 reset_ YES
    2025-10-15 09:43:11.308 INFO  [1543617] [SEvt::save@4370] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000 [genstep,hit]
    2025-10-15 09:47:37.591 INFO  [1543617] [QSim::simulate@541] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 770.900574 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 469.220245 (multilaunch loop begin to end) 
     (PCAT - LEND)/M  35.396347 (topfold concat and clear subfold) 
     (TAIL - BRES)/M 266.283020 (QSim::reset which saves hits) 
     tot_idt/M       433.556702 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          433.556447 int(tot_dt*M)    433556446 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M        35.529713 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-10-15 09:47:38.562  562835688 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges2   :(42, 5, )




Looks like the hit zeros were caused by truncation in NP::Concatenate
------------------------------------------------------------------------

::

    (ok) A[blyth@localhost A000]$ f
    Python 3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:02) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.1.0 -- An enhanced Interactive Python. Type '?' for help.
    Tip: Run your doctests from within IPython for development and debugging. The special %doctest_mode command toggles a mode where the prompt, output and exceptions display matches as closely as possible that of the default Python interpreter.
    f

    CMDLINE:/home/blyth/np/fold.py
    f.base:.

      : f.NPFold_index                                     :                (14,) : 0:03:06.335120 
      : f.hit                                              :    (598640516, 4, 4) : 0:06:18.482430 
      : f.f006                                             :                 None : 0:02:57.250153 
      : f.f005                                             :                 None : 0:02:58.446149 
      : f.sframe_meta                                      :                    7 : 0:02:25.789266 
      : f.f007                                             :                 None : 0:02:56.052157 
      : f.f010                                             :                 None : 0:02:39.460217 
      : f.f001                                             :                 None : 0:03:03.187132 
      : f.sframe                                           : NO ATTR  0:02:25.789266 
      : f.f003                                             :                 None : 0:03:00.812140 
      : f.NPFold_names                                     :                 (0,) : 0:02:25.790266 
      : f.f008                                             :                 None : 0:02:54.409163 
      : f.f011                                             :                 None : 0:02:25.790266 
      : f.f000                                             :                 None : 0:03:04.443127 
      : f.genstep                                          :          (120, 6, 4) : 0:06:51.822310 
      : f.NPFold_meta                                      :                   25 : 0:02:25.790266 
      : f.f002                                             :                 None : 0:03:01.993136 
      : f.f009                                             :                 None : 0:02:48.641184 
      : f.f004                                             :                 None : 0:02:59.632144 

     min_stamp : 2025-10-15 09:43:11.307817 
     max_stamp : 2025-10-15 09:47:37.340861 
     dif_stamp : 0:04:26.033044 
     age_stamp : 0:02:25.789266 

    In [1]: f.hit.shape
    Out[1]: (598640516, 4, 4)

    In [2]: f.hit[0]
    Out[2]: 
    array([[-10094.563, -10052.373, -13014.945,    131.872],
           [    -0.674,     -0.309,     -0.671,      0.   ],
           [     0.696,     -0.57 ,     -0.436,    426.277],
           [     0.   ,      0.   ,     -0.   ,      0.   ]], dtype=float32)

    In [3]: f.hit[-1]
    Out[3]: 
    array([[-11717.97 ,    960.966, -15470.643,    277.56 ],
           [    -0.147,     -0.963,     -0.227,      0.   ],
           [     0.879,     -0.233,      0.417,    426.099],
           [     0.   ,      0.   ,     -0.   ,      0.   ]], dtype=float32)

    In [4]: all_zero_mask = np.all(f.hit == 0, axis=(1, 2))

    In [5]: all_zero_mask
    Out[5]: array([False, False, False, False, False, ..., False, False, False, False, False], shape=(598640516,))

    In [6]: zero_indices = np.where(all_zero_mask)[0]

    In [7]: zero_indices
    Out[7]: array([], dtype=int64)

     

photon index does not fit in 31 bits
---------------------------------------

TEST=vvvlarge_evt cxs_min.sh ## opticks_num_genstep=120 ; opticks_num_photon=G3


::


    In [10]: ix = f.hit[:,3,2].view(np.uint32) & 0x7fffffff

    In [11]: ix
    Out[11]: array([        5,         9,        16,        21,        27, ..., 852516328, 852516331, 852516332, 852516335, 852516342], shape=(598640516,), dtype=uint32)

    In [12]: ix.min()
    Out[12]: np.uint32(5)

    In [13]: ix.max()
    Out[13]: np.uint32(2147483646)

    In [14]: hex(ix.max())
    Out[14]: '0x7ffffffe'



::

    In [35]: ix
    Out[35]: array([        5,         9,        16,        21,        27, ..., 852516328, 852516331, 852516332, 852516335, 852516342], shape=(598640516,), dtype=uint32)

    In [36]: iy = ix.astype(np.int64)

    In [37]: iy
    Out[37]: array([        5,         9,        16,        21,        27, ..., 852516328, 852516331, 852516332, 852516335, 852516342], shape=(598640516,))

    In [38]: diy = np.diff(iy)

    In [39]: np.where( diy < 0 )
    Out[39]: (array([428515601]),)



2.14 billion limit on the photon index
----------------------------------------

::

    2521 inline QSIM_METHOD void qsim::generate_photon(sphoton& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2522 {
    2523     const int& gencode = gs.q0.i.x ;
    2524     switch(gencode)
    2525     {
    2526         case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ;
    2527         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    2528 
    2529         case OpticksGenstep_G4Cerenkov_modified:
    2530         case OpticksGenstep_CERENKOV:
    2531                                               cerenkov->generate(    p, rng, gs, photon_id, genstep_id ) ; break ;
    2532 
    2533         case OpticksGenstep_DsG4Scintillation_r4695:
    2534         case OpticksGenstep_SCINTILLATION:
    2535                                               scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ;
    2536 
    2537         case OpticksGenstep_INPUT_PHOTON:    { p = evt->photon[photon_id] ; p.set_flag(TORCH) ; }        ; break ;
    2538         default:                             generate_photon_dummy(  p, rng, gs, photon_id, genstep_id)  ; break ;
    2539     }
    2540     p.set_idx(photon_id);
    2541 }



::

    150     SPHOTON_METHOD unsigned idx() const {      return orient_idx & 0x7fffffffu  ;  }
    151     SPHOTON_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; }
    152 
    153     SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it
    154     SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis




::

    (ok) A[blyth@localhost junosw]$ opticks-f set_orient
    ./ana/p.py:     72     SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
    ./ana/p.py:    109     set_orient( orient_ );
    ./sysrap/sphoton.h:    SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it
    ./sysrap/sphoton.h:    set_orient( orient_ );
    ./sysrap/squad.h:    SQUAD_METHOD void set_orient( float orient );
    ./sysrap/squad.h:SQUAD_METHOD void quad4::set_orient( float orient )  // not typically used as set_prd more convenient, but useful for debug
    ./sysrap/tests/squadTest.cc:        p.set_orient( orient[0] ); 
    ./u4/U4Step.h:    current_photon.set_orient( cosThetaSign );   // sets a bit : would be 0 if not set
    (ok) A[blyth@localhost opticks]$ 


    (ok) A[blyth@localhost A000]$ opticks-f orient\(\)
    ./sysrap/sphoton.h:    SPHOTON_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; }
    ./sysrap/sphoton.h:        << " or " << orient()
    ./sysrap/tests/squadTest.cc:void test_quad4_idx_orient()
    ./sysrap/tests/squadTest.cc:    test_quad4_idx_orient(); 
    (ok) A[blyth@localhost opticks]$ 

    ## HMM : PROBABLY USING FROM PYTHON


    330 /**
    331 sphoton::set_prd
    332 -----------------
    333 
    334 This is canonically invoked GPU side by qsim::propagate
    335 copying geometry info from the quad2 PRD struct into the sphoton.
    336 
    337 TODO: relocate identity - 1 offsetting into here as this
    338 marks the transition from geometry to event information
    339 and would allow the offsetting to be better hidden.
    340 
    341 See ~/opticks/notes/issues/sensor_identifier_offset_by_one_wrinkle.rst
    342 
    343 **/
    344 
    345 
    346 SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_, unsigned iindex_ )
    347 {
    348     set_boundary(boundary_);
    349     identity = identity_ ;
    350     set_orient( orient_ );
    351     iindex = iindex_ ;
    352 }
    353 


From qsim::propagate cosTheta and iindex come along together, so packing them is clean lifecycle wise::

    2218 inline QSIM_METHOD int qsim::propagate(const int bounce, RNG& rng, sctx& ctx )  // ::simulate
    2219 {
    2220     const unsigned boundary = ctx.prd->boundary() ;
    2221     const unsigned identity = ctx.prd->identity() ; // sensor_identifier+1, 0:not-a-sensor
    2222     const unsigned iindex = ctx.prd->iindex() ;
    2223     const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta
    2224 
    ....
    2247     // copy geometry info into the sphoton struct
    2248     ctx.p.set_prd(boundary, identity, cosTheta, iindex );  // HMM: lposcost not passed along



TEST=ref1 cxs_min.sh
------------------------

::

    In [2]: ii = f.photon[:,1,3].view(np.uint32)

    In [3]: ii.min(),ii.max()
    Out[3]: (np.uint32(0), np.uint32(48593))


    In [4]: np.c_[np.unique(ii, return_counts=True)]
    Out[4]: 
    array([[     0, 327770],
           [     2,      2],
           [     4,      3],
           [     5,      1],
           [    10,      2],
           ...,
           [ 48236,     11],
           [ 48237,     13],
           [ 48238,     16],
           [ 48239,     11],
           [ 48593,  66842]], shape=(36477, 2))

    In [5]: f.photon.shape
    Out[5]: (1000000, 4, 4)

    In [6]: f.hit.shape
    Out[6]: (200397, 4, 4)

    In [7]: ii.shape
    Out[7]: (1000000,)

    In [8]: u_ii, n_ii = np.unique(ii, return_counts=True)

    In [9]: u_ii
    Out[9]: array([    0,     2,     4,     5,    10, ..., 48236, 48237, 48238, 48239, 48593], shape=(36477,), dtype=uint32)

    In [10]: n_ii
    Out[10]: array([327770,      2,      3,      1,      2, ...,     11,     13,     16,     11,  66842], shape=(36477,))

    In [11]: np.where(n_ii > 1000)
    Out[11]: (array([    0, 36476]),)




    In [12]: id = f.photon[:,3,1].view(np.uint32)

    In [13]: id
    Out[13]: array([    0,  9095,   423, 35514, 34790, ...,    16,  4221, 12249,     0, 12478], shape=(1000000,), dtype=uint32)

    In [14]: u_id, n_id = np.unique(id, return_counts=True)

    In [15]: np.c_[u_id, n_id]
    Out[15]: 
    array([[     0, 418619],
           [     1,     22],
           [     2,     22],
           [     3,     20],
           [     4,     18],
           ...,
           [ 45596,      4],
           [ 45597,      1],
           [ 45598,      1],
           [ 45599,      1],
           [ 45600,      2]], shape=(34580, 2))

    In [16]: np.where( n_id > 1000 )
    Out[16]: (array([0]),)




