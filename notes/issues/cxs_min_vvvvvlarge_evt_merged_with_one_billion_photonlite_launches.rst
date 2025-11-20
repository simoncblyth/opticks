cxs_min_vvvvvlarge_evt_merged_with_one_billion_photonlite_launches
====================================================================



* Thrust has transient VRAM requirements, that force lots of headroom

::

    size_t SEventConfig::HeuristicMaxSlot( size_t totalGlobalMem_bytes, int lite, int merge  )
    {
        float vram_ceiling = float(totalGlobalMem_bytes)*0.87f ; // AIM not to exceed 87% (Grok:80%) of total VRAM
        float bytes_per_photon = (lite == 0) ? 64.f : 16.f;
        float headroom = (merge == 0) ? 1.75f : 3.3f;  // 1.75 was OK before adding GPU hit merging and lite mode
        return size_t(vram_ceiling / (bytes_per_photon * headroom) * 0.98f);
    }


After increase headroom ModeLite doing 8.35 billion in 16 launches
--------------------------------------------------------------------

::


    2025-11-20 17:18:00.560 INFO  [3703399] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-20 17:18:00.562 INFO  [3703399] [QSim::MaybeSaveIGS@734]  eventID 0 igs (512, 6, 4, ) igs_null NO  [QSim__SAVE_IGS_EVENTID] -1 [QSim__SAVE_IGS_PATH] $TMP/.opticks/igs.npy igs_path [/data1/blyth/tmp/.opticks/igs.npy] save_igs NO 
    2025-11-20 17:18:00.562 INFO  [3703399] [QSim::simulate@457]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 545000000 MaxSlot/M 545 sslice::Desc(igs_slice)
    sslice::Desc num_slice 16 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      33,         0, 531917892}531.917892
       1 : sslice {      33,      66, 531917892, 531917892}531.917892
       2 : sslice {      66,      99,1063835784, 531917892}531.917892
       3 : sslice {      99,     132,1595753676, 531917892}531.917892
       4 : sslice {     132,     165,2127671568, 531917892}531.917892
       5 : sslice {     165,     198,2659589460, 531917892}531.917892
       6 : sslice {     198,     231,3191507352, 531917892}531.917892
       7 : sslice {     231,     264,3723425244, 531917892}531.917892
       8 : sslice {     264,     297,4255343136, 531917892}531.917892
       9 : sslice {     297,     330,4787261028, 531917892}531.917892
      10 : sslice {     330,     363,5319178920, 531917892}531.917892
      11 : sslice {     363,     396,5851096812, 531917892}531.917892
      12 : sslice {     396,     429,6383014704, 531917892}531.917892
      13 : sslice {     429,     462,6914932596, 531917892}531.917892
      14 : sslice {     462,     495,7446850488, 531917892}531.917892
      15 : sslice {     495,     512,7978768380, 274018308}274.018308
                      start    stop     offset      count    count/M 
     num_slice 16
    2025-11-20 17:18:00.562 INFO  [3703399] [QSim::simulate@479]    0 : sslice {       0,      33,         0, 531917892}531.917892
    2025-11-20 17:19:19.975 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt   79.380918 slice    0 : sslice {       0,      33,         0, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106086153 merged 7015146 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:19:20.360 INFO  [3703399] [QSim::simulate@479]    1 : sslice {      33,      66, 531917892, 531917892}531.917892
    2025-11-20 17:20:40.896 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    1 dt   80.521225 slice    1 : sslice {      33,      66, 531917892, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106101354 merged 7017772 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:20:41.281 INFO  [3703399] [QSim::simulate@479]    2 : sslice {      66,      99,1063835784, 531917892}531.917892
    2025-11-20 17:22:01.846 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    2 dt   80.551822 slice    2 : sslice {      66,      99,1063835784, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106092815 merged 7015893 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:22:02.221 INFO  [3703399] [QSim::simulate@479]    3 : sslice {      99,     132,1595753676, 531917892}531.917892
    2025-11-20 17:23:22.776 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    3 dt   80.542252 slice    3 : sslice {      99,     132,1595753676, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106100978 merged 7017660 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:23:23.181 INFO  [3703399] [QSim::simulate@479]    4 : sslice {     132,     165,2127671568, 531917892}531.917892
    2025-11-20 17:24:44.893 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    4 dt   81.697846 slice    4 : sslice {     132,     165,2127671568, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106100042 merged 7014815 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:24:45.277 INFO  [3703399] [QSim::simulate@479]    5 : sslice {     165,     198,2659589460, 531917892}531.917892
    2025-11-20 17:26:06.653 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    5 dt   81.361646 slice    5 : sslice {     165,     198,2659589460, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106107720 merged 7014503 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:26:07.033 INFO  [3703399] [QSim::simulate@479]    6 : sslice {     198,     231,3191507352, 531917892}531.917892
    2025-11-20 17:27:28.544 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    6 dt   81.497683 slice    6 : sslice {     198,     231,3191507352, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106094205 merged 7015426 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:27:28.919 INFO  [3703399] [QSim::simulate@479]    7 : sslice {     231,     264,3723425244, 531917892}531.917892
    2025-11-20 17:28:50.107 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    7 dt   81.174386 slice    7 : sslice {     231,     264,3723425244, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106096116 merged 7016566 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:28:50.494 INFO  [3703399] [QSim::simulate@479]    8 : sslice {     264,     297,4255343136, 531917892}531.917892
    2025-11-20 17:30:11.223 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    8 dt   80.716089 slice    8 : sslice {     264,     297,4255343136, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106094849 merged 7017072 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:30:11.610 INFO  [3703399] [QSim::simulate@479]    9 : sslice {     297,     330,4787261028, 531917892}531.917892
    2025-11-20 17:31:32.556 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i    9 dt   80.932477 slice    9 : sslice {     297,     330,4787261028, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106096983 merged 7015262 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:31:32.940 INFO  [3703399] [QSim::simulate@479]   10 : sslice {     330,     363,5319178920, 531917892}531.917892
    2025-11-20 17:32:54.158 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i   10 dt   81.204544 slice   10 : sslice {     330,     363,5319178920, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106085776 merged 7016454 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:32:54.545 INFO  [3703399] [QSim::simulate@479]   11 : sslice {     363,     396,5851096812, 531917892}531.917892
    2025-11-20 17:34:15.115 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i   11 dt   80.556437 slice   11 : sslice {     363,     396,5851096812, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106108657 merged 7017926 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:34:15.489 INFO  [3703399] [QSim::simulate@479]   12 : sslice {     396,     429,6383014704, 531917892}531.917892
    2025-11-20 17:35:36.218 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i   12 dt   80.715908 slice   12 : sslice {     396,     429,6383014704, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106091006 merged 7016649 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:35:36.594 INFO  [3703399] [QSim::simulate@479]   13 : sslice {     429,     462,6914932596, 531917892}531.917892
    2025-11-20 17:36:57.678 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i   13 dt   81.070128 slice   13 : sslice {     429,     462,6914932596, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106088602 merged 7016138 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:36:58.057 INFO  [3703399] [QSim::simulate@479]   14 : sslice {     462,     495,7446850488, 531917892}531.917892
    2025-11-20 17:38:18.921 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i   14 dt   80.850055 slice   14 : sslice {     462,     495,7446850488, 531917892}531.917892
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 531917892 selected 106088016 merged 7018238 selected/in   0.199 merged/selected   0.066 
    2025-11-20 17:38:19.314 INFO  [3703399] [QSim::simulate@479]   15 : sslice {     495,     512,7978768380, 274018308}274.018308
    2025-11-20 17:39:01.222 INFO  [3703399] [QSim::simulate@505]  eventID 0 xxl YES i   15 dt   41.898689 slice   15 : sslice {     495,     512,7978768380, 274018308}274.018308
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 274018308 selected 54650793 merged 5627831 selected/in   0.199 merged/selected   0.103 
    2025-11-20 17:39:03.445 INFO  [3703399] [QSim::simulate@534]  num_slice 16 has_hlm YES hlm_final_merge YES
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 110873351 selected 110873351 merged 15394107 selected/in   1.000 merged/selected   0.139 
    2025-11-20 17:39:04.099 INFO  [3703399] [QSim::simulate_final_merge@645]  tot_ph 8252786688 hlm (110873351, 4, ) fin (15394107, 4, ) hlm/tot  0.0134 fin/hlm  0.1388
    2025-11-20 17:39:04.104 INFO  [3703399] [QSim::simulate@549]  eventID 0 tot_dt 1254.672103 tot_ph 8252786688 tot_ph/M 8252.787109 tot_ht   15394107 tot_ht/M  15.394107 tot_ht/tot_ph   0.001865 reset_ YES
    2025-11-20 17:39:04.105 INFO  [3703399] [SEvt::save@4505] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt_merge/A000 [genstep,hitlitemerged]
    2025-11-20 17:39:05.287 INFO  [3703399] [QSim::simulate@570] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 1264.725708 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 1260.878418 (multilaunch loop begin to end) 
     (PCAT - LEND)/M   2.663854 (topfold concat and clear subfold) 
     (TAIL - BRES)/M   1.181748 (QSim::reset which saves hits) 
     tot_idt/M       1254.672363 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          1254.672103 int(tot_dt*M)   1254672103 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M         5.972073 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-11-20 17:39:05.389  389267837 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 





hlm merged from subfold from the 32-launches perfectly matches the hlm from the 16 launches
--------------------------------------------------------------------------------------------


::

    ~/o/CSGOptiX/cxs_min.sh hlm

    ..

    20:25:15 | INFO     | opticks.sysrap.sphotonlite | [ photonlite.shape (176424702, 4) 
    20:25:19 | INFO     | opticks.sysrap.sphotonlite | -argsort[
    20:25:22 | INFO     | opticks.sysrap.sphotonlite | -argsort]
    20:25:24 | INFO     | opticks.sysrap.sphotonlite | -diff[
    20:25:25 | INFO     | opticks.sysrap.sphotonlite | -diff]
    20:25:25 | INFO     | opticks.sysrap.sphotonlite | -reducing n_groups 15394107 [
    20:26:40 | INFO     | opticks.sysrap.sphotonlite | -reducing n_groups 15394107 ]
    20:26:40 | INFO     | opticks.sysrap.sphotonlite | ] hlm.shape (15394107, 4) 


    TEST   # → 'vvvvvvlarge_evt_merge'
    f.hitlitemerged_meta   # → QSim__simulate_final_merge: tot_ph 8252786688 hlm (110873351, 4, ) fin (15394107, 4, ) hlm/tot  0.0134 fin/hlm  0.1388

    len(sub_hlm)   # → 32
    concat_hlm.shape   # → (176424702, 4)
    hlm2.shape   # → (15394107, 4)

    np.all( hlm == hlm2 )   # → np.True_


    In [1]: hlm
    Out[1]: 
    array([[1814691841, 1120016733, 2938013625,      11316],
           [ 502595585, 1120141318, 2105425674,      11572],
           [  79757313, 1120272442, 1079581664,      12084],
           [  67043329, 1120403464, 3550180968,      12084],
           [  71434241, 1120534540, 3389911965,      12084],
           ...,
           [    110536, 1138947281, 2320696252,      11300],
           [    110536, 1140859594, 3847896223,      11300],
           [    110536, 1141749945, 1708476262,      11812],
           [    110536, 1142373893, 3771439600,      11812],
           [    110536, 1142747641, 1470428383,      11316]], shape=(15394107, 4), dtype=uint32)

    In [2]: hlm2
    Out[2]: 
    array([[1814691841, 1120016733, 2938013625,      11316],
           [ 502595585, 1120141318, 2105425674,      11572],
           [  79757313, 1120272442, 1079581664,      12084],
           [  67043329, 1120403464, 3550180968,      12084],
           [  71434241, 1120534540, 3389911965,      12084],
           ...,
           [    110536, 1138947281, 2320696252,      11300],
           [    110536, 1140859594, 3847896223,      11300],
           [    110536, 1141749945, 1708476262,      11812],
           [    110536, 1142373893, 3771439600,      11812],
           [    110536, 1142747641, 1470428383,      11316]], shape=(15394107, 4), dtype=uint32)


















More than a billion launch on 32GB Ada 5000 : is a step too far
----------------------------------------------------------------

::

    2025-11-20 16:23:39.602 INFO  [3688062] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-20 16:23:39.604 INFO  [3688062] [QSim::MaybeSaveIGS@734]  eventID 0 igs (512, 6, 4, ) igs_null NO  [QSim__SAVE_IGS_EVENTID] -1 [QSim__SAVE_IGS_PATH] $TMP/.opticks/igs.npy igs_path [/data1/blyth/tmp/.opticks/igs.npy] save_igs NO 
    2025-11-20 16:23:39.604 INFO  [3688062] [QSim::simulate@457]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 1049000000 MaxSlot/M 1049 sslice::Desc(igs_slice)
    sslice::Desc num_slice 8 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      65,         0,1047717060}1047.717060
       1 : sslice {      65,     130,1047717060,1047717060}1047.717060
       2 : sslice {     130,     195,2095434120,1047717060}1047.717060
       3 : sslice {     195,     260,3143151180,1047717060}1047.717060
       4 : sslice {     260,     325,4190868240,1047717060}1047.717060
       5 : sslice {     325,     390,5238585300,1047717060}1047.717060
       6 : sslice {     390,     455,6286302360,1047717060}1047.717060
       7 : sslice {     455,     512,7334019420, 918767268}918.767268
                      start    stop     offset      count    count/M 
     num_slice 8
    2025-11-20 16:23:39.604 INFO  [3688062] [QSim::simulate@479]    0 : sslice {       0,      65,         0,1047717060}1047.717060
    2025-11-20 16:26:17.544 INFO  [3688062] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt  157.878467 slice    0 : sslice {       0,      65,         0,1047717060}1047.717060

    Thread 1 "CSGOptiXSMTest" received signal SIGABRT, Aborted.
    0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libnvidia-gpucomp-580.82.07-1.el9.x86_64 libnvidia-ml-580.82.07-1.el9.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64 nvidia-driver-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff483eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff4cb135a in __cxxabiv1::__terminate (handler=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff4cb13c5 in std::terminate () at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff4cb1658 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff70d1b30 <typeinfo for thrust::THRUST_200302_700_890_NS::system::detail::bad_alloc>, 
        dest=0x7ffff5ecc0b0 <thrust::THRUST_200302_700_890_NS::system::detail::bad_alloc::~bad_alloc()>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff57934c7 in thrust::THRUST_200302_700_890_NS::cuda_cub::malloc<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/malloc_and_free.h:85
    #7  0x00007ffff5793103 in thrust::THRUST_200302_700_890_NS::malloc<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/malloc_and_free.h:44
    #8  0x00007ffff5792b64 in thrust::THRUST_200302_700_890_NS::system::detail::generic::malloc<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/detail/generic/memory.inl:59
    #9  0x00007ffff579230d in thrust::THRUST_200302_700_890_NS::malloc<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/malloc_and_free.h:56
    #10 0x00007ffff5791553 in thrust::THRUST_200302_700_890_NS::system::detail::generic::get_temporary_buffer<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/detail/generic/temporary_buffer.inl:47
    #11 0x00007ffff5790652 in thrust::THRUST_200302_700_890_NS::get_temporary_buffer<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/temporary_buffer.h:66
    #12 0x00007ffff578fa48 in thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync>::allocate (this=0x7fffffff7630, cnt=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/allocator/temporary_allocator.inl:52
    #13 0x00007ffff577e9da in thrust::THRUST_200302_700_890_NS::detail::allocator_traits<thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::allocate(thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> >&, unsigned long)::workaround_warnings::allocate(thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> >&, unsigned long) (a=..., n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/allocator/allocator_traits.inl:378
    #14 0x00007ffff577e9ff in thrust::THRUST_200302_700_890_NS::detail::allocator_traits<thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::allocate (a=..., n=5126776447) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/allocator/allocator_traits.inl:382
    #15 0x00007ffff577d778 in thrust::THRUST_200302_700_890_NS::detail::contiguous_storage<unsigned char, thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::allocate (this=0x7fffffff7630, n=5126776447)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/contiguous_storage.inl:218
    #16 0x00007ffff577b1e7 in thrust::THRUST_200302_700_890_NS::detail::contiguous_storage<unsigned char, thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::contiguous_storage (this=0x7fffffff7630, n=5126776447, alloc=...)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/contiguous_storage.inl:76
    #17 0x00007ffff5774bc0 in thrust::THRUST_200302_700_890_NS::detail::temporary_array<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync>::temporary_array (this=0x7fffffff7630, system=..., 
        n=5126776447) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/temporary_array.inl:84
    #18 0x00007ffff577b698 in thrust::THRUST_200302_700_890_NS::cuda_cub::__radix_sort::radix_sort<cuda::std::__4::integral_constant<bool, true>, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, unsigned long, sphotonlite, long, thrust::THRUST_200302_700_890_NS::less<unsigned long> > (policy=..., keys=0x3c94a6e00, items=0x302000000, count=208971477)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:355
    #19 0x00007ffff5775146 in thrust::THRUST_200302_700_890_NS::cuda_cub::__smart_sort::smart_sort<cuda::std::__4::integral_constant<bool, true>, cuda::std::__4::integral_constant<bool, false>, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite>, thrust::THRUST_200302_700_890_NS::less<unsigned long> > (policy=..., 
        keys_first=..., keys_last=..., items_first=..., compare_op=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:461
    #20 0x00007ffff57744db in thrust::THRUST_200302_700_890_NS::cuda_cub::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite>, thrust::THRUST_200302_700_890_NS::less<unsigned long> > (policy=..., keys_first=..., keys_last=..., values=..., compare_op=...)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:551
    #21 0x00007ffff5773df6 in thrust::THRUST_200302_700_890_NS::cuda_cub::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite> > (policy=..., keys_first=..., keys_last=..., values=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:621
    #22 0x00007ffff5772c3f in thrust::THRUST_200302_700_890_NS::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite> > (exec=..., keys_first=..., keys_last=..., values_first=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/sort.inl:102
    #23 0x00007ffff576baf6 in SPM::merge_partial_select (d_in=0x7ffa36000000, num_in=1047717060, d_out=0x166c7a90, num_out=0x166c7a08, select_flagmask=8192, time_window=1, stream=0x0) at /home/blyth/opticks/sysrap/SPM.cu:262
    #24 0x00007ffff5ec1035 in QEvt::PerLaunchMerge (evt=0x166c7960, stream=0x0) at /home/blyth/opticks/qudarap/QEvt.cc:1103
    #25 0x00007ffff5ec05d1 in QEvt::gatherHitLiteMerged (this=0x18656c10) at /home/blyth/opticks/qudarap/QEvt.cc:1020
    #26 0x00007ffff5ec19a7 in QEvt::gatherComponent_ (this=0x18656c10, cmp=67108864) at /home/blyth/opticks/qudarap/QEvt.cc:1290
    #27 0x00007ffff5ec1635 in QEvt::gatherComponent (this=0x18656c10, cmp=67108864) at /home/blyth/opticks/qudarap/QEvt.cc:1265
    --Type <RET> for more, q to quit, c to continue without paging--






