beyond_X32_photon_event : Beyond 4.29 billion photon event ?
==============================================================

CSGOptiX7.cu
::

    538 extern "C" __global__ void __raygen__rg()
    539 {
    540     const uint3 idx = optixGetLaunchIndex();
    541     const uint3 dim = optixGetLaunchDimensions();



DONE : removed this limitation
---------------------------------

Each launch needs to be less than 4.29 billion as:

1. the launch index uses uint3
2. GPU VRAM is currently nowhere near enough to allow launches of 4.29 billion.
   Each photon is 4*4*4 bytes so just the photons would need 275G VRAM::

    In [3]: 4*4*4*1000000*4290/1e9
    Out[3]: 274.56



WIP : Check with 5 billion, if PIDX logging is controlled
------------------------------------------------------------

Some logging slipped thru::

    2025-10-23 11:31:33.845 INFO  [2755928] [QSim::simulate@489]  eventID 0 xxl YES i   14 dt   36.468866 slice   14 : sslice {     364,     390,3554687500, 253906250}253.906250
    2025-10-23 11:32:13.793 INFO  [2755928] [QSim::simulate@489]  eventID 0 xxl YES i   15 dt   36.469851 slice   15 : sslice {     390,     416,3808593750, 253906250}253.906250
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    2025-10-23 11:32:53.698 INFO  [2755928] [QSim::simulate@489]  eventID 0 xxl YES i   16 dt   36.683019 slice   16 : sslice {     416,     442,4062500000, 253906250}253.906250
    2025-10-23 11:33:33.661 INFO  [2755928] [QSim::simulate@489]  eventID 0 xxl YES i   17 dt   36.422368 slice   17 : sslice {     442,     468,4316406250, 253906250}253.906250




TEST=vvvvvvlarge_evt cxs_min.sh  # 8.2 billion photon, 1.6 billion hits
---------------------------------------------------------------------------


::

    2025-10-22 20:06:11.651 INFO  [2666545] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest f94d93c709d76d3f6c8cc0ad6c25e61a dynamic f94d93c709d76d3f6c8cc0ad6c25e61a
    2025-10-22 20:06:11.653 INFO  [2666545] [QSim::simulate@441]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 32 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      16,         0, 257899584}257.899584
       1 : sslice {      16,      32, 257899584, 257899584}257.899584
       2 : sslice {      32,      48, 515799168, 257899584}257.899584
       3 : sslice {      48,      64, 773698752, 257899584}257.899584
       4 : sslice {      64,      80,1031598336, 257899584}257.899584
       5 : sslice {      80,      96,1289497920, 257899584}257.899584
       6 : sslice {      96,     112,1547397504, 257899584}257.899584
       7 : sslice {     112,     128,1805297088, 257899584}257.899584
       8 : sslice {     128,     144,2063196672, 257899584}257.899584
       9 : sslice {     144,     160,2321096256, 257899584}257.899584
      10 : sslice {     160,     176,2578995840, 257899584}257.899584
      11 : sslice {     176,     192,2836895424, 257899584}257.899584
      12 : sslice {     192,     208,3094795008, 257899584}257.899584
      13 : sslice {     208,     224,3352694592, 257899584}257.899584
      14 : sslice {     224,     240,3610594176, 257899584}257.899584
      15 : sslice {     240,     256,3868493760, 257899584}257.899584
      16 : sslice {     256,     272,4126393344, 257899584}257.899584
      17 : sslice {     272,     288,4384292928, 257899584}257.899584
      18 : sslice {     288,     304,4642192512, 257899584}257.899584
      19 : sslice {     304,     320,4900092096, 257899584}257.899584
      20 : sslice {     320,     336,5157991680, 257899584}257.899584
      21 : sslice {     336,     352,5415891264, 257899584}257.899584
      22 : sslice {     352,     368,5673790848, 257899584}257.899584
      23 : sslice {     368,     384,5931690432, 257899584}257.899584
      24 : sslice {     384,     400,6189590016, 257899584}257.899584
      25 : sslice {     400,     416,6447489600, 257899584}257.899584
      26 : sslice {     416,     432,6705389184, 257899584}257.899584
      27 : sslice {     432,     448,6963288768, 257899584}257.899584
      28 : sslice {     448,     464,7221188352, 257899584}257.899584
      29 : sslice {     464,     480,7479087936, 257899584}257.899584
      30 : sslice {     480,     496,7736987520, 257899584}257.899584
      31 : sslice {     496,     512,7994887104, 257899584}257.899584
                      start    stop     offset      count    count/M 
     num_slice 32
    2025-10-22 20:06:48.067 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    0 dt   36.372397 slice    0 : sslice {       0,      16,         0, 257899584}257.899584
    2025-10-22 20:07:28.644 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    1 dt   37.126608 slice    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    2025-10-22 20:08:09.503 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    2 dt   37.438297 slice    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    2025-10-22 20:08:50.637 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    3 dt   37.624764 slice    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    2025-10-22 20:09:31.877 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    4 dt   37.705388 slice    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    2025-10-22 20:10:12.940 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    5 dt   37.597175 slice    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    2025-10-22 20:10:53.583 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    6 dt   37.388271 slice    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    2025-10-22 20:11:33.737 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    7 dt   37.171555 slice    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    2025-10-22 20:12:13.956 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    8 dt   37.384922 slice    8 : sslice {     128,     144,2063196672, 257899584}257.899584
    2025-10-22 20:12:54.211 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i    9 dt   37.278119 slice    9 : sslice {     144,     160,2321096256, 257899584}257.899584
    2025-10-22 20:13:34.758 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   10 dt   37.404079 slice   10 : sslice {     160,     176,2578995840, 257899584}257.899584
    2025-10-22 20:14:15.182 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   11 dt   37.272112 slice   11 : sslice {     176,     192,2836895424, 257899584}257.899584
    2025-10-22 20:14:55.161 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   12 dt   37.033712 slice   12 : sslice {     192,     208,3094795008, 257899584}257.899584
    2025-10-22 20:15:35.585 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   13 dt   37.513861 slice   13 : sslice {     208,     224,3352694592, 257899584}257.899584
    2025-10-22 20:16:16.202 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   14 dt   37.408353 slice   14 : sslice {     224,     240,3610594176, 257899584}257.899584
    2025-10-22 20:16:56.556 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   15 dt   37.245094 slice   15 : sslice {     240,     256,3868493760, 257899584}257.899584

    //qsim.propagate.head pidx      -1 : ctx.evt.index 0 evt.index 0 

    //qsim.propagate.head pidx      -1 : bnc 0 boundary 132 cosTheta 1.00000000 
    //qsim.propagate.head pidx      -1 : mom = np.array([0.57072037,0.82054293,0.03142488]) ; lmom = 1.00000000  
    //qsim.propagate.head pidx      -1 : pos = np.array([  57.07204,  82.05429,   3.14249]) ; lpos = 99.99999237 
    //qsim.propagate.head pidx      -1 : nrm = np.array([(0.57072037,0.82054293,0.03142488]) ; lnrm = 1.00000000  
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qsim.propagate_to_boundary.head pidx      -1 : u_absorption 0.72304481 logf(u_absorption) -0.32428402 absorption_length 40893.0938 absorption_distance 13260.976562 
    //qsim.propagate_to_boundary.head pidx      -1 : post = np.array([  57.07204,  82.05429,   3.14249,   0.00000]) 
    //qsim.propagate_to_boundary.head pidx      -1 : distance_to_boundary 17600.0000 absorption_distance 13260.9766 scattering_distance   154.6470 
    //qsim.propagate_to_boundary.head pidx      -1 : u_scattering     0.9934 u_absorption     0.7230 
    //qsim.propagate.body pidx      -1 bounce 0 command 2 flag 32 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.tail pidx      -1 bounce 0 command 2 flag 32 ctx.s.optical.y(ems) 1 

    //qsim.propagate.head pidx      -1 : ctx.evt.index 0 evt.index 0 

    //qsim.propagate.head pidx      -1 : bnc 1 boundary 132 cosTheta 0.99998546 
    //qsim.propagate.head pidx      -1 : mom = np.array([-0.55762243,-0.72653902,-0.40149507]) ; lmom = 1.00000000  
    //qsim.propagate.head pidx      -1 : pos = np.array([ 145.33224, 208.94881,   8.00225]) ; lpos = 254.64701843 
    //qsim.propagate.head pidx      -1 : nrm = np.array([(-0.55684042,-0.72441316,-0.40639180]) ; lnrm = 0.99999994  
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qsim.propagate_to_boundary.head pidx      -1 : u_absorption 0.87895429 logf(u_absorption) -0.12902236 absorption_length 40893.0938 absorption_distance 5276.123535 
    //qsim.propagate_to_boundary.head pidx      -1 : post = np.array([ 145.33224, 208.94881,   8.00225,   0.79524]) 
    //qsim.propagate_to_boundary.head pidx      -1 : distance_to_boundary 17935.8047 absorption_distance  5276.1235 scattering_distance 16148.5000 
    //qsim.propagate_to_boundary.head pidx      -1 : u_scattering     0.4993 u_absorption     0.8790 
    //qsim.propagate_to_boundary.body.BULK_ABSORB pidx      -1 : post = np.array([-2796.75269,-3624.36084,-2110.33545,  27.92657]) ; absorb_time_delta = 27.13132668   
    //qsim.propagate.body pidx      -1 bounce 1 command 1 flag 8 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.tail pidx      -1 bounce 1 command 1 flag 8 ctx.s.optical.y(ems) 1 
    2025-10-22 20:17:37.185 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   16 dt   37.379421 slice   16 : sslice {     256,     272,4126393344, 257899584}257.899584
    2025-10-22 20:18:17.634 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   17 dt   37.239186 slice   17 : sslice {     272,     288,4384292928, 257899584}257.899584
    2025-10-22 20:18:58.542 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   18 dt   37.423898 slice   18 : sslice {     288,     304,4642192512, 257899584}257.899584
    2025-10-22 20:19:39.510 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   19 dt   37.540160 slice   19 : sslice {     304,     320,4900092096, 257899584}257.899584
    2025-10-22 20:20:20.064 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   20 dt   37.366776 slice   20 : sslice {     320,     336,5157991680, 257899584}257.899584
    2025-10-22 20:21:00.345 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   21 dt   37.221078 slice   21 : sslice {     336,     352,5415891264, 257899584}257.899584
    2025-10-22 20:21:40.614 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   22 dt   37.315209 slice   22 : sslice {     352,     368,5673790848, 257899584}257.899584
    2025-10-22 20:22:21.413 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   23 dt   37.779618 slice   23 : sslice {     368,     384,5931690432, 257899584}257.899584
    2025-10-22 20:23:01.827 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   24 dt   37.138494 slice   24 : sslice {     384,     400,6189590016, 257899584}257.899584
    2025-10-22 20:23:41.887 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   25 dt   36.985423 slice   25 : sslice {     400,     416,6447489600, 257899584}257.899584
    2025-10-22 20:24:22.084 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   26 dt   37.112887 slice   26 : sslice {     416,     432,6705389184, 257899584}257.899584
    2025-10-22 20:25:02.625 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   27 dt   37.435371 slice   27 : sslice {     432,     448,6963288768, 257899584}257.899584
    2025-10-22 20:25:43.055 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   28 dt   37.246180 slice   28 : sslice {     448,     464,7221188352, 257899584}257.899584
    2025-10-22 20:26:23.620 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   29 dt   37.350592 slice   29 : sslice {     464,     480,7479087936, 257899584}257.899584
    2025-10-22 20:27:03.978 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   30 dt   37.094271 slice   30 : sslice {     480,     496,7736987520, 257899584}257.899584
    2025-10-22 20:27:44.384 INFO  [2666545] [QSim::simulate@489]  eventID 0 xxl YES i   31 dt   37.243868 slice   31 : sslice {     496,     512,7994887104, 257899584}257.899584
    2025-10-22 20:30:13.702 INFO  [2666545] [QSim::simulate@524]  eventID 0 tot_dt 1193.837140 tot_ph 8252786688 tot_ph/M 8252.787109 tot_ht 1646782603 tot_ht/M 1646.782593 tot_ht/tot_ph   0.199543 reset_ YES
    2025-10-22 20:30:13.704 INFO  [2666545] [SEvt::save@4384] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt/A000 [genstep,hit]
    2025-10-22 20:35:47.022 INFO  [2666545] [QSim::simulate@545] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 1774.903442 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 1295.774292 (multilaunch loop begin to end) 
     (PCAT - LEND)/M 146.274902 (topfold concat and clear subfold) 
     (TAIL - BRES)/M 332.851929 (QSim::reset which saves hits) 
     tot_idt/M       1193.838013 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          1193.837140 int(tot_dt*M)   1193837140 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M       101.630287 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-10-22 20:35:47.193  193548493 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    [sreport_Creator::init_SProf
    -sreport_Creator::init.SProf:runprof   :(2, 3, )
    -sreport_Creator::init_SProf.run       :(1, )
    -sreport_Creator::init_SProf.ranges2   :(101, 5, )
    ]sreport_Creator::init_SProf
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








TEST=vvvvvlarge_evt cxs_min.sh  # 5 billion photon, almost 1 billion hit
--------------------------------------------------------------------------

::

    2025-10-20 15:20:02.729 INFO  [2333764] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest f94d93c709d76d3f6c8cc0ad6c25e61a dynamic f94d93c709d76d3f6c8cc0ad6c25e61a
    2025-10-20 15:20:02.731 INFO  [2333764] [QSim::simulate@441]  eventID      0 igs (512, 6, 4, ) tot_ph_0 5000000000 tot_ph_0/M 5000 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 20 TotalPhoton 5000000000 TotalPhoton/M 5000.000000
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      26,         0, 253906250}253.906250
       1 : sslice {      26,      52, 253906250, 253906250}253.906250
       2 : sslice {      52,      78, 507812500, 253906250}253.906250
       3 : sslice {      78,     104, 761718750, 253906250}253.906250
       4 : sslice {     104,     130,1015625000, 253906250}253.906250
       5 : sslice {     130,     156,1269531250, 253906250}253.906250
       6 : sslice {     156,     182,1523437500, 253906250}253.906250
       7 : sslice {     182,     208,1777343750, 253906250}253.906250
       8 : sslice {     208,     234,2031250000, 253906250}253.906250
       9 : sslice {     234,     260,2285156250, 253906250}253.906250
      10 : sslice {     260,     286,2539062500, 253906250}253.906250
      11 : sslice {     286,     312,2792968750, 253906250}253.906250
      12 : sslice {     312,     338,3046875000, 253906250}253.906250
      13 : sslice {     338,     364,3300781250, 253906250}253.906250
      14 : sslice {     364,     390,3554687500, 253906250}253.906250
      15 : sslice {     390,     416,3808593750, 253906250}253.906250
      16 : sslice {     416,     442,4062500000, 253906250}253.906250
      17 : sslice {     442,     468,4316406250, 253906250}253.906250
      18 : sslice {     468,     494,4570312500, 253906250}253.906250
      19 : sslice {     494,     512,4824218750, 175781250}175.781250
                      start    stop     offset      count    count/M 
     num_slice 20
    2025-10-20 15:20:38.228 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    0 dt   35.455092 slice    0 : sslice {       0,      26,         0, 253906250}253.906250
    2025-10-20 15:21:17.496 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    1 dt   36.266780 slice    1 : sslice {      26,      52, 253906250, 253906250}253.906250
    2025-10-20 15:21:56.969 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    2 dt   36.554801 slice    2 : sslice {      52,      78, 507812500, 253906250}253.906250
    2025-10-20 15:22:36.935 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    3 dt   36.835774 slice    3 : sslice {      78,     104, 761718750, 253906250}253.906250
    2025-10-20 15:23:16.993 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    4 dt   36.912787 slice    4 : sslice {     104,     130,1015625000, 253906250}253.906250
    2025-10-20 15:23:57.169 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    5 dt   37.196386 slice    5 : sslice {     130,     156,1269531250, 253906250}253.906250
    2025-10-20 15:24:37.326 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    6 dt   37.092205 slice    6 : sslice {     156,     182,1523437500, 253906250}253.906250
    2025-10-20 15:25:17.386 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    7 dt   37.097305 slice    7 : sslice {     182,     208,1777343750, 253906250}253.906250
    2025-10-20 15:25:57.393 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    8 dt   37.036145 slice    8 : sslice {     208,     234,2031250000, 253906250}253.906250
    2025-10-20 15:26:37.603 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i    9 dt   37.213331 slice    9 : sslice {     234,     260,2285156250, 253906250}253.906250
    2025-10-20 15:27:17.763 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   10 dt   37.168065 slice   10 : sslice {     260,     286,2539062500, 253906250}253.906250
    2025-10-20 15:27:57.873 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   11 dt   37.045508 slice   11 : sslice {     286,     312,2792968750, 253906250}253.906250
    2025-10-20 15:28:37.962 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   12 dt   37.104472 slice   12 : sslice {     312,     338,3046875000, 253906250}253.906250
    2025-10-20 15:29:18.015 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   13 dt   37.004844 slice   13 : sslice {     338,     364,3300781250, 253906250}253.906250
    2025-10-20 15:29:58.185 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   14 dt   37.080189 slice   14 : sslice {     364,     390,3554687500, 253906250}253.906250
    2025-10-20 15:30:38.173 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   15 dt   36.940040 slice   15 : sslice {     390,     416,3808593750, 253906250}253.906250

    //qsim.propagate.head pidx      -1 : ctx.evt.index 0 evt.index 0 

    //qsim.propagate.head pidx      -1 : bnc 0 boundary 132 cosTheta 1.00000000 
    //qsim.propagate.head pidx      -1 : mom = np.array([0.57072037,0.82054293,0.03142488]) ; lmom = 1.00000000  
    //qsim.propagate.head pidx      -1 : pos = np.array([  57.07204,  82.05429,   3.14249]) ; lpos = 99.99999237 
    //qsim.propagate.head pidx      -1 : nrm = np.array([(0.57072037,0.82054293,0.03142488]) ; lnrm = 1.00000000  
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qsim.propagate_to_boundary.head pidx      -1 : u_absorption 0.72304481 logf(u_absorption) -0.32428402 absorption_length 40893.0938 absorption_distance 13260.976562 
    //qsim.propagate_to_boundary.head pidx      -1 : post = np.array([  57.07204,  82.05429,   3.14249,   0.00000]) 
    //qsim.propagate_to_boundary.head pidx      -1 : distance_to_boundary 17600.0000 absorption_distance 13260.9766 scattering_distance   154.6470 
    //qsim.propagate_to_boundary.head pidx      -1 : u_scattering     0.9934 u_absorption     0.7230 
    //qsim.propagate.body pidx      -1 bounce 0 command 2 flag 32 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.tail pidx      -1 bounce 0 command 2 flag 32 ctx.s.optical.y(ems) 1 

    //qsim.propagate.head pidx      -1 : ctx.evt.index 0 evt.index 0 

    //qsim.propagate.head pidx      -1 : bnc 1 boundary 132 cosTheta 0.99998546 
    //qsim.propagate.head pidx      -1 : mom = np.array([-0.55762243,-0.72653902,-0.40149507]) ; lmom = 1.00000000  
    //qsim.propagate.head pidx      -1 : pos = np.array([ 145.33224, 208.94881,   8.00225]) ; lpos = 254.64701843 
    //qsim.propagate.head pidx      -1 : nrm = np.array([(-0.55684042,-0.72441316,-0.40639180]) ; lnrm = 0.99999994  
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qsim.propagate_to_boundary.head pidx      -1 : u_absorption 0.87895429 logf(u_absorption) -0.12902236 absorption_length 40893.0938 absorption_distance 5276.123535 
    //qsim.propagate_to_boundary.head pidx      -1 : post = np.array([ 145.33224, 208.94881,   8.00225,   0.79524]) 
    //qsim.propagate_to_boundary.head pidx      -1 : distance_to_boundary 17935.8047 absorption_distance  5276.1235 scattering_distance 16148.5000 
    //qsim.propagate_to_boundary.head pidx      -1 : u_scattering     0.4993 u_absorption     0.8790 
    //qsim.propagate_to_boundary.body.BULK_ABSORB pidx      -1 : post = np.array([-2796.75269,-3624.36084,-2110.33545,  27.92657]) ; absorb_time_delta = 27.13132668   
    //qsim.propagate.body pidx      -1 bounce 1 command 1 flag 8 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.tail pidx      -1 bounce 1 command 1 flag 8 ctx.s.optical.y(ems) 1 
    2025-10-20 15:31:18.163 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   16 dt   36.948240 slice   16 : sslice {     416,     442,4062500000, 253906250}253.906250
    2025-10-20 15:31:58.253 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   17 dt   37.060448 slice   17 : sslice {     442,     468,4316406250, 253906250}253.906250
    2025-10-20 15:32:38.552 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   18 dt   37.145522 slice   18 : sslice {     468,     494,4570312500, 253906250}253.906250
    2025-10-20 15:33:07.251 INFO  [2333764] [QSim::simulate@489]  eventID 0 xxl YES i   19 dt   25.610661 slice   19 : sslice {     494,     512,4824218750, 175781250}175.781250
    2025-10-20 15:34:12.752 INFO  [2333764] [QSim::simulate@524]  eventID 0 tot_dt  726.768595 tot_ph 5000000000 tot_ph/M 5000.000000 tot_ht  997720522 tot_ht/M 997.720520 tot_ht/tot_ph   0.199544 reset_ YES
    2025-10-20 15:34:12.762 INFO  [2333764] [SEvt::save@4384] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt/A000 [genstep,hit]
    2025-10-20 15:41:57.298 INFO  [2333764] [QSim::simulate@545] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 1314.537354 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 786.665894 (multilaunch loop begin to end) 
     (PCAT - LEND)/M  63.354736 (topfold concat and clear subfold) 
     (TAIL - BRES)/M 464.515106 (QSim::reset which saves hits) 
     tot_idt/M       726.769165 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          726.768595 int(tot_dt*M)    726768594 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M        59.697079 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-10-20 15:41:58.861  861420440 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 



PIDX logging uses unsigned(-1) as default value that is never normally reached.
Cannot see any way to avoid this, other than to use a Release build which
does not do PIDX logging.


60 G of hits::

    /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt/A000

    A[blyth@localhost A000]$ du -h *
    3.1G	f000
    3.1G	f001
    3.1G	f002
    3.1G	f003
    3.1G	f004
    3.1G	f005
    3.1G	f006
    3.1G	f007
    2.0G	f008
    52K	genstep.npy
    60G	hit.npy
    4.0K	NPFold_index.txt
    A[blyth@localhost A000]$ 




PIDX -1 DUMPING : NEED TO CHANE PIDX TO ULL : AND USE X40 SENTINEL
-----------------------------------------------------------------------

::

    




Does the index clocking happen as expected ?
-------------------------------------------------

::

    In [1]: ix = f.hit.view(np.uint32)[:,3,2]

    In [2]: ix
    Out[2]: array([        5,         9,        16,        21,        27, ..., 705032677, 705032687, 705032699, 705032700, 705032702], shape=(997720522,), dtype=uint32)

    In [2]: ix  ## still looks clocked but changed, hopefully from no longer duplicating
    Out[2]: array([        5,         9,        16,        21,        27, ..., 705032683, 705032686, 705032696, 705032700, 705032703], shape=(997737665,), dtype=uint32)


    In [3]: iy = f.hit.view(np.uint32)[:,3,2].astype(np.uint64)

    In [4]: iy.min()
    Out[4]: np.uint64(5)

    In [5]: iy.max()
    Out[5]: np.uint64(4294967294)


    In [6]: np.where( iy == 4294967294 )
    Out[6]: (array([857047783]),)

    In [7]: j = 857047783

    In [8]: iy[j-5:j+5]
    Out[8]: array([4294967274, 4294967278, 4294967283, 4294967289, 4294967290, 4294967294,          5,          9,         16,         21], dtype=uint64)

    In [9]: iy[j-10:j+10]
    Out[9]: 
    array([4294967255, 4294967260, 4294967265, 4294967269, 4294967270, 4294967274, 4294967278, 4294967283, 4294967289, 4294967290, 4294967294,          5,          9,         16,         21,         27,
                   45,         47,         49,         83], dtype=uint64)

    In [10]: ix[:10]
    Out[10]: array([ 5,  9, 16, 21, 27, 45, 47, 49, 83, 91], dtype=uint32)




Photon index and photon duplication observed beyond the clocking : must be duplicating photons from clocked photon_idx
---------------------------------------------------------------------------------------------------------------------------

See repetition of indices and hits after the clocking ? 
Must be overwriting or other bug::

    In [11]: f.hit[0]
    Out[11]: 
    array([[-10094.563, -10052.373, -13014.945,    131.872],
           [    -0.674,     -0.309,     -0.671,     -0.   ],
           [     0.696,     -0.57 ,     -0.436,    426.277],
           [     0.   ,      0.   ,      0.   ,      0.   ]], dtype=float32)

    In [12]: f.hit[j]  ## j is max index, before the clocking
    Out[12]: 
    array([[  8884.436, -17051.502,  -1180.661,    111.548],
           [    -0.162,     -0.73 ,     -0.664,     -0.   ],
           [     0.969,      0.011,     -0.248,    420.   ],
           [     0.   ,      0.   ,        nan,      0.   ]], dtype=float32)

    In [13]: f.hit[j+1]
    Out[13]: 
    array([[-10094.563, -10052.373, -13014.945,    131.872],
           [    -0.674,     -0.309,     -0.671,     -0.   ],
           [     0.696,     -0.57 ,     -0.436,    426.277],
           [     0.   ,      0.   ,      0.   ,      0.   ]], dtype=float32)



One bug is that the photon_idx is clocked from use of unsigned when need ULL::

    372 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    373 {
    374     sevent* evt = params.evt ;
    375     if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon
    376 
    377     unsigned idx = launch_idx.x ;
    378     unsigned genstep_idx = evt->seed[idx] ;
    379     const quad6& gs = evt->genstep[genstep_idx] ;
    380     // genstep needs the raw index, from zero for each genstep slice sub-launch
    381 
    382     unsigned photon_idx = params.photon_slot_offset + idx ;  // 4.29 billion slots limit
    383     // rng_state access and array recording needs the absolute photon_idx
    384     // for multi-launch and single-launch simulation to match.
    385     // The offset hides the technicality of the multi-launch from output.
    386 
    387     qsim* sim = params.sim ;
    388 
    389 //#define OLD_WITHOUT_SKIPAHEAD 1
    390 #ifdef OLD_WITHOUT_SKIPAHEAD
    391     RNG rng = sim->rngstate[photon_idx] ;
    392 #else
    393     RNG rng ;
    394     sim->rng->init( rng, sim->evt->index, photon_idx );
    395 #endif
    396 




qrng.h::

    117 template<>
    118 struct qrng<Philox>
    119 {
    120     ULL  seed ;
    121     ULL  offset ;
    122     ULL  skipahead_event_offset ;
    123 
    124 #if defined(__CUDACC__) || defined(__CUDABE__)
    125     QRNG_METHOD void init(Philox& rng, unsigned event_idx, unsigned photon_idx )
    126     {
    127         ULL subsequence_ = photon_idx ;
    128         curand_init( seed, subsequence_, offset, &rng ) ;
    129         ULL skipahead_ = skipahead_event_offset*event_idx ;
    130         skipahead( skipahead_, &rng );
    131     }
    132 #else
    133     qrng(ULL seed_, ULL offset_, ULL skipahead_event_offset_ )
    134         :
    135         seed(seed_),
    136         offset(offset_),
    137         skipahead_event_offset(skipahead_event_offset_)
    138     {
    139     }
    140     void set_uploaded_states( void* ){}
    141 #endif
    142 };
    143 

::

     253 unsigned long long QEvent::get_photon_slot_offset() const
     254 {
     255     return gss ? gss->ph_offset : 0 ;
     256 }


Params.h widen photon_slot_offset to ULL::

     83     // simulation
     84     qsim*        sim ;
     85     sevent*      evt ;         // HMM: inside sim too ?
     86     int  event_index ;
     87     unsigned long long  photon_slot_offset ;   // for multi-launch to match single-launch
     88     float max_time ;           // ns



Try to avoid repetition from clocking the photon_idx
------------------------------------------------------

::

    2025-10-20 16:39:23.756 INFO  [2359523] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest f94d93c709d76d3f6c8cc0ad6c25e61a dynamic f94d93c709d76d3f6c8cc0ad6c25e61a
    2025-10-20 16:39:23.757 INFO  [2359523] [QSim::simulate@441]  eventID      0 igs (512, 6, 4, ) tot_ph_0 5000000000 tot_ph_0/M 5000 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 20 TotalPhoton 5000000000 TotalPhoton/M 5000.000000
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      26,         0, 253906250}253.906250
       1 : sslice {      26,      52, 253906250, 253906250}253.906250
       2 : sslice {      52,      78, 507812500, 253906250}253.906250
       3 : sslice {      78,     104, 761718750, 253906250}253.906250
       4 : sslice {     104,     130,1015625000, 253906250}253.906250
       5 : sslice {     130,     156,1269531250, 253906250}253.906250
       6 : sslice {     156,     182,1523437500, 253906250}253.906250
       7 : sslice {     182,     208,1777343750, 253906250}253.906250
       8 : sslice {     208,     234,2031250000, 253906250}253.906250
       9 : sslice {     234,     260,2285156250, 253906250}253.906250
      10 : sslice {     260,     286,2539062500, 253906250}253.906250
      11 : sslice {     286,     312,2792968750, 253906250}253.906250
      12 : sslice {     312,     338,3046875000, 253906250}253.906250
      13 : sslice {     338,     364,3300781250, 253906250}253.906250
      14 : sslice {     364,     390,3554687500, 253906250}253.906250
      15 : sslice {     390,     416,3808593750, 253906250}253.906250
      16 : sslice {     416,     442,4062500000, 253906250}253.906250
      17 : sslice {     442,     468,4316406250, 253906250}253.906250
      18 : sslice {     468,     494,4570312500, 253906250}253.906250
      19 : sslice {     494,     512,4824218750, 175781250}175.781250
                      start    stop     offset      count    count/M 
     num_slice 20
    2025-10-20 16:39:59.427 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    0 dt   35.628212 slice    0 : sslice {       0,      26,         0, 253906250}253.906250
    2025-10-20 16:40:38.735 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    1 dt   36.384852 slice    1 : sslice {      26,      52, 253906250, 253906250}253.906250
    2025-10-20 16:41:18.558 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    2 dt   36.853893 slice    2 : sslice {      52,      78, 507812500, 253906250}253.906250
    2025-10-20 16:41:58.918 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    3 dt   37.305328 slice    3 : sslice {      78,     104, 761718750, 253906250}253.906250
    2025-10-20 16:42:38.990 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    4 dt   37.035025 slice    4 : sslice {     104,     130,1015625000, 253906250}253.906250
    2025-10-20 16:43:19.158 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    5 dt   37.162806 slice    5 : sslice {     130,     156,1269531250, 253906250}253.906250
    2025-10-20 16:43:59.260 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    6 dt   37.061295 slice    6 : sslice {     156,     182,1523437500, 253906250}253.906250
    2025-10-20 16:44:39.213 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    7 dt   37.016884 slice    7 : sslice {     182,     208,1777343750, 253906250}253.906250
    2025-10-20 16:45:19.267 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    8 dt   37.028314 slice    8 : sslice {     208,     234,2031250000, 253906250}253.906250
    2025-10-20 16:45:59.356 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i    9 dt   36.991541 slice    9 : sslice {     234,     260,2285156250, 253906250}253.906250
    2025-10-20 16:46:39.570 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   10 dt   37.070125 slice   10 : sslice {     260,     286,2539062500, 253906250}253.906250
    2025-10-20 16:47:19.576 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   11 dt   36.976406 slice   11 : sslice {     286,     312,2792968750, 253906250}253.906250
    2025-10-20 16:47:59.506 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   12 dt   36.955921 slice   12 : sslice {     312,     338,3046875000, 253906250}253.906250
    2025-10-20 16:48:39.474 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   13 dt   36.946415 slice   13 : sslice {     338,     364,3300781250, 253906250}253.906250
    2025-10-20 16:49:19.324 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   14 dt   36.931239 slice   14 : sslice {     364,     390,3554687500, 253906250}253.906250
    2025-10-20 16:49:59.187 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   15 dt   36.916831 slice   15 : sslice {     390,     416,3808593750, 253906250}253.906250

    //qsim.propagate.head pidx      -1 : ctx.evt.index 0 evt.index 0 

    //qsim.propagate.head pidx      -1 : bnc 0 boundary 132 cosTheta 1.00000000 
    //qsim.propagate.head pidx      -1 : mom = np.array([0.57072037,0.82054293,0.03142488]) ; lmom = 1.00000000  
    //qsim.propagate.head pidx      -1 : pos = np.array([  57.07204,  82.05429,   3.14249]) ; lpos = 99.99999237 
    //qsim.propagate.head pidx      -1 : nrm = np.array([(0.57072037,0.82054293,0.03142488]) ; lnrm = 1.00000000  
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qsim.propagate_to_boundary.head pidx      -1 : u_absorption 0.72304481 logf(u_absorption) -0.32428402 absorption_length 40893.0938 absorption_distance 13260.976562 
    //qsim.propagate_to_boundary.head pidx      -1 : post = np.array([  57.07204,  82.05429,   3.14249,   0.00000]) 
    //qsim.propagate_to_boundary.head pidx      -1 : distance_to_boundary 17600.0000 absorption_distance 13260.9766 scattering_distance   154.6470 
    //qsim.propagate_to_boundary.head pidx      -1 : u_scattering     0.9934 u_absorption     0.7230 
    //qsim.propagate.body pidx      -1 bounce 0 command 2 flag 32 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.tail pidx      -1 bounce 0 command 2 flag 32 ctx.s.optical.y(ems) 1 

    //qsim.propagate.head pidx      -1 : ctx.evt.index 0 evt.index 0 

    //qsim.propagate.head pidx      -1 : bnc 1 boundary 132 cosTheta 0.99998546 
    //qsim.propagate.head pidx      -1 : mom = np.array([-0.55762243,-0.72653902,-0.40149507]) ; lmom = 1.00000000  
    //qsim.propagate.head pidx      -1 : pos = np.array([ 145.33224, 208.94881,   8.00225]) ; lpos = 254.64701843 
    //qsim.propagate.head pidx      -1 : nrm = np.array([(-0.55684042,-0.72441316,-0.40639180]) ; lnrm = 0.99999994  
    //qbnd.fill_state idx -1 boundary 132 line 528 wavelength   420.0000 m1_line 531 m2_line 528 su_line 530 s.optical.x 0  
    //qbnd.fill_state idx -1 boundary 132 [s.index.x-1](m1_index) 4 [s.index.y-1](m2_index) 15 [s.index.z-1](su_index) -1 
    //qsim.propagate_to_boundary.head pidx      -1 : u_absorption 0.87895429 logf(u_absorption) -0.12902236 absorption_length 40893.0938 absorption_distance 5276.123535 
    //qsim.propagate_to_boundary.head pidx      -1 : post = np.array([ 145.33224, 208.94881,   8.00225,   0.79524]) 
    //qsim.propagate_to_boundary.head pidx      -1 : distance_to_boundary 17935.8047 absorption_distance  5276.1235 scattering_distance 16148.5000 
    //qsim.propagate_to_boundary.head pidx      -1 : u_scattering     0.4993 u_absorption     0.8790 
    //qsim.propagate_to_boundary.body.BULK_ABSORB pidx      -1 : post = np.array([-2796.75269,-3624.36084,-2110.33545,  27.92657]) ; absorb_time_delta = 27.13132668   
    //qsim.propagate.body pidx      -1 bounce 1 command 1 flag 8 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.tail pidx      -1 bounce 1 command 1 flag 8 ctx.s.optical.y(ems) 1 
    2025-10-20 16:50:39.006 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   16 dt   36.909511 slice   16 : sslice {     416,     442,4062500000, 253906250}253.906250
    2025-10-20 16:51:18.824 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   17 dt   36.928301 slice   17 : sslice {     442,     468,4316406250, 253906250}253.906250
    2025-10-20 16:51:58.687 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   18 dt   36.917939 slice   18 : sslice {     468,     494,4570312500, 253906250}253.906250
    2025-10-20 16:52:27.143 INFO  [2359523] [QSim::simulate@489]  eventID 0 xxl YES i   19 dt   25.518892 slice   19 : sslice {     494,     512,4824218750, 175781250}175.781250
    2025-10-20 16:53:27.066 INFO  [2359523] [QSim::simulate@524]  eventID 0 tot_dt  726.539730 tot_ph 5000000000 tot_ph/M 5000.000000 tot_ht  997737665 tot_ht/M 997.737671 tot_ht/tot_ph   0.199548 reset_ YES
    2025-10-20 16:53:27.067 INFO  [2359523] [SEvt::save@4384] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt/A000 [genstep,hit]
    2025-10-20 16:58:48.239 INFO  [2359523] [QSim::simulate@545] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 1164.482544 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 785.406128 (multilaunch loop begin to end) 
     (PCAT - LEND)/M  57.902657 (topfold concat and clear subfold) 
     (TAIL - BRES)/M 321.171967 (QSim::reset which saves hits) 
     tot_idt/M       726.540283 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          726.539730 int(tot_dt*M)    726539729 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M        58.665901 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-10-20 16:58:49.767  767551197 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    [sreport.main  argv0 sreport dirp /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 


::

    In [8]: iy.max()
    Out[8]: np.uint64(4294967294)

    In [9]: 0xffffffff
    Out[9]: 4294967295

    In [10]: j = np.where( iy == 4294967294 )[0]

    In [11]: j
    Out[11]: array([857047783])

    In [12]: j = np.where( iy == 4294967294 )[0][0]

    In [13]: j
    Out[13]: np.int64(857047783)

    In [14]:  iy[j-5:j+5]
    Out[14]: array([4294967274, 4294967278, 4294967283, 4294967289, 4294967290, 4294967294,          4,         11,         15,         17], dtype=uint64)

    In [15]: iy
    Out[15]: array([        5,         9,        16,        21,        27, ..., 705032683, 705032686, 705032696, 705032700, 705032703], shape=(997737665,), dtype=uint64)



Photon duplication looks avoided::


    In [16]: f.hit[j]
    Out[16]: 
    array([[  8884.436, -17051.502,  -1180.661,    111.548],
           [    -0.162,     -0.73 ,     -0.664,     -0.   ],
           [     0.969,      0.011,     -0.248,    420.   ],
           [     0.   ,      0.   ,        nan,      0.   ]], dtype=float32)

    In [17]: f.hit[j+1]
    Out[17]: 
    array([[-9985.125, 13894.735, -8962.97 ,    98.002],
           [   -0.467,     0.726,    -0.505,    -0.   ],
           [    0.297,    -0.409,    -0.863,   420.   ],
           [    0.   ,     0.   ,     0.   ,     0.   ]], dtype=float32)

    In [18]: f.hit[0]
    Out[18]: 
    array([[-10094.563, -10052.373, -13014.945,    131.872],
           [    -0.674,     -0.309,     -0.671,     -0.   ],
           [     0.696,     -0.57 ,     -0.436,    426.277],
           [     0.   ,      0.   ,      0.   ,      0.   ]], dtype=float32)





Look for duplicates among the billion hits
---------------------------------------------

* ~/o/sysrap/tests/sdigest_duplicate_test/sdigest_duplicate_test.sh





