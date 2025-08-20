cxr_min_slow_load_from_cvmfs
===============================

Issue
-------

ljr/local_oj_release testing of a cxr_min.sh from cvmfs release using
default geomdir also on cvmfs is very noticeably slower to launch than
with local running and geomdir

Conclusion
-----------

By adding timestamped NPFold::load logging, eg with::

    lo ; CSGFoundry=INFO SSim=INFO NPFold__load_DUMP=1 cxr_min.sh

Observe that the bulk of load time is from the many thousands of meshgroup/0 tri/nrm/vtx from eg::

   /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/

Workaround is to not keep geometries on cvmfs ?


git lfs
---------

* https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage




Load from cvmfs : 6s to CSGFoundry::Load most of which from SSim::Load (factor of 5 slower than local load)
--------------------------------------------------------------------------------------------------------------

::

    ljr ; CSGFoundry=INFO SSim=INFO cxr_min.sh


Summary check again with SSim logging, again almost 6s to SSim::load_ from /cvmfs::

    2025-08-20 14:11:52.329 INFO  [671176] [CSGFoundry::Load@3028] [ argumentless 

    2025-08-20 14:11:52.330 INFO  [671176] [SSim::load_@455] [
    2025-08-20 14:11:52.330 INFO  [671176] [SSim::load_@459] [ top.load [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim]
    2025-08-20 14:11:58.295 INFO  [671176] [SSim::load_@465] ] top.load [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim] toploadtime/1e6  5.965725
    2025-08-20 14:11:58.431 INFO  [671176] [SSim::load_@473] ]

    2025-08-20 14:11:59.685 INFO  [671176] [CSGFoundry::Load@3056] ] argumentless 


Contrast with local load::

    ]NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/extra/jpmt/PMTSimParamData/CECOStheta rc 0]
    ]NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/extra/jpmt/PMTSimParamData rc 0]
    [NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/extra/jpmt/PMT_RINDEX]
    ]NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/extra/jpmt/PMT_RINDEX rc 0]
    ]NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/extra/jpmt rc 0]
    ]NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/extra rc 0]
    ]NPFold__load_DUMP 2025-08-20 16:58:23.191 : [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim rc 0]
    2025-08-20 16:58:23.191 INFO  [687348] [SSim::load_@465] ] top.load [/home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim] toploadtime/1e6  1.136827

Factor 5 slower from cvmfs::

    In [1]: 5.965725/1.136827
    Out[1]: 5.247698198582546




Summary log::

    2025-08-19 16:16:17.206 INFO  [3649064] [CSGFoundry::Load@3028] [ argumentless 
    2025-08-19 16:16:17.206 INFO  [3649064] [CSGFoundry::Load_@3143] [ SSim::Load cfbase /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2
    2025-08-19 16:16:22.704 INFO  [3649064] [CSGFoundry::Load_@3145] ] SSim::Load 
    2025-08-19 16:16:23.663 INFO  [3649064] [CSGFoundry::Load@3056] ] argumentless 


Fuller log::

    2025-08-19 16:16:17.206 INFO  [3649064] [CSGFoundry::Load@3028] [ argumentless 
    2025-08-19 16:16:17.206 INFO  [3649064] [CSGFoundry::Load_@3143] [ SSim::Load cfbase /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2
    2025-08-19 16:16:22.704 INFO  [3649064] [CSGFoundry::Load_@3145] ] SSim::Load 
    2025-08-19 16:16:22.904 INFO  [3649064] [CSGFoundry::load@2877] [ loaddir /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry
    2025-08-19 16:16:22.906 INFO  [3649064] [CSGFoundry::loadArray@3217]  ni    10 nj 3 nk 4 solid.npy
    2025-08-19 16:16:22.907 INFO  [3649064] [CSGFoundry::loadArray@3217]  ni  5515 nj 4 nk 4 prim.npy
    2025-08-19 16:16:22.909 INFO  [3649064] [CSGFoundry::loadArray@3217]  ni 28737 nj 4 nk 4 node.npy
    2025-08-19 16:16:22.910 INFO  [3649064] [CSGFoundry::loadArray@3217]  ni 11478 nj 4 nk 4 tran.npy
    2025-08-19 16:16:22.911 INFO  [3649064] [CSGFoundry::loadArray@3217]  ni 11478 nj 4 nk 4 itra.npy
    2025-08-19 16:16:22.913 INFO  [3649064] [CSGFoundry::loadArray@3217]  ni 47888 nj 4 nk 4 inst.npy
    2025-08-19 16:16:22.914 INFO  [3649064] [CSGFoundry::load@2904] ] loaddir /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry
    2025-08-19 16:16:22.914 INFO  [3649064] [CSGFoundry::CopySelect@3080] [
    2025-08-19 16:16:22.914 INFO  [3649064] [CSGFoundry::CopySelect@3082]    -        t: 315 : 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

    2025-08-19 16:16:22.914 INFO  [3649064] [CSGFoundry::CopySelect@3083] CSGFoundry::descELV elv.num_bits 315 num_include 315 num_exclude 0 is_all_set 1
    INCLUDE:315

    p:  0:midx:  0:mn:sTopRock_domeAir
    p:  1:midx:  1:mn:sTopRock_dome
    p:  2:midx:  2:mn:sDomeRockBox
    p:  3:midx:  3:mn:PoolCoversub
    ...
    p:311:midx:311:mn:sAirGap
    p:312:midx:312:mn:sPoolLining
    p:313:midx:313:mn:sBottomRock
    p:314:midx:314:mn:sWorld
    EXCLUDE:0


    2025-08-19 16:16:22.915 INFO  [3649064] [CSGFoundry::CopySelect@3084] CSGFoundry::descELV2 elv.num_bits 315 id.getNumName 315
      0 Y [sTopRock_domeAir] 
      1 Y [sTopRock_dome] 
      2 Y [sDomeRockBox] 
      3 Y [PoolCoversub] 
    ...
    312 Y [sPoolLining] 
    313 Y [sBottomRock] 
    314 Y [sWorld] 

    2025-08-19 16:16:23.367 INFO  [3649064] [CSGFoundry::CopySelect@3091] ]
    2025-08-19 16:16:23.367 INFO  [3649064] [CSGFoundry::Load@3040]  apply ELV selection to triangulated SScene 
    2025-08-19 16:16:23.663 INFO  [3649064] [CSGFoundry::getFrameE@3633] [CSGFoundry__getFrameE_VERBOSE] 0
    2025-08-19 16:16:23.663 INFO  [3649064] [CSGFoundry::getFrame@3503] [CSGFoundry__getFrame_VERBOSE] 0
    2025-08-19 16:16:23.663 INFO  [3649064] [CSGFoundry::AfterLoadOrCreate@3690] sframe::desc inst 0 frs PMT_3inch:0:0
     ekvid sframe_MOI_PMT_3inch_0_0 ek MOI ev PMT_3inch:0:0
     ce  (-5459.083,18213.496,3693.582,40.603)  is_zero 0
     m2w ( 0.055,-0.183, 0.982, 0.000) (-0.958,-0.287, 0.000, 0.000) ( 0.282,-0.940,-0.191, 0.000) (-5463.190,18227.199,3696.360, 1.000) 
     w2m ( 0.055,-0.958, 0.282, 0.000) (-0.183,-0.287,-0.940, 0.000) ( 0.982, 0.000,-0.191, 0.000) (-0.005, 0.029,19384.021, 1.000) 
     midx  135 mord    0 gord    0
     inst    0
     ix0     0 ix1     0 iy0     0 iy1     0 iz0     0 iz1     0 num_photon    0
     ins     0 gas     0 sensor_identifier        0 sensor_index      0
     propagate_epsilon    0.05000 is_hostside_simtrace NO

    2025-08-19 16:16:23.663 INFO  [3649064] [CSGFoundry::Load@3056] ] argumentless 




Local viz : About 2s to CSGFoundry::Load with more than 1s from SSim::Load
----------------------------------------------------------------------------

Compare with local viz::

    lo
    export CSGFoundry=INFO
    cxr_min.sh 

Smry log::

    2025-08-19 16:34:38.812 INFO  [3649947] [CSGFoundry::Load@3028] [ argumentless 
    2025-08-19 16:34:38.812 INFO  [3649947] [CSGFoundry::Load_@3143] [ SSim::Load cfbase /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug
    2025-08-19 16:34:39.973 INFO  [3649947] [CSGFoundry::Load_@3145] ] SSim::Load 
    2025-08-19 16:34:40.331 INFO  [3649947] [CSGFoundry::Load@3056] ] argumentless 


Fuller log::

    2025-08-19 16:34:38.812 INFO  [3649947] [CSGFoundry::Load@3028] [ argumentless 
    2025-08-19 16:34:38.812 INFO  [3649947] [CSGFoundry::Load_@3143] [ SSim::Load cfbase /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug
    2025-08-19 16:34:39.973 INFO  [3649947] [CSGFoundry::Load_@3145] ] SSim::Load 
    2025-08-19 16:34:39.982 INFO  [3649947] [CSGFoundry::load@2877] [ loaddir /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry
    2025-08-19 16:34:39.982 INFO  [3649947] [CSGFoundry::loadArray@3217]  ni    10 nj 3 nk 4 solid.npy
    2025-08-19 16:34:39.982 INFO  [3649947] [CSGFoundry::loadArray@3217]  ni  5515 nj 4 nk 4 prim.npy
    2025-08-19 16:34:39.983 INFO  [3649947] [CSGFoundry::loadArray@3217]  ni 28737 nj 4 nk 4 node.npy
    2025-08-19 16:34:39.984 INFO  [3649947] [CSGFoundry::loadArray@3217]  ni 11478 nj 4 nk 4 tran.npy
    2025-08-19 16:34:39.985 INFO  [3649947] [CSGFoundry::loadArray@3217]  ni 11478 nj 4 nk 4 itra.npy
    2025-08-19 16:34:39.986 INFO  [3649947] [CSGFoundry::loadArray@3217]  ni 47888 nj 4 nk 4 inst.npy
    2025-08-19 16:34:39.988 INFO  [3649947] [CSGFoundry::load@2904] ] loaddir /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry
    2025-08-19 16:34:39.988 INFO  [3649947] [CSGFoundry::CopySelect@3080] [
    2025-08-19 16:34:39.988 INFO  [3649947] [CSGFoundry::CopySelect@3082]    -t:314,313,16,14,15,2,0,1,2,13,3,311,310,309,308,312,101,102,302,303,301,300,125,134 315 : 000011111111100001111111111111111111111111111111111111111111111111111111111111111111111111111111111110011111111111111111111110111111110111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000011110000000

    2025-08-19 16:34:39.988 INFO  [3649947] [CSGFoundry::CopySelect@3083] CSGFoundry::descELV elv.num_bits 315 num_include 292 num_exclude 23 is_all_set 0
    INCLUDE:292

    p:  4:midx:  4:mn:Upper_LS_tube
    p:  5:midx:  5:mn:Upper_Steel_tube
    p:  6:midx:  6:mn:Upper_Chimney
    p:  7:midx:  7:mn:sBar_0
    ...
    p:304:midx:304:mn:mask_PMT_20inch_vetosMask
    p:305:midx:305:mn:PMT_20inch_veto_inner_solid_1_2
    p:306:midx:306:mn:PMT_20inch_veto_pmt_solid_1_2
    p:307:midx:307:mn:mask_PMT_20inch_vetosMask_virtual
    EXCLUDE:23

    p:  0:midx:  0:mn:sTopRock_domeAir
    p:  1:midx:  1:mn:sTopRock_dome
    p:  2:midx:  2:mn:sDomeRockBox
    p:  3:midx:  3:mn:PoolCoversub
    ...
    p:312:midx:312:mn:sPoolLining
    p:313:midx:313:mn:sBottomRock
    p:314:midx:314:mn:sWorld

    2025-08-19 16:34:39.989 INFO  [3649947] [CSGFoundry::CopySelect@3084] CSGFoundry::descELV2 elv.num_bits 315 id.getNumName 315
      0 N [sTopRock_domeAir] 
      1 N [sTopRock_dome] 
      2 N [sDomeRockBox] 
      3 N [PoolCoversub] 
    ....
    299 Y [sSurftube_38V1_1] 
    304 Y [mask_PMT_20inch_vetosMask] 
    305 Y [PMT_20inch_veto_inner_solid_1_2] 
    306 Y [PMT_20inch_veto_pmt_solid_1_2] 
    307 Y [mask_PMT_20inch_vetosMask_virtual] 

    2025-08-19 16:34:40.054 INFO  [3649947] [CSGFoundry::CopySelect@3091] ]
    2025-08-19 16:34:40.054 INFO  [3649947] [CSGFoundry::Load@3040]  apply ELV selection to triangulated SScene 
    2025-08-19 16:34:40.331 INFO  [3649947] [CSGFoundry::getFrameE@3633] [CSGFoundry__getFrameE_VERBOSE] 0
    2025-08-19 16:34:40.331 INFO  [3649947] [CSGFoundry::getFrame@3503] [CSGFoundry__getFrame_VERBOSE] 0
    2025-08-19 16:34:40.331 INFO  [3649947] [CSGFoundry::AfterLoadOrCreate@3690] sframe::desc inst 0 frs PMT_3inch:0:0
     ekvid sframe_MOI_PMT_3inch_0_0 ek MOI ev PMT_3inch:0:0
     ce  (-5459.083,18213.496,3693.582,40.603)  is_zero 0
     m2w ( 0.055,-0.183, 0.982, 0.000) (-0.958,-0.287, 0.000, 0.000) ( 0.282,-0.940,-0.191, 0.000) (-5463.190,18227.199,3696.360, 1.000) 
     w2m ( 0.055,-0.958, 0.282, 0.000) (-0.183,-0.287,-0.940, 0.000) ( 0.982, 0.000,-0.191, 0.000) (-0.005, 0.029,19384.021, 1.000) 
     midx  135 mord    0 gord    0
     inst    0
     ix0     0 ix1     0 iy0     0 iy1     0 iz0     0 iz1     0 num_photon    0
     ins     0 gas     0 sensor_identifier        0 sensor_index      0
     propagate_epsilon    0.05000 is_hostside_simtrace NO

    2025-08-19 16:34:40.331 INFO  [3649947] [CSGFoundry::Load@3056] ] argumentless 




Drill down into SSim::load is there any low hanging fruit ?
------------------------------------------------------------

::

    ljr ; CSGFoundry=INFO SSim=INFO NPFold__load_DUMP=1 cxr_min.sh


Summary check again with SSim logging, again almost 6s to SSim::load_ from /cvmfs::

    2025-08-20 14:11:52.329 INFO  [671176] [CSGFoundry::Load@3028] [ argumentless 

    2025-08-20 14:11:52.330 INFO  [671176] [SSim::load_@455] [
    2025-08-20 14:11:52.330 INFO  [671176] [SSim::load_@459] [ top.load [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim]
    2025-08-20 14:11:58.295 INFO  [671176] [SSim::load_@465] ] top.load [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim] toploadtime/1e6  5.965725
    2025-08-20 14:11:58.431 INFO  [671176] [SSim::load_@473] ]

    2025-08-20 14:11:59.685 INFO  [671176] [CSGFoundry::Load@3056] ] argumentless 


Hog is the NPFold::load call::

     459     LOG(LEVEL) << "[ top.load [" << dir << "]" ;
     460 
     461     int64_t t0 = sstamp::Now();
     462     top->load(dir) ;
     463     toploadtime = sstamp::Now() - t0 ;
     464 
     465     LOG(LEVEL) << "] top.load [" << dir << "] toploadtime/1e6 " << std::fixed << std::setw(9) << std::setprecision(6) << toploadtime/1e6 ;



Add GEOM config such that local opticks build loads from the release geomdir
-----------------------------------------------------------------------------

GEOM.sh::

     17 geom=J25_4_0_Opticks_v0_5_2
     ..
     38 export GEOM=$geom
     39 
     40 if [ "$GEOM" == "J25_4_0_Opticks_v0_5_2" ]; then
     41     export ${GEOM}_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/$GEOM
     42 

::

    lo ; CSGFoundry=INFO SSim=INFO NPFold__load_DUMP=1 cxr_min.sh




Looks like cvmfs is slow at loading the thousands of CSGFoundry/SSim/scene/meshgroup/0/5027 files ?::

    ...
    [NPFold__load_DUMP 2025-08-20 16:37:34.407 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/882]
    ]NPFold__load_DUMP 2025-08-20 16:37:34.408 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/882 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:34.408 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/883]
    ]NPFold__load_DUMP 2025-08-20 16:37:34.409 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/883 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:34.409 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/884]
    ]NPFold__load_DUMP 2025-08-20 16:37:34.410 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/884 rc 0]
    ...
    ]NPFold__load_DUMP 2025-08-20 16:37:38.457 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.457 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5023]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.458 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5023 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.458 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5024]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.459 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5024 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.459 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5025]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.460 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5025 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.460 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5026]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.461 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5026 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.461 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5027]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.462 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5027 rc 0]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.462 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.462 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1]
    [NPFold__load_DUMP 2025-08-20 16:37:38.463 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/0]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.464 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/0 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.464 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/1]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.465 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/1 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.465 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/2]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.466 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/2 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.466 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/3]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.467 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/3 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.467 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/4]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.468 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1/4 rc 0]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.468 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/1 rc 0]
    [NPFold__load_DUMP 2025-08-20 16:37:38.468 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/2]
    [NPFold__load_DUMP 2025-08-20 16:37:38.468 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/2/0]
    ]NPFold__load_DUMP 2025-08-20 16:37:38.469 : [/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/2/0 rc 0]


Hmm meshgroup zero is the globals::

    (ok) A[blyth@localhost sysrap]$ du -hs /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/*
    512	/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/NPFold_index.txt
    512	/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/NPFold_meta.txt
    512	/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/NPFold_names.txt
    3.5K	/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/nrm.npy
    6.5K	/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/tri.npy
    3.5K	/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.2/el9_amd64_gcc11/Tue/.opticks/GEOM/J25_4_0_Opticks_v0_5_2/CSGFoundry/SSim/scene/meshgroup/0/5022/vtx.npy
    (ok) A[blyth@localhost sysrap]$ 

But there is repetition there of the same solid with different transforms. But the geometry model only exploits that for instanced solids when have very
large numbers of repeats.  TODO: When have performance machinery operational again tune the instancing criteria.  It may also benefit loading time.



HMM : 5028:sWorld globals is about twice what it used to be : what new volumes are not being instanced ? 
-------------------------------------------------------------------------------------------------------------

::

    5028:sWorld
    5:PMT_3inch_pmt_solid
    9:NNVTMCPPMTsMask_virtual
    12:HamamatsuR12860sMask_virtual
    4:mask_PMT_20inch_vetosMask_virtual
    1:sStrutBallhead
    1:base_steel
    3:uni_acrylic1
    130:sPanel
    322:solidSJCLSanchor



Need to find some stree methods to dump the globals
-----------------------------------------------------

::

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=0 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 0 num_nodes 25600

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=1 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 1 num_nodes 12657

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=2 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 2 num_nodes 4955

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=3 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 3 num_nodes 2400

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=4 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 4 num_nodes 590

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=5 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 5 num_nodes 590

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=6 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 6 num_nodes 590

    (ok) A[blyth@localhost tests]$ TEST=desc_factor_nodes FIDX=7 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_factor_nodes idx 7 num_nodes 504



