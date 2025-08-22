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





HMM : 5028:sWorld globals is about twice what it used to be : what new volumes are not being instanced ?  353 new WP PMTS under current cut of 500
-----------------------------------------------------------------------------------------------------------------------------------------------------

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


DONE : added stree::desc_NRT
---------------------------------

::

    [stree::desc_NRT.R rem.npy
     vec.size 5028
    [stree::desc_unique_lvid
     some_lvid.size 5028
     u_lvid.size 114
     c_lvid.size 114
     o_lvid.size 114
     x_lvid.size 114
    [s_unique_desc
         num_uvals      114
         num_unams      315
         num_count      114
         num_order      114
         num_index      114
       num_inverse       -1
      num_original       -1

     uv    105 :  cn    370  ix   2346  un sStrut_0
     uv     46 :  cn    353  ix    994  un PMT_20inch_mcp_solid
     uv     45 :  cn    353  ix    993  un PMT_20inch_tube_solid
     uv     48 :  cn    353  ix    989  un PMT_20inch_pmt_solid_head
     uv     47 :  cn    353  ix    990  un PMT_20inch_inner_solid_head
     uv     43 :  cn    353  ix    991  un PMT_20inch_edge_solid
     uv     44 :  cn    353  ix    992  un PMT_20inch_plate_solid
     uv    106 :  cn    220  ix   2716  un sStrut_1

     In [4]: 353*6
     Out[4]: 2118

     In [5]: 353*6 + 370 + 220
     Out[5]: 2708


     uv     11 :  cn    126  ix     13  un sPlane
     uv     12 :  cn     63  ix     12  un sWall
     uv     90 :  cn     30  ix   2130  un GLb1.bt05_FlangeI_Web_FlangeII
     uv     56 :  cn     30  ix   1130  un GLw1.up06_up07_FlangeI_Web_FlangeII
     uv     89 :  cn     30  ix   2100  un GLb2.bt04_FlangeI_Web_FlangeII
     uv     53 :  cn     30  ix   1040  un GLw1.up09_up10_FlangeI_Web_FlangeII
     uv     54 :  cn     30  ix   1070  un GLw1.up08_up09_FlangeI_Web_FlangeII
     uv     55 :  cn     30  ix   1100  un GLw1.up07_up08_FlangeI_Web_FlangeII
     uv     59 :  cn     30  ix   1220  un GLw1.up03_up04_FlangeI_Web_FlangeII
     uv     91 :  cn     30  ix   2160  un GLb1.bt06_FlangeI_Web_FlangeII
     uv     77 :  cn     30  ix   1740  un GLb2.up08_FlangeI_Web_FlangeII
     uv     78 :  cn     30  ix   1770  un GLb2.up07_FlangeI_Web_FlangeII
     uv     42 :  cn     30  ix    957  un ZC2.A05_A05_FlangeI_Web_FlangeII
     uv     41 :  cn     30  ix    927  un ZC2.A03_A03_FlangeI_Web_FlangeII
     uv     40 :  cn     30  ix    897  un ZC2.B05_B05_FlangeI_Web_FlangeII
     uv     66 :  cn     30  ix   1430  un GLw2.bt03_bt04_FlangeI_Web_FlangeII
     uv     76 :  cn     30  ix   1710  un GLb3.up09_FlangeI_Web_FlangeII
     uv     75 :  cn     30  ix   1680  un GLb4.up10_FlangeI_Web_FlangeII
     uv     74 :  cn     30  ix   1650  un GLb3.up11_FlangeI_Web_FlangeII
     uv     72 :  cn     30  ix   1610  un GLw1.bt09_bt10_FlangeI_Web_FlangeII
     uv     71 :  cn     30  ix   1580  un GLw1.bt08_bt09_FlangeI_Web_FlangeII
     uv     70 :  cn     30  ix   1550  un GLw1.bt07_bt08_FlangeI_Web_FlangeII
     uv     69 :  cn     30  ix   1520  un GLw1.bt06_bt07_FlangeI_Web_FlangeII
     uv     68 :  cn     30  ix   1490  un GLw1.bt05_bt06_FlangeI_Web_FlangeII
     uv     67 :  cn     30  ix   1460  un GLw2.bt04_bt05_FlangeI_Web_FlangeII
     uv     57 :  cn     30  ix   1160  un GLw1.up05_up06_FlangeI_Web_FlangeII
     uv     65 :  cn     30  ix   1400  un GLw3.bt02_bt03_FlangeI_Web_FlangeII
     uv     64 :  cn     30  ix   1370  un GLw3.bt01_bt02_FlangeI_Web_FlangeII
     uv     63 :  cn     30  ix   1340  un GLw2.equ_bt01_FlangeI_Web_FlangeII
     uv     62 :  cn     30  ix   1310  un GLw2.equ_up01_FlangeI_Web_FlangeII
     uv     61 :  cn     30  ix   1280  un GLw1.up01_up02_FlangeI_Web_FlangeII
     uv     60 :  cn     30  ix   1250  un GLw1.up02_up03_FlangeI_Web_FlangeII
     uv     38 :  cn     30  ix    837  un ZC2.B01_B01_FlangeI_Web_FlangeII
     uv     58 :  cn     30  ix   1190  un GLw1.up04_up05_FlangeI_Web_FlangeII
     uv     79 :  cn     30  ix   1800  un GLb2.up06_FlangeI_Web_FlangeII
     uv     39 :  cn     30  ix    867  un ZC2.B03_B03_FlangeI_Web_FlangeII
     uv     19 :  cn     30  ix    267  un GZ1.A03_04_FlangeI_Web_FlangeII
     uv     18 :  cn     30  ix    237  un GZ1.A02_03_FlangeI_Web_FlangeII
     uv     17 :  cn     30  ix    207  un GZ1.A01_02_FlangeI_Web_FlangeII
     uv     92 :  cn     30  ix   2190  un GLb1.bt07_FlangeI_Web_FlangeII
     uv     93 :  cn     30  ix   2220  un GLb1.bt08_FlangeI_Web_FlangeII
     uv     94 :  cn     30  ix   2250  un GLb3.bt09_FlangeI_Web_FlangeII
     uv     95 :  cn     30  ix   2280  un GLb3.bt10_FlangeI_Web_FlangeII
     uv     96 :  cn     30  ix   2310  un GLb3.bt11_FlangeI_Web_FlangeII
     uv     20 :  cn     30  ix    297  un GZ1.A04_05_FlangeI_Web_FlangeII
     uv     80 :  cn     30  ix   1830  un GLb1.up05_FlangeI_Web_FlangeII
     uv     81 :  cn     30  ix   1860  un GLb1.up04_FlangeI_Web_FlangeII
     uv     82 :  cn     30  ix   1890  un GLb1.up03_FlangeI_Web_FlangeII
     uv     83 :  cn     30  ix   1920  un GLb1.up02_FlangeI_Web_FlangeII
     uv     84 :  cn     30  ix   1950  un GLb1.up01_FlangeI_Web_FlangeII
     uv     85 :  cn     30  ix   1980  un GLb2.equ_FlangeI_Web_FlangeII
     uv     86 :  cn     30  ix   2010  un GLb2.bt01_FlangeI_Web_FlangeII
     uv     87 :  cn     30  ix   2040  un GLb1.bt02_FlangeI_Web_FlangeII
     uv     88 :  cn     30  ix   2070  un GLb2.bt03_FlangeI_Web_FlangeII
     uv     22 :  cn     30  ix    357  un GZ1.A06_07_FlangeI_Web_FlangeII
     uv     37 :  cn     30  ix    807  un ZC2.A05_B06_FlangeI_Web_FlangeII
     uv     36 :  cn     30  ix    777  un ZC2.A04_B05_FlangeI_Web_FlangeII
     uv     35 :  cn     30  ix    747  un ZC2.A03_B04_FlangeI_Web_FlangeII
     uv     34 :  cn     30  ix    717  un ZC2.A02_B03_FlangeI_Web_FlangeII
     uv     33 :  cn     30  ix    687  un ZC2.A06_B06_FlangeI_Web_FlangeII
     uv     32 :  cn     30  ix    657  un ZC2.A05_B05_FlangeI_Web_FlangeII
     uv     31 :  cn     30  ix    627  un ZC2.A04_B04_FlangeI_Web_FlangeII
     uv     30 :  cn     30  ix    597  un ZC2.A03_B03_FlangeI_Web_FlangeII
     uv     29 :  cn     30  ix    567  un ZC2.A02_B02_FlangeI_Web_FlangeII
     uv     28 :  cn     30  ix    537  un GZ1.B06_07_FlangeI_Web_FlangeII
     uv     27 :  cn     30  ix    507  un GZ1.B05_06_FlangeI_Web_FlangeII
     uv     26 :  cn     30  ix    477  un GZ1.B04_05_FlangeI_Web_FlangeII
     uv     25 :  cn     30  ix    447  un GZ1.B03_04_FlangeI_Web_FlangeII
     uv     24 :  cn     30  ix    417  un GZ1.B02_03_FlangeI_Web_FlangeII
     uv     23 :  cn     30  ix    387  un GZ1.B01_02_FlangeI_Web_FlangeII
     uv     21 :  cn     30  ix    327  un GZ1.A05_06_FlangeI_Web_FlangeII
     uv     52 :  cn     10  ix   1030  un GLw1.up10_up11_FlangeI_Web_FlangeII
     uv     73 :  cn     10  ix   1640  un GLw1.bt10_bt11_FlangeI_Web_FlangeII
     uv     49 :  cn      5  ix    988  un sWaterAttLenMesaureEquipinner
     uv     50 :  cn      5  ix    987  un sWaterAttLenMesaureEquip
     uv    142 :  cn      1  ix   2939  un sChimneySteel
     uv    301 :  cn      1  ix   2340  un sInnerReflectorInCD_TSubWaterDistributor
     uv    300 :  cn      1  ix   2341  un sInnerWater_TSubWaterDistributor
     uv    102 :  cn      1  ix   2342  un sAcrylic_T
     uv    101 :  cn      1  ix   2343  un sTarget_T
     uv    104 :  cn      1  ix   2344  un sBotChimneySSEnclosure_PartI_II_III
     uv    103 :  cn      1  ix   2345  un sBotChimneySSEnclosureLS_PartI_II_III
     uv    143 :  cn      1  ix   2936  un sWaterTube
     uv    140 :  cn      1  ix   2937  un sChimneyAcrylic
     uv    141 :  cn      1  ix   2938  un sChimneyLS
     uv    314 :  cn      1  ix      0  un sWorld
     uv      4 :  cn      1  ix     10  un Upper_LS_tube
     uv     16 :  cn      1  ix      1  un sTopRock
     uv      2 :  cn      1  ix      2  un sDomeRockBox
     uv      1 :  cn      1  ix      3  un sTopRock_dome
     uv      0 :  cn      1  ix      4  un sTopRock_domeAir
     uv     15 :  cn      1  ix      5  un sExpRockBox
     uv     14 :  cn      1  ix      6  un sExpHall
     uv      3 :  cn      1  ix      7  un PoolCoversub
     uv      6 :  cn      1  ix      8  un Upper_Chimney
     uv      5 :  cn      1  ix      9  un Upper_Steel_tube
     uv    302 :  cn      1  ix   1029  un sOuterWaterInCD_TSubWaterDistributor
     uv     13 :  cn      1  ix     11  un sAirTT
     uv    313 :  cn      1  ix    201  un sBottomRock
     uv    312 :  cn      1  ix    202  un sPoolLining
     uv    311 :  cn      1  ix    203  un sAirGap
     uv    310 :  cn      1  ix    204  un sDeadWater_shell
     uv    309 :  cn      1  ix    205  un sTyvek_shell
     uv    308 :  cn      1  ix    206  un sOuterWaterPool
     uv     51 :  cn      1  ix   1027  un sWaterAttLenMeasureEquipShield
     uv    303 :  cn      1  ix   1028  un sOuterReflectorInCD_TSubWaterDistributor

    ]s_unique_desc

    ]stree::desc_unique_lvid
    ]stree::desc_NRT.R rem.npy
    [stree::desc_NRT.T tri.npy
     vec.size 322
    [stree::desc_unique_lvid
     some_lvid.size 322
     u_lvid.size 162
     c_lvid.size 162
     o_lvid.size 162
     x_lvid.size 162
    [s_unique_desc
         num_uvals      162
         num_unams      315
         num_count      162
         num_order      162
         num_index      162
       num_inverse       -1
      num_original       -1
     uv    112 :  cn     56  ix     54  un solidXJanchor
     uv    113 :  cn     56  ix    110  un solidXJfixture
     uv     98 :  cn     36  ix      2  un solidSJFixture
     uv     99 :  cn      8  ix     38  un solidSJReceiver
     uv    100 :  cn      8  ix     46  un solidSJReceiverFastern
     uv     97 :  cn      2  ix      0  un solidSJCLSanchor
     uv    240 :  cn      1  ix    263  un svacSurftube_9V1_1
     uv    258 :  cn      1  ix    281  un svacSurftube_18V1_1
     uv    243 :  cn      1  ix    264  un sSurftube_10V1_1
     uv    242 :  cn      1  ix    265  un svacSurftube_10V1_1
     uv    245 :  cn      1  ix    266  un sSurftube_11V1_1
     uv    244 :  cn      1  ix    267  un svacSurftube_11V1_1
     uv    247 :  cn      1  ix    268  un sSurftube_12V1_1
     uv    246 :  cn      1  ix    269  un svacSurftube_12V1_1
     uv    249 :  cn      1  ix    270  un sSurftube_13V1_1
     uv    248 :  cn      1  ix    271  un svacSurftube_13V1_1
     uv    251 :  cn      1  ix    272  un sSurftube_14V1_1
     uv    250 :  cn      1  ix    273  un svacSurftube_14V1_1
     uv    253 :  cn      1  ix    274  un sSurftube_15V1_1
     uv    252 :  cn      1  ix    275  un svacSurftube_15V1_1
     uv    255 :  cn      1  ix    276  un sSurftube_16V1_1
     uv    254 :  cn      1  ix    277  un svacSurftube_16V1_1
     uv    257 :  cn      1  ix    278  un sSurftube_17V1_1
     uv    256 :  cn      1  ix    279  un svacSurftube_17V1_1
     uv    259 :  cn      1  ix    280  un sSurftube_18V1_1





bbox of new WP PMT solids
-----------------------------

::


     uv    105 :  cn    370  ix   2346  un sStrut_0
     uv     46 :  cn    353  ix    994  un PMT_20inch_mcp_solid
     uv     45 :  cn    353  ix    993  un PMT_20inch_tube_solid
     uv     48 :  cn    353  ix    989  un PMT_20inch_pmt_solid_head
     uv     47 :  cn    353  ix    990  un PMT_20inch_inner_solid_head
     uv     43 :  cn    353  ix    991  un PMT_20inch_edge_solid
     uv     44 :  cn    353  ix    992  un PMT_20inch_plate_solid
     uv    106 :  cn    220  ix   2716  un sStrut_1


::

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=43 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 43 lvn PMT_20inch_edge_solid root Y sn::rbrief
      0 : sn::brief tc    2 cm  0 lv  43 xf Y pa Y bb Y pt N nc  2 dp  0 tg in bb.desc [-60000.000,-60000.000,-60000.000,60000.000,60000.000,60000.000]
      1 : sn::brief tc  105 cm  0 lv  43 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-47.900,-47.900,-10.000, 47.900, 47.900, 10.000]
      1 : sn::brief tc  105 cm  1 lv  43 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-46.900,-46.900,-10.100, 46.900, 46.900, 10.100]

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=44 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 44 lvn PMT_20inch_plate_solid root Y sn::rbrief
      0 : sn::brief tc    2 cm  0 lv  44 xf Y pa Y bb Y pt N nc  2 dp  0 tg in bb.desc [-60000.000,-60000.000,-60000.000,60000.000,60000.000,60000.000]
      1 : sn::brief tc  105 cm  0 lv  44 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-47.900,-47.900, -5.000, 47.900, 47.900,  5.000]
      1 : sn::brief tc  105 cm  1 lv  44 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-20.000,-20.000, -5.050, 20.000, 20.000,  5.050]

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=45 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 45 lvn PMT_20inch_tube_solid root Y sn::rbrief
      0 : sn::brief tc    2 cm  0 lv  45 xf Y pa Y bb Y pt N nc  2 dp  0 tg in bb.desc [-60000.000,-60000.000,-60000.000,60000.000,60000.000,60000.000]
      1 : sn::brief tc  105 cm  0 lv  45 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-42.900,-42.900,-21.112, 42.900, 42.900, 21.112]
      1 : sn::brief tc  105 cm  1 lv  45 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-41.900,-41.900,-21.323, 41.900, 41.900, 21.323]

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=46 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 46 lvn PMT_20inch_mcp_solid root Y sn::rbrief
      0 : sn::brief tc  105 cm  0 lv  46 xf Y pa Y bb Y pt N nc  0 dp  0 tg cy bb.desc [-20.000,-20.000, -1.000, 20.000, 20.000,  1.000]

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=47 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 47 lvn PMT_20inch_inner_solid_head root Y sn::rbrief
      0 : sn::brief tc  103 cm  0 lv  47 xf Y pa Y bb Y pt N nc  0 dp  0 tg zs bb.desc [-179.000,-179.000,-168.225,179.000,179.000,179.100]

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=48 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 48 lvn PMT_20inch_pmt_solid_head root Y sn::rbrief
      0 : sn::brief tc  103 cm  0 lv  48 xf Y pa Y bb Y pt N nc  0 dp  0 tg zs bb.desc [-184.001,-184.001,-173.226,184.001,184.001,184.101]

    (ok) A[blyth@localhost opticks]$ TEST=desc_solid LVID=49 ~/o/sysrap/tests/stree_load_test.sh run
    stree::desc_solid lvid 49 lvn sWaterAttLenMesaureEquipinner root Y sn::rbrief
      0 : sn::brief tc    1 cm  0 lv  49 xf Y pa Y bb Y pt N nc  2 dp  0 tg un bb.desc [-60000.000,-60000.000,-60000.000,60000.000,60000.000,60000.000]
      1 : sn::brief tc    1 cm  0 lv  49 xf Y pa Y bb Y pt Y nc  2 dp  0 tg un bb.desc [-60000.000,-60000.000,-60000.000,60000.000,60000.000,60000.000]
      2 : sn::brief tc  105 cm  0 lv  49 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-152.500,-152.500,-1740.000,152.500,152.500,-1399.000]
      2 : sn::brief tc  105 cm  0 lv  49 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-310.000,-310.000,-1400.000,310.000,310.000,-1000.000]
      1 : sn::brief tc  105 cm  0 lv  49 xf Y pa Y bb Y pt Y nc  0 dp  0 tg cy bb.desc [-190.000,-190.000,-1001.000,190.000,190.000,  0.000]
    (ok) A[blyth@localhost opticks]$


ELVID Node dumping
-----------------------

::

    (ok) A[blyth@localhost opticks]$ TEST=desc_node_elvid ELVID=43,44,45,46,47,48 ~/o/sysrap/tests/stree_load_test.sh
                       BASH_SOURCE : /home/blyth/o/sysrap/tests/stree_load_test.sh
                               opt : -DWITH_PLACEHOLDER -DWITH_CHILD
                              GEOM : J25_4_0_opticks_Debug
                               CFB : J25_4_0_opticks_Debug_CFBaseFromGEOM
                              FOLD : /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/stree
                               MOI : EXTENT:10000
                              TEST : desc_node_elvid
    [stree::desc_node_elvid
     nam rem.npy
     elvid YES
        989 : snode ix:  66509 dh: 9 sx:    0 pt:  66508 nc:    1 fc:  66510 ns:     -1 lv: 48 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 54 sn:-1 elvid YES
        990 : snode ix:  66510 dh:10 sx:    0 pt:  66509 nc:    4 fc:  66511 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
        991 : snode ix:  66511 dh:11 sx:    0 pt:  66510 nc:    0 fc:     -1 ns:  66512 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
        992 : snode ix:  66512 dh:11 sx:    1 pt:  66510 nc:    0 fc:     -1 ns:  66513 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
        993 : snode ix:  66513 dh:11 sx:    2 pt:  66510 nc:    0 fc:     -1 ns:  66514 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
        994 : snode ix:  66514 dh:11 sx:    3 pt:  66510 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES
        997 : snode ix:  66517 dh: 9 sx:    0 pt:  66516 nc:    1 fc:  66518 ns:     -1 lv: 48 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 54 sn:-1 elvid YES
        998 : snode ix:  66518 dh:10 sx:    0 pt:  66517 nc:    4 fc:  66519 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
        999 : snode ix:  66519 dh:11 sx:    0 pt:  66518 nc:    0 fc:     -1 ns:  66520 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       1000 : snode ix:  66520 dh:11 sx:    1 pt:  66518 nc:    0 fc:     -1 ns:  66521 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       1001 : snode ix:  66521 dh:11 sx:    2 pt:  66518 nc:    0 fc:     -1 ns:  66522 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       1002 : snode ix:  66522 dh:11 sx:    3 pt:  66518 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES
       1005 : snode ix:  66525 dh: 9 sx:    0 pt:  66524 nc:    1 fc:  66526 ns:     -1 lv: 48 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 54 sn:-1 elvid YES
       1006 : snode ix:  66526 dh:10 sx:    0 pt:  66525 nc:    4 fc:  66527 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       1007 : snode ix:  66527 dh:11 sx:    0 pt:  66526 nc:    0 fc:     -1 ns:  66528 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       1008 : snode ix:  66528 dh:11 sx:    1 pt:  66526 nc:    0 fc:     -1 ns:  66529 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       1009 : snode ix:  66529 dh:11 sx:    2 pt:  66526 nc:    0 fc:     -1 ns:  66530 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       1010 : snode ix:  66530 dh:11 sx:    3 pt:  66526 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES
       1013 : snode ix:  66533 dh: 9 sx:    0 pt:  66532 nc:    1 fc:  66534 ns:     -1 lv: 48 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 54 sn:-1 elvid YES
       1014 : snode ix:  66534 dh:10 sx:    0 pt:  66533 nc:    4 fc:  66535 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       1015 : snode ix:  66535 dh:11 sx:    0 pt:  66534 nc:    0 fc:     -1 ns:  66536 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       1016 : snode ix:  66536 dh:11 sx:    1 pt:  66534 nc:    0 fc:     -1 ns:  66537 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       1017 : snode ix:  66537 dh:11 sx:    2 pt:  66534 nc:    0 fc:     -1 ns:  66538 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       1018 : snode ix:  66538 dh:11 sx:    3 pt:  66534 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES
       1021 : snode ix:  66541 dh: 9 sx:    0 pt:  66540 nc:    1 fc:  66542 ns:     -1 lv: 48 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 54 sn:-1 elvid YES
       1022 : snode ix:  66542 dh:10 sx:    0 pt:  66541 nc:    4 fc:  66543 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       1023 : snode ix:  66543 dh:11 sx:    0 pt:  66542 nc:    0 fc:     -1 ns:  66544 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       1024 : snode ix:  66544 dh:11 sx:    1 pt:  66542 nc:    0 fc:     -1 ns:  66545 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       1025 : snode ix:  66545 dh:11 sx:    2 pt:  66542 nc:    0 fc:     -1 ns:  66546 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       1026 : snode ix:  66546 dh:11 sx:    3 pt:  66542 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2940 : snode ix: 382705 dh: 4 sx:    1 pt:  65723 nc:    1 fc: 382706 ns: 382711 lv: 48 cp:  52400 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2941 : snode ix: 382706 dh: 5 sx:    0 pt: 382705 nc:    4 fc: 382707 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2942 : snode ix: 382707 dh: 6 sx:    0 pt: 382706 nc:    0 fc:     -1 ns: 382708 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2943 : snode ix: 382708 dh: 6 sx:    1 pt: 382706 nc:    0 fc:     -1 ns: 382709 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2944 : snode ix: 382709 dh: 6 sx:    2 pt: 382706 nc:    0 fc:     -1 ns: 382710 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2945 : snode ix: 382710 dh: 6 sx:    3 pt: 382706 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2946 : snode ix: 382711 dh: 4 sx:    2 pt:  65723 nc:    1 fc: 382712 ns: 382717 lv: 48 cp:  52401 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2947 : snode ix: 382712 dh: 5 sx:    0 pt: 382711 nc:    4 fc: 382713 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2948 : snode ix: 382713 dh: 6 sx:    0 pt: 382712 nc:    0 fc:     -1 ns: 382714 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2949 : snode ix: 382714 dh: 6 sx:    1 pt: 382712 nc:    0 fc:     -1 ns: 382715 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2950 : snode ix: 382715 dh: 6 sx:    2 pt: 382712 nc:    0 fc:     -1 ns: 382716 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2951 : snode ix: 382716 dh: 6 sx:    3 pt: 382712 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2952 : snode ix: 382717 dh: 4 sx:    3 pt:  65723 nc:    1 fc: 382718 ns: 382723 lv: 48 cp:  52402 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2953 : snode ix: 382718 dh: 5 sx:    0 pt: 382717 nc:    4 fc: 382719 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2954 : snode ix: 382719 dh: 6 sx:    0 pt: 382718 nc:    0 fc:     -1 ns: 382720 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2955 : snode ix: 382720 dh: 6 sx:    1 pt: 382718 nc:    0 fc:     -1 ns: 382721 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2956 : snode ix: 382721 dh: 6 sx:    2 pt: 382718 nc:    0 fc:     -1 ns: 382722 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2957 : snode ix: 382722 dh: 6 sx:    3 pt: 382718 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2958 : snode ix: 382723 dh: 4 sx:    4 pt:  65723 nc:    1 fc: 382724 ns: 382729 lv: 48 cp:  52403 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2959 : snode ix: 382724 dh: 5 sx:    0 pt: 382723 nc:    4 fc: 382725 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2960 : snode ix: 382725 dh: 6 sx:    0 pt: 382724 nc:    0 fc:     -1 ns: 382726 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2961 : snode ix: 382726 dh: 6 sx:    1 pt: 382724 nc:    0 fc:     -1 ns: 382727 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2962 : snode ix: 382727 dh: 6 sx:    2 pt: 382724 nc:    0 fc:     -1 ns: 382728 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2963 : snode ix: 382728 dh: 6 sx:    3 pt: 382724 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2964 : snode ix: 382729 dh: 4 sx:    5 pt:  65723 nc:    1 fc: 382730 ns: 382735 lv: 48 cp:  52404 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2965 : snode ix: 382730 dh: 5 sx:    0 pt: 382729 nc:    4 fc: 382731 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2966 : snode ix: 382731 dh: 6 sx:    0 pt: 382730 nc:    0 fc:     -1 ns: 382732 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2967 : snode ix: 382732 dh: 6 sx:    1 pt: 382730 nc:    0 fc:     -1 ns: 382733 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2968 : snode ix: 382733 dh: 6 sx:    2 pt: 382730 nc:    0 fc:     -1 ns: 382734 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2969 : snode ix: 382734 dh: 6 sx:    3 pt: 382730 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2970 : snode ix: 382735 dh: 4 sx:    6 pt:  65723 nc:    1 fc: 382736 ns: 382741 lv: 48 cp:  52405 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2971 : snode ix: 382736 dh: 5 sx:    0 pt: 382735 nc:    4 fc: 382737 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2972 : snode ix: 382737 dh: 6 sx:    0 pt: 382736 nc:    0 fc:     -1 ns: 382738 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2973 : snode ix: 382738 dh: 6 sx:    1 pt: 382736 nc:    0 fc:     -1 ns: 382739 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2974 : snode ix: 382739 dh: 6 sx:    2 pt: 382736 nc:    0 fc:     -1 ns: 382740 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2975 : snode ix: 382740 dh: 6 sx:    3 pt: 382736 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES

       2976 : snode ix: 382741 dh: 4 sx:    7 pt:  65723 nc:    1 fc: 382742 ns: 382747 lv: 48 cp:  52406 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 elvid YES
       2977 : snode ix: 382742 dh: 5 sx:    0 pt: 382741 nc:    4 fc: 382743 ns:     -1 lv: 47 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 55 sn:-1 elvid YES
       2978 : snode ix: 382743 dh: 6 sx:    0 pt: 382742 nc:    0 fc:     -1 ns: 382744 lv: 43 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 56 sn:-1 elvid YES
       2979 : snode ix: 382744 dh: 6 sx:    1 pt: 382742 nc:    0 fc:     -1 ns: 382745 lv: 44 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 57 sn:-1 elvid YES
       2980 : snode ix: 382745 dh: 6 sx:    2 pt: 382742 nc:    0 fc:     -1 ns: 382746 lv: 45 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 58 sn:-1 elvid YES
       2981 : snode ix: 382746 dh: 6 sx:    3 pt: 382742 nc:    0 fc:     -1 ns:     -1 lv: 46 cp:      0 se:     -1 se:     -1 ri: 0 ro:   -1 bd: 59 sn:-1 elvid YES




dumping the 348 WP_ATM_LPMT using desc_node_EBOUNDARY EBOUNDARY=303
---------------------------------------------------------------------

jcv CopyNumber::

        014 enum PMTID_OFFSET_DETSIM {
         15   kOFFSET_CD_LPMT=0,     kOFFSET_CD_LPMT_END=17612, // CD-LPMT
         16   kOFFSET_CD_SPMT=20000, kOFFSET_CD_SPMT_END=45600,  // CD-SPMT
         17   kOFFSET_WP_PMT=50000,  kOFFSET_WP_PMT_END=52400,  // WP-LPMT
         18   kOFFSET_WP_ATM_LPMT=52400,  kOFFSET_WP_ATM_LPMT_END=52748, //WP-Atmosphere-LPMT
         19   kOFFSET_WP_ATM_MPMT=53000,  kOFFSET_WP_ATM_MPMT_END=53600,  //WP-Atmosphere-MPMT (Medium, 8 inch)
         20   kOFFSET_WP_WAL_PMT=54000,  kOFFSET_WP_WAL_PMT_END=54005 //WP-Water-attenuation-length
         21
         22 };


::

    (ok) A[blyth@localhost tests]$ TEST=desc_node_EBOUNDARY EBOUNDARY=303   ~/o/sysrap/tests/stree_load_test.sh
                       BASH_SOURCE : /home/blyth/o/sysrap/tests/stree_load_test.sh
                               opt : -DWITH_PLACEHOLDER -DWITH_CHILD
                              GEOM : J25_4_0_opticks_Debug
                               CFB : J25_4_0_opticks_Debug_CFBaseFromGEOM
                              FOLD : /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug/CSGFoundry/SSim/stree
                               MOI : PMT_20inch_mcp_solid:352:-2
                              TEST : desc_node_EBOUNDARY
    [stree::desc_node_elist
     etag EBOUNDARY
     tag BOUNDARY
     field_idx 13
     nam nds.npy
     elist YES
     382705 : snode ix: 382705 dh: 4 sx:    1 pt:  65723 nc:    1 fc: 382706 ns: 382711 lv: 48 cp:  52400 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382711 : snode ix: 382711 dh: 4 sx:    2 pt:  65723 nc:    1 fc: 382712 ns: 382717 lv: 48 cp:  52401 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382717 : snode ix: 382717 dh: 4 sx:    3 pt:  65723 nc:    1 fc: 382718 ns: 382723 lv: 48 cp:  52402 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382723 : snode ix: 382723 dh: 4 sx:    4 pt:  65723 nc:    1 fc: 382724 ns: 382729 lv: 48 cp:  52403 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382729 : snode ix: 382729 dh: 4 sx:    5 pt:  65723 nc:    1 fc: 382730 ns: 382735 lv: 48 cp:  52404 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382735 : snode ix: 382735 dh: 4 sx:    6 pt:  65723 nc:    1 fc: 382736 ns: 382741 lv: 48 cp:  52405 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382741 : snode ix: 382741 dh: 4 sx:    7 pt:  65723 nc:    1 fc: 382742 ns: 382747 lv: 48 cp:  52406 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382747 : snode ix: 382747 dh: 4 sx:    8 pt:  65723 nc:    1 fc: 382748 ns: 382753 lv: 48 cp:  52407 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382753 : snode ix: 382753 dh: 4 sx:    9 pt:  65723 nc:    1 fc: 382754 ns: 382759 lv: 48 cp:  52408 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382759 : snode ix: 382759 dh: 4 sx:   10 pt:  65723 nc:    1 fc: 382760 ns: 382765 lv: 48 cp:  52409 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382765 : snode ix: 382765 dh: 4 sx:   11 pt:  65723 nc:    1 fc: 382766 ns: 382771 lv: 48 cp:  52410 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382771 : snode ix: 382771 dh: 4 sx:   12 pt:  65723 nc:    1 fc: 382772 ns: 382777 lv: 48 cp:  52411 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382777 : snode ix: 382777 dh: 4 sx:   13 pt:  65723 nc:    1 fc: 382778 ns: 382783 lv: 48 cp:  52412 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382783 : snode ix: 382783 dh: 4 sx:   14 pt:  65723 nc:    1 fc: 382784 ns: 382789 lv: 48 cp:  52413 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382789 : snode ix: 382789 dh: 4 sx:   15 pt:  65723 nc:    1 fc: 382790 ns: 382795 lv: 48 cp:  52414 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382795 : snode ix: 382795 dh: 4 sx:   16 pt:  65723 nc:    1 fc: 382796 ns: 382801 lv: 48 cp:  52415 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382801 : snode ix: 382801 dh: 4 sx:   17 pt:  65723 nc:    1 fc: 382802 ns: 382807 lv: 48 cp:  52416 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382807 : snode ix: 382807 dh: 4 sx:   18 pt:  65723 nc:    1 fc: 382808 ns: 382813 lv: 48 cp:  52417 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382813 : snode ix: 382813 dh: 4 sx:   19 pt:  65723 nc:    1 fc: 382814 ns: 382819 lv: 48 cp:  52418 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382819 : snode ix: 382819 dh: 4 sx:   20 pt:  65723 nc:    1 fc: 382820 ns: 382825 lv: 48 cp:  52419 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382825 : snode ix: 382825 dh: 4 sx:   21 pt:  65723 nc:    1 fc: 382826 ns: 382831 lv: 48 cp:  52420 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382831 : snode ix: 382831 dh: 4 sx:   22 pt:  65723 nc:    1 fc: 382832 ns: 382837 lv: 48 cp:  52421 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382837 : snode ix: 382837 dh: 4 sx:   23 pt:  65723 nc:    1 fc: 382838 ns: 382843 lv: 48 cp:  52422 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382843 : snode ix: 382843 dh: 4 sx:   24 pt:  65723 nc:    1 fc: 382844 ns: 382849 lv: 48 cp:  52423 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382849 : snode ix: 382849 dh: 4 sx:   25 pt:  65723 nc:    1 fc: 382850 ns: 382855 lv: 48 cp:  52424 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382855 : snode ix: 382855 dh: 4 sx:   26 pt:  65723 nc:    1 fc: 382856 ns: 382861 lv: 48 cp:  52425 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382861 : snode ix: 382861 dh: 4 sx:   27 pt:  65723 nc:    1 fc: 382862 ns: 382867 lv: 48 cp:  52426 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382867 : snode ix: 382867 dh: 4 sx:   28 pt:  65723 nc:    1 fc: 382868 ns: 382873 lv: 48 cp:  52427 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382873 : snode ix: 382873 dh: 4 sx:   29 pt:  65723 nc:    1 fc: 382874 ns: 382879 lv: 48 cp:  52428 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382879 : snode ix: 382879 dh: 4 sx:   30 pt:  65723 nc:    1 fc: 382880 ns: 382885 lv: 48 cp:  52429 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382885 : snode ix: 382885 dh: 4 sx:   31 pt:  65723 nc:    1 fc: 382886 ns: 382891 lv: 48 cp:  52430 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382891 : snode ix: 382891 dh: 4 sx:   32 pt:  65723 nc:    1 fc: 382892 ns: 382897 lv: 48 cp:  52431 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382897 : snode ix: 382897 dh: 4 sx:   33 pt:  65723 nc:    1 fc: 382898 ns: 382903 lv: 48 cp:  52432 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382903 : snode ix: 382903 dh: 4 sx:   34 pt:  65723 nc:    1 fc: 382904 ns: 382909 lv: 48 cp:  52433 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382909 : snode ix: 382909 dh: 4 sx:   35 pt:  65723 nc:    1 fc: 382910 ns: 382915 lv: 48 cp:  52434 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382915 : snode ix: 382915 dh: 4 sx:   36 pt:  65723 nc:    1 fc: 382916 ns: 382921 lv: 48 cp:  52435 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382921 : snode ix: 382921 dh: 4 sx:   37 pt:  65723 nc:    1 fc: 382922 ns: 382927 lv: 48 cp:  52436 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382927 : snode ix: 382927 dh: 4 sx:   38 pt:  65723 nc:    1 fc: 382928 ns: 382933 lv: 48 cp:  52437 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382933 : snode ix: 382933 dh: 4 sx:   39 pt:  65723 nc:    1 fc: 382934 ns: 382939 lv: 48 cp:  52438 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382939 : snode ix: 382939 dh: 4 sx:   40 pt:  65723 nc:    1 fc: 382940 ns: 382945 lv: 48 cp:  52439 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382945 : snode ix: 382945 dh: 4 sx:   41 pt:  65723 nc:    1 fc: 382946 ns: 382951 lv: 48 cp:  52440 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382951 : snode ix: 382951 dh: 4 sx:   42 pt:  65723 nc:    1 fc: 382952 ns: 382957 lv: 48 cp:  52441 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382957 : snode ix: 382957 dh: 4 sx:   43 pt:  65723 nc:    1 fc: 382958 ns: 382963 lv: 48 cp:  52442 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382963 : snode ix: 382963 dh: 4 sx:   44 pt:  65723 nc:    1 fc: 382964 ns: 382969 lv: 48 cp:  52443 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382969 : snode ix: 382969 dh: 4 sx:   45 pt:  65723 nc:    1 fc: 382970 ns: 382975 lv: 48 cp:  52444 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382975 : snode ix: 382975 dh: 4 sx:   46 pt:  65723 nc:    1 fc: 382976 ns: 382981 lv: 48 cp:  52445 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382981 : snode ix: 382981 dh: 4 sx:   47 pt:  65723 nc:    1 fc: 382982 ns: 382987 lv: 48 cp:  52446 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382987 : snode ix: 382987 dh: 4 sx:   48 pt:  65723 nc:    1 fc: 382988 ns: 382993 lv: 48 cp:  52447 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382993 : snode ix: 382993 dh: 4 sx:   49 pt:  65723 nc:    1 fc: 382994 ns: 382999 lv: 48 cp:  52448 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     382999 : snode ix: 382999 dh: 4 sx:   50 pt:  65723 nc:    1 fc: 383000 ns: 383005 lv: 48 cp:  52449 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383005 : snode ix: 383005 dh: 4 sx:   51 pt:  65723 nc:    1 fc: 383006 ns: 383011 lv: 48 cp:  52450 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383011 : snode ix: 383011 dh: 4 sx:   52 pt:  65723 nc:    1 fc: 383012 ns: 383017 lv: 48 cp:  52451 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383017 : snode ix: 383017 dh: 4 sx:   53 pt:  65723 nc:    1 fc: 383018 ns: 383023 lv: 48 cp:  52452 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383023 : snode ix: 383023 dh: 4 sx:   54 pt:  65723 nc:    1 fc: 383024 ns: 383029 lv: 48 cp:  52453 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383029 : snode ix: 383029 dh: 4 sx:   55 pt:  65723 nc:    1 fc: 383030 ns: 383035 lv: 48 cp:  52454 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383035 : snode ix: 383035 dh: 4 sx:   56 pt:  65723 nc:    1 fc: 383036 ns: 383041 lv: 48 cp:  52455 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383041 : snode ix: 383041 dh: 4 sx:   57 pt:  65723 nc:    1 fc: 383042 ns: 383047 lv: 48 cp:  52456 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383047 : snode ix: 383047 dh: 4 sx:   58 pt:  65723 nc:    1 fc: 383048 ns: 383053 lv: 48 cp:  52457 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383053 : snode ix: 383053 dh: 4 sx:   59 pt:  65723 nc:    1 fc: 383054 ns: 383059 lv: 48 cp:  52458 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383059 : snode ix: 383059 dh: 4 sx:   60 pt:  65723 nc:    1 fc: 383060 ns: 383065 lv: 48 cp:  52459 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383065 : snode ix: 383065 dh: 4 sx:   61 pt:  65723 nc:    1 fc: 383066 ns: 383071 lv: 48 cp:  52460 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383071 : snode ix: 383071 dh: 4 sx:   62 pt:  65723 nc:    1 fc: 383072 ns: 383077 lv: 48 cp:  52461 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383077 : snode ix: 383077 dh: 4 sx:   63 pt:  65723 nc:    1 fc: 383078 ns: 383083 lv: 48 cp:  52462 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383083 : snode ix: 383083 dh: 4 sx:   64 pt:  65723 nc:    1 fc: 383084 ns: 383089 lv: 48 cp:  52463 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383089 : snode ix: 383089 dh: 4 sx:   65 pt:  65723 nc:    1 fc: 383090 ns: 383095 lv: 48 cp:  52464 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383095 : snode ix: 383095 dh: 4 sx:   66 pt:  65723 nc:    1 fc: 383096 ns: 383101 lv: 48 cp:  52465 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383101 : snode ix: 383101 dh: 4 sx:   67 pt:  65723 nc:    1 fc: 383102 ns: 383107 lv: 48 cp:  52466 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383107 : snode ix: 383107 dh: 4 sx:   68 pt:  65723 nc:    1 fc: 383108 ns: 383113 lv: 48 cp:  52467 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383113 : snode ix: 383113 dh: 4 sx:   69 pt:  65723 nc:    1 fc: 383114 ns: 383119 lv: 48 cp:  52468 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383119 : snode ix: 383119 dh: 4 sx:   70 pt:  65723 nc:    1 fc: 383120 ns: 383125 lv: 48 cp:  52469 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383125 : snode ix: 383125 dh: 4 sx:   71 pt:  65723 nc:    1 fc: 383126 ns: 383131 lv: 48 cp:  52470 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383131 : snode ix: 383131 dh: 4 sx:   72 pt:  65723 nc:    1 fc: 383132 ns: 383137 lv: 48 cp:  52471 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383137 : snode ix: 383137 dh: 4 sx:   73 pt:  65723 nc:    1 fc: 383138 ns: 383143 lv: 48 cp:  52472 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383143 : snode ix: 383143 dh: 4 sx:   74 pt:  65723 nc:    1 fc: 383144 ns: 383149 lv: 48 cp:  52473 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383149 : snode ix: 383149 dh: 4 sx:   75 pt:  65723 nc:    1 fc: 383150 ns: 383155 lv: 48 cp:  52474 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383155 : snode ix: 383155 dh: 4 sx:   76 pt:  65723 nc:    1 fc: 383156 ns: 383161 lv: 48 cp:  52475 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383161 : snode ix: 383161 dh: 4 sx:   77 pt:  65723 nc:    1 fc: 383162 ns: 383167 lv: 48 cp:  52476 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383167 : snode ix: 383167 dh: 4 sx:   78 pt:  65723 nc:    1 fc: 383168 ns: 383173 lv: 48 cp:  52477 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383173 : snode ix: 383173 dh: 4 sx:   79 pt:  65723 nc:    1 fc: 383174 ns: 383179 lv: 48 cp:  52478 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383179 : snode ix: 383179 dh: 4 sx:   80 pt:  65723 nc:    1 fc: 383180 ns: 383185 lv: 48 cp:  52479 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383185 : snode ix: 383185 dh: 4 sx:   81 pt:  65723 nc:    1 fc: 383186 ns: 383191 lv: 48 cp:  52480 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383191 : snode ix: 383191 dh: 4 sx:   82 pt:  65723 nc:    1 fc: 383192 ns: 383197 lv: 48 cp:  52481 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383197 : snode ix: 383197 dh: 4 sx:   83 pt:  65723 nc:    1 fc: 383198 ns: 383203 lv: 48 cp:  52482 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383203 : snode ix: 383203 dh: 4 sx:   84 pt:  65723 nc:    1 fc: 383204 ns: 383209 lv: 48 cp:  52483 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383209 : snode ix: 383209 dh: 4 sx:   85 pt:  65723 nc:    1 fc: 383210 ns: 383215 lv: 48 cp:  52484 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383215 : snode ix: 383215 dh: 4 sx:   86 pt:  65723 nc:    1 fc: 383216 ns: 383221 lv: 48 cp:  52485 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383221 : snode ix: 383221 dh: 4 sx:   87 pt:  65723 nc:    1 fc: 383222 ns: 383227 lv: 48 cp:  52486 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383227 : snode ix: 383227 dh: 4 sx:   88 pt:  65723 nc:    1 fc: 383228 ns: 383233 lv: 48 cp:  52487 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383233 : snode ix: 383233 dh: 4 sx:   89 pt:  65723 nc:    1 fc: 383234 ns: 383239 lv: 48 cp:  52488 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383239 : snode ix: 383239 dh: 4 sx:   90 pt:  65723 nc:    1 fc: 383240 ns: 383245 lv: 48 cp:  52489 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383245 : snode ix: 383245 dh: 4 sx:   91 pt:  65723 nc:    1 fc: 383246 ns: 383251 lv: 48 cp:  52490 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383251 : snode ix: 383251 dh: 4 sx:   92 pt:  65723 nc:    1 fc: 383252 ns: 383257 lv: 48 cp:  52491 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383257 : snode ix: 383257 dh: 4 sx:   93 pt:  65723 nc:    1 fc: 383258 ns: 383263 lv: 48 cp:  52492 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383263 : snode ix: 383263 dh: 4 sx:   94 pt:  65723 nc:    1 fc: 383264 ns: 383269 lv: 48 cp:  52493 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383269 : snode ix: 383269 dh: 4 sx:   95 pt:  65723 nc:    1 fc: 383270 ns: 383275 lv: 48 cp:  52494 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383275 : snode ix: 383275 dh: 4 sx:   96 pt:  65723 nc:    1 fc: 383276 ns: 383281 lv: 48 cp:  52495 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383281 : snode ix: 383281 dh: 4 sx:   97 pt:  65723 nc:    1 fc: 383282 ns: 383287 lv: 48 cp:  52496 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383287 : snode ix: 383287 dh: 4 sx:   98 pt:  65723 nc:    1 fc: 383288 ns: 383293 lv: 48 cp:  52497 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383293 : snode ix: 383293 dh: 4 sx:   99 pt:  65723 nc:    1 fc: 383294 ns: 383299 lv: 48 cp:  52498 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383299 : snode ix: 383299 dh: 4 sx:  100 pt:  65723 nc:    1 fc: 383300 ns: 383305 lv: 48 cp:  52499 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383305 : snode ix: 383305 dh: 4 sx:  101 pt:  65723 nc:    1 fc: 383306 ns: 383311 lv: 48 cp:  52500 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383311 : snode ix: 383311 dh: 4 sx:  102 pt:  65723 nc:    1 fc: 383312 ns: 383317 lv: 48 cp:  52501 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383317 : snode ix: 383317 dh: 4 sx:  103 pt:  65723 nc:    1 fc: 383318 ns: 383323 lv: 48 cp:  52502 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383323 : snode ix: 383323 dh: 4 sx:  104 pt:  65723 nc:    1 fc: 383324 ns: 383329 lv: 48 cp:  52503 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383329 : snode ix: 383329 dh: 4 sx:  105 pt:  65723 nc:    1 fc: 383330 ns: 383335 lv: 48 cp:  52504 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383335 : snode ix: 383335 dh: 4 sx:  106 pt:  65723 nc:    1 fc: 383336 ns: 383341 lv: 48 cp:  52505 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383341 : snode ix: 383341 dh: 4 sx:  107 pt:  65723 nc:    1 fc: 383342 ns: 383347 lv: 48 cp:  52506 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383347 : snode ix: 383347 dh: 4 sx:  108 pt:  65723 nc:    1 fc: 383348 ns: 383353 lv: 48 cp:  52507 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383353 : snode ix: 383353 dh: 4 sx:  109 pt:  65723 nc:    1 fc: 383354 ns: 383359 lv: 48 cp:  52508 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383359 : snode ix: 383359 dh: 4 sx:  110 pt:  65723 nc:    1 fc: 383360 ns: 383365 lv: 48 cp:  52509 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383365 : snode ix: 383365 dh: 4 sx:  111 pt:  65723 nc:    1 fc: 383366 ns: 383371 lv: 48 cp:  52510 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383371 : snode ix: 383371 dh: 4 sx:  112 pt:  65723 nc:    1 fc: 383372 ns: 383377 lv: 48 cp:  52511 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383377 : snode ix: 383377 dh: 4 sx:  113 pt:  65723 nc:    1 fc: 383378 ns: 383383 lv: 48 cp:  52512 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383383 : snode ix: 383383 dh: 4 sx:  114 pt:  65723 nc:    1 fc: 383384 ns: 383389 lv: 48 cp:  52513 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383389 : snode ix: 383389 dh: 4 sx:  115 pt:  65723 nc:    1 fc: 383390 ns: 383395 lv: 48 cp:  52514 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383395 : snode ix: 383395 dh: 4 sx:  116 pt:  65723 nc:    1 fc: 383396 ns: 383401 lv: 48 cp:  52515 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383401 : snode ix: 383401 dh: 4 sx:  117 pt:  65723 nc:    1 fc: 383402 ns: 383407 lv: 48 cp:  52516 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383407 : snode ix: 383407 dh: 4 sx:  118 pt:  65723 nc:    1 fc: 383408 ns: 383413 lv: 48 cp:  52517 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383413 : snode ix: 383413 dh: 4 sx:  119 pt:  65723 nc:    1 fc: 383414 ns: 383419 lv: 48 cp:  52518 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383419 : snode ix: 383419 dh: 4 sx:  120 pt:  65723 nc:    1 fc: 383420 ns: 383425 lv: 48 cp:  52519 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383425 : snode ix: 383425 dh: 4 sx:  121 pt:  65723 nc:    1 fc: 383426 ns: 383431 lv: 48 cp:  52520 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383431 : snode ix: 383431 dh: 4 sx:  122 pt:  65723 nc:    1 fc: 383432 ns: 383437 lv: 48 cp:  52521 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383437 : snode ix: 383437 dh: 4 sx:  123 pt:  65723 nc:    1 fc: 383438 ns: 383443 lv: 48 cp:  52522 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383443 : snode ix: 383443 dh: 4 sx:  124 pt:  65723 nc:    1 fc: 383444 ns: 383449 lv: 48 cp:  52523 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383449 : snode ix: 383449 dh: 4 sx:  125 pt:  65723 nc:    1 fc: 383450 ns: 383455 lv: 48 cp:  52524 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383455 : snode ix: 383455 dh: 4 sx:  126 pt:  65723 nc:    1 fc: 383456 ns: 383461 lv: 48 cp:  52525 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383461 : snode ix: 383461 dh: 4 sx:  127 pt:  65723 nc:    1 fc: 383462 ns: 383467 lv: 48 cp:  52526 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383467 : snode ix: 383467 dh: 4 sx:  128 pt:  65723 nc:    1 fc: 383468 ns: 383473 lv: 48 cp:  52527 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383473 : snode ix: 383473 dh: 4 sx:  129 pt:  65723 nc:    1 fc: 383474 ns: 383479 lv: 48 cp:  52528 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383479 : snode ix: 383479 dh: 4 sx:  130 pt:  65723 nc:    1 fc: 383480 ns: 383485 lv: 48 cp:  52529 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383485 : snode ix: 383485 dh: 4 sx:  131 pt:  65723 nc:    1 fc: 383486 ns: 383491 lv: 48 cp:  52530 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383491 : snode ix: 383491 dh: 4 sx:  132 pt:  65723 nc:    1 fc: 383492 ns: 383497 lv: 48 cp:  52531 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383497 : snode ix: 383497 dh: 4 sx:  133 pt:  65723 nc:    1 fc: 383498 ns: 383503 lv: 48 cp:  52532 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383503 : snode ix: 383503 dh: 4 sx:  134 pt:  65723 nc:    1 fc: 383504 ns: 383509 lv: 48 cp:  52533 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383509 : snode ix: 383509 dh: 4 sx:  135 pt:  65723 nc:    1 fc: 383510 ns: 383515 lv: 48 cp:  52534 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383515 : snode ix: 383515 dh: 4 sx:  136 pt:  65723 nc:    1 fc: 383516 ns: 383521 lv: 48 cp:  52535 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383521 : snode ix: 383521 dh: 4 sx:  137 pt:  65723 nc:    1 fc: 383522 ns: 383527 lv: 48 cp:  52536 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383527 : snode ix: 383527 dh: 4 sx:  138 pt:  65723 nc:    1 fc: 383528 ns: 383533 lv: 48 cp:  52537 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383533 : snode ix: 383533 dh: 4 sx:  139 pt:  65723 nc:    1 fc: 383534 ns: 383539 lv: 48 cp:  52538 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383539 : snode ix: 383539 dh: 4 sx:  140 pt:  65723 nc:    1 fc: 383540 ns: 383545 lv: 48 cp:  52539 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383545 : snode ix: 383545 dh: 4 sx:  141 pt:  65723 nc:    1 fc: 383546 ns: 383551 lv: 48 cp:  52540 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383551 : snode ix: 383551 dh: 4 sx:  142 pt:  65723 nc:    1 fc: 383552 ns: 383557 lv: 48 cp:  52541 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383557 : snode ix: 383557 dh: 4 sx:  143 pt:  65723 nc:    1 fc: 383558 ns: 383563 lv: 48 cp:  52542 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383563 : snode ix: 383563 dh: 4 sx:  144 pt:  65723 nc:    1 fc: 383564 ns: 383569 lv: 48 cp:  52543 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383569 : snode ix: 383569 dh: 4 sx:  145 pt:  65723 nc:    1 fc: 383570 ns: 383575 lv: 48 cp:  52544 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383575 : snode ix: 383575 dh: 4 sx:  146 pt:  65723 nc:    1 fc: 383576 ns: 383581 lv: 48 cp:  52545 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383581 : snode ix: 383581 dh: 4 sx:  147 pt:  65723 nc:    1 fc: 383582 ns: 383587 lv: 48 cp:  52546 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383587 : snode ix: 383587 dh: 4 sx:  148 pt:  65723 nc:    1 fc: 383588 ns: 383593 lv: 48 cp:  52547 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383593 : snode ix: 383593 dh: 4 sx:  149 pt:  65723 nc:    1 fc: 383594 ns: 383599 lv: 48 cp:  52548 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383599 : snode ix: 383599 dh: 4 sx:  150 pt:  65723 nc:    1 fc: 383600 ns: 383605 lv: 48 cp:  52549 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383605 : snode ix: 383605 dh: 4 sx:  151 pt:  65723 nc:    1 fc: 383606 ns: 383611 lv: 48 cp:  52550 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383611 : snode ix: 383611 dh: 4 sx:  152 pt:  65723 nc:    1 fc: 383612 ns: 383617 lv: 48 cp:  52551 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383617 : snode ix: 383617 dh: 4 sx:  153 pt:  65723 nc:    1 fc: 383618 ns: 383623 lv: 48 cp:  52552 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383623 : snode ix: 383623 dh: 4 sx:  154 pt:  65723 nc:    1 fc: 383624 ns: 383629 lv: 48 cp:  52553 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383629 : snode ix: 383629 dh: 4 sx:  155 pt:  65723 nc:    1 fc: 383630 ns: 383635 lv: 48 cp:  52554 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383635 : snode ix: 383635 dh: 4 sx:  156 pt:  65723 nc:    1 fc: 383636 ns: 383641 lv: 48 cp:  52555 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383641 : snode ix: 383641 dh: 4 sx:  157 pt:  65723 nc:    1 fc: 383642 ns: 383647 lv: 48 cp:  52556 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383647 : snode ix: 383647 dh: 4 sx:  158 pt:  65723 nc:    1 fc: 383648 ns: 383653 lv: 48 cp:  52557 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383653 : snode ix: 383653 dh: 4 sx:  159 pt:  65723 nc:    1 fc: 383654 ns: 383659 lv: 48 cp:  52558 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383659 : snode ix: 383659 dh: 4 sx:  160 pt:  65723 nc:    1 fc: 383660 ns: 383665 lv: 48 cp:  52559 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383665 : snode ix: 383665 dh: 4 sx:  161 pt:  65723 nc:    1 fc: 383666 ns: 383671 lv: 48 cp:  52560 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383671 : snode ix: 383671 dh: 4 sx:  162 pt:  65723 nc:    1 fc: 383672 ns: 383677 lv: 48 cp:  52561 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383677 : snode ix: 383677 dh: 4 sx:  163 pt:  65723 nc:    1 fc: 383678 ns: 383683 lv: 48 cp:  52562 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383683 : snode ix: 383683 dh: 4 sx:  164 pt:  65723 nc:    1 fc: 383684 ns: 383689 lv: 48 cp:  52563 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383689 : snode ix: 383689 dh: 4 sx:  165 pt:  65723 nc:    1 fc: 383690 ns: 383695 lv: 48 cp:  52564 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383695 : snode ix: 383695 dh: 4 sx:  166 pt:  65723 nc:    1 fc: 383696 ns: 383701 lv: 48 cp:  52565 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383701 : snode ix: 383701 dh: 4 sx:  167 pt:  65723 nc:    1 fc: 383702 ns: 383707 lv: 48 cp:  52566 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383707 : snode ix: 383707 dh: 4 sx:  168 pt:  65723 nc:    1 fc: 383708 ns: 383713 lv: 48 cp:  52567 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383713 : snode ix: 383713 dh: 4 sx:  169 pt:  65723 nc:    1 fc: 383714 ns: 383719 lv: 48 cp:  52568 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383719 : snode ix: 383719 dh: 4 sx:  170 pt:  65723 nc:    1 fc: 383720 ns: 383725 lv: 48 cp:  52569 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383725 : snode ix: 383725 dh: 4 sx:  171 pt:  65723 nc:    1 fc: 383726 ns: 383731 lv: 48 cp:  52570 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383731 : snode ix: 383731 dh: 4 sx:  172 pt:  65723 nc:    1 fc: 383732 ns: 383737 lv: 48 cp:  52571 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383737 : snode ix: 383737 dh: 4 sx:  173 pt:  65723 nc:    1 fc: 383738 ns: 383743 lv: 48 cp:  52572 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383743 : snode ix: 383743 dh: 4 sx:  174 pt:  65723 nc:    1 fc: 383744 ns: 383749 lv: 48 cp:  52573 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383749 : snode ix: 383749 dh: 4 sx:  175 pt:  65723 nc:    1 fc: 383750 ns: 383755 lv: 48 cp:  52574 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383755 : snode ix: 383755 dh: 4 sx:  176 pt:  65723 nc:    1 fc: 383756 ns: 383761 lv: 48 cp:  52575 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383761 : snode ix: 383761 dh: 4 sx:  177 pt:  65723 nc:    1 fc: 383762 ns: 383767 lv: 48 cp:  52576 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383767 : snode ix: 383767 dh: 4 sx:  178 pt:  65723 nc:    1 fc: 383768 ns: 383773 lv: 48 cp:  52577 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383773 : snode ix: 383773 dh: 4 sx:  179 pt:  65723 nc:    1 fc: 383774 ns: 383779 lv: 48 cp:  52578 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383779 : snode ix: 383779 dh: 4 sx:  180 pt:  65723 nc:    1 fc: 383780 ns: 383785 lv: 48 cp:  52579 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383785 : snode ix: 383785 dh: 4 sx:  181 pt:  65723 nc:    1 fc: 383786 ns: 383791 lv: 48 cp:  52580 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383791 : snode ix: 383791 dh: 4 sx:  182 pt:  65723 nc:    1 fc: 383792 ns: 383797 lv: 48 cp:  52581 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383797 : snode ix: 383797 dh: 4 sx:  183 pt:  65723 nc:    1 fc: 383798 ns: 383803 lv: 48 cp:  52582 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383803 : snode ix: 383803 dh: 4 sx:  184 pt:  65723 nc:    1 fc: 383804 ns: 383809 lv: 48 cp:  52583 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383809 : snode ix: 383809 dh: 4 sx:  185 pt:  65723 nc:    1 fc: 383810 ns: 383815 lv: 48 cp:  52584 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383815 : snode ix: 383815 dh: 4 sx:  186 pt:  65723 nc:    1 fc: 383816 ns: 383821 lv: 48 cp:  52585 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383821 : snode ix: 383821 dh: 4 sx:  187 pt:  65723 nc:    1 fc: 383822 ns: 383827 lv: 48 cp:  52586 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383827 : snode ix: 383827 dh: 4 sx:  188 pt:  65723 nc:    1 fc: 383828 ns: 383833 lv: 48 cp:  52587 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383833 : snode ix: 383833 dh: 4 sx:  189 pt:  65723 nc:    1 fc: 383834 ns: 383839 lv: 48 cp:  52588 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383839 : snode ix: 383839 dh: 4 sx:  190 pt:  65723 nc:    1 fc: 383840 ns: 383845 lv: 48 cp:  52589 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383845 : snode ix: 383845 dh: 4 sx:  191 pt:  65723 nc:    1 fc: 383846 ns: 383851 lv: 48 cp:  52590 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383851 : snode ix: 383851 dh: 4 sx:  192 pt:  65723 nc:    1 fc: 383852 ns: 383857 lv: 48 cp:  52591 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383857 : snode ix: 383857 dh: 4 sx:  193 pt:  65723 nc:    1 fc: 383858 ns: 383863 lv: 48 cp:  52592 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383863 : snode ix: 383863 dh: 4 sx:  194 pt:  65723 nc:    1 fc: 383864 ns: 383869 lv: 48 cp:  52593 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383869 : snode ix: 383869 dh: 4 sx:  195 pt:  65723 nc:    1 fc: 383870 ns: 383875 lv: 48 cp:  52594 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383875 : snode ix: 383875 dh: 4 sx:  196 pt:  65723 nc:    1 fc: 383876 ns: 383881 lv: 48 cp:  52595 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383881 : snode ix: 383881 dh: 4 sx:  197 pt:  65723 nc:    1 fc: 383882 ns: 383887 lv: 48 cp:  52596 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383887 : snode ix: 383887 dh: 4 sx:  198 pt:  65723 nc:    1 fc: 383888 ns: 383893 lv: 48 cp:  52597 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383893 : snode ix: 383893 dh: 4 sx:  199 pt:  65723 nc:    1 fc: 383894 ns: 383899 lv: 48 cp:  52598 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383899 : snode ix: 383899 dh: 4 sx:  200 pt:  65723 nc:    1 fc: 383900 ns: 383905 lv: 48 cp:  52599 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383905 : snode ix: 383905 dh: 4 sx:  201 pt:  65723 nc:    1 fc: 383906 ns: 383911 lv: 48 cp:  52600 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383911 : snode ix: 383911 dh: 4 sx:  202 pt:  65723 nc:    1 fc: 383912 ns: 383917 lv: 48 cp:  52601 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383917 : snode ix: 383917 dh: 4 sx:  203 pt:  65723 nc:    1 fc: 383918 ns: 383923 lv: 48 cp:  52602 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383923 : snode ix: 383923 dh: 4 sx:  204 pt:  65723 nc:    1 fc: 383924 ns: 383929 lv: 48 cp:  52603 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383929 : snode ix: 383929 dh: 4 sx:  205 pt:  65723 nc:    1 fc: 383930 ns: 383935 lv: 48 cp:  52604 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383935 : snode ix: 383935 dh: 4 sx:  206 pt:  65723 nc:    1 fc: 383936 ns: 383941 lv: 48 cp:  52605 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383941 : snode ix: 383941 dh: 4 sx:  207 pt:  65723 nc:    1 fc: 383942 ns: 383947 lv: 48 cp:  52606 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383947 : snode ix: 383947 dh: 4 sx:  208 pt:  65723 nc:    1 fc: 383948 ns: 383953 lv: 48 cp:  52607 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383953 : snode ix: 383953 dh: 4 sx:  209 pt:  65723 nc:    1 fc: 383954 ns: 383959 lv: 48 cp:  52608 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383959 : snode ix: 383959 dh: 4 sx:  210 pt:  65723 nc:    1 fc: 383960 ns: 383965 lv: 48 cp:  52609 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383965 : snode ix: 383965 dh: 4 sx:  211 pt:  65723 nc:    1 fc: 383966 ns: 383971 lv: 48 cp:  52610 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383971 : snode ix: 383971 dh: 4 sx:  212 pt:  65723 nc:    1 fc: 383972 ns: 383977 lv: 48 cp:  52611 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383977 : snode ix: 383977 dh: 4 sx:  213 pt:  65723 nc:    1 fc: 383978 ns: 383983 lv: 48 cp:  52612 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383983 : snode ix: 383983 dh: 4 sx:  214 pt:  65723 nc:    1 fc: 383984 ns: 383989 lv: 48 cp:  52613 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383989 : snode ix: 383989 dh: 4 sx:  215 pt:  65723 nc:    1 fc: 383990 ns: 383995 lv: 48 cp:  52614 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     383995 : snode ix: 383995 dh: 4 sx:  216 pt:  65723 nc:    1 fc: 383996 ns: 384001 lv: 48 cp:  52615 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384001 : snode ix: 384001 dh: 4 sx:  217 pt:  65723 nc:    1 fc: 384002 ns: 384007 lv: 48 cp:  52616 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384007 : snode ix: 384007 dh: 4 sx:  218 pt:  65723 nc:    1 fc: 384008 ns: 384013 lv: 48 cp:  52617 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384013 : snode ix: 384013 dh: 4 sx:  219 pt:  65723 nc:    1 fc: 384014 ns: 384019 lv: 48 cp:  52618 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384019 : snode ix: 384019 dh: 4 sx:  220 pt:  65723 nc:    1 fc: 384020 ns: 384025 lv: 48 cp:  52619 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384025 : snode ix: 384025 dh: 4 sx:  221 pt:  65723 nc:    1 fc: 384026 ns: 384031 lv: 48 cp:  52620 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384031 : snode ix: 384031 dh: 4 sx:  222 pt:  65723 nc:    1 fc: 384032 ns: 384037 lv: 48 cp:  52621 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384037 : snode ix: 384037 dh: 4 sx:  223 pt:  65723 nc:    1 fc: 384038 ns: 384043 lv: 48 cp:  52622 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384043 : snode ix: 384043 dh: 4 sx:  224 pt:  65723 nc:    1 fc: 384044 ns: 384049 lv: 48 cp:  52623 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384049 : snode ix: 384049 dh: 4 sx:  225 pt:  65723 nc:    1 fc: 384050 ns: 384055 lv: 48 cp:  52624 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384055 : snode ix: 384055 dh: 4 sx:  226 pt:  65723 nc:    1 fc: 384056 ns: 384061 lv: 48 cp:  52625 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384061 : snode ix: 384061 dh: 4 sx:  227 pt:  65723 nc:    1 fc: 384062 ns: 384067 lv: 48 cp:  52626 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384067 : snode ix: 384067 dh: 4 sx:  228 pt:  65723 nc:    1 fc: 384068 ns: 384073 lv: 48 cp:  52627 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384073 : snode ix: 384073 dh: 4 sx:  229 pt:  65723 nc:    1 fc: 384074 ns: 384079 lv: 48 cp:  52628 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384079 : snode ix: 384079 dh: 4 sx:  230 pt:  65723 nc:    1 fc: 384080 ns: 384085 lv: 48 cp:  52629 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384085 : snode ix: 384085 dh: 4 sx:  231 pt:  65723 nc:    1 fc: 384086 ns: 384091 lv: 48 cp:  52630 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384091 : snode ix: 384091 dh: 4 sx:  232 pt:  65723 nc:    1 fc: 384092 ns: 384097 lv: 48 cp:  52631 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384097 : snode ix: 384097 dh: 4 sx:  233 pt:  65723 nc:    1 fc: 384098 ns: 384103 lv: 48 cp:  52632 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384103 : snode ix: 384103 dh: 4 sx:  234 pt:  65723 nc:    1 fc: 384104 ns: 384109 lv: 48 cp:  52633 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384109 : snode ix: 384109 dh: 4 sx:  235 pt:  65723 nc:    1 fc: 384110 ns: 384115 lv: 48 cp:  52634 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384115 : snode ix: 384115 dh: 4 sx:  236 pt:  65723 nc:    1 fc: 384116 ns: 384121 lv: 48 cp:  52635 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384121 : snode ix: 384121 dh: 4 sx:  237 pt:  65723 nc:    1 fc: 384122 ns: 384127 lv: 48 cp:  52636 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384127 : snode ix: 384127 dh: 4 sx:  238 pt:  65723 nc:    1 fc: 384128 ns: 384133 lv: 48 cp:  52637 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384133 : snode ix: 384133 dh: 4 sx:  239 pt:  65723 nc:    1 fc: 384134 ns: 384139 lv: 48 cp:  52638 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384139 : snode ix: 384139 dh: 4 sx:  240 pt:  65723 nc:    1 fc: 384140 ns: 384145 lv: 48 cp:  52639 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384145 : snode ix: 384145 dh: 4 sx:  241 pt:  65723 nc:    1 fc: 384146 ns: 384151 lv: 48 cp:  52640 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384151 : snode ix: 384151 dh: 4 sx:  242 pt:  65723 nc:    1 fc: 384152 ns: 384157 lv: 48 cp:  52641 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384157 : snode ix: 384157 dh: 4 sx:  243 pt:  65723 nc:    1 fc: 384158 ns: 384163 lv: 48 cp:  52642 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384163 : snode ix: 384163 dh: 4 sx:  244 pt:  65723 nc:    1 fc: 384164 ns: 384169 lv: 48 cp:  52643 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384169 : snode ix: 384169 dh: 4 sx:  245 pt:  65723 nc:    1 fc: 384170 ns: 384175 lv: 48 cp:  52644 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384175 : snode ix: 384175 dh: 4 sx:  246 pt:  65723 nc:    1 fc: 384176 ns: 384181 lv: 48 cp:  52645 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384181 : snode ix: 384181 dh: 4 sx:  247 pt:  65723 nc:    1 fc: 384182 ns: 384187 lv: 48 cp:  52646 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384187 : snode ix: 384187 dh: 4 sx:  248 pt:  65723 nc:    1 fc: 384188 ns: 384193 lv: 48 cp:  52647 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384193 : snode ix: 384193 dh: 4 sx:  249 pt:  65723 nc:    1 fc: 384194 ns: 384199 lv: 48 cp:  52648 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384199 : snode ix: 384199 dh: 4 sx:  250 pt:  65723 nc:    1 fc: 384200 ns: 384205 lv: 48 cp:  52649 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384205 : snode ix: 384205 dh: 4 sx:  251 pt:  65723 nc:    1 fc: 384206 ns: 384211 lv: 48 cp:  52650 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384211 : snode ix: 384211 dh: 4 sx:  252 pt:  65723 nc:    1 fc: 384212 ns: 384217 lv: 48 cp:  52651 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384217 : snode ix: 384217 dh: 4 sx:  253 pt:  65723 nc:    1 fc: 384218 ns: 384223 lv: 48 cp:  52652 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384223 : snode ix: 384223 dh: 4 sx:  254 pt:  65723 nc:    1 fc: 384224 ns: 384229 lv: 48 cp:  52653 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384229 : snode ix: 384229 dh: 4 sx:  255 pt:  65723 nc:    1 fc: 384230 ns: 384235 lv: 48 cp:  52654 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384235 : snode ix: 384235 dh: 4 sx:  256 pt:  65723 nc:    1 fc: 384236 ns: 384241 lv: 48 cp:  52655 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384241 : snode ix: 384241 dh: 4 sx:  257 pt:  65723 nc:    1 fc: 384242 ns: 384247 lv: 48 cp:  52656 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384247 : snode ix: 384247 dh: 4 sx:  258 pt:  65723 nc:    1 fc: 384248 ns: 384253 lv: 48 cp:  52657 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384253 : snode ix: 384253 dh: 4 sx:  259 pt:  65723 nc:    1 fc: 384254 ns: 384259 lv: 48 cp:  52658 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384259 : snode ix: 384259 dh: 4 sx:  260 pt:  65723 nc:    1 fc: 384260 ns: 384265 lv: 48 cp:  52659 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384265 : snode ix: 384265 dh: 4 sx:  261 pt:  65723 nc:    1 fc: 384266 ns: 384271 lv: 48 cp:  52660 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384271 : snode ix: 384271 dh: 4 sx:  262 pt:  65723 nc:    1 fc: 384272 ns: 384277 lv: 48 cp:  52661 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384277 : snode ix: 384277 dh: 4 sx:  263 pt:  65723 nc:    1 fc: 384278 ns: 384283 lv: 48 cp:  52662 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384283 : snode ix: 384283 dh: 4 sx:  264 pt:  65723 nc:    1 fc: 384284 ns: 384289 lv: 48 cp:  52663 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384289 : snode ix: 384289 dh: 4 sx:  265 pt:  65723 nc:    1 fc: 384290 ns: 384295 lv: 48 cp:  52664 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384295 : snode ix: 384295 dh: 4 sx:  266 pt:  65723 nc:    1 fc: 384296 ns: 384301 lv: 48 cp:  52665 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384301 : snode ix: 384301 dh: 4 sx:  267 pt:  65723 nc:    1 fc: 384302 ns: 384307 lv: 48 cp:  52666 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384307 : snode ix: 384307 dh: 4 sx:  268 pt:  65723 nc:    1 fc: 384308 ns: 384313 lv: 48 cp:  52667 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384313 : snode ix: 384313 dh: 4 sx:  269 pt:  65723 nc:    1 fc: 384314 ns: 384319 lv: 48 cp:  52668 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384319 : snode ix: 384319 dh: 4 sx:  270 pt:  65723 nc:    1 fc: 384320 ns: 384325 lv: 48 cp:  52669 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384325 : snode ix: 384325 dh: 4 sx:  271 pt:  65723 nc:    1 fc: 384326 ns: 384331 lv: 48 cp:  52670 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384331 : snode ix: 384331 dh: 4 sx:  272 pt:  65723 nc:    1 fc: 384332 ns: 384337 lv: 48 cp:  52671 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384337 : snode ix: 384337 dh: 4 sx:  273 pt:  65723 nc:    1 fc: 384338 ns: 384343 lv: 48 cp:  52672 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384343 : snode ix: 384343 dh: 4 sx:  274 pt:  65723 nc:    1 fc: 384344 ns: 384349 lv: 48 cp:  52673 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384349 : snode ix: 384349 dh: 4 sx:  275 pt:  65723 nc:    1 fc: 384350 ns: 384355 lv: 48 cp:  52674 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384355 : snode ix: 384355 dh: 4 sx:  276 pt:  65723 nc:    1 fc: 384356 ns: 384361 lv: 48 cp:  52675 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384361 : snode ix: 384361 dh: 4 sx:  277 pt:  65723 nc:    1 fc: 384362 ns: 384367 lv: 48 cp:  52676 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384367 : snode ix: 384367 dh: 4 sx:  278 pt:  65723 nc:    1 fc: 384368 ns: 384373 lv: 48 cp:  52677 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384373 : snode ix: 384373 dh: 4 sx:  279 pt:  65723 nc:    1 fc: 384374 ns: 384379 lv: 48 cp:  52678 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384379 : snode ix: 384379 dh: 4 sx:  280 pt:  65723 nc:    1 fc: 384380 ns: 384385 lv: 48 cp:  52679 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384385 : snode ix: 384385 dh: 4 sx:  281 pt:  65723 nc:    1 fc: 384386 ns: 384391 lv: 48 cp:  52680 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384391 : snode ix: 384391 dh: 4 sx:  282 pt:  65723 nc:    1 fc: 384392 ns: 384397 lv: 48 cp:  52681 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384397 : snode ix: 384397 dh: 4 sx:  283 pt:  65723 nc:    1 fc: 384398 ns: 384403 lv: 48 cp:  52682 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384403 : snode ix: 384403 dh: 4 sx:  284 pt:  65723 nc:    1 fc: 384404 ns: 384409 lv: 48 cp:  52683 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384409 : snode ix: 384409 dh: 4 sx:  285 pt:  65723 nc:    1 fc: 384410 ns: 384415 lv: 48 cp:  52684 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384415 : snode ix: 384415 dh: 4 sx:  286 pt:  65723 nc:    1 fc: 384416 ns: 384421 lv: 48 cp:  52685 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384421 : snode ix: 384421 dh: 4 sx:  287 pt:  65723 nc:    1 fc: 384422 ns: 384427 lv: 48 cp:  52686 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384427 : snode ix: 384427 dh: 4 sx:  288 pt:  65723 nc:    1 fc: 384428 ns: 384433 lv: 48 cp:  52687 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384433 : snode ix: 384433 dh: 4 sx:  289 pt:  65723 nc:    1 fc: 384434 ns: 384439 lv: 48 cp:  52688 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384439 : snode ix: 384439 dh: 4 sx:  290 pt:  65723 nc:    1 fc: 384440 ns: 384445 lv: 48 cp:  52689 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384445 : snode ix: 384445 dh: 4 sx:  291 pt:  65723 nc:    1 fc: 384446 ns: 384451 lv: 48 cp:  52690 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384451 : snode ix: 384451 dh: 4 sx:  292 pt:  65723 nc:    1 fc: 384452 ns: 384457 lv: 48 cp:  52691 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384457 : snode ix: 384457 dh: 4 sx:  293 pt:  65723 nc:    1 fc: 384458 ns: 384463 lv: 48 cp:  52692 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384463 : snode ix: 384463 dh: 4 sx:  294 pt:  65723 nc:    1 fc: 384464 ns: 384469 lv: 48 cp:  52693 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384469 : snode ix: 384469 dh: 4 sx:  295 pt:  65723 nc:    1 fc: 384470 ns: 384475 lv: 48 cp:  52694 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384475 : snode ix: 384475 dh: 4 sx:  296 pt:  65723 nc:    1 fc: 384476 ns: 384481 lv: 48 cp:  52695 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384481 : snode ix: 384481 dh: 4 sx:  297 pt:  65723 nc:    1 fc: 384482 ns: 384487 lv: 48 cp:  52696 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384487 : snode ix: 384487 dh: 4 sx:  298 pt:  65723 nc:    1 fc: 384488 ns: 384493 lv: 48 cp:  52697 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384493 : snode ix: 384493 dh: 4 sx:  299 pt:  65723 nc:    1 fc: 384494 ns: 384499 lv: 48 cp:  52698 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384499 : snode ix: 384499 dh: 4 sx:  300 pt:  65723 nc:    1 fc: 384500 ns: 384505 lv: 48 cp:  52699 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384505 : snode ix: 384505 dh: 4 sx:  301 pt:  65723 nc:    1 fc: 384506 ns: 384511 lv: 48 cp:  52700 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384511 : snode ix: 384511 dh: 4 sx:  302 pt:  65723 nc:    1 fc: 384512 ns: 384517 lv: 48 cp:  52701 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384517 : snode ix: 384517 dh: 4 sx:  303 pt:  65723 nc:    1 fc: 384518 ns: 384523 lv: 48 cp:  52702 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384523 : snode ix: 384523 dh: 4 sx:  304 pt:  65723 nc:    1 fc: 384524 ns: 384529 lv: 48 cp:  52703 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384529 : snode ix: 384529 dh: 4 sx:  305 pt:  65723 nc:    1 fc: 384530 ns: 384535 lv: 48 cp:  52704 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384535 : snode ix: 384535 dh: 4 sx:  306 pt:  65723 nc:    1 fc: 384536 ns: 384541 lv: 48 cp:  52705 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384541 : snode ix: 384541 dh: 4 sx:  307 pt:  65723 nc:    1 fc: 384542 ns: 384547 lv: 48 cp:  52706 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384547 : snode ix: 384547 dh: 4 sx:  308 pt:  65723 nc:    1 fc: 384548 ns: 384553 lv: 48 cp:  52707 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384553 : snode ix: 384553 dh: 4 sx:  309 pt:  65723 nc:    1 fc: 384554 ns: 384559 lv: 48 cp:  52708 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384559 : snode ix: 384559 dh: 4 sx:  310 pt:  65723 nc:    1 fc: 384560 ns: 384565 lv: 48 cp:  52709 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384565 : snode ix: 384565 dh: 4 sx:  311 pt:  65723 nc:    1 fc: 384566 ns: 384571 lv: 48 cp:  52710 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384571 : snode ix: 384571 dh: 4 sx:  312 pt:  65723 nc:    1 fc: 384572 ns: 384577 lv: 48 cp:  52711 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384577 : snode ix: 384577 dh: 4 sx:  313 pt:  65723 nc:    1 fc: 384578 ns: 384583 lv: 48 cp:  52712 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384583 : snode ix: 384583 dh: 4 sx:  314 pt:  65723 nc:    1 fc: 384584 ns: 384589 lv: 48 cp:  52713 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384589 : snode ix: 384589 dh: 4 sx:  315 pt:  65723 nc:    1 fc: 384590 ns: 384595 lv: 48 cp:  52714 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384595 : snode ix: 384595 dh: 4 sx:  316 pt:  65723 nc:    1 fc: 384596 ns: 384601 lv: 48 cp:  52715 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384601 : snode ix: 384601 dh: 4 sx:  317 pt:  65723 nc:    1 fc: 384602 ns: 384607 lv: 48 cp:  52716 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384607 : snode ix: 384607 dh: 4 sx:  318 pt:  65723 nc:    1 fc: 384608 ns: 384613 lv: 48 cp:  52717 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384613 : snode ix: 384613 dh: 4 sx:  319 pt:  65723 nc:    1 fc: 384614 ns: 384619 lv: 48 cp:  52718 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384619 : snode ix: 384619 dh: 4 sx:  320 pt:  65723 nc:    1 fc: 384620 ns: 384625 lv: 48 cp:  52719 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384625 : snode ix: 384625 dh: 4 sx:  321 pt:  65723 nc:    1 fc: 384626 ns: 384631 lv: 48 cp:  52720 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384631 : snode ix: 384631 dh: 4 sx:  322 pt:  65723 nc:    1 fc: 384632 ns: 384637 lv: 48 cp:  52721 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384637 : snode ix: 384637 dh: 4 sx:  323 pt:  65723 nc:    1 fc: 384638 ns: 384643 lv: 48 cp:  52722 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384643 : snode ix: 384643 dh: 4 sx:  324 pt:  65723 nc:    1 fc: 384644 ns: 384649 lv: 48 cp:  52723 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384649 : snode ix: 384649 dh: 4 sx:  325 pt:  65723 nc:    1 fc: 384650 ns: 384655 lv: 48 cp:  52724 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384655 : snode ix: 384655 dh: 4 sx:  326 pt:  65723 nc:    1 fc: 384656 ns: 384661 lv: 48 cp:  52725 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384661 : snode ix: 384661 dh: 4 sx:  327 pt:  65723 nc:    1 fc: 384662 ns: 384667 lv: 48 cp:  52726 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384667 : snode ix: 384667 dh: 4 sx:  328 pt:  65723 nc:    1 fc: 384668 ns: 384673 lv: 48 cp:  52727 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384673 : snode ix: 384673 dh: 4 sx:  329 pt:  65723 nc:    1 fc: 384674 ns: 384679 lv: 48 cp:  52728 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384679 : snode ix: 384679 dh: 4 sx:  330 pt:  65723 nc:    1 fc: 384680 ns: 384685 lv: 48 cp:  52729 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384685 : snode ix: 384685 dh: 4 sx:  331 pt:  65723 nc:    1 fc: 384686 ns: 384691 lv: 48 cp:  52730 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384691 : snode ix: 384691 dh: 4 sx:  332 pt:  65723 nc:    1 fc: 384692 ns: 384697 lv: 48 cp:  52731 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384697 : snode ix: 384697 dh: 4 sx:  333 pt:  65723 nc:    1 fc: 384698 ns: 384703 lv: 48 cp:  52732 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384703 : snode ix: 384703 dh: 4 sx:  334 pt:  65723 nc:    1 fc: 384704 ns: 384709 lv: 48 cp:  52733 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384709 : snode ix: 384709 dh: 4 sx:  335 pt:  65723 nc:    1 fc: 384710 ns: 384715 lv: 48 cp:  52734 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384715 : snode ix: 384715 dh: 4 sx:  336 pt:  65723 nc:    1 fc: 384716 ns: 384721 lv: 48 cp:  52735 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384721 : snode ix: 384721 dh: 4 sx:  337 pt:  65723 nc:    1 fc: 384722 ns: 384727 lv: 48 cp:  52736 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384727 : snode ix: 384727 dh: 4 sx:  338 pt:  65723 nc:    1 fc: 384728 ns: 384733 lv: 48 cp:  52737 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384733 : snode ix: 384733 dh: 4 sx:  339 pt:  65723 nc:    1 fc: 384734 ns: 384739 lv: 48 cp:  52738 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384739 : snode ix: 384739 dh: 4 sx:  340 pt:  65723 nc:    1 fc: 384740 ns: 384745 lv: 48 cp:  52739 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384745 : snode ix: 384745 dh: 4 sx:  341 pt:  65723 nc:    1 fc: 384746 ns: 384751 lv: 48 cp:  52740 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384751 : snode ix: 384751 dh: 4 sx:  342 pt:  65723 nc:    1 fc: 384752 ns: 384757 lv: 48 cp:  52741 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384757 : snode ix: 384757 dh: 4 sx:  343 pt:  65723 nc:    1 fc: 384758 ns: 384763 lv: 48 cp:  52742 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384763 : snode ix: 384763 dh: 4 sx:  344 pt:  65723 nc:    1 fc: 384764 ns: 384769 lv: 48 cp:  52743 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384769 : snode ix: 384769 dh: 4 sx:  345 pt:  65723 nc:    1 fc: 384770 ns: 384775 lv: 48 cp:  52744 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384775 : snode ix: 384775 dh: 4 sx:  346 pt:  65723 nc:    1 fc: 384776 ns: 384781 lv: 48 cp:  52745 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384781 : snode ix: 384781 dh: 4 sx:  347 pt:  65723 nc:    1 fc: 384782 ns: 384787 lv: 48 cp:  52746 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     384787 : snode ix: 384787 dh: 4 sx:  348 pt:  65723 nc:    1 fc: 384788 ns:     -1 lv: 48 cp:  52747 se:     -1 se:     -1 ri: 0 ro:   -1 bd:303 sn:-1 tag BOUNDARY listed YES
     count 348
    ]stree::desc_node_elist

    (ok) A[blyth@localhost tests]$ echo $(( 12*29 ))
    348
    (ok) A[blyth@localhost tests]$




TODO : performance scan changing instancing cut from ~25 up to 500
---------------------------------------------------------------------






