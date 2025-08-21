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



TODO : performance scan changing instancing cut from ~25 up to 500
---------------------------------------------------------------------




