FIXED : EMM_selection_seems_not_working
=============================================

Fixed by use of ssys::envvar in SBit.cc to interpret the ekey "OPTICKS_EMM_SELECTION,EMM"
in comma delimited manner::


    EMM=0 ~/o/cx.sh 

::

     24 unsigned long long SGeoConfig::_EMM = SBit::FromEString(kEMM, "~0");
     40 unsigned long long SGeoConfig::EnabledMergedMesh(){  return _EMM ; }


    P[blyth@localhost opticks]$ opticks-f EnabledMergedMesh
    ./CSG/CSGFoundry.cc:way too rather than smearing ok->isEnabledMergedMesh all over CSGOptiX/SBT
    ./CSGOptiX/Six.cc:    emm(SGeoConfig::EnabledMergedMesh()),
    ./CSGOptiX/Six.cc:        if(SGeoConfig::IsEnabledMergedMesh(i))
    ./CSGOptiX/CSGOptiX.cc:    js["emm"] = SGeoConfig::EnabledMergedMesh() ;
    ./CSGOptiX/SBT.cc:    emm(SGeoConfig::EnabledMergedMesh()), 
    ./CSGOptiX/SBT.cc:        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx) ;
    ./CSGOptiX/SBT.cc:        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx)  ; 
    ./CSGOptiX/SBT.cc:        bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx)  ; 
    ./CSG_GGeo/CSG_GGeo_Convert.cc:        if(SGeoConfig::IsEnabledMergedMesh(repeatIdx))


::

     EMM=0 SBT=info ~/o/cx.sh

::

    P[blyth@localhost sysrap]$ EMM=t1,2,3 SGeoConfigTest
    SName::detail num_name 0

    2024-09-11 16:00:38.411 INFO  [399636] [main@62] 
            OPTICKS_GEOM,GEOM : -
    OPTICKS_EMM_SELECTION,EMM : ~0x0 0xffffffffffffffff
    OPTICKS_ELV_SELECTION,ELV : -
      OPTICKS_SOLID_SELECTION : -
        OPTICKS_SOLID_TRIMESH : -
        OPTICKS_FLIGHT_CONFIG : -
         OPTICKS_ARGLIST_PATH : -
             OPTICKS_CXSKIPLV : 
     OPTICKS_CXSKIPLV_IDXLIST : 



After fix::


    P[blyth@localhost sysrap]$ EMM=0,1,2,3 ~/o/sysrap/tests/SGeoConfigTest.sh
    2024-09-11 16:45:40.871 INFO  [97744] [test_EMM@55] SGeoConfig::DescEMM OPTICKS_EMM_SELECTION,EMM : 0xf 0xf
    0 1 2 3 

    P[blyth@localhost sysrap]$ EMM=t0,1,2,3 ~/o/sysrap/tests/SGeoConfigTest.sh
    2024-09-11 16:45:48.498 INFO  [97960] [test_EMM@55] SGeoConfig::DescEMM OPTICKS_EMM_SELECTION,EMM : ~0xf 0xfffffffffffffff0
    4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 
    P[blyth@localhost sysrap]$ 


    EMM=0, ~/o/cx.sh   ## only 0, "rem" 

    EMM=t0, ~/o/cx.sh   ## exclude 0 "rem"

    EMM=10, ~/o/cx.sh   ## only 10 "tri"

    EMM=10  ~/o/cx.sh   ## NB: WITHOUT COMMA IS VERY DIFFERENT TO ABOVE : BIT PATTERN FROM DECIMAL 10 

    EMM=1,2,3,4 ~/o/cx.sh

    EMM=1,2,3,4,5,6,7,8,9 ~/o/cx.sh 



