ELVSelection_issue_with_duplicated_prefix_names
=====================================================

FIXED this by switch to exact (starting:false) name matching for ELV selection::

    modified:   CSG/CSGFoundry.cc
    modified:   CSG/CSGFoundry.h
    modified:   sysrap/SGLFW.h
    modified:   sysrap/SGeoConfig.cc
    modified:   sysrap/SGeoConfig.hh
    modified:   sysrap/SName.h




~/j/cx.sh::

     17 export SGeoConfig__ELVSelection_VERBOSE=1
     18 
     19 elv=$(cat << EOS | tr "\n" ","
     20 sTopRock_domeAir
     21 sTopRock_dome
     22 sDomeRockBox
     23 PoolCoversub
     24 sBar_0
     25 sBar_1
     26 sPanelTape
     27 sPanel
     28 sPlane
     29 sWall
     30 sAirTT
     31 sExpHall
     32 sExpRockBox
     33 sTopRock
     34 sOuterWaterPool
     35 sTyvek_shell
     36 sDeadWater_shell
     37 HamamatsuR12860sMask_virtual
     38 NNVTMCPPMTsMask_virtual
     39 sWaterTube
     40 mask_PMT_20inch_vetosMask_virtual
     41 sAirGap
     42 sPoolLining
     43 sBottomRock
     44 sWorld
     45 EOS
     46 )
     47 
     48 echo $elv
     49 
     50 export CSGFoundry=INFO
     51 
     52 MOI=EXTENT:60000 ELV=t:$elv ~/o/cx.sh
     53 


::

    2024-10-13 20:48:05.696 INFO  [298859] [CSGFoundry::loadArray@3243]  ni  8236 nj 4 nk 4 itra.npy
    2024-10-13 20:48:05.700 INFO  [298859] [CSGFoundry::loadArray@3243]  ni 48478 nj 4 nk 4 inst.npy
    2024-10-13 20:48:05.703 INFO  [298859] [CSGFoundry::load@2861] ] loaddir /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/CSGFoundry
    SGeoConfig::ELVSelection [SGeoConfig__ELVSelection_VERBOSE]  elv_selection_ t:sTopRock_domeAir,sTopRock_dome,sDomeRockBox,PoolCoversub,sBar_0,sBar_1,sPanelTape,sPanel,sPlane,sWall,sAirTT,sExpHall,sExpRockBox,sTopRock,sOuterWaterPool,sTyvek_shell,sDeadWater_shell,HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual,sWaterTube,mask_PMT_20inch_vetosMask_virtual,sAirGap,sPoolLining,sBottomRock,sWorld,
    SGeoConfig::ELVSelection prefix t: strlen(prefix) 2
    SGeoConfig::ELVSelection after has_names Y
    2024-10-13 20:48:05.703 INFO  [298859] [CSGFoundry::ELVString@2956]  elv t:0,0,2,3,8,9,10,10,12,13,14,15,16,0,89,90,91,117,126,135,297,298,299,300,301
    2024-10-13 20:48:05.703 INFO  [298859] [CSGFoundry::ELV@2977]  num_meshname 302 elv_ t:0,0,2,3,8,9,10,10,12,13,14,15,16,0,89,90,91,117,126,135,297,298,299,300,301 elv    -t:0,0,2,3,8,9,10,10,12,13,14,15,16,0,89,90,91,117,126,135,297,298,299,300,301 302 : 01001111000100000111111111111111111111111111111111111111111111111111111111111111111111111000111111111111111111111111101111111101111111101111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000
    2024-10-13 20:48:05.703 INFO  [298859] [CSGFoundry::CopySelect@3090] [
    2024-10-13 20:48:05.703 INFO  [298859] [CSGFoundry::CopySelect@3092]    -t:0,0,2,3,8,9,10,10,12,13,14,15,16,0,89,90,91,117,126,135,297,298,299,300,301 302 : 01001111000100000111111111111111111111111111111111111111111111111111111111111111111111111000111111111111111111111111101111111101111111101111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000

    2024-10-13 20:48:05.703 INFO  [298859] [CSGFoundry::CopySelect@3093] CSGFoundry::descELV elv.num_bits 302 num_include 280 num_exclude 22 is_all_set 0
    INCLUDE:280




