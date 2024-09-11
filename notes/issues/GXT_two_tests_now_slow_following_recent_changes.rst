GXT_two_tests_now_slow_following_recent_changes
===================================================

::

    SLOW: tests taking longer that 15 seconds
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                       Passed                         33.70  
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test         Passed                         35.07  

::

    P[blyth@localhost tests]$ ctest --output-on-error
    Test project /data/blyth/opticks_Debug/build/g4cx/tests
        Start 1: G4CXTest.G4CXRenderTest
    1/2 Test #1: G4CXTest.G4CXRenderTest .................   Passed   35.09 sec
        Start 2: G4CXTest.G4CXOpticks_setGeometry_Test
    2/2 Test #2: G4CXTest.G4CXOpticks_setGeometry_Test ...   Passed   35.89 sec

    100% tests passed, 0 tests failed out of 2

    Total Test time (real) =  71.01 sec



These tests both start from gdml, time all in SetGeometry::

    gxt
    ./G4CXRenderTest.sh 


    ./G4CXOpticks_setGeometry_Test.sh

    2024-09-11 09:53:23.741 INFO  [241418] [main@16] [SetGeometry
    ...
    2024-09-11 09:53:59.388 INFO  [241418] [main@18] ]SetGeometry


Breakdown within there : primary time consumer is U4Tree::Create
------------------------------------------------------------------

Add bulk timing logging for::

     LOG=1 ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
     LOG=1 ~/o/g4cx/tests/G4CXRenderTest.sh


::

    P[blyth@localhost g4cx]$ LOG=1 ~/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh
    /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh : GEOM J_2024aug27 : no geomscript
    logging is a function
    logging () 
    { 
        type $FUNCNAME;
        export Dummy=INFO;
        export G4CXOpticks=INFO;
        export U4Tree=INFO
    }
    logging is a function
    logging () 
    { 
        type $FUNCNAME;
        export Dummy=INFO;
        export G4CXOpticks=INFO;
        export U4Tree=INFO
    }
    U4Tree=INFO
    G4CXOpticks=INFO
    Dummy=INFO
                       BASH_SOURCE : /home/blyth/o/g4cx/tests/G4CXOpticks_setGeometry_Test.sh 
                               arg : info_run_ana 
                              SDIR :  
                              GEOM : J_2024aug27 
                              FOLD : /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27 
                               bin : G4CXOpticks_setGeometry_Test 
                        geomscript :  
                            script : G4CXOpticks_setGeometry_Test.py 
                            origin : /home/blyth/.opticks/GEOM/J_2024aug27/origin.gdml 
    ./GXTestRunner.sh : FOUND GDMLPath /home/blyth/.opticks/GEOM/J_2024aug27/origin.gdml
                    HOME : /home/blyth
                     PWD : /data/blyth/junotop/opticks/g4cx/tests
                    GEOM : J_2024aug27
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXOpticks_setGeometry_Test
                    ARGS : 
    SLOG::EnvLevel adjusting loglevel by envvar   key G4CXOpticks level INFO fallback DEBUG upper_level INFO
    2024-09-11 14:44:55.883 INFO  [260407] [main@16] [SetGeometry
    ssys::getenvvar.is_path_prefixed  path $HOME/.opticks/GEOM/${GEOM}_meshname_stree__force_triangulate_solid.txt
    2024-09-11 14:44:55.884 INFO  [260407] [G4CXOpticks::init@107] CSGOptiX::Desc Version 7 WITH_CUSTOM4 
    G4CXOpticks::desc sim Y tr N wd N fd N cx N qs N
    2024-09-11 14:44:55.884 INFO  [260407] [G4CXOpticks::setGeometry@152]  argumentless 
    2024-09-11 14:44:55.885 INFO  [260407] [G4CXOpticks::setGeometry@172]  GEOM/U4VolumeMaker::PV 
    U4VolumeMaker::PV name J_2024aug27
    2024-09-11 14:44:55.885 FATAL [260407] [SOpticksResource::GDMLPath@452]  TODO: ELIMINATE THIS : INSTEAD USE GDMLPathFromGEOM 
    U4VolumeMaker::PVG_ name J_2024aug27 gdmlpath /home/blyth/.opticks/GEOM/J_2024aug27/origin.gdml sub - exists 1
    G4GDML: Reading '/home/blyth/.opticks/GEOM/J_2024aug27/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/.opticks/GEOM/J_2024aug27/origin.gdml' done!
    U4GDML::read                   yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    2024-09-11 14:44:57.923 INFO  [260407] [G4CXOpticks::setGeometry@243] [ G4VPhysicalVolume world 0x122fc00
    2024-09-11 14:44:57.923 INFO  [260407] [G4CXOpticks::setGeometry@250] [U4Tree::Create 
    2024-09-11 14:44:57.923 INFO  [260407] [U4Tree::Create@214] [new U4Tree
    2024-09-11 14:44:57.968 INFO  [260407] [U4Tree::init@270] -initRayleigh
    2024-09-11 14:44:57.968 INFO  [260407] [U4Tree::init@272] -initMaterials
    2024-09-11 14:44:58.075 INFO  [260407] [U4Tree::init@274] -initMaterials_NoRINDEX
    2024-09-11 14:44:58.075 INFO  [260407] [U4Tree::init@277] -initScint
    2024-09-11 14:44:58.076 INFO  [260407] [U4Tree::init@280] -initSurfaces
    2024-09-11 14:44:58.080 INFO  [260407] [U4Tree::init@283] -initSolids                    ## 3s
    [U4Tree::initSolids
    sn::decrease_zmin_ lvid 94 _zmin   -6.50 dz    1.00 new_zmin   -7.50
    sn::increase_zmax_ lvid 102 _zmax  -15.00 dz    1.00 new_zmax  -14.00
    sn::decrease_zmin_ lvid 102 _zmin -101.00 dz    1.00 new_zmin -102.00
    sn::increase_zmax_ lvid 102 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 102 _zmin  -15.00 dz    1.00 new_zmin  -16.00
    sn::increase_zmax_cone lvid 103 z2 0.00 r2 450.00 dz 1.00 new_z2 1.00 new_r2 451.79
    sn::increase_zmax_ lvid 104 _zmax    6.50 dz    1.00 new_zmax    7.50
    sn::uncoincide sn__uncoincide_dump_lvid 107 lvid 107
    sn::uncoincide_ lvid 107 num_prim 6
    sn::uncoincide_ lvid 107 num_prim 6 coincide 0

    sn::increase_zmax_ lvid 115 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 115 _zmin    0.00 dz    1.00 new_zmin   -1.00
    sn::increase_zmax_ lvid 116 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::decrease_zmin_ lvid 116 _zmin    0.00 dz    1.00 new_zmin   -1.00
    sn::increase_zmax_ lvid 117 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::increase_zmax_ lvid 117 _zmax  100.00 dz    1.00 new_zmax  101.00
    sn::increase_zmax_ lvid 126 _zmax    0.00 dz    1.00 new_zmax    1.00
    sn::increase_zmax_ lvid 126 _zmax   97.00 dz    1.00 new_zmax   98.00
    ]U4Tree::initSolids
    2024-09-11 14:45:01.731 INFO  [260407] [U4Tree::init@285] -initNodes                    ## 9s
    2024-09-11 14:45:10.973 INFO  [260407] [U4Tree::init@287] -initSurfaces_Serialize
    2024-09-11 14:45:11.005 INFO  [260407] [U4Tree::init@290] -initStandard
    U4Tree::init U4Tree::desc
     st Y
     top Y
     sid Y
     level 0
     lvidx 302
     pvs 382197
     materials 23
     surfaces 95
     solids 302
     enable_osur YES
     enable_isur YES

    2024-09-11 14:45:11.193 INFO  [260407] [U4Tree::Create@216] ]new U4Tree
    2024-09-11 14:45:11.193 INFO  [260407] [U4Tree::Create@218] [stree::factorize                 ## 14s  
    2024-09-11 14:45:25.608 INFO  [260407] [U4Tree::Create@220] ]stree::factorize
    2024-09-11 14:45:25.608 INFO  [260407] [U4Tree::Create@222] [U4Tree::identifySensitive
    2024-09-11 14:45:25.762 INFO  [260407] [U4Tree::Create@224] ]U4Tree::identifySensitive
    2024-09-11 14:45:25.762 INFO  [260407] [U4Tree::Create@227] [stree::add_inst
    2024-09-11 14:45:26.475 INFO  [260407] [U4Tree::Create@229] ]stree::add_inst
    2024-09-11 14:45:26.475 INFO  [260407] [U4Tree::Create@234] [stree::postcreate
    [stree::postcreate
    stree::desc_sensor
     sensor_id.size 0
     sensor_count 0
     sensor_name.size 0
    sensor_name[
    ]
    [stree::desc_sensor_nd
     edge            0
     num_nd          382197
     num_nd_sensor   0
     num_sid         0
    ]stree::desc_sensor_nd
    stree::desc_sensor_id sensor_id.size 0
    [
    ]]stree::postcreate
    2024-09-11 14:45:26.478 INFO  [260407] [U4Tree::Create@236] ]stree::postcreate
    2024-09-11 14:45:26.478 INFO  [260407] [G4CXOpticks::setGeometry@252] ]U4Tree::Create 
    2024-09-11 14:45:26.478 INFO  [260407] [G4CXOpticks::setGeometry@255] [SSim::initSceneFromTree
    2024-09-11 14:45:27.333 INFO  [260407] [G4CXOpticks::setGeometry@257] ]SSim::initSceneFromTree
    2024-09-11 14:45:27.333 INFO  [260407] [G4CXOpticks::setGeometry@260] [CSGFoundry::CreateFromSim
    [CSGImport::importPrim.dump_LVID:1 node.lvid 101 LVID -1 name uni1 soname uni1 primIdx 0 bn 7 ln(subset of bn) 1 num_sub_total 8
    .CSGImport::importPrim dumping as ln > 0 : solid contains listnode
    2024-09-11 14:45:27.984 INFO  [260407] [CSGImport::importPrim@427] s_bb::IncludeAABB 
     inital_aabb  [  0.000,  0.000,  0.000,  0.000,  0.000,  0.000] ALL_ZERO
     other_aabb   [-206.200,-206.200, -7.000,206.200,206.200,  7.000] ADOPT OTHER AS STARTING
     updated_aabb [-206.200,-206.200, -7.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200, -7.000,206.200,206.200,  7.000]
     other_aabb   [151.000,-13.000,-115.000,177.000, 13.000,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [102.966,102.966,-115.000,128.966,128.966,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [-13.000,151.000,-115.000, 13.000,177.000,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [-128.966,102.966,-115.000,-102.966,128.966,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [-177.000,-13.000,-115.000,-151.000, 13.000,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [-128.966,-128.966,-115.000,-102.966,-102.966,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [-13.000,-177.000,-115.000, 13.000,-151.000,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
    s_bb::IncludeAABB 
     inital_aabb  [-206.200,-206.200,-115.000,206.200,206.200,  7.000]
     other_aabb   [102.966,-128.966,-115.000,128.966,-102.966,-15.000]COMBINE
     updated_aabb [-206.200,-206.200,-115.000,206.200,206.200,  7.000]

    ]CSGImport::importPrim.dump_LVID:1 node.lvid 101 LVID -1 name uni1 soname uni1
    2024-09-11 14:45:28.108 INFO  [260407] [G4CXOpticks::setGeometry@262] ]CSGFoundry::CreateFromSim
    2024-09-11 14:45:28.109 INFO  [260407] [G4CXOpticks::setGeometry@265] [setGeometry(fd_)
    2024-09-11 14:45:28.109 INFO  [260407] [G4CXOpticks::setGeometry_@319] [ fd 0x11021930
    2024-09-11 14:45:28.109 INFO  [260407] [G4CXOpticks::setGeometry_@325] [ CSGOptiX::Create        ## 2s
    2024-09-11 14:45:30.101 INFO  [260407] [CSGOptiX::initPIDXYZ@703]  params->pidxyz (4294967295,4294967295,4294967295) 
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::setGeometry_@327] ] CSGOptiX::Create 
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::setGeometry_@338]  cx Y qs Y QSim::Get Y
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::setGeometry_@346] ] fd 0x11021930
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::setGeometry_@352] [ G4CXOpticks__setGeometry_saveGeometry 
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::saveGeometry@494]  dir_ /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::saveGeometry@496] [ /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27
    2024-09-11 14:45:30.103 INFO  [260407] [G4CXOpticks::saveGeometry@497] [ /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27
    G4CXOpticks::saveGeometry [ /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27
    2024-09-11 14:45:30.103 INFO  [260407] [U4GDML::write@291]  ekey U4GDML_GDXML_FIX_DISABLE U4GDML_GDXML_FIX_DISABLE 0 U4GDML_GDXML_FIX 1
    2024-09-11 14:45:30.103 INFO  [260407] [U4GDML::write_@318] [
    2024-09-11 14:45:30.108 INFO  [260407] [U4GDML::write_@327]  path /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin_raw.gdml exists YES rc 0
    G4GDML: Writing '/data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin_raw.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin_raw.gdml' done !
    2024-09-11 14:45:32.107 INFO  [260407] [U4GDML::write_@335] ]
    2024-09-11 14:45:32.107 INFO  [260407] [U4GDML::write@305] [ Apply GDXML::Fix  rawpath /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin_raw.gdml dstpath /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin.gdml
    2024-09-11 14:45:33.568 INFO  [260407] [U4GDML::write@307] ] Apply GDXML::Fix  rawpath /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin_raw.gdml dstpath /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27/origin.gdml
    2024-09-11 14:45:41.264 INFO  [260407] [G4CXOpticks::saveGeometry@503] ] /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27
    2024-09-11 14:45:41.264 INFO  [260407] [G4CXOpticks::setGeometry_@354] ] G4CXOpticks__setGeometry_saveGeometry 
    2024-09-11 14:45:41.264 INFO  [260407] [G4CXOpticks::setGeometry@267] ]setGeometry(fd_)
    2024-09-11 14:45:41.264 INFO  [260407] [G4CXOpticks::setGeometry@269] CSGOptiX::Desc Version 7 WITH_CUSTOM4 
    2024-09-11 14:45:41.264 INFO  [260407] [G4CXOpticks::setGeometry@271] ] G4VPhysicalVolume world 0x122fc00
    2024-09-11 14:45:41.264 INFO  [260407] [main@18] ]SetGeometry


