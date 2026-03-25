FIXED : U4Polycone_improve_phicut_error_message
===================================================


::

    ~/j/oj_test/doublemuon/doublemuon.sh dbg


    (gdb) bt
    #0  0x00007ffff748bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff743eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff7428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff742875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff7437886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007fffc0efa50e in U4Polycone::init (this=0x7ffffffe8c90) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:344
    #6  0x00007fffc0efa10b in U4Polycone::U4Polycone (this=0x7ffffffe8c90, polycone_=0xb520220, lvid_=87, depth_=0, level_=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:252
    #7  0x00007fffc0ef94d0 in U4Polycone::Convert (polycone=0xb520220, lvid=87, depth=0, level=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:159
    #8  0x00007fffc0efdda3 in U4Solid::init_Polycone (this=0x7ffffffe8fd0) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:887
    #9  0x00007fffc0efbffa in U4Solid::init_Constituents (this=0x7ffffffe8fd0) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:406
    #10 0x00007fffc0efbea6 in U4Solid::init (this=0x7ffffffe8fd0) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:381
    #11 0x00007fffc0efbd78 in U4Solid::U4Solid (this=0x7ffffffe8fd0, solid_=0xb520220, lvid_=87, depth_=0, level_=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:368
    #12 0x00007fffc0efbc84 in U4Solid::Convert (solid=0xb520220, lvid=87, depth=0, level=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:349
    #13 0x00007fffc0f0229e in U4Tree::initSolid (this=0xd092b40, so=0xb520220, lvid=87) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:713
    #14 0x00007fffc0f021f6 in U4Tree::initSolid (this=0xd092b40, lv=0xb5216d0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:681
    #15 0x00007fffc0f0218c in U4Tree::initSolids_r (this=0xd092b40, pv=0xb5218a0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:674
    #16 0x00007fffc0f02127 in U4Tree::initSolids_r (this=0xd092b40, pv=0xb5e87d0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #17 0x00007fffc0f02127 in U4Tree::initSolids_r (this=0xd092b40, pv=0xaf2f940) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #18 0x00007fffc0f02127 in U4Tree::initSolids_r (this=0xd092b40, pv=0xaf2f9a0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #19 0x00007fffc0f02127 in U4Tree::initSolids_r (this=0xd092b40, pv=0xaf1c5d0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #20 0x00007fffc0f01e97 in U4Tree::initSolids (this=0xd092b40) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:614
    #21 0x00007fffc0f0067c in U4Tree::init (this=0xd092b40) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:299
    #22 0x00007fffc0f000b3 in U4Tree::U4Tree (this=0xd092b40, st_=0x67ebb60, top_=0xaf1c5d0, sid_=0x0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:270
    #23 0x00007fffc0eff5dc in U4Tree::Create (st=0x67ebb60, top=0xaf1c5d0, sid=0x0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:226
    #24 0x00007fffc0ea4108 in G4CXOpticks::setGeometry (this=0xd0af140, world=0xaf1c5d0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:305
    #25 0x00007fffc0ea2ac1 in G4CXOpticks::SetGeometry (world=0xaf1c5d0) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:79
    #26 0x00007fffc0ea2d70 in G4CXOpticks::SetGeometry_JUNO (world=0xaf1c5d0, sd=0x9f0b5f0, jpmt=0xc3c7900, jlut=0xc3c9780) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:117
    #27 0x00007fffbd545372 in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0xaf1c5d0, sd=0x9f0b5f0, ppd=0x6730a10, psd=0x66a61c0, pmtscan=0x0) at /builds/JUNO/offline/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc:47
    #28 0x00007fffbd509138 in LSExpDetectorConstruction::setupOpticks (this=0xacb7f30, world=0xaf1c5d0) at /builds/JUNO/offline/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:472
    #29 0x00007fffbd5089d8 in LSExpDetectorConstruction::Construct (this=0xacb7f30) at /builds/JUNO/offline/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:393
    #30 0x00007fffc6cb492e in G4RunManager::InitializeGeometry() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.7.2/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #31 0x00007fffc6cb4afc in G4RunManager::Initialize() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.7.2/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so


Added some logging for the error::

    U4Polycone::init_phicut level -1 phi_start       0.0873 phi_end         1.7628 phi_delta       1.6755 has_phicut YES has_half NO  ENABLE_PHICUT NO
    U4Polycone::init FATAL geometry with unsupported phicut :  enable experimental support with envvar  [U4Polycone__ENABLE_PHICUT] G4Polycone.GetName [s_EMFcoil_holder_ring8_seg1]
    python: /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:347: void U4Polycone::init(): Assertion `0' failed.
     *** Break *** abort


HUH

* dont get this error with OJ_LOCAL but do with OJ_61 ? Missing some env ?

::

    export U4Solid__IsFlaggedLVID=87




CONFIRMED CAUSE : FORGOTTEN local_envset.sh FOR EMF THAT ALLOWS PHICUT
----------------------------------------------------------------------

* NB the local_envset.sh is by design not committed : which requires that

::

    A[blyth@localhost junosw]$ l InstallArea/*/local_envset.sh
    4 -rw-r--r--. 1 blyth blyth 1464 Jan 29 21:41 InstallArea/Chenjing-EMFcoilsgeometry/local_envset.sh


    A[blyth@localhost junosw]$ cat InstallArea/Chenjing-EMFcoilsgeometry/local_envset.sh
    #  InstallArea/Chenjing-EMFcoilsgeometry/local_envset.sh
    #
    # OJ Opticks+JUNOSW installations include an envset.sh file
    # within the top level of the installation prefix.
    # The oj_initialize.sh from the top level of the source tree
    # sources the envset.sh and if it exists the local_envset.sh
    # sibling to the envset.sh.
    #
    # NB local_envset.sh settings are because they are not part of
    # the installation local and temporary changes to the geometry.
    # Once settings have been decided they need to be added
    # to the opticks_juno.sh repository file in order for the
    # settings to become a standard part of the geometry.
    #
    #
    ## settings for Chenjing-EMFcoilsgeometry branch with EMF enabled
    export OJ_INITIALIZE_TUT_DETSIM_OPTION="--emf-coils-system --emf-support-bars"

    ## even when using triangulated for the phicut solids
    ## the below analytic settings are still
    ## needed to avoid asserts from the conversion

    export U4Polycone__ENABLE_PHICUT=1
    export sn__PhiCut_PACMAN_ALLOWED=1


    export U4Mesh__NumberOfRotationSteps_solidName_STARTING_pfx_0=s_EMFcoil_holder_ring
    export U4Mesh__NumberOfRotationSteps_solidName_STARTING_val_0=480

    export U4Mesh__NumberOfRotationSteps_solidName_STARTING_pfx_1=s_EMFsupport_ring
    export U4Mesh__NumberOfRotationSteps_solidName_STARTING_val_1=480


    ##############################################
    ## logging controls are better to not commit
    export U4Mesh__NumberOfRotationSteps_DUMP=1
    #unset U4Mesh__NumberOfRotationSteps_DUMP

    A[blyth@localhost junosw]$




Is the triangulation now automatic ?

