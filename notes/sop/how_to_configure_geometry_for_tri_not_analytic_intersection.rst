how_to_configure_geometry_for_tri_not_analytic_intersection
============================================================


Getting a geometry persisted even with issues
----------------------------------------------

When a geometry has something that the Opticks translation
is unable to handle asserts will typically be triggered with errors and backtrace like::

    sn::increase_zmax_ lvid 65 _zmax   37.50 dz    1.00 new_zmax   38.50
    sn::increase_zmax_ lvid 65 _zmax  -33.50 dz    1.00 new_zmax  -32.50
    sn::PhiCut phi0 0.000174533 phi1 6.28301 expect NO  expect_phi YES expect_cross_product NO  cross_product -0.000349066 is_wedge NO  is_pacman YES PACMAN_ALLOWED [sn__PhiCut_PACMAN_ALLOWED] 0
    python: /data1/blyth/local/opticks_Debug/include/SysRap/sn.h:3383: static sn* sn::PhiCut(double, double): Assertion `expect_cross_product' failed.
     *** Break *** abort

    #10 0x00007f2d60c37886 in __assert_fail () from /lib64/libc.so.6
    #11 0x00007f2d2a71e164 in sn::PhiCut (phi0=0.00017453292519943296, phi1=6.2830107742543868) at /data1/blyth/local/opticks_Debug/include/SysRap/sn.h:3369
    #12 0x00007f2d2a739c8a in U4Polycone::init_phicut (this=0x7fffd8c4b0f0) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:468
    #13 0x00007f2d2a7395f0 in U4Polycone::init (this=0x7fffd8c4b0f0) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:307
    #14 0x00007f2d2a73938f in U4Polycone::U4Polycone (this=0x7fffd8c4b0f0, polycone_=0xb416300, lvid_=82, depth_=0, level_=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:252
    #15 0x00007f2d2a738754 in U4Polycone::Convert (polycone=0xb416300, lvid=82, depth=0, level=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Polycone.h:159
    #16 0x00007f2d2a73d027 in U4Solid::init_Polycone (this=0x7fffd8c4b430) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:887
    #17 0x00007f2d2a73b27e in U4Solid::init_Constituents (this=0x7fffd8c4b430) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:406
    #18 0x00007f2d2a73b12a in U4Solid::init (this=0x7fffd8c4b430) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:381
    #19 0x00007f2d2a73affc in U4Solid::U4Solid (this=0x7fffd8c4b430, solid_=0xb416300, lvid_=82, depth_=0, level_=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:368
    #20 0x00007f2d2a73af08 in U4Solid::Convert (solid=0xb416300, lvid=82, depth=0, level=-1) at /data1/blyth/local/opticks_Debug/include/U4/U4Solid.h:349
    #21 0x00007f2d2a741522 in U4Tree::initSolid (this=0xd8a8200, so=0xb416300, lvid=82) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:713
    #22 0x00007f2d2a74147a in U4Tree::initSolid (this=0xd8a8200, lv=0xb417680) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:681
    #23 0x00007f2d2a741410 in U4Tree::initSolids_r (this=0xd8a8200, pv=0xb416240) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:674
    #24 0x00007f2d2a7413ab in U4Tree::initSolids_r (this=0xd8a8200, pv=0xbe4ce90) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #25 0x00007f2d2a7413ab in U4Tree::initSolids_r (this=0xd8a8200, pv=0xb421070) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #26 0x00007f2d2a7413ab in U4Tree::initSolids_r (this=0xd8a8200, pv=0xb4210d0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #27 0x00007f2d2a7413ab in U4Tree::initSolids_r (this=0xd8a8200, pv=0xb40e030) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:671
    #28 0x00007f2d2a74111b in U4Tree::initSolids (this=0xd8a8200) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:614
    #29 0x00007f2d2a73f900 in U4Tree::init (this=0xd8a8200) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:299
    #30 0x00007f2d2a73f337 in U4Tree::U4Tree (this=0xd8a8200, st_=0x74a33f0, top_=0xb40e030, sid_=0x0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:270
    #31 0x00007f2d2a73e860 in U4Tree::Create (st=0x74a33f0, top=0xb40e030, sid=0x0) at /data1/blyth/local/opticks_Debug/include/U4/U4Tree.h:226
    #32 0x00007f2d2a6e4108 in G4CXOpticks::setGeometry (this=0xd8a9040, world=0xb40e030) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:305
    #33 0x00007f2d2a6e2ac1 in G4CXOpticks::SetGeometry (world=0xb40e030) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:79
    #34 0x00007f2d2a6e2d70 in G4CXOpticks::SetGeometry_JUNO (world=0xb40e030, sd=0x7484700, jpmt=0xd516ae0, jlut=0xd58b740) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:117
    #35 0x00007f2d26b42414 in LSExpDetectorConstruction_Opticks::Setup (opticksMode=1, world=0xb40e030, sd=0x7484700, ppd=0x7261670, psd=0x72075b0, pmtscan=0x0) at /home/blyth/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc:47
    #36 0x00007f2d26b064a6 in LSExpDetectorConstruction::setupOpticks (this=0xafffee0, world=0xb40e030) at /home/blyth/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:472
    #37 0x00007f2d26b05d46 in LSExpDetectorConstruction::Construct (this=0xafffee0) at /home/blyth/junosw/Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc:393
    #38 0x00007f2d2ffb692e in G4RunManager::InitializeGeometry() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.7.2/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so


In order to get the geometry persisted for investigation it will often be necessary to add envvar exclusions that 
avoid the triggered asserts and set those envvars in the script that configures the options for geometry conversion
eg Examples/Tutorial/share/opticks_juno.sh with the block::

   ######## CHANGES FOR EMFCoils #########################################
   export OJ_INITIALIZE_TUT_DETSIM_OPTION=--emf-coils-system
   export U4Polycone__ENABLE_PHICUT=1 
   export sn__PhiCut_PACMAN_ALLOWED=1
   #######################################################################


Where is the tri/ana decision made ? 
--------------------------------------

All solids have triangulated versions created at translation but
configuration is needed to select the use of the triangles. There is
also some automation of switching to tri for solids that contain G4Torus
for example.

stree.h::

    0649     static void FindForceTriangulateLVID(std::vector<int>& lvid, const std::vector<std::string>& _sonames, const char* _force_triangulate_solid, char delim=','  );
     650     std::string descForceTriangulateLVID() const ;
     651     bool        is_force_triangulate( int lvid ) const ; // HMM: is_manual_triangulate would be better name
     652     bool        is_auto_triangulate( int lvid ) const ;  // WIP: automate decision, avoiding hassle with geometry updates that change/add solid names
     653     bool        is_triangulate(int lvid) const ;  // OR of the above


Added "phicut,halfspace" to the auto triangulate default::

    inline bool stree::is_auto_triangulate( int lvid ) const
    {
        const sn* root = sn::GetLVRoot(lvid);
        assert( root );

        const char* names = "torus,notsupported,cutcylinder,phicut,halfspace" ;
        const char* NAMES = ssys::getenvvar(stree__is_auto_triangulate_NAMES, names);
        std::vector<int> tcq ;
        CSG::TypeCodeVec(tcq, NAMES, ','); 
        int minsubdepth = 0; 
        int count = root->typecodes_count(tcq, minsubdepth );
        return count > 0 ;
    }









