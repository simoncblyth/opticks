gxs_high_level_translation_of_U4VolumeMaker_Geant4_geometry
=============================================================

Overview
----------

gx is using Opticks cx to do GPU simulation within a geometry auto-translated from Geant4. 

Will not be able to run the simulation on laptop, but can start with getting the translation 
of test Geant4 geometries to work.


Build and Run
---------------

NB gx and some deps are not standardly built, so build with::

    o
    oo ## build the standard set 
    b7 ## rebuilds cx with OptiX 7
    u4 ; om 
    gx ; om 

    jps ; om   ## when need PMTSim updated 


run::

    gx ; ./gxs.sh 
    gx ; ./gxs.sh dbg 



u4 and gx deps
-----------------

u4/CMakeLists.txt::

     14 find_package(SysRap REQUIRED CONFIG)
     15 find_package(G4 REQUIRED MODULE)
     17 find_package(CLHEP  REQUIRED CONFIG)
     20 find_package(OpticksXercesC REQUIRED MODULE)
     22 find_package(PMTSim CONFIG)

gx/CMakeLists.txt::

    find_package(U4       REQUIRED CONFIG)
    find_package(ExtG4    REQUIRED CONFIG)
    find_package(CSG_GGeo REQUIRED CONFIG)
    find_package(CSGOptiX REQUIRED CONFIG)

gx/tests/G4CXSimulateTest.cc::

     09 int main(int argc, char** argv)
     10 {
     11     OPTICKS_LOG(argc, argv);
     12 
     13     U4Material::LoadBnd();   // "back" creation of G4 material properties from the Opticks bnd.npy obtained from SSim::Load 
     14     // this is needed for U4VolumeMaker::PV to find the G4 materials
     15 
     16     SEventConfig::SetRGMode("simulate");
     17 
     18     // GGeo creation done when starting from gdml or live G4, still needs Opticks instance,  
     19     // TODO: avoid this by replacing with automated SOpticks instanciated by OPTICKS_LOG
     20     Opticks::Configure(argc, argv, "--gparts_transform_offset" );
     21 
     22 
     23     G4CXOpticks gx ;
     24 
     25     //gx.setGeometry(SPath::SomeGDMLPath()); 
     26     //gx.setGeometry(CSGFoundry::Load()); 
     27     gx.setGeometry( U4VolumeMaker::PV() );   // sensitive to GEOM envvar
     28 
     29 
     30     gx.simulate();
     31 
     32     return 0 ;
     33 }



Shakedown Issues
--------------------

::

    epsilon:~ blyth$ gx
    /Users/blyth/opticks/g4cx

    epsilon:g4cx blyth$ ./gxs.sh 
    === ../bin/GEOM_.sh : GEOM RaindropRockAirWater2
    2022-07-03 17:02:01.598 INFO  [36224120] [Opticks::postconfigureSize@3335]  ssize 1920,1080,2 sizescale 1 sz uvec4(1920, 1080, 2, 0) ssz uvec4(1920, 1080, 2, 0)
    SStr::LoadList split  into 0
    2022-07-03 17:02:01.602 INFO  [36224120] [U4VolumeMaker::RaindropRockAirWater_Configure@531] U4VolumeMaker_RaindropRockAirWater_HALFSIDE 100
    2022-07-03 17:02:01.602 INFO  [36224120] [U4VolumeMaker::RaindropRockAirWater_Configure@532] U4VolumeMaker_RaindropRockAirWater_FACTOR 1
    2022-07-03 17:02:01.607 INFO  [36224120] [X4PhysicalVolume::convertMaterials@264]  num_mt 3
       0 :                          Water :  num_prop   5               RINDEX              GROUPVEL              RAYLEIGH             ABSLENGTH        REEMISSIONPROB 
       1 :                            Air :  num_prop   5               RINDEX              GROUPVEL              RAYLEIGH             ABSLENGTH        REEMISSIONPROB 
       2 :                           Rock :  num_prop   5               RINDEX              GROUPVEL              RAYLEIGH             ABSLENGTH        REEMISSIONPROB 

    2022-07-03 17:02:01.618 INFO  [36224120] [X4PhysicalVolume::convertMaterials@273]  used_materials.size 3 num_material_with_efficiency 0
    2022-07-03 17:02:01.618 INFO  [36224120] [GMaterialLib::dumpSensitiveMaterials@1273] X4PhysicalVolume::convertMaterials num_sensitive_materials 0
    2022-07-03 17:02:01.619 INFO  [36224120] [GSurfaceLib::dumpImplicitBorderSurfaces@765] X4PhysicalVolume::convertSurfaces
     num_implicit_border_surfaces 0 edgeitems 100

    2022-07-03 17:02:01.621 INFO  [36224120] [GSurfaceLib::dumpSurfaces@907] X4PhysicalVolume::convertSurfaces num_surfaces 5 edgeitems 100
     index :  0 is_sensor : N type :        bordersurface name :                                        air_rock_bs bpv1 air_pv bpv2 rock_pv .
     index :  1 is_sensor : Y type :          testsurface name :                               perfectDetectSurface .
     index :  2 is_sensor : N type :          testsurface name :                               perfectAbsorbSurface .
     index :  3 is_sensor : N type :          testsurface name :                             perfectSpecularSurface .
     index :  4 is_sensor : N type :          testsurface name :                              perfectDiffuseSurface .
    2022-07-03 17:02:01.621 INFO  [36224120] [GPropertyLib::dumpSensorIndices@1088] X4PhysicalVolume::convertSurfaces  NumSensorIndices 1 ( 1  ) 
    2022-07-03 17:02:01.622 INFO  [36224120] [X4PhysicalVolume::convertSolid@951]  lvname Water_lv soname Water_solid [--x4skipsolidname] n
    2022-07-03 17:02:01.632 INFO  [36224120] [X4PhysicalVolume::convertSolid@951]  lvname Air_lv soname Air_solid [--x4skipsolidname] n
    2022-07-03 17:02:01.633 INFO  [36224120] [X4PhysicalVolume::convertSolid@951]  lvname Rock_lv soname Rock_solid [--x4skipsolidname] n
    2022-07-03 17:02:01.634 INFO  [36224120] [X4PhysicalVolume::dumpLV@1215]  m_lvidx.size() 3 m_lvlist.size() 3 edgeitems 100
     i     0 idx     0 lvname                                           Water_lv soname                                        Water_solid
     i     1 idx     1 lvname                                             Air_lv soname                                          Air_solid
     i     2 idx     2 lvname                                            Rock_lv soname                                         Rock_solid
    2022-07-03 17:02:01.635 INFO  [36224120] [X4PhysicalVolume::convertStructure@1325] [ creating large tree of GVolume instances
    2022-07-03 17:02:01.637 INFO  [36224120] [X4PhysicalVolume::postConvert@215]  GGeo::getNumVolumes() 0 GGeo::getNumSensorVolumes() 0
     GGeo::getSensorBoundaryReport() 
                          sensor_total      0

    2022-07-03 17:02:01.637 INFO  [36224120] [X4PhysicalVolume::postConvert@225] GBndLib::getAddBoundaryReport edgeitems 100 num_boundary_add 3
     boundary   0 b+1   1 add_count      1 Rock///Rock
     boundary   1 b+1   2 add_count      1 Rock//air_rock_bs/Air
     boundary   2 b+1   3 add_count      1 Air///Water
                          add_total      3

    2022-07-03 17:02:01.637 INFO  [36224120] [GGeo::prepare@673] [
    2022-07-03 17:02:01.637 INFO  [36224120] [GGeo::prepareVolumes@1379] [ creating merged meshes from the volume tree 
    2022-07-03 17:02:01.637 INFO  [36224120] [GInstancer::dumpDigests@557] before sort
    2022-07-03 17:02:01.637 INFO  [36224120] [GInstancer::dumpDigests@557] after sort
    2022-07-03 17:02:01.638 INFO  [36224120] [GInstancer::findRepeatCandidates@373]  nall 3 repeat_min 400 vertex_min 0 num_repcan 0
    2022-07-03 17:02:01.638 ERROR [36224120] [*GGeoLib::makeMergedMesh@346] mm index   0 geocode   T                  numVolumes          3 numFaces         552 numITransforms           0 numITransforms*numVolumes           0 GParts N GPts Y
    2022-07-03 17:02:01.638 INFO  [36224120] [GInstancer::dump@1032] GGeo::prepareVolumes
    2022-07-03 17:02:01.638 INFO  [36224120] [GInstancer::dumpMeshset@976]  numRepeats 0 numRidx 1 (slot 0 for global non-instanced) 
     ridx 0 ms 3 ( 0 1 2  ) 
    2022-07-03 17:02:01.638 INFO  [36224120] [GInstancer::dumpCSGSkips@1008] 
    2022-07-03 17:02:01.638 INFO  [36224120] [GGeo::prepareVolumes@1413] GNodeLib::descOriginMap m_origin2index.size 3
    2022-07-03 17:02:01.638 INFO  [36224120] [GGeo::prepareVolumes@1414] ]
    2022-07-03 17:02:01.643 INFO  [36224120] [GGeo::prepare@694] ]
    2022-07-03 17:02:01.644 INFO  [36224120] [GGeo::save@717] 
    GGeo::save GGeoLib numMergedMesh 1 ptr 0x7fcf8c4509a0
    mm index   0 geocode   T                  numVolumes          3 numFaces         552 numITransforms           1 numITransforms*numVolumes           3 GParts N GPts Y
     num_remainder_volumes 3 num_instanced_volumes 0 num_remainder_volumes + num_instanced_volumes 3 num_total_faces 552 num_total_faces_woi 552 (woi:without instancing) 
       0 pts Y  GPts.NumPt     3 lvIdx ( 2 1 0) 0 1 2 all_same_count 1

    2022-07-03 17:02:01.672 INFO  [36224120] [GMeshLib::addAltMeshes@133]  num_indices_with_alt 0
    2022-07-03 17:02:01.672 INFO  [36224120] [GMeshLib::dump@279] addAltMeshes meshnames 3 meshes 3
     i   0 aidx   0 midx   0 name                                        Water_solid mesh  nv    267 nf    528
     i   1 aidx   1 midx   1 name                                          Air_solid mesh  nv      8 nf     12
     i   2 aidx   2 midx   2 name                                         Rock_solid mesh  nv      8 nf     12
    2022-07-03 17:02:01.895 FATAL [36224120] [*GScintillatorLib::legacyCreateBuffer@231]  using legacy approach, avoid this by GScintillatorLib::setGeant4InterpolatedICDF  
    2022-07-03 17:02:01.896 INFO  [36224120] [BMeta::dump@202] GGeo::saveCacheMeta
    {
        "GEOCACHE_CODE_VERSION": 15,
        "argline": "G4CXSimulateTest ",
        "cwd": "/Users/blyth/opticks/g4cx",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20220703_170201",
        "runfolder": "G4CXSimulateTest",
        "runlabel": "R0_cvd_",
        "runstamp": 1656864121
    }
    2022-07-03 17:02:01.897 INFO  [36224120] [GParts::add@1369]  --gparts_transform_offset IS ENABLED, COUNT  1 ridx 0 tranOffset 0
    2022-07-03 17:02:01.897 INFO  [36224120] [GParts::add@1369]  --gparts_transform_offset IS ENABLED, COUNT  2 ridx 0 tranOffset 1
    2022-07-03 17:02:01.897 INFO  [36224120] [GParts::add@1369]  --gparts_transform_offset IS ENABLED, COUNT  3 ridx 0 tranOffset 2
    GGeo::reportMeshUsage
     meshIndex, nvert, nface, nodeCount, nodeCount*nvert, nodeCount*nface, meshName, nmm, mm[0] 
         0 ( v  267 f  528 ) :       1 :        267 :        528 :                                        Water_solid :  1 :    0
         1 ( v    8 f   12 ) :       1 :          8 :         12 :                                          Air_solid :  1 :    0
         2 ( v    8 f   12 ) :       1 :          8 :         12 :                                         Rock_solid :  1 :    0
     tot  node :       3 vert :     283 face :     552
    2022-07-03 17:02:01.898 INFO  [36224120] [GGeo::postDirectTranslationDump@648] GGeo::postDirectTranslationDump NOT --dumpsensor numSensorVolumes 0
    2022-07-03 17:02:01.898 ERROR [36224120] [*CSG_GGeo_Convert::Translate@36] [ convert ggeo 
    SName::findIndicesfromNames FAILED to find q [HamamatsuR12860sMask0x]
    Assertion failed: (found), function findIndicesFromNames, file /Users/blyth/opticks/sysrap/SName.h, line 280.
    ./gxs.sh: line 23: 55464 Abort trap: 6           G4CXSimulateTest
    ./gxs.sh run error
    epsilon:g4cx blyth$ 


FIXED : SName::findIndicesFromNames was asserting when names not found 
-------------------------------------------------------------------------

As name checking is used to identify a geometry cannot require to always find the names::

    ./gxs.sh dbg 

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff72d94b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff72f5f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff72cf01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff72cb81ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010ad7f91d libSysRap.dylib`SName::findIndicesFromNames(this=0x000000010b7905d0, idxs=size=0, qq=size=3) const at SName.h:280
        frame #5: 0x000000010ad7f60d libSysRap.dylib`SName::hasNames(this=0x000000010b7905d0, qq=size=3) const at SName.h:300
        frame #6: 0x000000010ad7ef02 libSysRap.dylib`SName::hasNames(this=0x000000010b7905d0, qq_="HamamatsuR12860sMask0x,HamamatsuR12860_PMT_20inch,NNVTMCPPMT_PMT_20inch", delim=',') const at SName.h:295
        frame #7: 0x000000010ad7f41c libSysRap.dylib`SGeoConfig::GeometrySpecificSetup(id=0x000000010b7905d0) at SGeoConfig.cc:170
        frame #8: 0x00000001064e5321 libCSG_GGeo.dylib`CSG_GGeo_Convert::init(this=0x00007ffeefbfe4a8) at CSG_GGeo_Convert.cc:86
        frame #9: 0x00000001064e5191 libCSG_GGeo.dylib`CSG_GGeo_Convert::CSG_GGeo_Convert(this=0x00007ffeefbfe4a8, foundry_=0x000000010b7909f0, ggeo_=0x000000010b7647e0) at CSG_GGeo_Convert.cc:73
        frame #10: 0x00000001064e38a5 libCSG_GGeo.dylib`CSG_GGeo_Convert::CSG_GGeo_Convert(this=0x00007ffeefbfe4a8, foundry_=0x000000010b7909f0, ggeo_=0x000000010b7647e0) at CSG_GGeo_Convert.cc:67
        frame #11: 0x00000001064e342d libCSG_GGeo.dylib`CSG_GGeo_Convert::Translate(ggeo=0x000000010b7647e0) at CSG_GGeo_Convert.cc:37
        frame #12: 0x0000000100142a99 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, gg_=0x000000010b7647e0) at G4CXOpticks.cc:38
        frame #13: 0x0000000100142a68 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, world=0x000000010b7636c0) at G4CXOpticks.cc:33
        frame #14: 0x000000010002c569 G4CXSimulateTest`main(argc=1, argv=0x00007ffeefbfe798) at G4CXSimulateTest.cc:27
        frame #15: 0x00007fff72c44015 libdyld.dylib`start + 1
        frame #16: 0x00007fff72c44015 libdyld.dylib`start + 1
    (lldb) 
    (lldb) f 8
    frame #8: 0x00000001064e5321 libCSG_GGeo.dylib`CSG_GGeo_Convert::init(this=0x00007ffeefbfe4a8) at CSG_GGeo_Convert.cc:86
       83  	    ggeo->getMergedMeshLabels(foundry->mmlabel); 
       84  	    // boundary names now travel with the NP bnd.names 
       85  	
    -> 86  	    SGeoConfig::GeometrySpecificSetup(foundry->id);
       87  	
       88  	    const char* cxskiplv = SGeoConfig::CXSkipLV() ; 
       89  	    const char* cxskiplv_idxlist = SGeoConfig::CXSkipLV_IDXList() ;  
    (lldb) p foundry->id
    (SName *) $0 = 0x000000010b7905d0
    (lldb) 


    (lldb) f 7
    frame #7: 0x000000010ad7f41c libSysRap.dylib`SGeoConfig::GeometrySpecificSetup(id=0x000000010b7905d0) at SGeoConfig.cc:170
       167 	void SGeoConfig::GeometrySpecificSetup(const SName* id)  // static
       168 	{
       169 	    const char* JUNO_names = "HamamatsuR12860sMask0x,HamamatsuR12860_PMT_20inch,NNVTMCPPMT_PMT_20inch" ;  
    -> 170 	    bool JUNO_detected = id->hasNames(JUNO_names); 
       171 	    LOG(info) << " JUNO_detected " << JUNO_detected ; 
       172 	    if(JUNO_detected)
       173 	    {
    (lldb) 

    (lldb) f 5
    frame #5: 0x000000010ad7f60d libSysRap.dylib`SName::hasNames(this=0x000000010b7905d0, qq=size=3) const at SName.h:300
       297 	inline bool SName::hasNames( const std::vector<std::string>& qq ) const 
       298 	{
       299 	    std::vector<unsigned> idxs ; 
    -> 300 	    findIndicesFromNames(idxs, qq); 
       301 	    bool has_all = qq.size() == idxs.size() ; 
       302 	    return has_all ; 
       303 	}
    (lldb) 



FIXED : issue : QScint tripped up by test geometry without any scintillator
---------------------------------------------------------------------------------

::

    lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff72d94b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff72f5f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff72cf01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff72cb81ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010a9b18cf libQUDARap.dylib`QScint::MakeScintTex(src=0x000000010c192b00, hd_factor=20) at QScint.cc:82
        frame #5: 0x000000010a9b17d2 libQUDARap.dylib`QScint::QScint(this=0x000000010c192ad0, icdf=0x000000010b60c790, hd_factor=20) at QScint.cc:39
        frame #6: 0x000000010a9b2063 libQUDARap.dylib`QScint::QScint(this=0x000000010c192ad0, icdf=0x000000010b60c790, hd_factor=20) at QScint.cc:42
        frame #7: 0x000000010a8f83ab libQUDARap.dylib`QSim::UploadComponents(ssim=0x000000010c110d00) at QSim.cc:118
        frame #8: 0x00000001069091d8 libCSGOptiX.dylib`CSGOptiX::InitSim(ssim=0x000000010c110d00) at CSGOptiX.cc:154
        frame #9: 0x00000001069099dc libCSGOptiX.dylib`CSGOptiX::Create(fd=0x000000010c111da0) at CSGOptiX.cc:172
        frame #10: 0x0000000100142ad9 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, fd_=0x000000010c111da0) at G4CXOpticks.cc:44
        frame #11: 0x0000000100142aaa libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, gg_=0x000000010b799c10) at G4CXOpticks.cc:39
        frame #12: 0x0000000100142a68 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, world=0x000000010b798af0) at G4CXOpticks.cc:33
        frame #13: 0x000000010002c569 G4CXSimulateTest`main(argc=1, argv=0x00007ffeefbfe798) at G4CXSimulateTest.cc:27
        frame #14: 0x00007fff72c44015 libdyld.dylib`start + 1
        frame #15: 0x00007fff72c44015 libdyld.dylib`start + 1
    (lldb) 


Need cleaner way to cope with no scint::

    2022-07-03 17:39:14.013 ERROR [16606] [*CSG_GGeo_Convert::Translate@53] ] convert ggeo 
    2022-07-03 17:39:14.014 ERROR [16606] [*CSGFoundry::getOriginCFBase@2027]  CAUTION HOW YOU USE THIS : MISUSE CAN EASILY LEAD TO INCONSISTENCY BETWEEN RESULTS AND GEOMETRY 
    2022-07-03 17:39:14.014 INFO  [16606] [*CSGOptiX::Create@169] fd.descBase CSGFoundry.descBase  CFBase - OriginCFBase -
    fd.descBase 2022-07-03 17:39:14.015 ERROR [16606] [*CSGFoundry::getOriginCFBase@2027]  CAUTION HOW YOU USE THIS : MISUSE CAN EASILY LEAD TO INCONSISTENCY BETWEEN RESULTS AND GEOMETRY 
    CSGFoundry.descBase  CFBase - OriginCFBase -
    2022-07-03 17:39:15.462 FATAL [16606] [*QScint::MakeScintTex@83]  unexpected shape of src (0, 4096, 1, )
    Assertion failed: (expected_shape), function MakeScintTex, file /Users/blyth/opticks/qudarap/QScint.cc, line 85.
    ./gxs.sh: line 23:  2769 Abort trap: 6           G4CXSimulateTest
    ./gxs.sh run error
    epsilon:g4cx blyth$ 


HMM where does SSim come from::

    148 void CSGOptiX::InitSim( const SSim* ssim  )
    149 {
    150     if(SEventConfig::IsRGModeRender()) return ;
    151     if(ssim == nullptr) LOG(fatal) << "simulate/simtrace modes require SSim/QSim setup" ;
    152     assert(ssim);
    153 
    154     QSim::UploadComponents(ssim);
    155 
    156     QSim* sim = new QSim ;
    157     LOG(info) << sim->desc() ;
    158 }

Coming from fd->sim::

    167 CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
    168 {
    169     LOG(info) << "fd.descBase " << ( fd ? fd->descBase() : "-" ) ;
    170     std::cout << "fd.descBase " << ( fd ? fd->descBase() : "-" ) << std::endl ;
    171 
    172     InitSim(fd->sim);
    173     InitGeo(fd);
    174 
    175     CSGOptiX* cx = new CSGOptiX(fd) ;
    176 
    177     QSim* qs = QSim::Get() ;
    178 
    179     qs->setLauncher(cx);
    180 
    181     QEvent* event = qs->event ;
    182     event->setMeta( fd->meta.c_str() );
    183 
    184     // DONE: setup QEvent as SCompProvider of NP arrays allowing SEvt to drive QEvent download
    185     return cx ;
    186 }

::

     35 void G4CXOpticks::setGeometry(const GGeo* gg_)
     36 {
     37     gg = gg_ ;
     38     CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ;
     39     setGeometry(fd_);
     40 }
     41 void G4CXOpticks::setGeometry(CSGFoundry* fd_)
     42 {
     43     fd = fd_ ;
     44     cx = CSGOptiX::Create(fd);
     45     qs = cx->sim ;
     46 }


Find where SSim coming from in translation::

    epsilon:opticks blyth$ opticks-f SSim | grep -v sysrap
    ./CSGOptiX/CSGOptiX.h:struct SSim ; 
    ./CSGOptiX/CSGOptiX.h:    static void InitSim( const SSim* ssim ); 
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:with standard CFBASE basis CSGFoundry/SSim input arrays. 
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:Notice that the standard SSim input arrays are loaded without the corresponding standard geometry
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:using the intentional arms length (SSim subdirectory/NPFold) relationship between CSGFoundry and SSim. 
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:#include "SSim.hh"
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:    const SSim* ssim = SSim::Load() ;  // standard $CFBase/CSGFoundry/SSim
    ./CSGOptiX/tests/CSGOptiXSimulateTest.cc:    fdl->setOverrideSim(ssim);    // local geometry with standard SSim inputs 
    ./CSGOptiX/tests/CXRaindropTest.cc:#include "SSim.hh"
    ./CSGOptiX/tests/CXRaindropTest.cc:    SSim* ssim = SSim::Load();
    ./CSGOptiX/tests/CXRaindropTest.cc:    ssim->save("$CFBASE_LOCAL/CSGFoundry/SSim" ); // DIRTY: FOR PYTHON CONSUMPTION
    ./CSGOptiX/tests/CXRaindropTest.cc:    CSGOptiX* cx = CSGOptiX::Create(fdl); // encumbent SSim used for QSim setup in here 
    ./CSGOptiX/tests/CSGOptiXSimTest.cc:    CSGFoundry* fd = CSGFoundry::Load() ;  // standard OPTICKS_KEY CFBase/CSGFoundry geometry and SSim
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.cc:#include "SSim.hh"
    ./CSGOptiX/CSGOptiX.cc:void CSGOptiX::InitSim( const SSim* ssim  )
    ./CSGOptiX/CSGOptiX.cc:    if(ssim == nullptr) LOG(fatal) << "simulate/simtrace modes require SSim/QSim setup" ;
    ./CSGOptiX/cxsim.sh:cxsim.sh : CSGOptiXSimTest : standard geometry and SSim inputs 
    ./CSG/CSGFoundry.h:struct SSim ; 
    ./CSG/CSGFoundry.h:    void setOverrideSim( const SSim* ssim ); 
    ./CSG/CSGFoundry.h:    const SSim* getSim() const ; 
    ./CSG/CSGFoundry.h:    const SSim* sim ; 
    ./CSG/CSGFoundry.py:class SSim(NPFold):
    ./CSG/CSGFoundry.py:        sim = cls(fold=os.path.join(simbase, "SSim"))  
    ./CSG/CSGFoundry.py:        self.sim = SSim.Load(fold)
    ./CSG/CSGFoundry.cc:#include "SSim.hh"
    ./CSG/CSGFoundry.cc:#include "SSim.hh"
    ./CSG/CSGFoundry.cc:    sim(SSim::Create()),
    ./CSG/CSGFoundry.cc:    mismatch += SSim::Compare( a->sim, b->sim, true ); 
    ./CSG/CSGFoundry.cc:        LOG(fatal) << " SSim::save " << dir ;  
    ./CSG/CSGFoundry.cc:        sim->save(dir, "SSim");  
    ./CSG/CSGFoundry.cc:        LOG(fatal) << " CANNOT SSim::save AS sim null " ;  
    ./CSG/CSGFoundry.cc:    sim = NP::Exists(dir, "SSim") ? SSim::Load(dir, "SSim") : nullptr ; 
    ./CSG/CSGFoundry.cc:    // pass the SSim pointer from the loaded src instance, 
    ./CSG/CSGFoundry.cc:    // overriding the empty dst SSim instance 
    ./CSG/CSGFoundry.cc:void CSGFoundry::setOverrideSim( const SSim* override_sim )
    ./CSG/CSGFoundry.cc:const SSim* CSGFoundry::getSim() const 
    ./ggeo/GGeo.hh:struct SSim ; 
    ./ggeo/GGeo.hh:        void convertSim_BndLib(SSim* sim) const ; 
    ./ggeo/GGeo.hh:        void convertSim_ScintillatorLib(SSim* sim) const ; 
    ./ggeo/GGeo.hh:        void convertSim_Prop(SSim* sim) const ; 
    ./ggeo/GGeo.hh:        void convertSim_MultiFilm(SSim* sim) const ; 
    ./ggeo/GGeo.cc:#include "SSim.hh"
    ./ggeo/GGeo.cc:    SSim* sim = SSim::Get();
    ./ggeo/GGeo.cc:    if(sim == nullptr) LOG(fatal) << "SSim should have been created by CSGFoundry::CSGFoundry " ; 
    ./ggeo/GGeo.cc:void GGeo::convertSim_BndLib(SSim* sim) const 
    ./ggeo/GGeo.cc:        sim->add(SSim::BND, bnd ); 
    ./ggeo/GGeo.cc:        sim->add(SSim::OPTICAL, optical ); 
    ./ggeo/GGeo.cc:void GGeo::convertSim_ScintillatorLib(SSim* sim) const 
    ./ggeo/GGeo.cc:    sim->add(SSim::ICDF, icdf); 
    ./ggeo/GGeo.cc:void GGeo::convertSim_Prop(SSim* sim) const 
    ./ggeo/GGeo.cc:    sim->add(SSim::PROPCOM, propcom); 
    ./ggeo/GGeo.cc:void GGeo::convertSim_MultiFilm(SSim* sim) const 
    ./ggeo/GGeo.cc:        sim->add(SSim::MULTIFILM, multifilm ); 
    ./qudarap/tests/QSimTest.cc:#include "SSim.hh"
    ./qudarap/tests/QSimTest.cc:    SSim* ssim = SSim::Load(); 
    ./qudarap/tests/QPrdTest.cc:    NP* bnd = NP::Load(cfbase, "CSGFoundry/SSim/bnd.npy"); 
    ./qudarap/tests/QSimWithEventTest.cc:#include "SSim.hh"
    ./qudarap/tests/QSimWithEventTest.cc:    const SSim* ssim = SSim::Load(); 
    ./qudarap/tests/QBndTest.cc:#include "SSim.hh"
    ./qudarap/tests/QBndTest.cc:    NP* bnd = NP::Load(cfbase, "CSGFoundry/SSim/bnd.npy"); 
    ./qudarap/tests/QOpticalTest.cc:    bool exists = NP::Exists(cfbase, "CSGFoundry/SSim/optical.npy") ; 
    ./qudarap/tests/QOpticalTest.cc:    NP* optical = exists ? NP::Load(cfbase, "CSGFoundry/SSim/optical.npy") : nullptr ; 
    ./qudarap/QDebug.cc:         << " TO FIX THIS YOU PROBABLY NEED TO RERUN THE GEOMETRY CONVERSION TO UPDATE THE PERSISTED SSim IN CSGFoundry/SSim "
    ./qudarap/QBnd.hh:      as done in SSim::addFake_ 
    ./qudarap/QSim.hh:struct SSim ; 
    ./qudarap/QSim.hh:    static void UploadComponents(const SSim* ssim);   
    ./qudarap/QBnd.cc:#include "SSim.hh"
    ./qudarap/QBnd.cc:    src(SSim::NarrowIfWide(buf)),
    ./qudarap/QSim.cc:#include "SSim.hh"
    ./qudarap/QSim.cc:void QSim::UploadComponents( const SSim* ssim  )
    ./qudarap/QSim.cc:    const NP* optical = ssim->get(SSim::OPTICAL); 
    ./qudarap/QSim.cc:    const NP* bnd = ssim->get(SSim::BND); 
    ./qudarap/QSim.cc:    const NP* propcom = ssim->get(SSim::PROPCOM); 
    ./qudarap/QSim.cc:    const NP* icdf = ssim->get(SSim::ICDF); 
    ./qudarap/QSim.cc:    const NP* multifilm = ssim->get(SSim::MULTIFILM); 
    ./u4/tests/U4RecorderTest.cc:    U4Material::LoadBnd();   // "back" creation of G4 material properties from the Opticks bnd.npy obtained from SSim::Load 
    ./u4/tests/U4MaterialTest.cc:#include "SSim.hh"
    ./u4/U4Material.cc:#include "SSim.hh"
    ./u4/U4Material.cc:Load the material properties from the SSim::get_bnd array using SBnd::getPropertyGroup 
    ./u4/U4Material.cc:    SSim* sim = SSim::Load(); 
    ./g4cx/tests/G4CXSimulateTest.cc:    U4Material::LoadBnd();   // "back" creation of G4 material properties from the Opticks bnd.npy obtained from SSim::Load 
    epsilon:opticks blyth$ 



SSim instanciated by CSGFoundry::CSGFoundry and populated by GGeo::convertSim::

    2334 void GGeo::convertSim() const
    2335 {
    2336     SSim* sim = SSim::Get();
    2337     if(sim == nullptr) LOG(fatal) << "SSim should have been created by CSGFoundry::CSGFoundry " ;
    2338     assert(sim);
    2339 
    2340     convertSim_BndLib(sim);
    2341     convertSim_ScintillatorLib(sim);
    2342     convertSim_Prop(sim);
    2343     convertSim_MultiFilm(sim);
    2344 }
    2345 

::

    2387 void GGeo::convertSim_ScintillatorLib(SSim* sim) const
    2388 {
    2389     GScintillatorLib* slib = getScintillatorLib();
    2390     NP* icdf = slib->getBuf();   // assuming 1 scintillator
    2391     
    2392     LOG(error) << " icdf " << ( icdf ? icdf->sstr() : "-" ) ; 
    2393     
    2394     sim->add(SSim::ICDF, icdf);
    2395 }   

::

    2022-07-03 17:51:25.391 ERROR [121677] [GGeo::convertSim_ScintillatorLib@2392]  icdf (0, 4096, 1, )


::

    198 /**
    199 GScintillatorLib::setGeant4InterpolatedICDF
    200 ---------------------------------------------
    201 
    202 Invoked from X4PhysicalVolume::createScintillatorGeant4InterpolatedICDF
    203 which trumps the ICDF from GScintillatorLib::legacyCreateBuffer
    204 
    205 **/
    206 
    207 void GScintillatorLib::setGeant4InterpolatedICDF( NPY<double>* g4icdf )
    208 {
    209     m_g4icdf = g4icdf ;
    210 }
    211 NPY<double>* GScintillatorLib::getGeant4InterpolatedICDF() const
    212 {
    213     return m_g4icdf ;
    214 }
    215 


     389 void X4PhysicalVolume::createScintillatorGeant4InterpolatedICDF()
     390 {
     391     unsigned num_scint = m_sclib->getNumRawOriginal() ;
     392     if( num_scint == 0 ) return ;
     393     //assert( num_scint == 1 ); 
     394 
     395     typedef GPropertyMap<double> PMAP ;
     396     PMAP* pmap_en = m_sclib->getRawOriginal(0u);
     397     assert( pmap_en );
     398     assert( pmap_en->hasOriginalDomain() );
     399 
     400     NPY<double>* slow_en = pmap_en->getProperty("SLOWCOMPONENT")->makeArray();
     401     NPY<double>* fast_en = pmap_en->getProperty("FASTCOMPONENT")->makeArray();
     402 
     403     //slow_en->save("/tmp/slow_en.npy"); 
     404     //fast_en->save("/tmp/fast_en.npy"); 
     405 
     406     X4Scintillation xs(slow_en, fast_en);
     407 
     408     unsigned num_bins = 4096 ;
     409     unsigned hd_factor = 20 ;
     410     const char* material_name = pmap_en->getName() ;
     411 
     412     NPY<double>* g4icdf = xs.createGeant4InterpolatedInverseCDF(num_bins, hd_factor, material_name ) ;
     413 
     414     LOG(info)
     415         << " num_scint " << num_scint
     416         << " slow_en " << slow_en->getShapeString()
     417         << " fast_en " << fast_en->getShapeString()
     418         << " num_bins " << num_bins
     419         << " hd_factor " << hd_factor
     420         << " material_name " << material_name
     421         << " g4icdf " << g4icdf->getShapeString()
     422         ;
     423 
     424     m_sclib->setGeant4InterpolatedICDF(g4icdf);   // trumps legacyCreateBuffer
     425     m_sclib->close();   // creates and sets "THE" buffer 
     426 }



Avoid the bad icdf shape by returning nullptr when no scintillator::

     
    .NPY<double>* GScintillatorLib::createBuffer()
     {
    -    return m_g4icdf ? m_g4icdf : legacyCreateBuffer() ; 
    +    //return m_g4icdf ? m_g4icdf : legacyCreateBuffer() ; 
    +    return m_g4icdf ; 
     }
     
     GItemList*  GScintillatorLib::createNames()
     {
    -    return m_g4icdf ? geant4ICDFCreateNames() : legacyCreateNames() ;  
    +    //return m_g4icdf ? geant4ICDFCreateNames() : legacyCreateNames() ;  
    +    return m_g4icdf ? geant4ICDFCreateNames() : nullptr ;  
     }



FIXED : issue : QEvent instanciated before SEvt, need to instanciate SEvt in main 
-------------------------------------------------------------------------------------

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff6a490b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff6a65b080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff6a3ec1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff6a3b41ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010a91f017 libQUDARap.dylib`QEvent::init(this=0x000000010b609530) at QEvent.cc:88
        frame #5: 0x000000010a91ee76 libQUDARap.dylib`QEvent::QEvent(this=0x000000010b609530) at QEvent.cc:70
        frame #6: 0x000000010a91f1c5 libQUDARap.dylib`QEvent::QEvent(this=0x000000010b609530) at QEvent.cc:68
        frame #7: 0x000000010a8f87cf libQUDARap.dylib`QSim::QSim(this=0x000000010b609490) at QSim.cc:149
        frame #8: 0x000000010a8f8fe5 libQUDARap.dylib`QSim::QSim(this=0x000000010b609490) at QSim.cc:163
        frame #9: 0x0000000106909200 libCSGOptiX.dylib`CSGOptiX::InitSim(ssim=0x000000010d204490) at CSGOptiX.cc:156
        frame #10: 0x00000001069099dc libCSGOptiX.dylib`CSGOptiX::Create(fd=0x000000010d204a50) at CSGOptiX.cc:172
        frame #11: 0x0000000100142ad9 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, fd_=0x000000010d204a50) at G4CXOpticks.cc:44
        frame #12: 0x0000000100142aaa libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, gg_=0x000000010b794230) at G4CXOpticks.cc:39
        frame #13: 0x0000000100142a68 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x00007ffeefbfe708, world=0x000000010b792e10) at G4CXOpticks.cc:33
        frame #14: 0x000000010002c569 G4CXSimulateTest`main(argc=1, argv=0x00007ffeefbfe798) at G4CXSimulateTest.cc:27
        frame #15: 0x00007fff6a340015 libdyld.dylib`start + 1
        frame #16: 0x00007fff6a340015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x000000010a91f017 libQUDARap.dylib`QEvent::init(this=0x000000010b609530) at QEvent.cc:88
       85  	{
       86  	    if(!sev) LOG(fatal) << "QEvent instanciated before SEvt instanciated : this is not going to fly " ; 
       87  	
    -> 88  	    assert(sev); 
       89  	    assert(evt); 
       90  	    assert(selector); 
       91  	
    (lldb) 



FIXED : issue : no gensteps : need to set OPTICKS_INPUT_PHOTON SEventConfig or envvar
---------------------------------------------------------------------------------------

::

    N[blyth@localhost g4cx]$ ./gxs.sh dbg

    2022-07-04 03:22:42.174 INFO  [284805] [SBT::checkHitgroup@907]  num_sbt (sbt.hitgroupRecordCount) 3 num_solid 1 num_prim 3
    2022-07-04 03:22:42.174 INFO  [284805] [SBT::createGeom@109] ]
    2022-07-04 03:22:42.175 INFO  [284805] [SBT::getAS@584]  spec i0 c i idx 0
    2022-07-04 03:22:42.175 FATAL [284805] [QEvent::setGenstep@151] Must SEvt::AddGenstep before calling QEvent::setGenstep 
    2022-07-04 03:22:42.175 ERROR [284805] [QSim::simulate@228]  QEvent::setGenstep ERROR : no gensteps collected : will skip cx.simulate 


Despite starting from a Geant4 geometry this test needs to follow  
aspects from the cx/cxs_raindrop.sh 

However the setup of input photon running is common to both contexts, being done in SEventConfig + SEvt. 


issue : QEvent null
---------------------

::

    2022-07-04 03:54:20.956 INFO  [301410] [SBT::createGeom@109] ]
    2022-07-04 03:54:20.956 INFO  [301410] [SBT::getAS@584]  spec i0 c i idx 0

    Program received signal SIGSEGV, Segmentation fault.
    0x00007fffec5b2caa in QEvent::setGenstep (this=0x0, gs_=0x211e930) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:158
    158	    gs = gs_ ; 
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffec5b2caa in QEvent::setGenstep (this=0x0, gs_=0x211e930) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:158
    #1  0x00007fffec5b2c52 in QEvent::setGenstep (this=0x0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:153
    #2  0x00007fffec5a3b50 in QSim::simulate (this=0x2156e70) at /data/blyth/junotop/opticks/qudarap/QSim.cc:227
    #3  0x00007ffff7bb71a8 in G4CXOpticks::simulate (this=0x7fffffff5660) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:58
    #4  0x000000000040a393 in main (argc=1, argv=0x7fffffff5b38) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:31
    (gdb) f 4
    #4  0x000000000040a393 in main (argc=1, argv=0x7fffffff5b38) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:31
    31	    gx.simulate(); 
    (gdb) f 3
    #3  0x00007ffff7bb71a8 in G4CXOpticks::simulate (this=0x7fffffff5660) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:58
    58	    qs->simulate(); 
    (gdb) f 2
    #2  0x00007fffec5a3b50 in QSim::simulate (this=0x2156e70) at /data/blyth/junotop/opticks/qudarap/QSim.cc:227
    227	   int rc = event->setGenstep(); 
    (gdb) f 1
    #1  0x00007fffec5b2c52 in QEvent::setGenstep (this=0x0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:153
    153	    return gs == nullptr ? -1 : setGenstep(gs) ; 
    (gdb) 

    (gdb) f 2
    #2  0x00007fffec5a3b50 in QSim::simulate (this=0x2156e70) at /data/blyth/junotop/opticks/qudarap/QSim.cc:227
    227	   int rc = event->setGenstep(); 
    (gdb) p event
    $1 = (QEvent *) 0x0
    (gdb) 


::

    167 CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
    168 {   
    169     LOG(info) << "fd.descBase " << ( fd ? fd->descBase() : "-" ) ;  
    170     std::cout << "fd.descBase " << ( fd ? fd->descBase() : "-" ) << std::endl ;
    171     
    172     InitSim(fd->sim);
    173     InitGeo(fd);
    174     
    175     CSGOptiX* cx = new CSGOptiX(fd) ;
    176     
    177     QSim* qs = QSim::Get() ;
    178     
    179     qs->setLauncher(cx);
    180     
    181     QEvent* event = qs->event ; 
    182     event->setMeta( fd->meta.c_str() );
    183     
    184     // DONE: setup QEvent as SCompProvider of NP arrays allowing SEvt to drive QEvent download
    185     return cx ;
    186 }


Hmm the QEvent was created within QSim, something is stomping on it, or have two QSim instances::

    2022-07-04 16:48:23.244 INFO  [312850] [QSim::UploadComponents@122]  multifilm null 
    2022-07-04 16:48:23.244 INFO  [312850] [QEvent::init@93]  QEvent::init calling SEvt::setCompProvider 
    2022-07-04 16:48:23.244 INFO  [312850] [QSim::QSim@164]  QSim::QSim instanciating QEvent 
    2022-07-04 16:48:23.244 INFO  [312850] [QSim::init@203] QSim sim->rngstate 0x7fffae000000 sim->base0x7fffbdc00000 sim->bnd 0x7fffbdc00800 sim->scint 0 sim->cerenkov 0x7fffbdc01000 sim 0x15a0580 d_sim 0x7fffbdc01400
    2022-07-04 16:48:23.244 INFO  [312850] [QSim::init@204] 

Two QSim, so the second lacks the event::

    2022-07-04 17:33:20.275 INFO  [319886] [CSGOptiX::init@255] ]
    2022-07-04 17:33:20.275 ERROR [319886] [QSim::simulate@227]  event null QSim.hh this 0x2156de0 INSTANCE 0x159fa10 QEvent.hh:event 0 qsim.h:sim 0x2156e00 qsim.h:d_sim 0x159f688 sim->rngstate 0x7fffbdc02000 sim->base 0xd80dc0 sim->bnd 0xd0 sim->scint 0x2156e00 sim->cerenkov 0x5



Suspected culprit id the OPTIX_VERSION switch maybe dealing with the wrong version of CSGOptiX header 
so cx->sim gives garbage pointer ? 

Despite this::

    2022-07-04 21:24:58.658 INFO  [434219] [main@26] G4CXOpticks::Desc CSGOptiX::Desc Version 7 PTXNAME CSGOptiX7 GEO_PTXNAME -

The problem could maifest from a macro not being there when using the header. 

Confirmed, the problem is avoided by adding the Dummy pointers in the OPTIX_VERSION macro branch::

    084 
     85 #if OPTIX_VERSION < 70000
     86     Six* six ;
     87     Dummy* dummy0 ;
     88     Dummy* dummy1 ;
     89 #else
     90     Ctx* ctx ;
     91     PIP* pip ;
     92     SBT* sbt ;
     93 #endif
     94 
     95     Frame* frame ;
     96     SMeta* meta ;
     97     double dt ;
     98 
     99     QSim*        sim ;
    100     QEvent*      event ;
    101 


HMM: this demonstates that having version macros that change members in commonly used headers 
should be avoided, as it then becomes necessary to ensure the same macros are defined
for all uses of that header otherwise get mismatch and wierd bugs from trying to access
some address as wrong pointer type. 

Although CSGOptiX.h does::

   #include <optix.h>

But the OPTIX_VERSION resulting from that depends on the CMake environment 
that b7 cooks up when building CX.

Do not particularly want uses of CX to need to do the same setup.  



shakedown : issue 1 : all getting absorbed : geometry or input photon issue ?
---------------------------------------------------------------------------------

::

    In [1]: t.seq[:,0]
    Out[1]: array([77, 77, 77, 77, 77, ..., 77, 77, 77, 77, 77], dtype=uint64)

    In [2]: np.all( t.seq[:,0] == 77 )
    Out[2]: True

    In [3]: t
    Out[3]: 
    t

    CMDLINE:/Users/blyth/opticks/g4cx/tests/G4CXSimulateTest.py
    t.base:/tmp/blyth/opticks/G4CXSimulateTest

      : t.genstep                                          :            (1, 6, 4) : 0:02:58.141744 
      : t.seed                                             :             (10000,) : 0:02:50.119603 
      : t.seq                                              :           (10000, 2) : 0:02:50.118744 
      : t.record_meta                                      :                    1 : 0:02:50.120041 
      : t.rec_meta                                         :                    1 : 0:02:51.416620 
      : t.rec                                              :    (10000, 10, 2, 4) : 0:02:51.416961 
      : t.NPFold_meta                                      :                    2 : 0:03:03.796718 
      : t.record                                           :    (10000, 10, 4, 4) : 0:02:50.120325 
      : t.domain                                           :            (2, 4, 4) : 0:03:03.796202 
      : t.inphoton                                         :        (10000, 4, 4) : 0:02:54.312348 
      : t.flat                                             :          (10000, 48) : 0:02:58.142259 
      : t.NPFold_index                                     :                   11 : 0:03:03.797301 
      : t.prd                                              :    (10000, 10, 2, 4) : 0:02:51.833685 
      : t.photon                                           :        (10000, 4, 4) : 0:02:52.282276 
      : t.domain_meta                                      :                    2 : 0:03:03.795836 
      : t.tag                                              :           (10000, 4) : 0:02:50.117545 

     min_stamp : 2022-07-04 15:40:05.162773 
     max_stamp : 2022-07-04 15:40:18.842529 
     dif_stamp : 0:00:13.679756 
     age_stamp : 0:02:50.117545 

    In [4]: seqhis_(77)
    Out[4]: 'TO AB'



Looks like input photons starting in Rock and getting AB immediately::

    In [7]: t.record[0,:2]
    Out[7]: 
    array([[[   4.295,    4.959, -990.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.756,   -0.655,    0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   4.295,    4.959, -989.998,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.756,   -0.655,    0.   ,  501.   ],
            [   0.   ,    0.   ,   -0.   ,    0.   ]]], dtype=float32)







