SOpticksKey_in_new_workflow
==============================

Having to instanciate Opticks is a hangover from the old workflow
that will have to coped with until the geometry translation is revisted. 
As Opticks still needed by the translation from Geant4->GGeo

::

    epsilon:g4cx blyth$ BP=SOpticksKey::SOpticksKey ./gxs.sh dbg 
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                       TMP_GEOMDIR : /tmp/blyth/opticks/J000 
                           GEOMDIR : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo 

                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON_.sh
                              GEOM : J000
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_FRAME : NNVT:0:1000
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_LABEL : DownXZ1000
                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON.sh 
                         ScriptDir : ./../bin 
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy 
        OPTICKS_INPUT_PHOTON_FRAME : NNVT:0:1000 
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy 

    /Applications/Xcode/Xcode_10_1.app/Contents/Developer/usr/bin/lldb -f G4CXSimulateTest -o "b SOpticksKey::SOpticksKey" -o b --
    (lldb) target create "/usr/local/opticks/lib/G4CXSimulateTest"
    Current executable set to '/usr/local/opticks/lib/G4CXSimulateTest' (x86_64).
    (lldb) b SOpticksKey::SOpticksKey
    Breakpoint 1: 2 locations.
    (lldb) b
    Current breakpoints:
    1: name = 'SOpticksKey::SOpticksKey', locations = 2
      1.1: where = libSysRap.dylib`SOpticksKey::SOpticksKey(char const*) + 20 at SOpticksKey.cc:153, address = libSysRap.dylib[0x00000000001312f4], unresolved, hit count = 0 
      1.2: where = libSysRap.dylib`SOpticksKey::SOpticksKey(char const*) + 32 at SOpticksKey.cc:141, address = libSysRap.dylib[0x0000000000132ba0], unresolved, hit count = 0 

    (lldb) r
    Process 3737 launched: '/usr/local/opticks/lib/G4CXSimulateTest' (x86_64)
    PLOG::EnvLevel adjusting loglevel by envvar   key SEvt level INFO fallback DEBUG
    SCF::Create BUT no CFBASE envvar 
    PLOG::EnvLevel adjusting loglevel by envvar   key U4VolumeMaker level INFO fallback DEBUG
    SCF::Create BUT no CFBASE envvar 
    PLOG::EnvLevel adjusting loglevel by envvar   key CSGFoundry level INFO fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key G4CXOpticks level INFO fallback DEBUG
    Process 3737 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x000000010ae1b2f4 libSysRap.dylib`SOpticksKey::SOpticksKey(this=0x000000010c0565f0, spec="OKX4Test.X4PhysicalVolume.lWorld0x5780b30_PV.5303cd587554cb16682990189831ae83") at SOpticksKey.cc:153
       150 	    m_layout( LAYOUT ),
       151 	    m_current_exename( SAr::Instance ? SAr::Instance->exename() : "OpticksEmbedded" ), 
       152 	    m_live(false)
    -> 153 	{
       154 	    std::vector<std::string> elem ; 
       155 	    SStr::Split(spec, '.', elem ); 
       156 	
    Target 0: (G4CXSimulateTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x000000010ae1b2f4 libSysRap.dylib`SOpticksKey::SOpticksKey(this=0x000000010c0565f0, spec="OKX4Test.X4PhysicalVolume.lWorld0x5780b30_PV.5303cd587554cb16682990189831ae83") at SOpticksKey.cc:153
        frame #1: 0x000000010ae1b0ad libSysRap.dylib`SOpticksKey::SetKey(spec="OKX4Test.X4PhysicalVolume.lWorld0x5780b30_PV.5303cd587554cb16682990189831ae83") at SOpticksKey.cc:100
        frame #2: 0x0000000109a9e05a libOpticksCore.dylib`Opticks::envkey(this=0x000000010c055890) at Opticks.cc:337
        frame #3: 0x0000000109a9f002 libOpticksCore.dylib`Opticks::Opticks(this=0x000000010c055890, argc=1, argv=0x00007ffeefbfe5b0, argforced="--gparts_transform_offset") at Opticks.cc:370
        frame #4: 0x0000000109a9de0b libOpticksCore.dylib`Opticks::Opticks(this=0x000000010c055890, argc=1, argv=0x00007ffeefbfe5b0, argforced="--gparts_transform_offset") at Opticks.cc:426
        frame #5: 0x0000000109a9e82e libOpticksCore.dylib`Opticks::Configure(argc=1, argv=0x00007ffeefbfe5b0, argforced="--gparts_transform_offset") at Opticks.cc:352
        frame #6: 0x000000010003115d G4CXSimulateTest`main(argc=1, argv=0x00007ffeefbfe5b0) at G4CXSimulateTest.cc:22
        frame #7: 0x00007fff583b7015 libdyld.dylib`start + 1
    (lldb) 




::

    0323 bool Opticks::envkey()
     324 {
     325     LOG(LEVEL);
     326     bool allownokey = isAllowNoKey();
     327     if(allownokey)
     328     {
     329         LOG(fatal) << " --allownokey option prevents key checking : this is for debugging of geocache creation " ;
     330         return false ;
     331     }
     332 
     333     bool key_is_set(false) ;
     334     key_is_set = SOpticksKey::IsSet() ;
     335     if(key_is_set) return true ;
     336 
     337     SOpticksKey::SetKey(NULL) ;  // use keyspec from OPTICKS_KEY envvar 
     338 
     339     key_is_set = SOpticksKey::IsSet() ;
     340     assert( key_is_set == true && "valid geocache and key are required, for operation without geocache use --allownokey " );
     341 
     342     return key_is_set ;
     343 }


    0357 Opticks::Opticks(int argc, char** argv, const char* argforced )
     358     :
     359     m_log(new SLog("Opticks::Opticks","",debug)),
     360     m_ok(this),
     361     m_sargs(new SArgs(argc, argv, argforced)),
     362     m_geo(nullptr),
     363     m_argc(m_sargs->argc),
     364     m_argv(m_sargs->argv),
     365     m_lastarg(m_argc > 1 ? strdup(m_argv[m_argc-1]) : NULL),
     366     m_mode(new OpticksMode(this)),
     367     m_composition(new Composition(this)),
     368     m_dumpenv(m_sargs->hasArg("--dumpenv")),
     369     m_allownokey(m_sargs->hasArg("--allownokey")),
     370     m_envkey(envkey()),


    1234 /**
    1235 Opticks::isAllowNoKey
    1236 -----------------------
    1237 
    1238 As this is needed prior to configure it directly uses
    1239 the bool set early in instanciation.
    1240 
    1241 **/
    1242 
    1243 bool Opticks::isAllowNoKey() const   // --allownokey
    1244 {
    1245     return m_allownokey ;
    1246 }
    1247 


Commenting the setting of OPTICKS_KEY::

    N[blyth@localhost g4cx]$ vi ~/.opticks_key
    N[blyth@localhost g4cx]$ 
    N[blyth@localhost g4cx]$ ./gxs.sh 
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                       TMP_GEOMDIR : /tmp/blyth/opticks/J000 
                           GEOMDIR : /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo 

                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON_.sh
                              GEOM : J000
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_FRAME : NNVT:0:1000
      OPTICKS_INPUT_PHOTON_ABSPATH : /home/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_LABEL : DownXZ1000
                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON.sh 
                         ScriptDir : ./../bin 
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy 
        OPTICKS_INPUT_PHOTON_FRAME : NNVT:0:1000 
      OPTICKS_INPUT_PHOTON_ABSPATH : /home/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy 

    PLOG::EnvLevel adjusting loglevel by envvar   key SOpticksKey level INFO fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key SEvt level INFO fallback DEBUG
    SCF::Create BUT no CFBASE envvar 
    PLOG::EnvLevel adjusting loglevel by envvar   key CSGFoundry level INFO fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key U4VolumeMaker level INFO fallback DEBUG
    SCF::Create BUT no CFBASE envvar 
    PLOG::EnvLevel adjusting loglevel by envvar   key G4CXOpticks level INFO fallback DEBUG
    2022-07-24 21:51:04.391 INFO  [177425] [SOpticksKey::SetKey@95] from OPTICKS_KEY envvar (null)
    2022-07-24 21:51:04.392 INFO  [177425] [SOpticksKey::SetKey@98]  spec (null)
    G4CXSimulateTest: /data/blyth/junotop/opticks/optickscore/Opticks.cc:340: bool Opticks::envkey(): Assertion `key_is_set == true && "valid geocache and key are required, for operation without geocache use --allownokey "' failed.
    ./gxs.sh: line 103: 177425 Aborted                 (core dumped) $bin
    ./gxs.sh run G4CXSimulateTest error
    N[blyth@localhost g4cx]$ 





Can "--allownokey" be argforced ? Not without tidying up U4Material::LoadBnd
------------------------------------------------------------------------------

::

     19 int main(int argc, char** argv)
     20 {
     21     OPTICKS_LOG(argc, argv);
     22     Opticks::Configure(argc, argv, "--gparts_transform_offset --allownokey" );
     23 
     24     U4Material::LoadBnd();
     25     // TODO: relocate inside G4CXOpticks::setGeometry
     26     // create G4 materials from SSim::Load bnd.npy, used by U4VolumeMaker::PV PMTSim
     27     // HMM: probably dont want to do this when running from GDML
     28 

::

    N[blyth@localhost g4cx]$ ./gxs.sh dbg
    ...
    2022-07-24 21:54:00.663 FATAL [177759] [Opticks::envkey@329]  --allownokey option prevents key checking : this is for debugging of geocache creation 
    2022-07-24 21:54:00.670 INFO  [177759] [SOpticksKey::SetKey@95] from OPTICKS_KEY envvar (null)
    2022-07-24 21:54:00.671 INFO  [177759] [SOpticksKey::SetKey@98]  spec (null)
    2022-07-24 21:54:00.674 FATAL [177759] [BOpticksResource::initViaKey@711]  m_key is nullptr : early exit 
    2022-07-24 21:54:00.681 FATAL [177759] [OpticksResource::init@122]  CAUTION : are allowing no key 
    2022-07-24 21:54:00.681 FATAL [177759] [Opticks::initResource@1032]  idpath NULL 
    2022-07-24 21:54:00.682 FATAL [177759] [Opticks::defineEventSpec@3075]  resource_pfx (null) config_pfx (null) pfx default_pfx cat (null) udet g4live typ TORCH tag 1
    2022-07-24 21:54:00.683 INFO  [177759] [SOpticksKey::SetKey@95] from OPTICKS_KEY envvar (null)
    2022-07-24 21:54:00.683 INFO  [177759] [SOpticksKey::SetKey@98]  spec (null)
    G4CXSimulateTest: /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:186: static const char* SOpticksResource::CGDir_(bool, const char*): Assertion `idpath' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffebb10387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007fffebb10387 in raise () from /lib64/libc.so.6
    #1  0x00007fffebb11a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffebb091a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffebb09252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffed7238b8 in SOpticksResource::CGDir_ (setkey=true, rel=0x7fffed7d8124 "CSG_GGeo") at /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:186
    #5  0x00007fffed72386d in SOpticksResource::CGDir (setkey=true) at /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:182
    #6  0x00007fffed723930 in SOpticksResource::CFBase () at /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:223
    #7  0x00007fffed723d3c in SOpticksResource::Get (key=0x7713c1 "CFBase") at /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:383
    #8  0x00007fffed6d8ef3 in SPath::Resolve (spec_=0x7fffed7dfe11 "$CFBase/CSGFoundry/SSim", create_dirs=2) at /data/blyth/junotop/opticks/sysrap/SPath.cc:179
    #9  0x00007fffed7483ef in SSim::Load (base_=0x0) at /data/blyth/junotop/opticks/sysrap/SSim.cc:45
    #10 0x00007ffff79325b4 in U4Material::LoadBnd (ssimdir=0x0) at /data/blyth/junotop/opticks/u4/U4Material.cc:693
    #11 0x000000000040f4e2 in main (argc=3, argv=0x7fffffff64f8) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:24
    (gdb) 


::

    ./gxs.sh   ## running from GDML with J001

    ...

    pdig 4c29bcd2a52a397de5036b415af92efe ndig    504 nprog    129 placements    504 n pPanel_0_f_0x71b2850
    G4CXSimulateTest: /data/blyth/junotop/opticks/ggeo/GGeo.cc:707: void GGeo::save(): Assertion `idpath' failed.
    ./gxs.sh: line 103: 181076 Aborted                 (core dumped) $bin
    ./gxs.sh run G4CXSimulateTest error
    N[blyth@localhost g4cx]$ 
    N[blyth@localhost g4cx]$ 


     pdig 4c29bcd2a52a397de5036b415af92efe ndig    504 nprog    129 placements    504 n pPanel_0_f_0x71b2850
    G4CXSimulateTest: /data/blyth/junotop/opticks/ggeo/GGeo.cc:707: void GGeo::save(): Assertion `idpath' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffebb10387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007fffebb10387 in raise () from /lib64/libc.so.6
    #1  0x00007fffebb11a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffebb091a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffebb09252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff01abf1e in GGeo::save (this=0x7fc7d0) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:707
    #5  0x00007ffff01ab22c in GGeo::postDirectTranslation (this=0x7fc7d0) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:606
    #6  0x00007ffff764a7ae in X4Geo::Translate (top=0x7fc2d0) at /data/blyth/junotop/opticks/extg4/X4Geo.cc:29
    #7  0x00007ffff7bae690 in G4CXOpticks::setGeometry (this=0x7fffffff5b60, world=0x7fc2d0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:124
    #8  0x00007ffff7bae2de in G4CXOpticks::setGeometry (this=0x7fffffff5b60) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:101
    #9  0x000000000040f4b3 in main (argc=3, argv=0x7fffffff6198) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:36
    (gdb) 

    (gdb) f 9
    #9  0x000000000040f4b3 in main (argc=3, argv=0x7fffffff6198) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:36
    36	    gx.setGeometry(); 
    (gdb) f 8
    #8  0x00007ffff7bae2de in G4CXOpticks::setGeometry (this=0x7fffffff5b60) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:101
    101	        setGeometry( U4VolumeMaker::PV() );  // this may load GDML using U4VolumeMaker::PVG if "GEOM"_GDMLPath is defined   
    (gdb) f 7
    #7  0x00007ffff7bae690 in G4CXOpticks::setGeometry (this=0x7fffffff5b60, world=0x7fc2d0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:124
    124	    GGeo* gg_ = X4Geo::Translate(wd) ; 
    (gdb) f 6
    #6  0x00007ffff764a7ae in X4Geo::Translate (top=0x7fc2d0) at /data/blyth/junotop/opticks/extg4/X4Geo.cc:29
    29	    gg->postDirectTranslation();    
    (gdb) f 5
    #5  0x00007ffff01ab22c in GGeo::postDirectTranslation (this=0x7fc7d0) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:606
    606	    save();
    (gdb) f 4
    #4  0x00007ffff01abf1e in GGeo::save (this=0x7fc7d0) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:707
    707	    assert( idpath ); 
    (gdb) p idpath
    $1 = 0x0
    (gdb) 

::

     594 void GGeo::postDirectTranslation()
     595 {
     596     LOG(LEVEL) << "[" ;
     597 
     598     prepare();     // instances are formed here     
     599 
     600     LOG(LEVEL) << "( GBndLib::fillMaterialLineMap " ;
     601     GBndLib* blib = getBndLib();
     602     blib->fillMaterialLineMap();
     603     LOG(LEVEL) << ") GBndLib::fillMaterialLineMap " ;
     604 
     605     LOG(LEVEL) << "( GGeo::save " ;
     606     save();
     607     LOG(LEVEL) << ") GGeo::save " ;
     608 
     609 
     610     deferred();
     611 
     612     postDirectTranslationDump();
     613 
     614     LOG(LEVEL) << "]" ;
     615 }

::

    0704 void GGeo::save()
     705 {
     706     const char* idpath = m_ok->getIdPath() ;
     707     assert( idpath );
     708     LOG(LEVEL) << "[" << " idpath " << idpath ;
     709 
     710     if(!m_prepared)
     711     {
     712         LOG(info) << "preparing before save " ;
     713         prepare();
     714     }
     715 
     716     std::string s = m_geolib->summary("GGeo::save");
     717     LOG(LEVEL) << std::endl << s ;
     718 
     719     LOG(LEVEL) << " before saves " ;
     720 
     721     LOG(LEVEL) << " m_geolib.save " ;
     722     m_geolib->save(); // in here GGeoLib::saveConstituents invokes the save of both triangulated GMergedMesh and analytic GParts 
     723     LOG(LEVEL) << " m_meshlib.save " ;
     724     m_meshlib->save();
     725     LOG(LEVEL) << " m_nodelib.save " ;
     726     m_nodelib->save();
     727     LOG(LEVEL) << " m_materiallib.save " ;
     728     m_materiallib->save();
     729     LOG(LEVEL) << " m_surfacelib.save " ;
     730     m_surfacelib->save();
     731     LOG(LEVEL) << " m_scintillatorlib.save " ;
     732     m_scintillatorlib->save();
     733     LOG(LEVEL) << " m_sourcelib.save " ;
     734     m_sourcelib->save();
     735     LOG(LEVEL) << " m_bndlib.save " ;
     736     m_bndlib->save();
     737 
     738     LOG(LEVEL) << " after saves " ;
     739 
     740     saveCacheMeta();
     741 
     742     LOG(LEVEL) << "]" ;
     743 }


    164 void GGeoLib::save()
    165 {
    166     const char* idpath = m_ok->getIdPath() ;
    167     saveConstituents(idpath);
    168 }
    169 



::

    2022-07-24 23:31:41.341 INFO  [195245] [CSGFoundry::addInstance@1522]  ins_idx 48473 gas_idx 9 ias_idx 0
    2022-07-24 23:31:41.341 INFO  [195245] [CSGFoundry::addInstance@1522]  ins_idx 48474 gas_idx 9 ias_idx 0
    2022-07-24 23:31:41.341 INFO  [195245] [CSGFoundry::addInstance@1522]  ins_idx 48475 gas_idx 9 ias_idx 0
    2022-07-24 23:31:41.341 INFO  [195245] [CSGFoundry::addInstance@1522]  ins_idx 48476 gas_idx 9 ias_idx 0
    2022-07-24 23:31:41.387 INFO  [195245] [SOpticksKey::SetKey@95] from OPTICKS_KEY envvar (null)
    2022-07-24 23:31:41.387 INFO  [195245] [SOpticksKey::SetKey@98]  spec (null)
    NP::load Failed to load from path /tmp/blyth/opticks/GScintillatorLib/LS_ori/RINDEX.npy
    2022-07-24 23:31:41.387 INFO  [195245] [G4CXOpticks::setGeometry@136]  fd_ 0x14ec97700
    G4CXSimulateTest: /data/blyth/junotop/opticks/qudarap/QCerenkov.cc:75: static qcerenkov* QCerenkov::MakeInstance(): Assertion `prop' failed.
    ./gxs.sh: line 103: 195245 Aborted                 (core dumped) $bin
    ./gxs.sh run G4CXSimulateTest error
    N[blyth@localhost g4cx]$ ./gxs.sh dbg
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                       TMP_GEOMDIR : /tmp/blyth/opticks/J001 


::

    134 void G4CXOpticks::setGeometry(CSGFoundry* fd_)
    135 {
    136     LOG(LEVEL) << " fd_ " << fd_ ;
    137 #ifdef __APPLE__
    138     return ;
    139 #endif
    140     fd = fd_ ;
    141     cx = CSGOptiX::Create(fd);   // uploads geometry to GPU 
    142     qs = cx->sim ;
    143     LOG(LEVEL)  << " cx " << cx << " qs " << qs << " QSim::Get " << QSim::Get() ;
    144 }

    188 CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
    189 {
    190     LOG(LEVEL) << "fd.descBase " << ( fd ? fd->descBase() : "-" ) ;
    191 
    192     InitSim(fd->sim);
    193     InitGeo(fd);
    194 
    195     CSGOptiX* cx = new CSGOptiX(fd) ;
    196 
    197     bool render_mode = SEventConfig::IsRGModeRender() ;
    198     if(render_mode == false)
    199     {
    200         QSim* qs = QSim::Get() ;
    201         qs->setLauncher(cx);
    202         QEvent* event = qs->event ;
    203         event->setMeta( fd->meta.c_str() );
    204     }
    205 
    206     return cx ;
    207 }

    168 void CSGOptiX::InitSim( const SSim* ssim  )
    169 {
    170     if(SEventConfig::IsRGModeRender()) return ;
    171 
    172     if(ssim == nullptr) LOG(fatal) << "simulate/simtrace modes require SSim/QSim setup" ;
    173     assert(ssim);
    174 
    175     QSim::UploadComponents(ssim);
    176 
    177     QSim* qs = QSim::Create() ;
    178     LOG(LEVEL) << qs->desc() ;
    179 }



::

    2022-07-24 23:34:37.499 INFO  [197744] [CSGFoundry::addInstance@1522]  ins_idx 48474 gas_idx 9 ias_idx 0
    2022-07-24 23:34:37.499 INFO  [197744] [CSGFoundry::addInstance@1522]  ins_idx 48475 gas_idx 9 ias_idx 0
    2022-07-24 23:34:37.499 INFO  [197744] [CSGFoundry::addInstance@1522]  ins_idx 48476 gas_idx 9 ias_idx 0
    2022-07-24 23:34:37.545 INFO  [197744] [SOpticksKey::SetKey@95] from OPTICKS_KEY envvar (null)
    2022-07-24 23:34:37.545 INFO  [197744] [SOpticksKey::SetKey@98]  spec (null)
    NP::load Failed to load from path /tmp/blyth/opticks/GScintillatorLib/LS_ori/RINDEX.npy
    2022-07-24 23:34:37.545 INFO  [197744] [G4CXOpticks::setGeometry@136]  fd_ 0x14e951700
    Missing separate debuginfo for /lib64/libcuda.so.1

    (gdb) bt
    #0  0x00007fffebb0f387 in raise () from /lib64/libc.so.6
    #1  0x00007fffebb10a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffebb081a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffebb08252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffedb31587 in QCerenkov::MakeInstance () at /data/blyth/junotop/opticks/qudarap/QCerenkov.cc:75
    #5  0x00007fffedb316c0 in QCerenkov::QCerenkov (this=0x151a58e50) at /data/blyth/junotop/opticks/qudarap/QCerenkov.cc:120
    #6  0x00007fffedad17e6 in QSim::UploadComponents (ssim=0x32ae900) at /data/blyth/junotop/opticks/qudarap/QSim.cc:127
    #7  0x00007fffefcc2705 in CSGOptiX::InitSim (ssim=0x32ae900) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:175
    #8  0x00007fffefcc297a in CSGOptiX::Create (fd=0x14e951700) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:192
    #9  0x00007ffff7bae8e0 in G4CXOpticks::setGeometry (this=0x7fffffff5a30, fd_=0x14e951700) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:141
    #10 0x00007ffff7bae7c0 in G4CXOpticks::setGeometry (this=0x7fffffff5a30, gg_=0x7fc7d0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:132
    #11 0x00007ffff7bae6aa in G4CXOpticks::setGeometry (this=0x7fffffff5a30, world=0x7fc2d0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:125
    #12 0x00007ffff7bae2de in G4CXOpticks::setGeometry (this=0x7fffffff5a30) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:101
    #13 0x000000000040f4b3 in main (argc=3, argv=0x7fffffff6068) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:36
    (gdb) 


    066 qcerenkov* QCerenkov::MakeInstance() // static 
     67 {
     68     const QBase* base = QBase::Get();
     69     assert( base );
     70 
     71     const QBnd* bnd = QBnd::Get();
     72     assert( bnd );
     73 
     74     const QProp<float>* prop = QProp<float>::Get();
     75     assert(prop);
     76 
     77     qcerenkov* cerenkov= new qcerenkov ;
     78     cerenkov->base = base->d_base ;
     79     cerenkov->bnd = bnd->d_bnd ;
     80     cerenkov->prop = prop->d_prop ;
     81     return cerenkov ;
     82 }


::

    pdig 4c29bcd2a52a397de5036b415af92efe ndig    504 nprog    129 placements    504 n pPanel_0_f_0x71b2850
    2022-07-24 23:52:36.349 ERROR [199551] [GGeo::save@718] cannot save as no idpath set
    2022-07-24 23:52:37.264 INFO  [199551] [G4CXOpticks::setGeometry@129] 
    2022-07-24 23:52:37.266 INFO  [199551] [CSGFoundry::setMeta@127]                      : -
    2022-07-24 23:52:37.266 INFO  [199551] [CSGFoundry::setMeta@127]                 HOME : /home/blyth
    2022-07-24 23:52:37.266 INFO  [199551] [CSGFoundry::setMeta@127]                 USER : blyth
    2022-07-24 23:52:37.266 INFO  [199551] [CSGFoundry::setMeta@127]               SCRIPT : -
    2022-07-24 23:52:37.266 INFO  [199551] [CSGFoundry::setMeta@127]                  PWD : /data/blyth/junotop/opticks/g4cx
    2022-07-24 23:52:37.266 INFO  [199551] [CSGFoundry::setMeta@127]              CMDLINE : -
    2022-07-24 23:52:37.333 ERROR [199551] [CSG_GGeo_Convert::convertSolid@314]  cxskip 1 meshIdx 117 mname NNVTMCPPMTsMask_virtual0x5f5f0e0
    2022-07-24 23:52:37.337 ERROR [199551] [CSG_GGeo_Convert::convertSolid@314]  cxskip 1 meshIdx 110 mname HamamatsuR12860sMask_virtual0x5f50520
    2022-07-24 23:52:37.337 ERROR [199551] [CSG_GGeo_Convert::convertSolid@314]  cxskip 1 meshIdx 134 mname mask_PMT_20inch_vetosMask_virtual0x5f62620

    Breakpoint 1, SOpticksKey::SetKey (spec=0x0) at /data/blyth/junotop/opticks/sysrap/SOpticksKey.cc:84
    84	    if(SOpticksKey::IsSet())
    (gdb) bt
    #0  SOpticksKey::SetKey (spec=0x0) at /data/blyth/junotop/opticks/sysrap/SOpticksKey.cc:84
    #1  0x00007fffed7227e3 in SOpticksResource::IDPath (setkey=true) at /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:167
    #2  0x00007fffed722d17 in SOpticksResource::Get (key=0x2c9e661 "IDPath") at /data/blyth/junotop/opticks/sysrap/SOpticksResource.cc:382
    #3  0x00007fffed6d7ef3 in SPath::Resolve (spec_=0x7ffff01e7920 "$IDPath/GScintillatorLib/LS_ori/RINDEX.npy", create_dirs=0) at /data/blyth/junotop/opticks/sysrap/SPath.cc:179
    #4  0x00007fffed74dda5 in SProp::MockupCombination (path_=0x7ffff01e7920 "$IDPath/GScintillatorLib/LS_ori/RINDEX.npy") at /data/blyth/junotop/opticks/sysrap/SProp.cc:31
    #5  0x00007ffff01b4ecc in GGeo::convertSim_Prop (this=0x7fc810, sim=0x32aa070) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:2412
    #6  0x00007ffff01b482c in GGeo::convertSim (this=0x7fc810) at /data/blyth/junotop/opticks/ggeo/GGeo.cc:2355
    #7  0x00007ffff0469734 in CSG_GGeo_Convert::convertSim (this=0x7fffffff42b0) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:182
    #8  0x00007ffff0468b48 in CSG_GGeo_Convert::convert (this=0x7fffffff42b0) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:105
    #9  0x00007ffff04683b8 in CSG_GGeo_Convert::Translate (ggeo=0x7fc810) at /data/blyth/junotop/opticks/CSG_GGeo/CSG_GGeo_Convert.cc:39
    #10 0x00007ffff7bae7a6 in G4CXOpticks::setGeometry (this=0x7fffffff57a0, gg_=0x7fc810) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:131
    #11 0x00007ffff7bae6aa in G4CXOpticks::setGeometry (this=0x7fffffff57a0, world=0x7fc310) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:125
    #12 0x00007ffff7bae2de in G4CXOpticks::setGeometry (this=0x7fffffff57a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:101
    #13 0x000000000040f4b3 in main (argc=3, argv=0x7fffffff5dd8) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimulateTest.cc:36
    (gdb) 


Hmm looks like GGeo->CF is assuming a saved IDPath. 

::

    2347 void GGeo::convertSim() const
    2348 {
    2349     SSim* sim = SSim::Get();
    2350     if(sim == nullptr) LOG(fatal) << "SSim should have been created by CSGFoundry::CSGFoundry " ;
    2351     assert(sim);
    2352 
    2353     convertSim_BndLib(sim);
    2354     convertSim_ScintillatorLib(sim);
    2355     convertSim_Prop(sim);
    2356     convertSim_MultiFilm(sim);
    2357 }


::

    2418 void GGeo::convertSim_Prop(SSim* sim) const
    2419 {
    2420     const char* idpath = m_ok->getIdPath() ;
    2421     const char* ri_prop = "$IDPath/GScintillatorLib/LS_ori/RINDEX.npy" ;
    2422     if( idpath == nullptr )
    2423     {
    2424         LOG(error) << " SSim cannot add ri_prop as no idpath " << ri_prop ;
    2425     }
    2426     else
    2427     {
    2428         const NP* propcom = SProp::MockupCombination(ri_prop);
    2429         sim->add(SSim::PROPCOM, propcom);
    2430     }
    2431 }








