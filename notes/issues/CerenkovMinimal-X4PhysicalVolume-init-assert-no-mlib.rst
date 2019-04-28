CerenkovMinimal-X4PhysicalVolume-init-assert-no-mlib
======================================================


Overview
-------------

Theres a difference between a key under creation and one referring to a pre-existing cache
that GGeo::init was not distinguishing. So added a live argument to GGeo that (for now) 
ignores preexisting cache directory for GGeo population from G4Opticks::translateGeometry.

This avoids the assert.

This shows up now following my resource booting rearrangement that forces 
Opticks::configure earlier. 



Issue : G4Opticks translateGeometry assert from lack of GGeo libs
----------------------------------------------------------------------


::

    -- Set runtime path of "/home/blyth/local/opticks/lib/CerenkovMinimal" to "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/optix/lib64"
    /home/blyth/local/opticks/lib/CerenkovMinimal

            ############################################
            !!! WARNING - FPE detection is activated !!!
            ############################################

    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02    (25-May-2018)
                           Copyright : Geant4 Collaboration
                          References : NIM A 506 (2003), 250-303
                                     : IEEE-TNS 53 (2006), 270-278
                                     : NIM A 835 (2016), 186-225
                                 WWW : http://geant4.org/
    **************************************************************

    DetectorConstruction::Construct DONE 
    PhysicsList<T>::ConstructOp AddDiscreteProcess to OpticalPhoton 


    ###[ RunAction::BeginOfRunAction G4Opticks.setGeometry


    2019-04-28 15:07:41.106 INFO  [40780] [G4Opticks::G4Opticks@114] ctor : DISABLE FPE detection : as it breaks OptiX launches

            C4FPEDetection::InvalidOperationDetection_Disable
            ############################################
            !!! WARNING - FPE detection is DISABLED  !!!
            ############################################

    2019-04-28 15:07:41.107 FATAL [40780] [G4Opticks::setGeometry@121] [[[
    2019-04-28 15:07:41.107 FATAL [40780] [G4Opticks::setGeometry@124] ( translateGeometry 
    2019-04-28 15:07:41.107 ERROR [40780] [G4Opticks::translateGeometry@160] SetKey CerenkovMinimal.X4PhysicalVolume.World.792496b5e2cc08bdf5258cc12e63de9f
    2019-04-28 15:07:41.107 INFO  [40780] [G4Opticks::translateGeometry@166] EmbeddedCommandLine : [ --gltf 3 --compute --save --embedded --natural --printenabled --pindex 0 ]
    2019-04-28 15:07:41.107 INFO  [40780] [G4Opticks::translateGeometry@168] ( Opticks
    2019-04-28 15:07:41.110 INFO  [40780] [BOpticksResource::init@131] layout : 0
    2019-04-28 15:07:41.111 INFO  [40780] [BOpticksResource::setupViaKey@495] 
                 BOpticksKey  : KEYSOURCE
          spec (OPTICKS_KEY)  : CerenkovMinimal.X4PhysicalVolume.World.792496b5e2cc08bdf5258cc12e63de9f
                     exename  : CerenkovMinimal
             current_exename  : CerenkovMinimal
                       class  : X4PhysicalVolume
                     volname  : World
                      digest  : 792496b5e2cc08bdf5258cc12e63de9f
                      idname  : CerenkovMinimal_World_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2019-04-28 15:07:41.111 INFO  [40780] [BOpticksResource::setupViaKey@533]  idname CerenkovMinimal_World_g4live idfile g4ok.gltf srcdigest 792496b5e2cc08bdf5258cc12e63de9f idpath /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1
    2019-04-28 15:07:41.112 ERROR [40780] [OpticksResource::initRunResultsDir@262] /tmp/blyth/opticks/results/CerenkovMinimal/runlabel/20190428_150741
    2019-04-28 15:07:41.112 INFO  [40780] [G4Opticks::translateGeometry@174] ) Opticks /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1
    2019-04-28 15:07:41.112 INFO  [40780] [G4Opticks::translateGeometry@177] ( CGDML
    2019-04-28 15:07:41.112 INFO  [40780] [CGDML::Export@46] export to /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/g4ok.gdml
    G4GDML: Writing '/home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/g4ok.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/g4ok.gdml' done !
    2019-04-28 15:07:41.118 INFO  [40780] [G4Opticks::translateGeometry@179] ) CGDML
    2019-04-28 15:07:41.118 INFO  [40780] [G4Opticks::translateGeometry@181] ( GGeo instanciate
    2019-04-28 15:07:41.118 INFO  [40780] [G4Opticks::translateGeometry@183] ) GGeo instanciate 
    2019-04-28 15:07:41.118 INFO  [40780] [G4Opticks::translateGeometry@185] ( GGeo populate
    CerenkovMinimal: /home/blyth/opticks/extg4/X4PhysicalVolume.cc:118: X4PhysicalVolume::X4PhysicalVolume(GGeo*, const G4VPhysicalVolume*): Assertion `m_mlib && msg' failed.
    ./go.sh: line 43: 40780 Aborted                 (core dumped) $exe
    [blyth@localhost CerenkovMinimal]$ 
    [blyth@localhost CerenkovMinimal]$ 




The resource rearrangement forced Opticks::configure earlier to avoid idpath assert

::

    096 X4PhysicalVolume::X4PhysicalVolume(GGeo* ggeo, const G4VPhysicalVolume* const top)
     97     :
     98     X4Named("X4PhysicalVolume"),
     99     m_ggeo(ggeo),
    100     m_top(top),
    101     m_ok(m_ggeo->getOpticks()),
    102     m_lvsdname(m_ok->getLVSDName()),
    103     m_query(m_ok->getQuery()),
    104     m_gltfpath(m_ok->getGLTFPath()),
    105     m_g4codegen(m_ok->isG4CodeGen()),
    106     m_g4codegendir(m_ok->getG4CodeGenDir()),
    107     m_mlib(m_ggeo->getMaterialLib()),
    108     m_slib(m_ggeo->getSurfaceLib()),
    109     m_blib(m_ggeo->getBndLib()),
    110     m_xform(new nxform<X4Nd>(0,false)),
    111     m_verbosity(m_ok->getVerbosity()),
    112     m_node_count(0),
    113     m_selected_node_count(0)
    114 {
    115     const char* msg = "GGeo ctor argument of X4PhysicalVolume must have mlib, slib and blib already " ;
    116 
    117     // trying to Opticks::configure earlier, from Opticks::init trips these asserts
    118     assert( m_mlib && msg );
    119     assert( m_slib && msg );
    120     assert( m_blib && msg );
    121 
    122     init();
    123 }
    124 


Hmm, the libs cannot contain anything at this juncture...

::

    156 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    157 {
    158     LOG(verbose) << "( key" ;
    159     const char* keyspec = X4PhysicalVolume::Key(top) ;
    160     LOG(error) << "SetKey " << keyspec  ;
    161     BOpticksKey::SetKey(keyspec);
    162     LOG(verbose) << ") key" ;
    163 
    164     const char* g4opticks_debug = SSys::getenvvar("G4OPTICKS_DEBUG") ;
    165     std::string ecl = EmbeddedCommandLine(g4opticks_debug) ;
    166     LOG(info) << "EmbeddedCommandLine : [" << ecl << "]" ;
    167 
    168     LOG(info) << "( Opticks" ;
    169     Opticks* ok = new Opticks(0,0, ecl.c_str() );  // Opticks instanciation must be after BOpticksKey::SetKey
    170     ok->configure();
    171 
    172     const char* idpath = ok->getIdPath();
    173     assert(idpath);
    174     LOG(info) << ") Opticks " << idpath ;
    175 
    176     const char* gdmlpath = ok->getGDMLPath();   // inside geocache, not SrcGDMLPath from opticksdata
    177     LOG(info) << "( CGDML" ;
    178     CGDML::Export( gdmlpath, top );
    179     LOG(info) << ") CGDML" ;
    180 
    181     LOG(info) << "( GGeo instanciate" ;
    182     GGeo* gg = new GGeo(ok) ;
    183     LOG(info) << ") GGeo instanciate " ;
    184 
    185     LOG(info) << "( GGeo populate" ;
    186     X4PhysicalVolume xtop(gg, top) ;
    187     LOG(info) << ") GGeo populate" ;
    188 
    189     LOG(info) << "( GGeo::postDirectTranslation " ;
    190     gg->postDirectTranslation();
    191     LOG(info) << ") GGeo::postDirectTranslation " ;
    192 
    193     int root = 0 ;
    194     const char* gltfpath = ok->getGLTFPath();   // inside geocache
    195     LOG(info) << "( gltf " ;
    196     GGeoGLTF::Save(gg, gltfpath, root );
    197     LOG(info) << ") gltf " ;
    198 
    199     return gg ;
    200 }



Probably sees the cache due to OPTICKS_KEY being in env::

     413 void GGeo::init()
     414 {
     415    const char* idpath = m_ok->getIdPath() ;
     416    LOG(verbose)
     417          << " idpath " << ( idpath ? idpath : "NULL" )
     418          ;
     419 
     420    assert(idpath && "GGeo::init idpath is required" );
     421 
     422    fs::path geocache(idpath);
     423    bool cache_exists = fs::exists(geocache) && fs::is_directory(geocache) ;
     424    bool cache_requested = m_ok->isGeocache() ;
     425 
     426    m_loaded = cache_exists && cache_requested ;
     427 
     428    LOG(error)
     429         << " idpath " << idpath
     430         << " cache_exists " << cache_exists
     431         << " cache_requested " << cache_requested
     432         << " m_loaded " << m_loaded
     433         ;
     434 
     435    if(m_loaded) return ;
     436 
     437    //////////////  below only when operating pre-cache //////////////////////////
     438 
     439    m_bndlib = new GBndLib(m_ok);
     440    m_materiallib = new GMaterialLib(m_ok);
     441    m_surfacelib  = new GSurfaceLib(m_ok);
     442 
     443    m_bndlib->setMaterialLib(m_materiallib);
     444    m_bndlib->setSurfaceLib(m_surfacelib);
     445 
     446    // NB this m_analytic is always false
     447    //    the analytic versions of these libs are born in GScene
     448    assert( m_analytic == false );
     449    bool testgeo = false ;
     450 
     451    m_meshlib = new GMeshLib(m_ok, m_analytic);
     452    m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
     453    m_nodelib = new GNodeLib(m_ok, m_analytic, testgeo );
     454 
     455    m_instancer = new GInstancer(m_ok, m_geolib, m_nodelib, m_ok->getSceneConfig() ) ;



