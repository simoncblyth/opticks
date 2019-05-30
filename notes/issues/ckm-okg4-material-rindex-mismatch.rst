ckm-okg4-material-rindex-mismatch
======================================

ckm-okg4
-----------

::

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural
    }



Issue : material energy range persisted in the genstep mismatches that read from the G4 material, tripping an assert
------------------------------------------------------------------------------------------------------------------------

* this is due to an inconsistent application of standardization in the two executables 

  * TODO: review the material info flow in both cases, to decide where to standardize 

::

    199 
    200     G4MaterialPropertyVector* Rindex = GetRINDEX(materialIndex) ;  // NB straight G4, no range standardization
    201 
    202     G4double Pmin2 = Rindex->GetMinLowEdgeEnergy();
    203     G4double Pmax2 = Rindex->GetMaxLowEdgeEnergy();
    204     G4double dp2 = Pmax2 - Pmin2;
    205 
    206     G4double epsilon = 1e-6 ;
    207     bool Pmin_match = std::abs( Pmin2 - Pmin ) < epsilon ;
    208     bool Pmax_match = std::abs( Pmax2 - Pmax ) < epsilon ;
    209 
    210     if(!Pmin_match || !Pmax_match)
    211         LOG(fatal)
    212             << " Pmin " << Pmin
    213             << " Pmin2 (MinLowEdgeEnergy) " << Pmin2
    214             << " dif " << std::abs( Pmin2 - Pmin )
    215             << " epsilon " << epsilon
    216             << " Pmin(nm) " << h_Planck*c_light/Pmin/nm
    217             << " Pmin2(nm) " << h_Planck*c_light/Pmin2/nm
    218             ;
    219 
    220     if(!Pmax_match || !Pmin_match)
    221         LOG(fatal)
    222             << " Pmax " << Pmax
    223             << " Pmax2 (MaxLowEdgeEnergy) " << Pmax2
    224             << " dif " << std::abs( Pmax2 - Pmax )
    225             << " epsilon " << epsilon
    226             << " Pmax(nm) " << h_Planck*c_light/Pmax/nm
    227             << " Pmax2(nm) " << h_Planck*c_light/Pmax2/nm



OKG4Test
------------

::

    [blyth@localhost issues]$ DEBUG=1 ckm-okg4

    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/OKG4Test --compute --envkey --embedded --save --natural


    2019-05-29 22:31:56.672 INFO  [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@135]  genstep_idx 0 num_gs 1 materialLine 7 materialIndex 1      post  0.000   0.000   0.000   0.000 

    2019-05-29 22:31:56.672 INFO  [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@168]  From Genstep :  Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 preVelocity 276.074 postVelocity 273.253
    2019-05-29 22:31:56.672 ERROR [195702] [CCerenkovGenerator::GetRINDEX@73]  aMaterial 0x9d5310 aMaterial.Name Water materialIndex 1 num_material 3 Rindex 0x9d6930 Rindex2 0x9d6930
    2019-05-29 22:31:56.672 FATAL [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@211]  Pmin 1.512e-06 Pmin2 (MinLowEdgeEnergy) 2.034e-06 dif 5.21998e-07 epsilon 1e-06 Pmin(nm) 820 Pmin2(nm) 609.558
    2019-05-29 22:31:56.672 FATAL [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@221]  Pmax 2.0664e-05 Pmax2 (MaxLowEdgeEnergy) 4.136e-06 dif 1.6528e-05 epsilon 1e-06 Pmax(nm) 60 Pmax2(nm) 299.768
    OKG4Test: /home/blyth/opticks/cfg4/CCerenkovGenerator.cc:234: static G4VParticleChange* CCerenkovGenerator::GeneratePhotonsFromGenstep(const OpticksGenstep*, unsigned int): Assertion `Pmax_match && "material mismatches genstep source material"' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe2038207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2038207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20398f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2031026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20310d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefd6d182 in CCerenkovGenerator::GeneratePhotonsFromGenstep (gs=0x8f2470, idx=0) at /home/blyth/opticks/cfg4/CCerenkovGenerator.cc:234
    #5  0x00007fffefdf2000 in CGenstepSource::generatePhotonsFromOneGenstep (this=0x933c80) at /home/blyth/opticks/cfg4/CGenstepSource.cc:94
    #6  0x00007fffefdf1f19 in CGenstepSource::GeneratePrimaryVertex (this=0x933c80, event=0x21636b0) at /home/blyth/opticks/cfg4/CGenstepSource.cc:70
    #7  0x00007fffefdc5940 in CPrimaryGeneratorAction::GeneratePrimaries (this=0x8f35f0, event=0x21636b0) at /home/blyth/opticks/cfg4/CPrimaryGeneratorAction.cc:15
    #8  0x00007fffec6b5ba7 in G4RunManager::GenerateEvent (this=0x706cd0, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:460
    #9  0x00007fffec6b563c in G4RunManager::ProcessOneEvent (this=0x706cd0, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:398
    #10 0x00007fffec6b54d7 in G4RunManager::DoEventLoop (this=0x706cd0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #11 0x00007fffec6b4d2d in G4RunManager::BeamOn (this=0x706cd0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #12 0x00007fffefdeec4f in CG4::propagate (this=0x7003b0) at /home/blyth/opticks/cfg4/CG4.cc:331
    #13 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffd760) at /home/blyth/opticks/okg4/OKG4Mgr.cc:144
    #14 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffd760) at /home/blyth/opticks/okg4/OKG4Mgr.cc:84
    #15 0x00000000004039a7 in main (argc=6, argv=0x7fffffffda98) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 

    (gdb) f 5
    #5  0x00007fffefdf2000 in CGenstepSource::generatePhotonsFromOneGenstep (this=0x933c80) at /home/blyth/opticks/cfg4/CGenstepSource.cc:94
    94          case CERENKOV:      pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(m_gs,m_idx) ; break ; 
    (gdb) l
    89      unsigned gencode = m_gs->getGencode(m_idx) ; 
    90      G4VParticleChange* pc = NULL ; 
    91  
    92      switch( gencode )
    93      { 
    94          case CERENKOV:      pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(m_gs,m_idx) ; break ; 
    95          case SCINTILLATION: pc = NULL                                                       ; break ;  
    96          default:            pc = NULL ; 
    97      }
    98  
    (gdb) 

    (gdb) l
    229 
    230     bool with_key = Opticks::HasKey() ; 
    231     if(with_key)
    232     {
    233         assert( Pmin_match && "material mismatches genstep source material" ); 
    234         assert( Pmax_match && "material mismatches genstep source material" ); 
    235     }
    236     else
    237     {
    238         LOG(warning) << "permissive generation for legacy gensteps " ;
    (gdb) 

    (gdb) p Pmin2
    $1 = 2.0339999999999999e-06
    (gdb) p Pmin
    $2 = 1.5120023135750671e-06
    (gdb) p Pmax2
    $3 = 4.1359999999999999e-06
    (gdb) p Pmax
    $4 = 2.0664030671468936e-05
    (gdb) 


Review OKG4Test which is just OKG4Mgr instanciatiom, propagate, visualize
----------------------------------------------------------------------------

::

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural
    }


OKG4Mgr::OKG4Mgr as used by OKG4test 
------------------------------------------
    
m_hub(OpticksHub)
    loads geometry from geocache into GGeo 
    
m_g4(CG4)
    when "--load" option is NOT used (TODO:change "--load" to "--loadevent" ) 
    geometry is loaded from GDML into Geant4 model by  
    The .gdml file was persisted into geocache at its creation. 

m_viz(OpticksViz)
    when "--compute" option is NOT used instanciate from m_hub    


Note frailty of having two sources of geometry here. I recall previous
matching activity where I avoided this by creating the Geant4 geometry 
from the Opticks one : but I think that was just for simple test geometries. 

Of course the geocache was created from the same initial source Geant4 geometry,
but still there are more layers of code.


Perhaps a more direct way...
-------------------------------

Hmm do I need OKX4Mgr ?  To encapsulate whats done in OKX4Test and make it reusable.
That starts from GDML uses G4GDMLParser to get G4VPhysicalVolume 
does the direct X4 conversion to populate a GGeo, persists to cache and 
then uses OKMgr to pop the geometry up to GPU for propagation.

This OKX4Test direct way is intended to be the same as what G4Opticks::TranslateGeometry is doing.

* BUT do not want to complicate the CerenkovMinimal or other example with 
  the CFG4 gorilla instrumentation : hence the desire to split that into 2nd executable


Back translation from Opticks to Geant4 geometry ? 
-----------------------------------------------------

* too much effort (and not needed) to do fully, but code that back translates materials 
  already exists (CMaterial/CMaterialLib/CMaterialBridge?) and can avoid the mismatch 
  problem

* having two sets of geometry, means two sets of materials but they are in different lingo 


Compare Material Information Flow with various executables
-------------------------------------------------------------

Direct : OKX4Test
~~~~~~~~~~~~~~~~~~

* read from GDML into G4Material instances
* convertd by X4PhysicalVolume::convertMaterials X4MaterialTable::Convert
  populating m_mlib(GMaterialLib) in GGeo 

* conversion adds both standardized and as-is materials into GMaterialLib

::

     55 void X4MaterialTable::init()
     56 {
     57     unsigned nmat = G4Material::GetNumberOfMaterials();
     ...
     61     for(unsigned i=0 ; i < nmat ; i++)
     62     {   
     63         G4Material* material = Get(i) ; 
     64         G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
     ...
     76         GMaterial* mat = X4Material::Convert( material );
     ...
     80         m_mlib->add(mat) ;    // creates standardized material
     81         m_mlib->addRaw(mat) ; // stores as-is
     82     }
     83 }



Within OKMgr:

* m_hub(OpticksHub) instanciation adopts GGeo
* m_propagator(OKPropagator) m_engine(OpEngine) m_scene(OScene) instanciation 

* GBndLib::load can trigger interpolation of properties with "--finebndtex" option
* GBndLib::createBufferForTex2d does memcpy zip of material and surface properties 
* OBndLib instanciation uploads properties into GPU texture


Direct : CerenkovMinimal via G4Opticks::TranslateGeometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* RunAction::BeginOfRunAction 

::

     23     bool standardize_geant4_materials = true ;   // required for alignment 
     24     G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );


* X4PhysicalVolume instanciation invokes X4PhysicalVolume::convertMaterials, just like above 



G4Opticks::standardizeGeant4MaterialProperties
-----------------------------------------------

Invoked by G4Opticks::setGeometry when argument requests.

Standardize G4 material properties to use the Opticks standard domain, 
this works by replacing existing Geant4 MPT 
with one converted from the Opticks property map, which are 
standardized on material collection.


X4MaterialLib::Standardize
----------------------------

* requires: both Geant4 G4MaterialTable and Opticks GMaterialLib 

* must be same number/names/order of the materials from both 

* for Geant4 materials with MPT (G4MaterialPropertiesTable) replaces it
  with an MPT converted from the Opticks GMaterial property map

* "Standardize" not a good name, its more "AdoptOpticksMaterialProperties"
  
   * BUT on the other hand it does standardize, because Opticks standardizes 
     materials to common wavelength domain when they are added to the GMaterialLib

* this is currently invoked ONLY BY G4Opticks::TranslateGeometry



Add standardizarion in CGDMLDetector::init
------------------------------------------------


::

    -    addMPT();
    +    addMPTLegacyGDML(); 
    +    standardizeGeant4MaterialProperties();
     


    +void CGDMLDetector::standardizeGeant4MaterialProperties()   
    +{
    +    LOG(info) << "[" ;
    +    X4MaterialLib::Standardize() ;
    +    LOG(info) << "]" ;
    +}




Material Ordering difference
-----------------------------

Formerly the material order was a user input, thats not appropriate in direct workflow. 

Where does the order get changed ? 





::

    ckm-okg4
    ...

    2019-05-30 14:04:54.697 INFO  [319555] [CGDMLDetector::init@69] parse /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml
    G4GDML: Reading '/home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml' done!
    2019-05-30 14:04:54.785 INFO  [319555] [CMaterialSort::dump@37] after size : 3
    2019-05-30 14:04:54.785 INFO  [319555] [CMaterialSort::dump@41]  i   0 name Glass
    2019-05-30 14:04:54.785 INFO  [319555] [CMaterialSort::dump@41]  i   1 name Water
    2019-05-30 14:04:54.785 INFO  [319555] [CMaterialSort::dump@41]  i   2 name Air
    2019-05-30 14:04:54.785 INFO  [319555] [CDetector::setTop@94] .
    2019-05-30 14:04:54.785 INFO  [319555] [CTraverser::Summary@106] CDetector::traverse numMaterials 3 numMaterialsWithoutMPT 0
    2019-05-30 14:04:54.785 ERROR [319555] [CGDMLDetector::addMPTLegacyGDML@144]  Looks like GDML has succeded to load material MPTs   nmat 3 nmat_without_mpt 0 skipping the fixup 
    2019-05-30 14:04:54.785 INFO  [319555] [CGDMLDetector::standardizeGeant4MaterialProperties@209] [
    2019-05-30 14:04:54.786 FATAL [319555] [X4MaterialLib::init@77]  MATERIAL NAME MISMATCH  index 0 pmap_name Air m4_name Glass
    OKG4Test: /home/blyth/opticks/extg4/X4MaterialLib.cc:84: void X4MaterialLib::init(): Assertion `name_match' failed.

    Program received signal SIGABRT, Aborted.



ckm : DetectorConstruction::Construct creation of G4 Materials in order : Air, Water, Glass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Geant4 reversed materials order after taking a trip thru the GDML ?

::

    171 G4VPhysicalVolume* DetectorConstruction::Construct()
    172 {
    173     G4Material* air = MakeAir();
    174     G4Box* so_0 = new G4Box("World",1000.,1000.,1000.);
    175     G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,air,"World",0,0,0);
    176 
    177     G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0 ,"World",0,false,0);
    178 
    179     G4Material* water = MakeWater();
    180     G4Box* so_1 = new G4Box("Obj",500.,500.,500.);
    181     G4LogicalVolume* lv_1 = new G4LogicalVolume(so_1,water,"Obj",0,0,0);
    182     G4VPhysicalVolume* pv_1 = new G4PVPlacement(0,G4ThreeVector(),lv_1 ,"Obj",lv_0,false,0);
    183     assert( pv_1 );
    184 
    185     G4Material* glass = MakeGlass();    // slab of sensitive glass in the water 
    186     AddProperty(glass, "EFFICIENCY", MakeConstantProperty(0.5));
    187 



Opticks order in geocache matches the creation order::

    [blyth@localhost ~]$ kcd
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1
    rundate
    20190529_220906
    runstamp
    1559138946
    argline
     /home/blyth/local/opticks/lib/CerenkovMinimal
    runlabel
    R0_cvd_
    runfolder
    CerenkovMinimal

    [blyth@localhost 1]$ cat GItemList/GMaterialLib.txt 
    Air
    Water
    Glass
        


Fix Geant4 GDML loaded material ordering using the Opticks order
--------------------------------------------------------------------

::

     void CGDMLDetector::sortMaterials()
     {
         GMaterialLib* mlib = getGMaterialLib();     
    -    const std::map<std::string, unsigned>& order = mlib->getOrder(); 
    +
    +    //const std::map<std::string, unsigned>& order = mlib->getOrder();  
    +    //  old order was from preferences
    +
    +    std::map<std::string, unsigned> order ;  
    +    mlib->getCurrentOrder(order); 
    +    // new world order, just use the current Opticks material order : which should correspond to Geant4 creation order
    +    // unlike following a trip thru  GDML that reverses the material order 
    +    // see notes/issues/ckm-okg4-material-rindex-mismatch.rst
    + 
         CMaterialSort msort(order);  
     }
     



::

    2019-05-30 15:02:51.385 INFO  [422891] [CGDMLDetector::CGDMLDetector@42] [
    2019-05-30 15:02:51.385 INFO  [422891] [CGDMLDetector::init@69] parse /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml
    G4GDML: Reading '/home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml' done!
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dumpOrder@26] order from ctor argument
     v     0 k                            Air
     v     2 k                          Glass
     v     1 k                          Water
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@37] before size : 3 G4 materials from G4Material::GetMaterialTable 
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@41]  i   0 name Glass
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@41]  i   1 name Water
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@41]  i   2 name Air
    2019-05-30 15:02:51.395 FATAL [422891] [CMaterialSort::sort@55]  sorting G4MaterialTable using order kv 3
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@37] after size : 3 G4 materials from G4Material::GetMaterialTable 
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@41]  i   0 name Air
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@41]  i   1 name Water
    2019-05-30 15:02:51.395 INFO  [422891] [CMaterialSort::dump@41]  i   2 name Glass
    2019-05-30 15:02:51.395 INFO  [422891] [CDetector::setTop@94] .
    2019-05-30 15:02:51.395 INFO  [422891] [CTraverser::Summary@106] CDetector::traverse numMaterials 3 numMaterialsWithoutMPT 0
    2019-05-30 15:02:51.395 ERROR [422891] [CGDMLDetector::addMPTLegacyGDML@153]  Looks like GDML has succeded to load material MPTs   nmat 3 nmat_without_mpt 0 skipping the fixup 
    2019-05-30 15:02:51.395 INFO  [422891] [CGDMLDetector::standardizeGeant4MaterialProperties@218] [
    2019-05-30 15:02:51.396 INFO  [422891] [CGDMLDetector::standardizeGeant4MaterialProperties@220] ]

















