U4_CollectGenstep_G4Cerenkov_modified_needs_access_to_matline_or_need_to_convert_from_materialIndex_later 
============================================================================================================

U4HitGet uses SEvt to access the frame transforms using SEvt::cf  (SGeo*)

Need to do something a bit similar in order to convert Geant4 material index into the matline. 

* stree::lookup_mtline can do the translation but requires SBnd::fillMaterialLine
  to be applied to the stree first which needs bnd specs
  and then stree::init_mtindex_to_mtline for fill the map. 

  * see sysrap/tests/stree_material_test.cc 
  * DONE in SSim::import_bnd

* Who should hold onto stree pointer ? SSim would seem appropriate as it holds bnd. 
  Currently held inside G4CXOpticks/U4Tree : but that is too high a level for 
  the low level generality of stree  

* this suggests that stree should live inside SSim then 
  SGeo/CSGFoundry can vend the SSim:sim/stree functionality  

* DONE : moved SSim::Create into G4CXOpticks::setGeometry to instanciated 
  it sooner at higher level that former place in CSGFoundry::CSGFoundry
  thence SSim::Get is used from CSGFoundry

* in some sense SSim and stree need to switch places : the use of stree at highest 
  level is not really appropriate, but SSim at higest level makes sense

::

     02 /**
      3 SSim.hh : Manages input arrays for QUDARap/QSim : Using Single NPFold Member
      4 ==================================================================================
      5 
      6 Canonically instanciated by CSGFoundry::CSGFoundry 
      7 and populated by GGeo::convertSim which is for example invoked 
      8 during GGeo to CSGFoundry conversion within CSG_GGeo_Convert::convertSim
      9 
     10 Currently the SSim instance is persisted within CSGFoundry/SSim 
     11 using NPFold functionality.  
     12 
     13 The SSim instance provides the input arrays to QSim
     14 which does the uploading to the device. 
     15 
     16 **/
     17 


Considered deferring this translation until a later stage, keeping genstep collection simple. 
But that means would have to edit the genstep later adding more moving parts. 
Other SGeo info is used by U4HitGet via SEvt, seems no reason for the mapping 
not to do something similar. 

::

    85 struct CSG_API CSGFoundry : public SGeo


::

    136 static quad6 MakeGenstep_G4Cerenkov_modified(
    ...
    163     quad6 _gs ;
    164     _gs.zero() ;
    165     
    166     scerenkov& gs = (scerenkov&)_gs ;
    167     
    168     gs.gentype = OpticksGenstep_G4Cerenkov_modified ;
    169     gs.trackid = aTrack->GetTrackID() ;
    170     gs.matline = 1000000 + aMaterial->GetIndex() ;  // gets converted from "1000000 + materialIndex" into "matline" later. Where ? 
    171     gs.numphoton = numPhotons ;
    172     


Interim solution : use the GGeo bnd 
-------------------------------------

* need to reposition stree into SSim to make this convenient 
  and preferably to not need to change when cut out GGeo 

::

    174 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    175 {
    176     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    177     assert(world);
    178     wd = world ;
    179     tr = U4Tree::Create(world, SensorIdentifier ) ;
    180 
    181 #ifdef __APPLE__
    182     return ;
    183 #endif
    184 
    185     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    186     Opticks::Configure("--gparts_transform_offset --allownokey" );
    187 
    188     GGeo* gg_ = X4Geo::Translate(wd) ;
    189     setGeometry(gg_);
    190 }
    191 


New Workflow : needs to create the bnd.npy 
----------------------------------------------

The bnd list has sets of 4 integers. 
The bnd buffer interleaves items from material and surface buffers, 
using standard domain values. 

Collect the G4Material indices together with the names in U4Tree::convertMaterial

U4Tree.h::

    133 inline void U4Tree::convertMaterial(const G4Material* const mt)
    134 {
    135     materials.push_back(mt);
    136     const G4String& mtname = mt->GetName() ;
    137     unsigned mtindex = mt->GetIndex() ;
    138 
    139     st->mtname.push_back(mtname);
    140     st->mtindex.push_back(mtindex);
    141 }

stree.h::

   111     std::vector<std::string> mtname ;       // unique material names
   112     std::vector<int>         mtindex ;      // G4Material::GetIndex 0-based creation indices


HMM: need to reproduce boundary collection


::

    2484 void GGeo::convertSim_BndLib(SSim* sim) const
    2485 {
    2486     LOG(LEVEL) << "[" ;
    2487     GBndLib* blib = getBndLib();
    2488 
    2489     bool can_create = blib->canCreateBuffer() ;
    2490     NP* bnd = nullptr ;
    2491     NP* optical = nullptr ;
    2492 
    2493     if( can_create )
    2494     {
    2495         blib->createDynamicBuffers();
    2496         // hmm perhaps this is done already on loading now ?
    2497         bnd = blib->getBuf();
    2498 
    2499         LOG(LEVEL) << " bnd.desc " << bnd->desc() ;
    2500 
    2501         optical = blib->getOpticalBuf();
    2502 
    2503         const std::vector<std::string>& bndnames = blib->getNameList();
    2504         bnd->set_names( bndnames );
    2505 
    2506         LOG(LEVEL) << " bnd.set_names " << bndnames.size() ;
    2507 
    2508         sim->add(SSim::BND, bnd );
    2509         sim->add(SSim::OPTICAL, optical );
    2510     }
    2511     else
    2512     {
    2513         LOG(error) << "cannot create GBndLib buffer : no materials ? " ;
    2514     }
    2515 }


::

     233 /**
     234 GPropertyLib::getBuf
     235 ----------------------
     236 
     237 Convert NPY into NP with metadata and names passed along 
     238 
     239 **/
     240 
     241 NP* GPropertyLib::getBuf() const
     242 {
     243     NP* buf = m_buffer ? m_buffer->spawn() : nullptr ;
     244     const std::vector<std::string>& names = getNameList();
     245     if(buf && names.size() > 0)
     246     {
     247         buf->set_names(names);
     248     }
     249     return buf ;
     250 }





Old workflow : OpticksGen::setMaterialLine : goes via the string material name for every genstep 
--------------------------------------------------------------------------------------------------

::

    392 void OpticksGen::setMaterialLine( GenstepNPY* gs )
    393 {
    394     if(!m_blib)
    395     {
    396         LOG(warning) << "no blib, skip setting material line " ;
    397         return ;
    398     }
    399    const char* material = gs->getMaterial() ;
    400 
    401    if(material == NULL)
    402       LOG(fatal) << "NULL material from GenstepNPY, probably missed material in torch config" ;
    403    assert(material);
    404 
    405    unsigned int matline = m_blib->getMaterialLine(material);
    406    gs->setMaterialLine(matline);
    407 
    408    LOG(debug) << "OpticksGen::setMaterialLine"
    409               << " material " << material
    410               << " matline " << matline
    411               ;
    412 }


Better to create the materialIndex->matline mapping once (just a num_materials length vector)
and simply do a lookup to do the mapping.  

Trace bnd
------------

Need to review bnd preparation. 

::

     32 qbnd* QBnd::MakeInstance(const QTex<float4>* tex, const std::vector<std::string>& names )
     33 {
     34     qbnd* bnd = new qbnd ;
     35 
     36     bnd->boundary_tex = tex->texObj ;
     37     bnd->boundary_meta = tex->d_meta ;
     38     bnd->boundary_tex_MaterialLine_Water = SBnd::GetMaterialLine("Water", names) ;
     39     bnd->boundary_tex_MaterialLine_LS    = SBnd::GetMaterialLine("LS", names) ;
     40 


::

    260 inline unsigned SBnd::GetMaterialLine( const char* material, const std::vector<std::string>& specs ) // static
    261 {   
    262     unsigned line = MISSING ; 
    263     for(unsigned i=0 ; i < specs.size() ; i++)
    264     {   
    265         std::vector<std::string> elem ; 
    266         SStr::Split(specs[i].c_str(), '/', elem );
    267         const char* omat = elem[0].c_str();
    268         const char* imat = elem[3].c_str();
    269         
    270         if(strcmp( material, omat ) == 0 )
    271         {   
    272             line = i*4 + 0 ;
    273             break ;
    274         }
    275         if(strcmp( material, imat ) == 0 )
    276         {   
    277             line = i*4 + 3 ;
    278             break ;
    279         }
    280     }
    281     return line ;
    282 }




matline
---------

::

    epsilon:g4ok blyth$ opticks-f matline 
    ./opticksgeo/OpticksGen.cc:just need to avoid trying to translate the matline later.
    ./opticksgeo/OpticksGen.cc:   unsigned int matline = m_blib->getMaterialLine(material);
    ./opticksgeo/OpticksGen.cc:   gs->setMaterialLine(matline);  
    ./opticksgeo/OpticksGen.cc:              << " matline " << matline
    ./sysrap/SEvt.hh:index and photon offset in addition to  gentype/trackid/matline/numphotons 
    ./sysrap/scarrier.h:   SCARRIER_METHOD static void FillGenstep( scarrier& gs, unsigned matline, unsigned numphoton_per_genstep ) ; 
    ./sysrap/scarrier.h:inline void scarrier::FillGenstep( scarrier& gs, unsigned matline, unsigned numphoton_per_genstep ) 
    ./sysrap/scerenkov.h:    unsigned matline ;   // formerly MaterialIndex, used by qbnd::boundary_lookup 
    ./sysrap/scerenkov.h:   static void FillGenstep( scerenkov& gs, unsigned matline, unsigned numphoton_per_genstep ) ; 
    ./sysrap/scerenkov.h:* NB matline is crucial as that determines which materials RINDEX is used 
    ./sysrap/scerenkov.h:inline void scerenkov::FillGenstep( scerenkov& gs, unsigned matline, unsigned numphoton_per_genstep )
    ./sysrap/scerenkov.h:    gs.matline = matline ; 
    ./sysrap/storch.h:    unsigned matline ; 
    ./sysrap/storch.h:    printf("//storch::generate photon_id %3d genstep_id %3d  gs gentype/trackid/matline/numphoton(%3d %3d %3d %3d) type %d \n", 
    ./sysrap/storch.h:       gs.matline, 
    ./sysrap/sscint.h:    unsigned matline ; 
    ./sysrap/sscint.h:    gs.matline = 0u ;
    ./qudarap/qcerenkov.h:    //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline ); 
    ./qudarap/qcerenkov.h:        float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u); 
    ./qudarap/QDebug.cc:    unsigned cerenkov_matline = qb ? qb->bnd->boundary_tex_MaterialLine_LS : 0 ;   
    ./qudarap/QDebug.cc:         << "AS NO QBnd at QDebug::MakeInstance the qdebug cerenkov genstep is using default matline of zero " << std::endl 
    ./qudarap/QDebug.cc:         << " cerenkov_matline " << cerenkov_matline  << std::endl
    ./qudarap/QDebug.cc:    scerenkov::FillGenstep( cerenkov_gs, cerenkov_matline, 100 ); 
    ./u4/U4.cc:    gs.matline = 0u ; //  aMaterial->GetIndex()   // not used for scintillation
    ./u4/U4.cc:    gs.matline = 1000000 + aMaterial->GetIndex() ;  // gets converted from "1000000 + materialIndex" into "matline" later. Where ? 
    epsilon:opticks blyth$ 



G4Material::GetIndex
------------------------

g4-cls G4Material::

    261   //the index of this material in the Table:    
    262   inline size_t GetIndex() const {return fIndexInTable;}


g4-cls G4MaterialTable::

     41 #include <vector>
     42 
     43 class G4Material;
     44 
     45 typedef std::vector<G4Material*> G4MaterialTable;
     46 


G4Material::fIndexInTable is 0-based material creation index
----------------------------------------------------------------

::

    094 G4Material::G4Material(const G4String& name, G4double z,
     95                        G4double a, G4double density,
     96                        G4State state, G4double temp, G4double pressure)
     97   : fName(name)
     98 {
     99   InitializePointers();
    100 

    258 void G4Material::InitializePointers()
    259 {
    ...
    288   // Store in the static Table of Materials
    289   fIndexInTable = theMaterialTable.size();
    290   for(size_t i=0; i<fIndexInTable; ++i) {
    291     if(theMaterialTable[i]->GetName() == fName) {
    292       G4cout << "G4Material WARNING: duplicate name of material "
    293          << fName << G4endl;
    294       break;
    295     } 
    296   } 
    297   theMaterialTable.push_back(this);
    298 } 


