review_optical_surface_handling_with_view_to_qpmt_special_surfaces
======================================================================


How to do the equivalent of the below on GPU ? 
--------------------------------------------------

C4OpBoundaryProcess.cc::

     502             //[OpticalSurface.mpt.CustomPrefix
     503             if( OpticalSurfaceName0 == '@' || OpticalSurfaceName0 == '#' )  // only customize specially named OpticalSurfaces 
     504             {
     505                 if( m_custom_art->local_z(aTrack) < 0. ) // lower hemi : No customization, standard boundary  
     506                 {   
     507                     m_custom_status = 'Z' ;
     508                 }
     509                 else if( OpticalSurfaceName0 == '@') // upper hemi with name starting @ : ART transmit thru into PMT
     510                 {   
     511                     m_custom_status = 'Y' ;
     512                     
     513                     m_custom_art->doIt(aTrack, aStep) ;
     ...
     527                     type = dielectric_dielectric ;
     528                     theModel = glisur ;
     529                     theFinish = polished ; // to make Rindex2 get picked up below, plus use theGlobalNormal as theFacetNormal 
     534                 }
     535                 else if( OpticalSurfaceName0 == '#' ) // upper hemi with name starting # : Traditional PHC Detection 
     536                 {
     537                     m_custom_status = '-' ;
     538 
     539                     type = dielectric_metal ;
     540                     theModel = glisur ;
     541                     theReflectivity = 0. ;
     542                     theTransmittance = 0. ;
     543                     theEfficiency = 1. ;
     544                 }
     545             }



where to handle special boundary/surface enum ? not at boundary level 
-----------------------------------------------------------------------

Seems not appropriate at boundary level U4Tree::initNodes_r, 
because are dealing with G4LogicalSurface not OpticalSurface there.::

    350     const G4Material* const imat_ = lv->GetMaterial() ;
    351     const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 
    352 
    353     const G4LogicalSurface* const osur_ = U4Surface::Find( pv_p, pv );
    354     const G4LogicalSurface* const isur_ = U4Surface::Find( pv  , pv_p );
    355 
    356     int imat = GetPointerIndex<G4Material>(materials, imat_);
    357     int omat = GetPointerIndex<G4Material>(materials, omat_);
    358     int isur = GetPointerIndex<G4LogicalSurface>(surfaces, isur_);
    359     int osur = GetPointerIndex<G4LogicalSurface>(surfaces, osur_);
    360 
    361     int4 bd = {omat, osur, isur, imat } ;
    362     bool new_boundary = GetValueIndex<int4>( st->bd, bd ) == -1 ;
    363     if(new_boundary)
    364     {
    365         st->bd.push_back(bd) ;
    366         std::string bdn = getBoundaryName(bd,'/') ;
    367         st->bdname.push_back(bdn.c_str()) ;
    368         // HMM: better to use higher level stree::add_boundary if can get names at stree level 
    369     }
    370     int boundary = GetValueIndex<int4>( st->bd, bd ) ;
    371     assert( boundary > -1 );

    // the boundary is incorporated into the snode


where to handle special boundary/surface enum ? Has to be surface level
--------------------------------------------------------------------------

::

    235 inline void U4Tree::initSurfaces()
    236 {
    237     U4Surface::Collect(surfaces);
    238     st->surface = U4Surface::MakeFold(surfaces);
    239 
    240     for(unsigned i=0 ; i < surfaces.size() ; i++)
    241     {
    242         const G4LogicalSurface* s = surfaces[i] ;
    243         const G4String& _sn = s->GetName() ;
    244         const char* sn = _sn.c_str();
    245         st->add_surface( sn, i );
    246     }
    247 }

::

    264 inline NPFold* U4Surface::MakeFold(const std::vector<const G4LogicalSurface*>& surfaces ) // static
    265 {
    266     NPFold* fold = new NPFold ;
    267     for(unsigned i=0 ; i < surfaces.size() ; i++)
    268     {   
    269         const G4LogicalSurface* ls = surfaces[i] ;
    270         const G4String& name = ls->GetName() ; 
    271         const char* key = name.c_str() ; 
    272 
    273         G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(ls->GetSurfaceProperty());
    274         
    275         G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    276         assert(mpt); 
    277         NPFold* sub = U4MaterialPropertiesTable::MakeFold(mpt) ;
    278         
    279         const char* osn = os->GetName().c_str() ;
    280         sub->set_meta<std::string>("osn", osn) ;  // ADDED for specials handling 
    281         
    282         const G4LogicalBorderSurface* bs = dynamic_cast<const G4LogicalBorderSurface*>(ls) ;
    283         const G4LogicalSkinSurface*   ks = dynamic_cast<const G4LogicalSkinSurface*>(ls) ; 
    284         
    285         if(bs)
    286         {


createOpticalBuffer
---------------------

::

    epsilon:ggeo blyth$ opticks-f createOpticalBuffer
    ./ggeo/GBndLib.cc:    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    ./ggeo/GBndLib.cc:    NPY<unsigned int>* optical_buffer = createOpticalBuffer();
    ./ggeo/GBndLib.cc:GBndLib::createOpticalBuffer
    ./ggeo/GBndLib.cc:NPY<unsigned>* GBndLib::createOpticalBuffer()
    ./ggeo/GSurfaceLib.hh:       NPY<unsigned int>* createOpticalBuffer();  
    ./ggeo/GSurfaceLib.cc:    NPY<unsigned int>* ibuf = createOpticalBuffer();
    ./ggeo/GSurfaceLib.cc:NPY<unsigned int>* GSurfaceLib::createOpticalBuffer()
    ./ggeo/GBndLib.hh:       NPY<unsigned int>* createOpticalBuffer();
    ./qudarap/qbnd.h:    GBndLib::createOpticalBuffer 
    ./qudarap/qsim.h:   as acted upon above. See also GBndLib::createOpticalBuffer   
    epsilon:opticks blyth$ 


GGeo::convertSim_BndLib adds SSim::BND and SSim::OPTICAL to SSim 
-----------------------------------------------------------------

::

    2543 void GGeo::convertSim_BndLib(SSim* sim) const
    2544 {   
    2545     LOG(LEVEL) << "[" ; 
    2546     GBndLib* blib = getBndLib();
    2547     
    2548     bool can_create = blib->canCreateBuffer() ;
    2549     NP* bnd = nullptr ; 
    2550     NP* optical = nullptr ;
    2551     
    2552     if( can_create )
    2553     {    
    2554         blib->createDynamicBuffers();  
    2555         // hmm perhaps this is done already on loading now ?
    2556         bnd = blib->getBuf();
    2557         
    2558         LOG(LEVEL) << " bnd.desc " << bnd->desc() ;
    2559         
    2560         optical = blib->getOpticalBuf();
    2561         
    2562         const std::vector<std::string>& bndnames = blib->getNameList();
    2563         bnd->set_names( bndnames );
    2564         
    2565         LOG(LEVEL) << " bnd.set_names " << bndnames.size() ;
    2566         
    2567         sim->add(SSim::BND, bnd ); 
    2568         sim->add(SSim::OPTICAL, optical );
    2569     }   
    2570     else
    2571     {    
    2572         LOG(LEVEL) << "cannot create GBndLib buffer : no materials ? " ;
    2573     }
    2574 }

::

     208 void GBndLib::createDynamicBuffers()
     209 {
     210     // there is not much difference between this and doing a close ??? 
     211 
     212     GItemList* names = createNames();     // added Aug 21, 2018
     213     setNames(names);
     214 
     215     NPY<double>* buf = createBuffer();  // createBufferForTex2d
     216     setBuffer(buf);
     217 
     218     NPY<unsigned int>* optical_buffer = createOpticalBuffer();
     219     setOpticalBuffer(optical_buffer);
     220 


::

     324 NP* GBndLib::getOpticalBuf() const
     325 {
     326     assert( m_optical_buffer );
     327 
     328     NP* optical = m_optical_buffer->spawn() ;
     329     std::string shape0 = optical->sstr() ;
     330 
     331     assert( optical->shape.size() == 3 );
     332 
     333     unsigned ni = optical->shape[0] ;
     334     unsigned nj = optical->shape[1] ;
     335     unsigned nk = optical->shape[2] ;
     336 
     337     assert( ni > 0 && nj == 4 && nk == 4 );
     338 
     339     optical->change_shape( ni*nj , nk );
     340     LOG(LEVEL) << " changed optical shape from " << shape0  << " -> " << optical->sstr() ;
     341 
     342     return optical ;
     343 }


NPY.hpp::

    212     public:
    213        // between old and new array types
    214        NP* spawn() const ;



TODO : U4Tree::initBoundary reimplement GBndLib::createBufferForTex2d GBndLib::createOpticalBuffer
-------------------------------------------------------------------------------------------------------

* reimplement bnd+optical buffer creation in U4Tree/sstree world without GGeo
* needs to be based off U4Tree material + surface data
* should get bit perfect match with the old GGeo arrays 

* U4Tree::initBoundary NEEDED NOW, as makes no sense to make changes for special surface enum in old GGeo world plus U4Tree world 


TODO : generalize optical.x to carry special surface enum (plenty of bits available)
---------------------------------------------------------------------------------------

* once have perfect match between old and new bnd+optical can change the optical 
  to support special surface enum with some bit packing 


QOptical : uploads d_optical buffer 
---------------------------------------

::

     18 union quad ; 
     19 struct NP ; 
     20 template <typename T> struct QBuf ;  
     21        
     22 struct QUDARAP_API QOptical
     23 {   
     24     static const plog::Severity LEVEL ;
     25     static const QOptical*      INSTANCE ;
     26     static const QOptical*      Get();
     27 
     28     QOptical(const NP* optical);
     29     std::string desc() const ;
     30     void check() const ;
     31     
     32     const NP*       optical ;
     33     QBuf<unsigned>* buf ;
     34     quad*           d_optical ;
     35          
     36 };


QSim::UploadComponents instanciates QOptical with SSim::OPTICAL "optical.npy" from SSim 
-------------------------------------------------------------------------------------------

::

     101 void QSim::UploadComponents( const SSim* ssim  )
     ...
     119     const NP* optical = ssim->get(SSim::OPTICAL);
     120     const NP* bnd = ssim->get(SSim::BND);
     121 
     122     if( optical == nullptr && bnd == nullptr )
     123     {
     124         LOG(error) << " optical and bnd null  SSim::OPTICAL " << SSim::OPTICAL << " SSim::BND " << SSim::BND  ;
     125     }
     126     else
     127     {
     128        // note that QOptical and QBnd are tightly coupled, perhaps add constraints to tie them together
     129         QOptical* qopt = new QOptical(optical);
     130         LOG(LEVEL) << qopt->desc();
     131 
     132         QBnd* qbnd = new QBnd(bnd); // boundary texture with standard domain, used for standard fast property lookup 
     133         LOG(LEVEL) << qbnd->desc();
     134     }
     135 




backtrace from qsim.h point of use of ctx.s.optical.x 
--------------------------------------------------------


::

    1455     int command = propagate_to_boundary( flag, rng, ctx );
    1456 #ifdef DEBUG_PIDX
    1457     if( ctx.idx == base->pidx )
    1458     printf("//qsim.propagate idx %d bounce %d command %d flag %d s.optical.x %d \n", ctx.idx, bounce, command, flag, ctx.s.optical.x       );
    1459 #endif
    1460 
    1461     if( command == BOUNDARY )
    1462     {
    1463         command = ctx.s.optical.x == 0 ?
    1464                                       propagate_at_boundary( flag, rng, ctx )
    1465                                   :
    1466                                       propagate_at_surface( flag, rng, ctx )
    1467                                   ;
    1468 
    1469 
    1470     }
    1471 
    1472     ctx.p.set_flag(flag);    // hmm could hide this ?
    1473 
    1474     return command ;
    1475 }
    1476 /**
    1477 Q: Where does ctx.s.optical come from ?
    1478 A: Populated in qbnd::fill_state based on boundary and cosTheta sign to get the 
    1479    su_line index of the optical buffer which distinguishes surface from boundary,
    1480    as acted upon above. See also GBndLib::createOpticalBuffer   
    1481 
    1482 Q: How to implement CustomBoundary (actually CustomSurface might be better name?) "propagate_at_custom_surface ?"
    1483 A: HMM: need 3-way split, but for local_z < 0 ordinary surface applies, 
    1484    so maybe effect the split within propagate_at_surface.
    1485 
    1486    Actually can use the sign of prd->lposcost() which is already available, so could split here, 
    1487    maybe using a negated ctx.s.optical.x ?
    1488 
    1489 


::

   ctx:sctx.h
   ctx.s:sstate.h 
   ctx.s.optical:uint4 
   
   25 struct sstate
   26 {
   27     float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
   28     float4 m1group2 ;     // group_velocity/spare1/spare2/spare3
   29     float4 material2 ;
   30     float4 surface ;      // detect/absorb/reflect_specular/reflect_diffuse
   31 
   32     uint4  optical ;      // x/y/z/w index/type/finish/value  
   33     uint4  index ;        // indices of m1/m2/surf/sensor
   34 

::

    030 struct qbnd
     31 {
     32     cudaTextureObject_t boundary_tex ;
     33     quad4*              boundary_meta ;
     34     unsigned            boundary_tex_MaterialLine_Water ;
     35     unsigned            boundary_tex_MaterialLine_LS ;
     36     quad*               optical ;
     37 

    177 inline QBND_METHOD void qbnd::fill_state(sstate& s, unsigned boundary, float wavelength, float cosTheta, unsigned idx  )
    178 {
    179     const int line = boundary*_BOUNDARY_NUM_MATSUR ;      // now that are not signing boundary use 0-based
    180 
    181     const int m1_line = cosTheta > 0.f ? line + IMAT : line + OMAT ;
    182     const int m2_line = cosTheta > 0.f ? line + OMAT : line + IMAT ;
    183     const int su_line = cosTheta > 0.f ? line + ISUR : line + OSUR ;
    184 
    185 
    186     s.material1 = boundary_lookup( wavelength, m1_line, 0);   // refractive_index, absorption_length, scattering_length, reemission_pro    b
    187     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);   // group_velocity ,  (unused          , unused           , unused)  
    188     s.material2 = boundary_lookup( wavelength, m2_line, 0);   // refractive_index, (absorption_length, scattering_length, reemission_pr    ob) only m2:refractive index actually used  
    189     s.surface   = boundary_lookup( wavelength, su_line, 0);   //  detect,        , absorb            , (reflect_specular), reflect_diff    use     [they add to 1. so one not used] 
    190 
    191     //printf("//qsim.fill_state boundary %d line %d wavelength %10.4f m1_line %d \n", boundary, line, wavelength, m1_line ); 
    192 
    193     s.optical = optical[su_line].u ;   
            // 1-based-surface-index-0-meaning-boundary/type/finish/value  (type,finish,value not used currently)
    194 




