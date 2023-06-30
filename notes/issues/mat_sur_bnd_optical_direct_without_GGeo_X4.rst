mat_sur_bnd_optical_direct_without_GGeo_X4
=============================================

Context
----------

* :doc:`review_optical_surface_handling_with_view_to_qpmt_special_surfaces`


Aiming to integrate QPMT/qpmt into QSim.  
To do this need to create a surface type enum and change the 
optical buffer. 

BUT : optical+bnd buffers are still coming from old workflow
(actually it kinda unholy mixture of old and new currently).
As do not want to do any significant dev in old workflow anymore
it means need to bring mat/sur/bnd/optical handling into new workflow.

Initially there should be near-bit-perfect matches between old and new 
arrays can clearly see when have done this correctly. 

Developments
-------------

sysrap/tests/stree_mat_test.sh  
   excercise stree::create_mat

sysrap/stree.h 
   stree::create_mat
   stree::create_sur 
   stree::create_bnd

   * coordination, with impls elsewhere 
   * stree already holds material and surface NPFold with
     all Geant4 material and surface properties. 
  

sysrap/smaterial.h 
   smaterial::create from list of names and the stree::materials NPFold

sysrap/sprop.h
sysrap/sprop_Material.h
    standard props 

sysrap/sdomain.h 
    added NP interface 

sysrap/tests/sdomain_test.sh 
sysrap/tests/sdomain_test.cc

sysrap/NP.hh  
    NP::ARange NP::MakeFromValues 


1st: need to develop the addition of standard material and surfaces without GGeo/X4
---------------------------------------------------------------------------------------

Thats reimplementing GMaterialLib::createStandardMaterial


Surface props, conversion where ?
------------------------------------

::

     620 void X4PhysicalVolume::convertSurfaces()
     621 {
     622     LOG(LEVEL) << "[" ;
     623 
     624     size_t num_surf0, num_surf1 ;
     625     num_surf0 = m_slib->getNumSurfaces() ;
     626     assert( num_surf0 == 0 );
     627 
     628     char mode_g4interpolate = 'G' ;
     629     //char mode_oldstandardize = 'S' ; 
     630     //char mode_asis = 'A' ; 
     631     char mode = mode_g4interpolate ;
     632 
     633     X4LogicalBorderSurfaceTable::Convert(m_slib, mode);
     634     num_surf1 = m_slib->getNumSurfaces() ;
     635 
     636     size_t num_lbs = num_surf1 - num_surf0 ; num_surf0 = num_surf1 ;
     637 
     638     X4LogicalSkinSurfaceTable::Convert(m_slib, mode);
     639     num_surf1 = m_slib->getNumSurfaces() ;
     640 
     641     size_t num_sks = num_surf1 - num_surf0 ; num_surf0 = num_surf1 ;
     642 
     643     const G4VPhysicalVolume* pv = m_top ;
     644     int depth = 0 ;
     645     convertImplicitSurfaces_r(pv, depth);
     646     num_surf1 = m_slib->getNumSurfaces() ;
     647 
     648     size_t num_ibs = num_surf1 - num_surf0 ; num_surf0 = num_surf1 ;
     649 
     650 
     651     //m_slib->dumpImplicitBorderSurfaces("X4PhysicalVolume::convertSurfaces");  
     652 
     653     m_slib->addPerfectSurfaces();



::

    110 void X4LogicalBorderSurfaceTable::init()
    111 {
    112     unsigned num_src = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
    113     assert( num_src == m_src->size() );
    114 
    115     LOG(LEVEL) << " NumberOfBorderSurfaces " << num_src ;
    116 
    117     for(size_t i=0 ; i < m_src->size() ; i++)
    118     {
    119 
    120         G4LogicalBorderSurface* src = (*m_src)[i] ;
    121 
    122         LOG(LEVEL) << src->GetName() ;
    123 
    124         GBorderSurface* dst = X4LogicalBorderSurface::Convert( src, m_mode );
    125 
    126         assert( dst );
    127 
    128         m_dst->add(dst) ; // GSurfaceLib
    129     }
    130 }

::

     41 GBorderSurface* X4LogicalBorderSurface::Convert(const G4LogicalBorderSurface* src, char mode)
     42 {
     43     const char* name = X4::Name( src );
     44     size_t index = X4::GetOpticksIndex( src ) ;
     45 
     46     G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
     47     assert( os );
     48     GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ;
     49     assert( optical_surface );
     50 
     51     GBorderSurface* dst = new GBorderSurface( name, index, optical_surface) ;
     52     // standard domain is set by GBorderSurface::init
     53 
     54     X4LogicalSurface::Convert( dst, src, mode);
     55 
     56     const G4VPhysicalVolume* pv1 = src->GetVolume1();
     57     const G4VPhysicalVolume* pv2 = src->GetVolume2();
     58     assert( pv1 && pv2 ) ;
     59 
     60     dst->setBorderSurface( X4::Name(pv1), X4::Name(pv2) );
     61 
     62     LOG(LEVEL) << name << " is_sensor " << dst->isSensor() ;
     63 
     64     return dst ;
     65 }


::

     34 void X4LogicalSurface::Convert(GPropertyMap<double>* dst,  const G4LogicalSurface* src, char mode )
     35 {   
     36     LOG(LEVEL) << "[" ; 
     37     const G4SurfaceProperty*  psurf = src->GetSurfaceProperty() ;   
     38     const G4OpticalSurface* opsurf = dynamic_cast<const G4OpticalSurface*>(psurf);
     39     assert( opsurf );   
     40     G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable() ;
     41     X4MaterialPropertiesTable::Convert( dst, mpt, mode );
     42     
     43     LOG(LEVEL) << "]" ;
     44 }




ANSWERED : Where is old source of the standard wavelength domain ? Is it same as sdomain.h ? YES sdomain is used by GDomain
-----------------------------------------------------------------------------------------------------------------------------

::

    epsilon:ggeo blyth$ opticks-f GetDefaultDomain
    ./extg4/tests/X4PhysicsVectorTest.cc:    GDomain<double>* dom = GDomain<double>::GetDefaultDomain() ; 
    ./extg4/X4MaterialPropertiesTable.cc:    GDomain<double>* dom = GDomain<double>::GetDefaultDomain(); 
    ./ggeo/GDomain.cc:GDomain<T>* GDomain<T>::GetDefaultDomain()  // static
    ./ggeo/GPropertyLib.cc:    return GDomain<double>::GetDefaultDomain(); 
    ./ggeo/GPropertyLib.cc:        m_standard_domain = GDomain<double>::GetDefaultDomain(); 
    ./ggeo/GSkinSurface.cc:    setStandardDomain( GDomain<double>::GetDefaultDomain()) ;   
    ./ggeo/GDomain.hh:     static GDomain<T>* GetDefaultDomain() ; 
    ./ggeo/GMaterial.cc:    setStandardDomain( GDomain<double>::GetDefaultDomain()) ;   
    ./ggeo/GPropertyMap.cc:        standard_domain = GDomain<T>::GetDefaultDomain();
    ./ggeo/GBorderSurface.cc:    setStandardDomain( GDomain<double>::GetDefaultDomain()) ;   
    epsilon:opticks blyth$ 



::

     38 template <typename T>
     39 GDomain<T>* GDomain<T>::GetDefaultDomain()  // static
     40 {
     41     if(fDefaultDomain == NULL)
     42     {
     43         fDefaultDomain = MakeDefaultDomain();
     44     }
     45     return fDefaultDomain ;
     46 }
     47 
     48 template <typename T>
     49 GDomain<T>* GDomain<T>::MakeDefaultDomain()  // static
     50 {
     51     GDomain<T>* domain = nullptr ;
     52     switch(sdomain::DOMAIN_TYPE)
     53     {
     54         case 'F': domain = MakeFineDomain() ; break ;
     55         case 'C': domain = MakeCoarseDomain() ; break ;
     56     }
     57     return domain ;
     58 }
     59 
     60 template <typename T>
     61 GDomain<T>* GDomain<T>::MakeCoarseDomain()  // static
     62 {
     63     return  new GDomain<T>(sdomain::DOMAIN_LOW, sdomain::DOMAIN_HIGH, sdomain::DOMAIN_STEP );
     64 }
     65 
     66 template <typename T>
     67 GDomain<T>* GDomain<T>::MakeFineDomain()  // static
     68 {
     69     return new GDomain<T>(sdomain::DOMAIN_LOW, sdomain::DOMAIN_HIGH, sdomain::FINE_DOMAIN_STEP );
     70 }
     71 
     72 





ANSWERED : Where in old workflow is the energy to wavelength switch done ?
------------------------------------------------------------------------------

* starting point is X4PhysicalVolume::init esp 

::

     265 void X4PhysicalVolume::convertMaterials()
     266 {
     267     OK_PROFILE("_X4PhysicalVolume::convertMaterials");
     268     LOG(LEVEL) << "[" ;
     269 
     270     const G4VPhysicalVolume* pv = m_top ;
     271     int depth = 0 ;
     272     convertMaterials_r(pv, depth);
     273 
     274     LOG(LEVEL) << X4Material::Desc(m_mtlist);
     275 
     276     const std::vector<G4Material*>& used_materials = m_mtlist ;
     277     X4MaterialTable::Convert(m_mlib, m_material_with_efficiency, used_materials );
     278     size_t num_material_with_efficiency = m_material_with_efficiency.size() ;
     279 
     280     m_mlib->close();   // may change order if prefs dictate


::

    105 void X4MaterialTable::init()
    106 {   
    107     unsigned num_input_materials = m_input_materials.size() ;
    108     
    109     LOG(LEVEL) << ". G4 nmat " << num_input_materials ;
    110     
    111     for(unsigned i=0 ; i < num_input_materials ; i++)
    112     {   
    113         G4Material* material = m_input_materials[i] ; 
    114         G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    115         
    116         if( mpt == NULL )
    117         {   
    118             LOG(LEVEL) << "PROCEEDING TO convert material with no mpt " << material->GetName() ;
    119         }
    120         else
    121         {   
    122             LOG(LEVEL) << " converting material with mpt " <<  material->GetName() ;
    123         }
    124         
    125         //char mode_oldstandardized = 'S' ;
    126         char mode_g4interpolated = 'G' ;
    127         GMaterial* mat = X4Material::Convert( material, mode_g4interpolated );   
    128         if(mat->hasProperty("EFFICIENCY")) m_materials_with_efficiency.push_back(material);
    129         m_mlib->add(mat) ;
    130         
    131         char mode_asis_nm = 'A' ;
    132         GMaterial* rawmat = X4Material::Convert( material, mode_asis_nm );
    133         m_mlib->addRaw(rawmat) ;
    134         
    135         char mode_asis_en = 'E' ;
    136         GMaterial* rawmat_en = X4Material::Convert( material, mode_asis_en );   
    137         GPropertyMap<double>* pmap_rawmat_en = dynamic_cast<GPropertyMap<double>*>(rawmat_en) ;
    138         m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib
    139 


::

     66 /**
     67 X4Material::Convert
     68 ----------------------
     69 
     70 Canonically invoked from X4MaterialTable::init, mode:
     71 
     72 'S'
     73     old_standardized no longer in use
     74 'G'
     75     g4interpolated onto the domain 
     76 'A'
     77     asis_nm not interpolated just converted to nm domain
     78 'E'
     79     asis_en not interpolated and with original (energy) domain left with no change to units  
     80 
     81 
     82 The default approach is to convert energy domain to wavelength domain in nm, when 
     83 such conversion is **NOT** done with mode 'E' the setOriginalDomain label is set.
     84 
     85 **/
     86 


::

    298 void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt, char mode )   //      static
    299 {
    300     typedef G4MaterialPropertyVector MPV ;
    301 
    302     std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    303     LOG(LEVEL) << " MaterialPropertyNames pns.size " << pns.size() ;
    304 
    305     GDomain<double>* dom = GDomain<double>::GetDefaultDomain();
    306     unsigned pns_null = 0 ;
    307 
    308     for( unsigned i=0 ; i < pns.size() ; i++)
    309     {  
    310         const std::string& pname = pns[i];
    311         G4int pidx = X4MaterialPropertiesTable::GetPropertyIndex(mpt, pname.c_str());
    312         assert( pidx > -1 ); 
    313         MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);
    314         LOG(LEVEL)
    315             << " pname : "
    316             << std::setw(30) << pname 
    317             << " pidx : "
    318             << std::setw(5) << pidx
    319             << " pvec : "
    320             << std::setw(16) << pvec
    321             ;  
    322 
    323         if(pvec == NULL)
    324         {   
    325             pns_null += 1 ;
    326             continue ;
    327         }
    328 
    329         GProperty<double>* prop = nullptr ; 
    330 
    331         if( mode == 'G' )           // Geant4 src interpolation onto the domain 
    332         {
    333             prop = X4PhysicsVector<double>::Interpolate(pvec, dom) ;
    334             pmap->addPropertyAsis( pname.c_str(), prop );
    335         }
    336         else if( mode == 'S' )      // Opticks pmap interpolation onto standard domain   
    337         {
    338             bool nm_domain = true ;
    339             prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ;
    340             pmap->addPropertyStandardized( pname.c_str(), prop );
    341         }
    342         else if( mode == 'A' )      //  asis : no interpolation, but converted to nm  
    343         {
    344             bool nm_domain = true ;
    345             prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ;
    346             pmap->addPropertyAsis( pname.c_str(), prop );
    347         }
    348         else if( mode == 'E' )      //  asis : no interpolation, NOT converted to nm : Energy domain 
    349         {
    350             bool nm_domain = false ;
    351             prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ;
    352             pmap->addPropertyAsis( pname.c_str(), prop );
    353         }
    354         else





    290 /**
    291 X4MaterialPropertiesTable::AddProperties
    292 -------------------------------------------
    293 
    294 Used from X4Material::Convert/X4Material::init
    295 
    296 **/
    297 
    298 void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt, char mode )   //      static
    299 {
    300     typedef G4MaterialPropertyVector MPV ;
    301 
    302     std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    303     LOG(LEVEL) << " MaterialPropertyNames pns.size " << pns.size() ;
    304 



    330 
    331         if( mode == 'G' )           // Geant4 src interpolation onto the domain 
    332         {
    333             prop = X4PhysicsVector<double>::Interpolate(pvec, dom) ;
    334             pmap->addPropertyAsis( pname.c_str(), prop );
    335         }
    336         else if( mode == 'S' )      // Opticks pmap interpolation onto standard domain   
    337         {
    338             bool nm_domain = true ;
    339             prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ;
    340             pmap->addPropertyStandardized( pname.c_str(), prop );
    341         }
    342         else if( mode == 'A' )      //  asis : no interpolation, but converted to nm  
    343         {
    344             bool nm_domain = true ;
    345             prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ;
    346             pmap->addPropertyAsis( pname.c_str(), prop );
    347         }
    348         else if( mode == 'E' )      //  asis : no interpolation, NOT converted to nm : Energy domain 
    349         {
    350             bool nm_domain = false ;
    351             prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ;
    352             pmap->addPropertyAsis( pname.c_str(), prop );
    353         }
    354         else
    355         {
    356             LOG(fatal) << " mode must be one of G/S/A/E " ;
    357             assert(0);
    358         }



::

    161 /**
    162 X4PhysicsVector::getInterpolatedValues
    163 ---------------------------------------
    164 
    165 Each of the domain wavelength_nm values is converted 
    166 into energy_eV which is used by the m_vec G4PhysicsVector::Value 
    167 to get an interpolated value stored into the new array. 
    168 Thus the energy domain is swapped out for a different 
    169 interpolated wavelength domain.  Notice that no reversal 
    170 is needed because the wavelength_nm array is just directing 
    171 a bunch of interpolation Value calls to the m_vec. 
    172 
    173 **/
    174 
    175 template <typename T>
    176 T* X4PhysicsVector<T>::getInterpolatedValues(T* wavelength_nm, size_t n, T hc_eVnm_ ) const
    177 {
    178     T* a = new T[n] ;
    179     
    180     T hc_eVnm = hc_eVnm_ > 1239. && hc_eVnm_ < 1241. ? hc_eVnm_ : _hc_eVnm()  ;
    181     
    182     for (size_t i=0; i<n; i++)
    183     {
    184         T wl_nm = wavelength_nm[i] ;
    185         T en_eV = hc_eVnm/wl_nm ;  
    186         T value = m_vec->Value(en_eV*eV);     // eV = electronvolt = 1.e-6  "g4-cls SystemOfUnits" 
    187         a[i] = value ;
    188 

