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


Strategy
---------

Initial try was to do this at sysrap level with sstandard::mat 
from the NPFold using NP interpolation
but comparing that with the old mat shows quite a few differences
in GROUPVEL etc.. 

So try attacking at higher level in U4Tree::initMaterials U4Tree::initSurfaces
using U4Material::MakeStandardArray U4Surface::MakeStandardArray which 
uses sproplist but intead of NP interpolation uses Geant4 interpolation
direct from the Geant4 materials and surfaces. 


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



bd diff
----------

::

    st
    ./stree_mat_test.sh ana


    epsilon:tests blyth$ diff -y /tmp/SBnd_test/bd_names.txt /tmp/stree_mat_test/bd_names.txt
    Galactic///Galactic						Galactic///Galactic
    Galactic///Rock							Galactic///Rock
    Rock///Galactic							Rock///Galactic
    Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air	      |	Rock///Air
    Rock///Rock							Rock///Rock
    Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air	      <
    Air///Steel							Air///Steel
    Air///Air							Air///Air
    Air///LS							Air///LS
    Air///Tyvek							Air///Tyvek
    Air///Aluminium							Air///Aluminium
    Aluminium///Adhesive						Aluminium///Adhesive
    Adhesive///TiO2Coating						Adhesive///TiO2Coating
    TiO2Coating///Scintillator					TiO2Coating///Scintillator
    Rock///Tyvek							Rock///Tyvek
    Tyvek//VETOTyvekSurface/vetoWater				Tyvek//VETOTyvekSurface/vetoWater
    vetoWater///LatticedShellSteel					vetoWater///LatticedShellSteel
    vetoWater/CDTyvekSurface//Tyvek					vetoWater/CDTyvekSurface//Tyvek
    Tyvek//CDInnerTyvekSurface/Water				Tyvek//CDInnerTyvekSurface/Water
    Water///Acrylic							Water///Acrylic
    Acrylic///LS							Acrylic///LS
    LS///Acrylic							LS///Acrylic
    LS///PE_PA							LS///PE_PA
    Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel	Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel
    Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutStee	Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutStee
    Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/Steel	      <
    Water///Steel							Water///Steel



Why lots of constant GROUPVEL ?
---------------------------------

::

    In [33]: np.all( t.oldmat[1,1,:,0] == t.oldmat[1,1,0,0] )
    Out[33]: True

    In [34]: t.oldmat[1,1,0,0]
    Out[34]: 299.792458


Changed the sproplist.h default but need to rerun U4Tree creation 
back on workstation. So kludge that::

    In [7]: np.where( t.mat == 300. )
    Out[7]: 
    (array([ 1,  1,  1,  1,  1, ..., 15, 15, 15, 15, 15]),
     array([1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1]),
     array([  0,   1,   2,   3,   4, ..., 756, 757, 758, 759, 760]),
     array([0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0]))

    In [8]: t.mat[np.where( t.mat == 300. )] = 299.792458

::

    In [13]: np.array(t.mat_names)[np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])]
    Out[13]: array(['LS', 'Acrylic', 'AcrylicMask', 'Vacuum', 'Pyrex', 'Water', 'vetoWater'], dtype='<U18')



::

    np.all( np.array( t.mat_names) == np.array( t.oldmat_names ))  
    True
    t.mat.shape == t.oldmat.shape
    True
    np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])
    [ 4 11 14 17 18 19]
    np.array(t.mat_names)[np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])] 
    ['LS' 'Acrylic' 'AcrylicMask' 'Pyrex' 'Water' 'vetoWater']
    np.max(ab, axis=2).reshape(-1,8)   # max deviation across wavelength domain 
    [[      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.891       0.09        0.         49.062       0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.018       0.          0.         47.025       0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.002       0.          0.         47.025       0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.          0.          0.          0.          0.          0.          0.   ]
     [      0.          0.001       0.          0.          4.898       0.          0.          0.   ]
     [      0.          0.051 1763077.672       0.          8.413       0.          0.          0.   ]
     [      0.          0.051       0.          0.          8.413       0.          0.          0.   ]]

         RINDEX     ABSLENGTH  RAYLEIGH   REEMISSIONPROB   GROUPVEL 

    In [1]: np.array(t.mat_names)[19]
    Out[1]: 'vetoWater'


Comparing GROUPVEL plots between oldmat and mat : clearly related but different bin handling 
---------------------------------------------------------------------------------------------

::
    
    st
    ./stree_mat_test.sh 

    GROUPVEL 4 LS 
    GROUPVEL 11 Acrylic 
    GROUPVEL 14 AcrylicMask 
    GROUPVEL 17 Pyrex 
    GROUPVEL 18 Water 
    GROUPVEL 19 vetoWater 





Water RAYLEIGH is very discrepant 
----------------------------------

::

    In [3]: np.all( t.mat[18,0,:,2] == 1e6 ) ## Water : NEED TO TAP INTO Water/RAYLEIGH SPECIAL CASING 
    Out[3]: True

    In [5]: np.all( t.mat[19,0,:,2] == 1e6 ) ## VetoWater 
    Out[5]: True

    In [7]: t.oldmat[18,0,:,2]   ## this is Geant4 special cased Water RAYLEIGH
    Out[7]: 
    array([    283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,
               283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     283.98 ,     284.422,     327.186,     368.908,     409.623,     449.37 ,     488.181,     526.09 ,     563.127,
               599.322,     634.705,     669.3  ,     703.136,     736.236,     768.624,     800.323,     831.354,     861.739,     891.498,     920.649,     949.212,     977.203,    1004.64 ,
              1031.539,    1057.915,    1083.784,    1109.161,    1134.059,    1158.491,    1182.471,    1206.011,    1229.123,    1251.819,    1274.109,    1296.005,    1317.516,    1338.654,


    In [6]: np.all( t.oldmat[19,0,:,2] == 1e6 )   ## HMM vetoWater RAYLEIGH is constant, unlike Water 
    Out[6]: True                                  ## that looks like junosw bug  




TODO : review X4MaterialWater X4OpRayleigh and do something similar in U4Water.h
------------------------------------------------------------------------------------



Nature of GROUPVEL diff : Looks like different calc
-----------------------------------------------------

::


    In [13]: np.c_[t.mat[4,1,:,0]-t.oldmat[4,1,:,0], t.mat[4,1,:,0], t.oldmat[4,1,:,0] ][140:160]
    Out[13]: 
    array([[ 10.999, 182.723, 171.723],
           [ 23.781, 182.241, 158.46 ],
           [ 23.076, 181.764, 158.688],
           [ 22.377, 181.292, 158.915],
           [ 21.684, 180.824, 159.14 ],
           [ 20.998, 180.362, 159.364],
           [ 20.318, 179.903, 159.586],
           [ 19.643, 179.449, 159.806],




mat diff
----------


Default GROUPVEL diff::

    In [19]: np.all( o.mat[0,1,:,0] == 299.711 )
    Out[19]: False

    In [20]: np.all( o.mat[0,1,:,0] == o.mat[0,1,0,0] )
    Out[20]: True

    In [21]: o.mat[0,1,0,0]
    Out[21]: 299.7106369961001

    In [22]: np.all( o.mat[0,1,:,0] == 299.7106369961001 )
    Out[22]: True


Looks like this is GMaterialLib::replaceGROUPVEL calculating it from the RINDEX.

This is presumably duplicating Geant4 calc, better to not bother 
and just use the Geant4 calc ?::

    366 G4MaterialPropertyVector* G4MaterialPropertiesTable::CalculateGROUPVEL()


::

    epsilon:opticks blyth$ g4-cc CalculateGROUPVEL
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:      CalculateGROUPVEL();
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:      CalculateGROUPVEL();
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:G4MaterialPropertyVector* G4MaterialPropertiesTable::CalculateGROUPVEL()
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:    G4Exception("G4MaterialPropertiesTable::CalculateGROUPVEL()", "mat205",
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:      G4Exception("G4MaterialPropertiesTable::CalculateGROUPVEL()", "mat205",
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:        G4Exception("G4MaterialPropertiesTable::CalculateGROUPVEL()", "mat205",
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:  message += "Use G4MaterialPropertiesTable::CalculateGROUPVEL() instead";
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/materials/src/G4MaterialPropertiesTable.cc:  return CalculateGROUPVEL();
    epsilon:opticks blyth$ 




::

     317 template <typename T>
     318 GProperty<T>* GProperty<T>::make_GROUPVEL(GProperty<T>* rindex)
     319 {
     320     /*
     321     :param rindex: refractive_index assumed to have standard wavelength domain and order
     322     */
     323     GAry<T>* wl0 = rindex->getDomain();
     324 
     325     GProperty<T>* riE = rindex->createReversedReciprocalDomain(GConstant::hc_eVnm);
     326     GAry<T>* en = riE->getDomain();
     327     GAry<T>* ri = riE->getValues();
     328 
     329     GAry<T>* ds = make_dispersion_term(riE);
     330 
     331     GAry<T>* ee = en->g4_groupvel_bintrick();
     332     GAry<T>* nn = ri->g4_groupvel_bintrick();
     333     GAry<T>* nn_plus_ds = GAry<T>::add( nn, ds );
     334 
     335     GAry<T>* vg0 = nn->reciprocal(GConstant::c_light);
     336     GAry<T>* vg  = nn_plus_ds->reciprocal(GConstant::c_light);
     337 
     338     assert(vg0->getLength() == vg->getLength());
     339     unsigned len = vg0->getLength();
     340 
     341     GAry<T>* ze =  GAry<T>::zeros(len);
     342 
     343     GAry<T>* vgc = vg->clip(ze, vg0, vg0, vg0 );
     344 
     345     // interpolate back onto original energy domain: en   
     346     GAry<T>* vgi = GAry<T>::np_interp( en , ee, vgc ) ;
     347 
     348     // clip again after the interpolation to avoid tachyons
     349     GAry<T>* vgic = vgi->clip(ze, vg0, vg0, vg0 );
     350 
     351     GProperty<T>* vgE = new GProperty<T>( vgic, en );



RINDEX differences
--------------------

Maybe G4/NP interpolation difference::

    In [36]: np.abs( o.mat[4,0,:,0]-t.mat[4,0,:,0] ).max()
    Out[36]: 3.5602402803647237e-07

    In [25]: np.where( o.mat[:,0,:,0] != t.mat[:,0,:,0] )
    Out[25]: 
    (array([ 4,  4,  4,  4,  4,  4,  4,  4, ..., 19, 19, 19, 19, 19, 19, 19, 19]),
     array([ 20,  21,  22,  23,  24,  25,  26,  27, ..., 732, 733, 734, 735, 736, 737, 738, 739]))

    In [26]: m, w = np.where( o.mat[:,0,:,0] != t.mat[:,0,:,0] )

    In [27]: m
    Out[27]: array([ 4,  4,  4,  4,  4,  4,  4,  4, ..., 19, 19, 19, 19, 19, 19, 19, 19])

    In [28]: np.unique(m)
    Out[28]: array([ 4, 11, 14, 17, 18, 19])

    In [29]: mn = np.array(o.mat_names)
    In [30]: mn[np.unique(m)]
    Out[30]: array(['LS', 'Acrylic', 'AcrylicMask', 'Pyrex', 'Water', 'vetoWater'], dtype='<U18')



No large RINDEX deviations::

    In [48]: np.where( np.abs( o.mat[:,0,:,0] - t.mat[:,0,:,0] ) > 1e-5 )  # no very different rindex
    Out[48]: (array([], dtype=int64), array([], dtype=int64))



ABSLENGTH has deviations::

    In [54]: x, y = np.where( np.abs( o.mat[:,0,:,1] - t.mat[:,0,:,1] ) > 1e-5 )
    In [55]: np.unique(x)
    Out[55]: array([ 4, 11, 14, 17, 18, 19])
    In [56]: np.array(o.mat_names)[np.unique(x)]
    Out[56]: array(['LS', 'Acrylic', 'AcrylicMask', 'Pyrex', 'Water', 'vetoWater'], dtype='<U18')



    In [59]: x, y = np.where( np.abs( o.mat[:,0,:,1] - t.mat[:,0,:,1] ) > 0.1 )
    In [61]: np.unique(x)                                                                                                                     
    Out[61]: array([4])

    In [62]: o.mat_names[4]                                                                                                                   
    Out[62]: 'LS'





Comparing new s (U4Material::MakeStandardArray) with old o:(old mat from bnd)
--------------------------------------------------------------------------------

General impression to be confirmed is that: 

* GROUPVEL has large discrepancy for most materials.

  * looks like actual calculation difference not just precision issue, 
    need to investigate where Geant4 CalculateGROUPVEL happens 
  * possibly the old GGeo calculation of this has diverged 


* other props are discrepant but it could be float/double interpol difference 

* WIP: arrange to get a mat array direct from old workflow 
  in double precision without using the bnd reconstruction 
  which limits to float. This will allow comparison without
  being clouded by precision diffs. 

Did this in::

    2543 void GGeo::convertSim_BndLib(SSim* sim) const
    2544 {
    ....
    2571 
    2572         // OLD WORKFLOW ADDITION TO CHECK NEW WORKFLOW 
    2573         GMaterialLib* mlib = getMaterialLib();
    2574         GSurfaceLib*  slib = getSurfaceLib();
    2575         NP* oldmat = mlib->getBuf();
    2576         NP* oldsur = slib->getBuf();
    2577         sim->add(SSim::OLDMAT, oldmat );
    2578         sim->add(SSim::OLDSUR, oldsur );
    2579     }




::

     i : 0  Air 
     j : 0 
     len(np.where( np.abs( o.mat[0,0,:,0] - s.mat[0,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[0,0,:,1] - s.mat[0,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[0,0,:,2] - s.mat[0,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[0,0,:,3] - s.mat[0,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[0,1,:,0] - s.mat[0,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[0,1,:,1] - s.mat[0,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[0,1,:,2] - s.mat[0,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[0,1,:,3] - s.mat[0,1,:,3] ) > 1e-4)[0]) : 0 

     i : 1  Rock 
     j : 0 
     len(np.where( np.abs( o.mat[1,0,:,0] - s.mat[1,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[1,0,:,1] - s.mat[1,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[1,0,:,2] - s.mat[1,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[1,0,:,3] - s.mat[1,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[1,1,:,0] - s.mat[1,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[1,1,:,1] - s.mat[1,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[1,1,:,2] - s.mat[1,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[1,1,:,3] - s.mat[1,1,:,3] ) > 1e-4)[0]) : 0 

     i : 2  Galactic 
     j : 0 
     len(np.where( np.abs( o.mat[2,0,:,0] - s.mat[2,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[2,0,:,1] - s.mat[2,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[2,0,:,2] - s.mat[2,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[2,0,:,3] - s.mat[2,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[2,1,:,0] - s.mat[2,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[2,1,:,1] - s.mat[2,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[2,1,:,2] - s.mat[2,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[2,1,:,3] - s.mat[2,1,:,3] ) > 1e-4)[0]) : 0 

     i : 3  Steel 
     j : 0 
     len(np.where( np.abs( o.mat[3,0,:,0] - s.mat[3,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[3,0,:,1] - s.mat[3,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[3,0,:,2] - s.mat[3,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[3,0,:,3] - s.mat[3,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[3,1,:,0] - s.mat[3,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[3,1,:,1] - s.mat[3,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[3,1,:,2] - s.mat[3,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[3,1,:,3] - s.mat[3,1,:,3] ) > 1e-4)[0]) : 0 

     i : 4  LS 
     j : 0 
     len(np.where( np.abs( o.mat[4,0,:,0] - s.mat[4,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[4,0,:,1] - s.mat[4,0,:,1] ) > 1e-4)[0]) : 433 
     len(np.where( np.abs( o.mat[4,0,:,2] - s.mat[4,0,:,2] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[4,0,:,3] - s.mat[4,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[4,1,:,0] - s.mat[4,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[4,1,:,1] - s.mat[4,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[4,1,:,2] - s.mat[4,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[4,1,:,3] - s.mat[4,1,:,3] ) > 1e-4)[0]) : 0 

Interpolation diffs on big values::

    In [6]: np.c_[o.mat[4,0,:,1] - s.mat[4,0,:,1], o.mat[4,0,:,1], s.mat[4,0,:,1]][w]
    Out[6]: 
    array([[   0.   ,   18.647,   18.647],
           [   0.   ,   22.73 ,   22.73 ],
           [   0.   ,   25.433,   25.432],
           [   0.   ,   28.823,   28.823],
           [   0.   ,   33.234,   33.234],
           ...,
           [  -0.008, 6082.314, 6082.322],
           [   0.004, 6002.164, 6002.16 ],
           [  -0.004, 6054.658, 6054.662],
           [  -0.011, 5973.25 , 5973.261],
           [  -0.011, 5822.29 , 5822.301]])




     i : 5  Tyvek 
     j : 0 
     len(np.where( np.abs( o.mat[5,0,:,0] - s.mat[5,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[5,0,:,1] - s.mat[5,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[5,0,:,2] - s.mat[5,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[5,0,:,3] - s.mat[5,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[5,1,:,0] - s.mat[5,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[5,1,:,1] - s.mat[5,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[5,1,:,2] - s.mat[5,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[5,1,:,3] - s.mat[5,1,:,3] ) > 1e-4)[0]) : 0 

     i : 6  Scintillator 
     j : 0 
     len(np.where( np.abs( o.mat[6,0,:,0] - s.mat[6,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[6,0,:,1] - s.mat[6,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[6,0,:,2] - s.mat[6,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[6,0,:,3] - s.mat[6,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[6,1,:,0] - s.mat[6,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[6,1,:,1] - s.mat[6,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[6,1,:,2] - s.mat[6,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[6,1,:,3] - s.mat[6,1,:,3] ) > 1e-4)[0]) : 0 

     i : 7  TiO2Coating 
     j : 0 
     len(np.where( np.abs( o.mat[7,0,:,0] - s.mat[7,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[7,0,:,1] - s.mat[7,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[7,0,:,2] - s.mat[7,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[7,0,:,3] - s.mat[7,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[7,1,:,0] - s.mat[7,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[7,1,:,1] - s.mat[7,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[7,1,:,2] - s.mat[7,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[7,1,:,3] - s.mat[7,1,:,3] ) > 1e-4)[0]) : 0 

     i : 8  Adhesive 
     j : 0 
     len(np.where( np.abs( o.mat[8,0,:,0] - s.mat[8,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[8,0,:,1] - s.mat[8,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[8,0,:,2] - s.mat[8,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[8,0,:,3] - s.mat[8,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[8,1,:,0] - s.mat[8,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[8,1,:,1] - s.mat[8,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[8,1,:,2] - s.mat[8,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[8,1,:,3] - s.mat[8,1,:,3] ) > 1e-4)[0]) : 0 

     i : 9  Aluminium 
     j : 0 
     len(np.where( np.abs( o.mat[9,0,:,0] - s.mat[9,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[9,0,:,1] - s.mat[9,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[9,0,:,2] - s.mat[9,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[9,0,:,3] - s.mat[9,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[9,1,:,0] - s.mat[9,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[9,1,:,1] - s.mat[9,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[9,1,:,2] - s.mat[9,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[9,1,:,3] - s.mat[9,1,:,3] ) > 1e-4)[0]) : 0 

     i : 10  LatticedShellSteel 
     j : 0 
     len(np.where( np.abs( o.mat[10,0,:,0] - s.mat[10,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[10,0,:,1] - s.mat[10,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[10,0,:,2] - s.mat[10,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[10,0,:,3] - s.mat[10,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[10,1,:,0] - s.mat[10,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[10,1,:,1] - s.mat[10,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[10,1,:,2] - s.mat[10,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[10,1,:,3] - s.mat[10,1,:,3] ) > 1e-4)[0]) : 0 

     i : 11  Acrylic 
     j : 0 
     len(np.where( np.abs( o.mat[11,0,:,0] - s.mat[11,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[11,0,:,1] - s.mat[11,0,:,1] ) > 1e-4)[0]) : 60 
     len(np.where( np.abs( o.mat[11,0,:,2] - s.mat[11,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[11,0,:,3] - s.mat[11,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[11,1,:,0] - s.mat[11,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[11,1,:,1] - s.mat[11,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[11,1,:,2] - s.mat[11,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[11,1,:,3] - s.mat[11,1,:,3] ) > 1e-4)[0]) : 0 


Hmm that could just be float/double diff::

    In [13]: np.c_[o.mat[11,0,:,1] - s.mat[11,0,:,1], o.mat[11,0,:,1], s.mat[11,0,:,1]][w]                                                    
    Out[13]: 
    array([[   0.   ,   45.766,   45.766],
           [   0.   ,   52.154,   52.154],
           [   0.   ,   58.508,   58.508],
           [   0.   ,   64.828,   64.828],
           [   0.   ,   71.115,   71.114],
           [   0.   ,   77.368,   77.367],
           [   0.   ,   83.587,   83.587],
           [   0.   ,   89.774,   89.774],
           [   0.   ,   95.928,   95.928],
           [   0.   ,  102.05 ,  102.05 ],
           [   0.001,  118.482,  118.481],
           [   0.001,  139.672,  139.672],
           [   0.001,  160.752,  160.751],
           [   0.001,  181.722,  181.722],
           [   0.001,  202.583,  202.583],
           [   0.001,  223.336,  223.336],
           [   0.001,  243.982,  243.981],
           [   0.001,  264.521,  264.521],
           [   0.001,  284.955,  284.955],
           [   0.002,  308.9  ,  308.898],
           [   0.002,  361.139,  361.137],
           [   0.002,  413.112,  413.11 ],
           [   0.002,  464.82 ,  464.819],
           [   0.002,  516.266,  516.265],
           [   0.002,  567.452,  567.45 ],
           [   0.002,  618.379,  618.377],




     i : 12  PE_PA 
     j : 0 
     len(np.where( np.abs( o.mat[12,0,:,0] - s.mat[12,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[12,0,:,1] - s.mat[12,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[12,0,:,2] - s.mat[12,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[12,0,:,3] - s.mat[12,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[12,1,:,0] - s.mat[12,1,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[12,1,:,1] - s.mat[12,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[12,1,:,2] - s.mat[12,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[12,1,:,3] - s.mat[12,1,:,3] ) > 1e-4)[0]) : 0 

     i : 13  StrutSteel 
     j : 0 
     len(np.where( np.abs( o.mat[13,0,:,0] - s.mat[13,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[13,0,:,1] - s.mat[13,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[13,0,:,2] - s.mat[13,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[13,0,:,3] - s.mat[13,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[13,1,:,0] - s.mat[13,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[13,1,:,1] - s.mat[13,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[13,1,:,2] - s.mat[13,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[13,1,:,3] - s.mat[13,1,:,3] ) > 1e-4)[0]) : 0 

     i : 14  AcrylicMask 
     j : 0 
     len(np.where( np.abs( o.mat[14,0,:,0] - s.mat[14,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[14,0,:,1] - s.mat[14,0,:,1] ) > 1e-4)[0]) : 149 
     len(np.where( np.abs( o.mat[14,0,:,2] - s.mat[14,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[14,0,:,3] - s.mat[14,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[14,1,:,0] - s.mat[14,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[14,1,:,1] - s.mat[14,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[14,1,:,2] - s.mat[14,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[14,1,:,3] - s.mat[14,1,:,3] ) > 1e-4)[0]) : 0 

     i : 15  CDReflectorSteel 
     j : 0 
     len(np.where( np.abs( o.mat[15,0,:,0] - s.mat[15,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[15,0,:,1] - s.mat[15,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[15,0,:,2] - s.mat[15,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[15,0,:,3] - s.mat[15,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[15,1,:,0] - s.mat[15,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[15,1,:,1] - s.mat[15,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[15,1,:,2] - s.mat[15,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[15,1,:,3] - s.mat[15,1,:,3] ) > 1e-4)[0]) : 0 

     i : 16  Vacuum 
     j : 0 
     len(np.where( np.abs( o.mat[16,0,:,0] - s.mat[16,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[16,0,:,1] - s.mat[16,0,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[16,0,:,2] - s.mat[16,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[16,0,:,3] - s.mat[16,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[16,1,:,0] - s.mat[16,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[16,1,:,1] - s.mat[16,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[16,1,:,2] - s.mat[16,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[16,1,:,3] - s.mat[16,1,:,3] ) > 1e-4)[0]) : 0 

     i : 17  Pyrex 
     j : 0 
     len(np.where( np.abs( o.mat[17,0,:,0] - s.mat[17,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[17,0,:,1] - s.mat[17,0,:,1] ) > 1e-4)[0]) : 442 
     len(np.where( np.abs( o.mat[17,0,:,2] - s.mat[17,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[17,0,:,3] - s.mat[17,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[17,1,:,0] - s.mat[17,1,:,0] ) > 1e-4)[0]) : 760 
     len(np.where( np.abs( o.mat[17,1,:,1] - s.mat[17,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[17,1,:,2] - s.mat[17,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[17,1,:,3] - s.mat[17,1,:,3] ) > 1e-4)[0]) : 0 

     i : 18  Water 
     j : 0 
     len(np.where( np.abs( o.mat[18,0,:,0] - s.mat[18,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[18,0,:,1] - s.mat[18,0,:,1] ) > 1e-4)[0]) : 619 
     len(np.where( np.abs( o.mat[18,0,:,2] - s.mat[18,0,:,2] ) > 1e-4)[0]) : 757 
     len(np.where( np.abs( o.mat[18,0,:,3] - s.mat[18,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[18,1,:,0] - s.mat[18,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[18,1,:,1] - s.mat[18,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[18,1,:,2] - s.mat[18,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[18,1,:,3] - s.mat[18,1,:,3] ) > 1e-4)[0]) : 0 


    In [18]: w = np.where( np.abs( o.mat[18,0,:,1] - s.mat[18,0,:,1] ) > 1e-2)[0]

    In [19]: np.c_[o.mat[18,0,:,1] - s.mat[18,0,:,1], o.mat[18,0,:,1], s.mat[18,0,:,1]][w]
    Out[19]: 
    array([[   0.021,  557.549,  557.528],
           [   0.01 ,  908.361,  908.351],
           [  -0.016, 1369.246, 1369.262],
           [   0.015, 1617.745, 1617.73 ],
           [   0.011, 1938.504, 1938.493],
           ...,
           [   0.015,  618.672,  618.657],
           [   0.018,  626.91 ,  626.892],
           [  -0.019,  667.067,  667.086],
           [  -0.028,  691.929,  691.957],
           [  -0.01 ,  705.131,  705.141]])





     i : 19  vetoWater 
     j : 0 
     len(np.where( np.abs( o.mat[19,0,:,0] - s.mat[19,0,:,0] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[19,0,:,1] - s.mat[19,0,:,1] ) > 1e-4)[0]) : 619 
     len(np.where( np.abs( o.mat[19,0,:,2] - s.mat[19,0,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[19,0,:,3] - s.mat[19,0,:,3] ) > 1e-4)[0]) : 0 
     j : 1 
     len(np.where( np.abs( o.mat[19,1,:,0] - s.mat[19,1,:,0] ) > 1e-4)[0]) : 761 
     len(np.where( np.abs( o.mat[19,1,:,1] - s.mat[19,1,:,1] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[19,1,:,2] - s.mat[19,1,:,2] ) > 1e-4)[0]) : 0 
     len(np.where( np.abs( o.mat[19,1,:,3] - s.mat[19,1,:,3] ) > 1e-4)[0]) : 0 







In [6]: w = np.where( np.abs(s.mat[11,0,:,1] - o.mat[11,0,:,1]) > 1e-3 )[0]

In [7]: np.c_[s.mat[11,0,:,1] - o.mat[11,0,:,1], s.mat[11,0,:,1], o.mat[11,0,:,1]][w] 
Out[7]: 
array([[  -0.002,  308.898,  308.9  ],
       [  -0.002,  361.137,  361.139],
       [  -0.002,  413.11 ,  413.112],
       [  -0.002,  464.819,  464.82 ],
       [  -0.002,  516.265,  516.266],
       [  -0.002,  567.45 ,  567.452],
       [  -0.002,  618.377,  618.379],
       [  -0.002,  669.047,  669.049],
       [  -0.002,  719.463,  719.465],
       [  -0.002,  769.626,  769.628],
       [  -0.004,  822.136,  822.139],
       [  -0.004,  922.621,  922.625],
       [  -0.004, 1022.607, 1022.611],
       [  -0.004, 1122.097, 1122.101],
       [  -0.004, 1221.095, 1221.098],
       [  -0.003, 1319.603, 1319.606],
       [  -0.003, 1417.626, 1417.63 ],
       [  -0.003, 1515.168, 1515.171],
       [  -0.003, 1612.231, 1612.234],
       [  -0.003, 1708.82 , 1708.823],
       [  -0.003, 1804.937, 1804.941],
       [  -0.008, 1961.465, 1961.473],
       [  -0.008, 2189.596, 2189.604],
       [  -0.008, 2416.623, 2416.631],
       [  -0.008, 2642.553, 2642.561],
       [  -0.008, 2867.394, 2867.402],
       [  -0.008, 3091.154, 3091.162],
       [  -0.008, 3313.84 , 3313.849],
       [  -0.008, 3535.462, 3535.47 ],
       [  -0.008, 3756.025, 3756.033],
       [  -0.008, 3975.539, 3975.547],
       [  -0.018, 4386.355, 4386.373],
       [  -0.018, 4871.871, 4871.889],
       [  -0.018, 5355.09 , 5355.108],
       [  -0.018, 5836.03 , 5836.048],
       [  -0.018, 6314.707, 6314.725],
       [  -0.018, 6791.137, 6791.154],
       [  -0.018, 7265.335, 7265.353],
       [  -0.018, 7737.317, 7737.335],
       [  -0.018, 8207.099, 8207.116],
       [  -0.018, 8674.696, 8674.713]])



Building Opticks qu needs updated c4
---------------------------------------

HMM. Need to bump the version. 

::

   N[blyth@localhost junoenv]$ bash junoenv libs all custom4



