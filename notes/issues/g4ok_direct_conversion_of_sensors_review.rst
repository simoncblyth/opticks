g4ok_direct_conversion_of_sensors_review
============================================


G4Opticks::translateGeometry
------------------------------

::

    139 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    140 {
    141     const char* keyspec = X4PhysicalVolume::Key(top) ;
    142     BOpticksKey::SetKey(keyspec);
    143     LOG(error) << " SetKey " << keyspec  ;
    144 
    145     Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    146 
    147     const char* gdmlpath = ok->getGDMLPath();   // inside geocache, not SrcGDMLPath from opticksdata
    148     CGDML::Export( gdmlpath, top );
    149 
    150     GGeo* gg = new GGeo(ok) ;
    151     X4PhysicalVolume xtop(gg, top) ;   // <-- populates gg 
    152     gg->postDirectTranslation();
    153 
    154     int root = 0 ;
    155     const char* gltfpath = ok->getGLTFPath();   // inside geocache
    156     GGeoGLTF::Save(gg, gltfpath, root );
    157 
    158     return gg ;
    159 }


X4PhysicalVolume steers the work
-----------------------------------

::

    125 void X4PhysicalVolume::init()
    126 {
    127     LOG(info) << "query : " << m_query->desc() ;
    128 
    129     convertMaterials();
    130     convertSurfaces();
    131     convertSensors();  // before closeSurfaces as may add some SensorSurfaces
    132     closeSurfaces();
    133     convertSolids();
    134     convertStructure();
    135     convertCheck();
    136 }


X4PhysicalVolume::convertSensors
-----------------------------------

::

    147 void X4PhysicalVolume::convertSensors()
    148 {
    149     LOG(fatal) << "[" ;
    150 
    151     convertSensors_r(m_top, 0);
    152 
    153     unsigned num_clv = m_ggeo->getNumCathodeLV();
    154     LOG(error)
    155          << " m_lvsdname " << m_lvsdname
    156          << " num_clv " << num_clv
    157          ;
    158 
    159     unsigned num_bds = m_ggeo->getNumBorderSurfaces() ;
    160     unsigned num_sks0 = m_ggeo->getNumSkinSurfaces() ;
    161 
    162     GGeoSensor::AddSensorSurfaces(m_ggeo) ;
    163 
    164     unsigned num_sks1 = m_ggeo->getNumSkinSurfaces() ;
    165     assert( num_bds == m_ggeo->getNumBorderSurfaces()  );
    166 
    167     LOG(error)
    168          << " num_bds " << num_bds
    169          << " num_sks0 " << num_sks0
    170          << " num_sks1 " << num_sks1
    171          ;
    172 
    173     LOG(fatal) << "]" ;
    174 }



GGeoSensor::AddSensorSurfaces
------------------------------

Sensor surfaces added based on GGeo CathodeLV::

     36 void GGeoSensor::AddSensorSurfaces( GGeo* gg )
     37 {
     38     GMaterial* cathode_props = gg->getCathode() ;
     39     if(!cathode_props)
     40     { 
     41         LOG(fatal) << " require a cathode material to AddSensorSurfaces " ;
     42         return ; 
     43     }
     44 
     45     unsigned nclv = gg->getNumCathodeLV();
     46 
     47     for(unsigned i=0 ; i < nclv ; i++) 
     48     {   
     49         const char* sslv = gg->getCathodeLV(i);
     50 
     51         unsigned num_mat = gg->getNumMaterials()  ;
     52         unsigned num_sks = gg->getNumSkinSurfaces() ;
     53         unsigned num_bds = gg->getNumBorderSurfaces() ;
     54 
     55         unsigned index = num_mat + num_sks + num_bds ;
     56         // standard materials/surfaces use the originating aiMaterial index, 
     57         // extend that for fake SensorSurface by toting up all 
     58 
     59         LOG(info) << "GGeoSensor::AddSensorSurfaces"
     60                   << " i " << i
     61                   << " sslv " << sslv
     62                   << " index " << index
     63                   << " num_mat " << num_mat
     64                   << " num_sks " << num_sks
     65                   << " num_bds " << num_bds
     66                   ;
     67 
     68         GSkinSurface* gss = MakeSensorSurface(sslv, index);
     69         gss->setStandardDomain();  // default domain 
     70         gss->setSensor();
     71         gss->add(cathode_props);
     72 
     73         LOG(info) << " gss " << gss->description();
     74 
     75         gg->add(gss);
     76 
     77         {
     78             // not setting sensor or domain : only the standardized need those
     79             GSkinSurface* gss_raw = MakeSensorSurface(sslv, index);
     80             gss_raw->add(cathode_props);
     81             gg->addRaw(gss_raw);
     82         }
     83     }
     84 }




Huh convertSensors_r should notice the SD and addLVSD::

    196 void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
    197 {
    198     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    199     const char* lvname = lv->GetName().c_str();
    200     G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
    201 
    202     bool is_lvsdname = m_lvsdname && BStr::Contains(lvname, m_lvsdname, ',' ) ;
    203     bool is_sd = sd != NULL ;
    204 
    205     const std::string sdn = sd ? sd->GetName() : "SD?" ;   // perhaps GetFullPathName() 
    206 
    207     if( is_lvsdname || is_sd )
    208     {
    209         std::string name = BFile::Name(lvname);
    210         std::string nameref = SGDML::GenerateName( name.c_str() , lv , true );
    211         LOG(info)
    212             << " is_lvsdname " << is_lvsdname
    213             << " is_sd " << is_sd
    214             << " name " << name
    215             << " nameref " << nameref
    216             ;
    217 
    218         m_ggeo->addLVSD(nameref.c_str(), sdn.c_str()) ;
    219     }
    220 
    221     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    222     {
    223         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
    224         convertSensors_r(child_pv, depth+1 );
    225     }
    226 }




::

    2019-03-13 20:26:52.701 FATAL [1534889] [X4PhysicalVolume::convertSensors@149] [
    2019-03-13 20:26:52.701 INFO  [1534889] [X4PhysicalVolume::convertSensors_r@211]  is_lvsdname 0 is_sd 1 name Det nameref Det0x110caa210
    2019-03-13 20:26:52.701 ERROR [1534889] [X4PhysicalVolume::convertSensors@154]  m_lvsdname (null) num_clv 1
    2019-03-13 20:26:52.701 INFO  [1534889] [GGeoSensor::AddSensorSurfaces@59] GGeoSensor::AddSensorSurfaces i 0 sslv Det0x110caa210 index 3 num_mat 3 num_sks 0 num_bds 0
    2019-03-13 20:26:52.701 FATAL [1534889] [*GGeoSensor::MakeOpticalSurface@103]  sslv Det0x110caa210 name Det0x110caa210SensorSurface
    2019-03-13 20:26:52.701 ERROR [1534889] [GPropertyMap<float>::setStandardDomain@278]  setStandardDomain(NULL) -> default_domain  GDomain  low 60 high 820 step 20 length 39
    2019-03-13 20:26:52.701 INFO  [1534889] [GGeoSensor::AddSensorSurfaces@73]  gss GSS:: GPropertyMap<T>::  3    skinsurface s: GOpticalSurface  type 0 model 1 finish 3 value     1   Det0x110caa210SensorSurface k:RINDEX EFFICIENCY GROUPVEL
    2019-03-13 20:26:52.701 FATAL [1534889] [*GGeoSensor::MakeOpticalSurface@103]  sslv Det0x110caa210 name Det0x110caa210SensorSurface
    2019-03-13 20:26:52.701 ERROR [1534889] [X4PhysicalVolume::convertSensors@167]  num_bds 0 num_sks0 0 num_sks1 1
    2019-03-13 20:26:52.701 FATAL [1534889] [X4PhysicalVolume::convertSensors@173] ]



G4OK bouncemax zero, historical for checking generation::

    42 const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  ;


Removing "--bouncemax 0" yields 42 Opticks side hits, less than 91+17 from G4::

    2019-03-13 20:48:56.582 ERROR [1545649] [EventAction::EndOfEventAction@42]  num_hits 42 hits 0x136a757e0
    2019-03-13 20:48:56.582 INFO  [1545649] [SensitiveDetector::DumpHitCollections@159]  query SD0/OpHitCollectionA hcid    0 hc 0x110ccbd00 hc.entries 91
    2019-03-13 20:48:56.582 INFO  [1545649] [SensitiveDetector::DumpHitCollections@159]  query SD0/OpHitCollectionB hcid    1 hc 0x110ccbd48 hc.entries 17
    2019-03-13 20:48:56.583 INFO  [1545649] [RunAction::EndOfRunAction@30] .
    2019-03-13 20:48:56.583 INFO  [1545649] [RunAction::EndOfRunAction@32] G4Opticks ok 0x110f54ac0 opmgr 0x110f72b60


::



