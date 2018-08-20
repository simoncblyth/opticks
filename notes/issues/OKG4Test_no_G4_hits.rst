OKG4Test_no_G4_hits
=====================

* parent :doc:`OKG4Test_direct_two_executable_shakedown` 


Approach:

0. add CSensitiveDetector and CHit to CG4 based on the ckm- ones
1. add CSensitiveDetector argument to CDetector and CDetector::hookupSD
   invoked from init using traverser to access the LV by name from the CTraverser
   using the getCathodeLV from GGeo

   * BUT : it dont work : GGeo doesnt persist the cathode lv
   * so move to NMeta based LVSD, that gets further but run into GDML names with pointers   
   * fixed that, twas stale GDML 




Hmm prediction of the GDML LV name not correct::

    epsilon:1 blyth$ cat cachemeta.json
    {"answer":42,"argline":" /usr/local/opticks/lib/CerenkovMinimal","lv2sd":{"Det0x110e97c30":"SD0"},"question":"huh?"}epsilon:1 blyth$ 
    epsilon:1 blyth$ 
    epsilon:1 blyth$ 
    epsilon:1 blyth$ vi g4ok.gdml
    epsilon:1 blyth$ grep Det g4ok.gdml
        <box lunit="mm" name="Det0x110f979c0" x="800" y="800" z="20"/>
        <volume name="Det0x110f97a30">
          <solidref ref="Det0x110f979c0"/>
          <physvol name="Det0x110f97b00">
            <volumeref ref="Det0x110f97a30"/>
            <position name="Det0x110f97b00_pos" unit="mm" x="0" y="0" z="100"/>
    epsilon:1 blyth$ 


Moving to always writing GDML in CGDML::Export avoids stale pointers in GDML not matching those in cachemeta.json LV2SD::

    epsilon:1 blyth$ l
    total 56
    ...
    -rw-r--r--  1 blyth  staff   144 Aug 20 20:53 primaries.npy
    -rw-r--r--  1 blyth  staff   173 Aug 20 20:53 solids.txt
    -rw-r--r--  1 blyth  staff  5927 Aug 20 20:53 g4ok.gltf
    -rw-r--r--  1 blyth  staff   116 Aug 20 20:53 cachemeta.json
    drwxr-xr-x  6 blyth  staff   192 Aug 20 20:53 GMeshLib
    ...
    epsilon:1 blyth$ grep Det g4ok.gdml 
        <box lunit="mm" name="Det0x110c9a890" x="800" y="800" z="20"/>
        <volume name="Det0x110c9a900">
          <solidref ref="Det0x110c9a890"/>
          <physvol name="Det0x110c9a9d0">
            <volumeref ref="Det0x110c9a900"/>
            <position name="Det0x110c9a9d0_pos" unit="mm" x="0" y="0" z="100"/>
    epsilon:1 blyth$ cat cachemeta.json
    {"answer":42,"argline":" /usr/local/opticks/lib/CerenkovMinimal","lv2sd":{"Det0x110c9a900":"SD0"},"question":"huh?"}epsilon:1 blyth$ 
    epsilon:1 blyth$ 





ckm CerenokovMimimal in executable one associates an SD with the LV::

    171 G4VPhysicalVolume* DetectorConstruction::Construct()
    172 {
    ...
    184     G4Material* glass = MakeGlass();    // slab of sensitive glass in the water 
    185     AddProperty(glass, "EFFICIENCY", MakeConstantProperty(0.5));
    186 
    187     G4Box* so_2 = new G4Box("Det",400.,400.,10.);  // half sizes 
    188     G4LogicalVolume* lv_2 = new G4LogicalVolume(so_2,glass,"Det",0,0,0);
    189     G4VPhysicalVolume* pv_2 = new G4PVPlacement(0,G4ThreeVector(0,0,100.),lv_2 ,"Det",lv_1,false,0);
    190     assert( pv_2 );
    191 
    192     G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist();        assert( SDMan && " SDMan should have been created before now " );
    193     G4VSensitiveDetector* sd = SDMan->FindSensitiveDetector(sdname); assert( sd && " failed for find sd with sdname " );
    194     lv_2->SetSensitiveDetector(sd);
    195 

    016 DetectorConstruction::DetectorConstruction( const char* sdname_ )
     17     :
     18     G4VUserDetectorConstruction(),
     19     sdname(strdup(sdname_))
     20 {
     21 }

::

    epsilon:extg4 blyth$ grep GetSensitiveDetector *.*
    X4PhysicalVolume.cc:    G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ; 


    169 /**
    170 X4PhysicalVolume::convertSensors_r
    171 -----------------------------------
    172 
    173 Sensors are identified by two approaches:
    174 
    175 1. logical volume having an associated sensitive detector G4VSensitiveDetector
    176 2. name of logical volume matching one of a comma delimited list 
    177    of strings provided by the "LV sensitive detector name" option
    178    eg  "--lvsdname Cathode,cathode,Sensor,SD" 
    179 
    180 The second approach is useful as a workaround when operating 
    181 with a GDML loaded geometry, as GDML does not yet(?) persist 
    182 the SD LV association.
    183 
    184 Names of sensitive LV are inserted into a set datastructure in GGeo. 
    185 
    186 **/
    187 
    188 void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
    189 {
    190     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    191     const char* lvname = lv->GetName().c_str();
    192     G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
    193 
    194     bool is_lvsdname = m_lvsdname && BStr::Contains(lvname, m_lvsdname, ',' ) ;
    195     bool is_sd = sd != NULL ;
    196 
    197     if( is_lvsdname || is_sd )
    198     {
    199         std::string name = BFile::Name(lvname);
    200         LOG(info)
    201             << " is_lvsdname " << is_lvsdname
    202             << " is_sd " << is_sd
    203             << " name " << name
    204             ;
    205 
    206         m_ggeo->addCathodeLV(name.c_str()) ;
    207     }
    208 
    209     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    210     {
    211         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
    212         convertSensors_r(child_pv, depth+1 );
    213     }
    214 }



