ami_reports_assembly_volume_somehow_giving_NoRINDEX_implicits_surfaces_incorrectly
====================================================================================

Do geometry translation
--------------------------

Run the translation and persist the geometry with::

   cd ~/opticks/g4cx/tests
   ./example_pet.sh 


::

    epsilon:tests blyth$ cat example_pet.sh
    #!/bin/bash -l 

    export GEOM=example_pet
    export G4CXOpticks__setGeometry_saveGeometry=$HOME/.opticks/GEOM/$GEOM

    ./G4CXOpticks_setGeometry_Test.sh



Observations on the source GDML
---------------------------------

::

    vi ~/geant_pet/example_pet_opticks/small_pet.gdml


solids
~~~~~~~~~

::

    1026   <solids>
    1027     <box name="WorldBox" x="114.01000000000002" y="147.70600000000002" z="147.70600000000002" lunit="mm"/>

    1028     <box name="lysoBox001" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1029     <box name="lysoBox002" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1030     <box name="lysoBox003" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1031     <box name="lysoBox004" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1032     <box name="lysoBox005" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1033     <box name="lysoBox006" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1034     <box name="lysoBox007" x="27.8" y="27.8" z="8.0" lunit="mm"/>
    1035     <box name="lysoBox008" x="27.8" y="27.8" z="8.0" lunit="mm"/>

    ////     8 identical lysoBox differing only in their names

    1036     <box name="glassBox001" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1037     <box name="glassBox002" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1038     <box name="glassBox003" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1039     <box name="glassBox004" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1040     <box name="glassBox005" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1041     <box name="glassBox006" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1042     <box name="glassBox007" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1043     <box name="glassBox008" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1044     <box name="glassBox009" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1045     <box name="glassBox010" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1046     <box name="glassBox011" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1047     <box name="glassBox012" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1048     <box name="glassBox013" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    ....
    1538     <box name="glassBox503" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1539     <box name="glassBox504" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1540     <box name="glassBox505" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1541     <box name="glassBox506" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1542     <box name="glassBox507" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1543     <box name="glassBox508" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1544     <box name="glassBox509" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1545     <box name="glassBox510" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1546     <box name="glassBox511" x="3.22" y="3.22" z="2.0" lunit="mm"/>
    1547     <box name="glassBox512" x="3.22" y="3.22" z="2.0" lunit="mm"/>

    ////      512 identical glassBox differing only in their names  

    1548     <sphere name="Sphere" rmin="0.0" rmax="5.0" startphi="0.0" deltaphi="360.0" starttheta="0.0" deltatheta="180.0" aunit="deg" lunit="mm"/>
    1549     <opticalsurface name="ScintWrap" model="glisur" finish="0" type="0" value="1.0">
    1550       <property name="EFFICIENCY" ref="EFFICIENCY"/>
    1551       <property name="REFLECTIVITY" ref="REFLECTIVITY"/>
    1552     </opticalsurface>
    1553   </solids>


Why all this pointless duplication of solids ?
You should create the geometry much more simply and efficiently 
using one lysoBox and one glassBox.  That single instance
of solid is then referenced from the logical volumes. 

structure
~~~~~~~~~~~~

::

    1554   <structure>
    1555     <volume name="LV_Sphere">
    1556       <materialref ref="G4_TISSUE_SOFT_ICRP"/>
    1557       <solidref ref="Sphere"/>
    1558       <auxiliary auxtype="Color" auxvalue="#9cff0000"/>
    1559     </volume>

    1560     <volume name="V-glassBox512">
    1561       <materialref ref="Glass"/>
    1562       <solidref ref="glassBox512"/>
    1563       <auxiliary auxtype="Color" auxvalue="#fea30000"/>
    1564       <auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>
    1565     </volume>
    1566     <volume name="V-glassBox511">
    1567       <materialref ref="Glass"/>
    1568       <solidref ref="glassBox511"/>
    1569       <auxiliary auxtype="Color" auxvalue="#fea30000"/>
    1570       <auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>
    1571     </volume>
    1572     <volume name="V-glassBox510">
    1573       <materialref ref="Glass"/>
    1574       <solidref ref="glassBox510"/>
    1575       <auxiliary auxtype="Color" auxvalue="#fea30000"/>
    1576       <auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>
    1577     </volume>
    ////     .......
    ////     .......
    ////     .......
    4614     <volume name="V-glassBox003">
    4615       <materialref ref="Glass"/>
    4616       <solidref ref="glassBox003"/>
    4617       <auxiliary auxtype="Color" auxvalue="#fea30000"/>
    4618       <auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>
    4619     </volume>
    4620     <volume name="V-glassBox002">
    4621       <materialref ref="Glass"/>
    4622       <solidref ref="glassBox002"/>
    4623       <auxiliary auxtype="Color" auxvalue="#fea30000"/>
    4624       <auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>
    4625     </volume>
    4626     <volume name="V-glassBox001">
    4627       <materialref ref="Glass"/>
    4628       <solidref ref="glassBox001"/>
    4629       <auxiliary auxtype="Color" auxvalue="#fea30000"/>
    4630       <auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>
    4631     </volume>

    ////
    ////      512 V-glassBoxNNN logical volumes referencing corresponding glassBoxNNN solids
    ////      all these should be referencing a single glassBox solid  
    ////
    ////      All these 512 logical volumes are identical, 
    ////      you only need one such V-glassBox logical volume. 
    ////


    4632     <assembly name="Part">
    4633       <physvol name="PV-V-glassBox001">
    4634         <volumeref ref="V-glassBox001"/>
    4635         <positionref ref="P-V-glassBox0019"/>
    4636         <rotationref ref="identity"/>
    4637       </physvol>
    4638       <physvol name="PV-V-glassBox002">
    4639         <volumeref ref="V-glassBox002"/>
    4640         <positionref ref="P-V-glassBox00210"/>
    4641         <rotationref ref="identity"/>
    4642       </physvol>

    ////       ....

    7183       <physvol name="PV-V-glassBox511">
    7184         <volumeref ref="V-glassBox511"/>
    7185         <positionref ref="P-V-glassBox511519"/>
    7186         <rotationref ref="R-V-glassBox511454"/>
    7187       </physvol>
    7188       <physvol name="PV-V-glassBox512">
    7189         <volumeref ref="V-glassBox512"/>
    7190         <positionref ref="P-V-glassBox512520"/>
    7191         <rotationref ref="R-V-glassBox512455"/>
    7192       </physvol>
    7193     </assembly>

    ////      "Part" assembly with 512 PV-V-glassBox001...512
    ////
    ////      Because the 512 pv have different transforms they need
    ////      to be separate but they should all be referencing the same 
    ////      V-glassBox logical volume. 
    ////


    7194     <volume name="V-lysoBox008">
    7195       <materialref ref="LYSO"/>
    7196       <solidref ref="lysoBox008"/>
    7197       <auxiliary auxtype="Color" auxvalue="#80808000"/>
    7198     </volume>
    7199     <volume name="V-lysoBox007">
    7200       <materialref ref="LYSO"/>
    7201       <solidref ref="lysoBox007"/>
    7202       <auxiliary auxtype="Color" auxvalue="#80808000"/>
    7203     </volume>
    7204     <volume name="V-lysoBox006">
    7205       <materialref ref="LYSO"/>
    7206       <solidref ref="lysoBox006"/>
    7207       <auxiliary auxtype="Color" auxvalue="#80808000"/>
    7208     </volume>

    ////    .....

    7229     <volume name="V-lysoBox001">
    7230       <materialref ref="LYSO"/>
    7231       <solidref ref="lysoBox001"/>
    7232       <auxiliary auxtype="Color" auxvalue="#80808000"/>
    7233     </volume>


    ////     8 V-lysoBoxNNN logical volumes referencing corresponding lysoBoxNNN
    ////     
    ////     They should be referencing a single lysoBox solid. 
    ////
    ////     These 8 logical volumes are identical : 
    ////     you only need a single such logical volume. 
    ////


    7234     <assembly name="Part001">
    7235       <physvol name="PV-V-lysoBox001">
    7236         <volumeref ref="V-lysoBox001"/>
    7237         <positionref ref="P-V-lysoBox0011"/>
    7238         <rotationref ref="identity"/>
    7239       </physvol>
    7240       <physvol name="PV-V-lysoBox002">
    7241         <volumeref ref="V-lysoBox002"/>
    7242         <positionref ref="P-V-lysoBox0022"/>
    7243         <rotationref ref="R-V-lysoBox0021"/>
    7244       </physvol>

    ////        .......

    7265       <physvol name="PV-V-lysoBox007">
    7266         <volumeref ref="V-lysoBox007"/>
    7267         <positionref ref="P-V-lysoBox0077"/>
    7268         <rotationref ref="R-V-lysoBox0076"/>
    7269       </physvol>
    7270       <physvol name="PV-V-lysoBox008">
    7271         <volumeref ref="V-lysoBox008"/>
    7272         <positionref ref="P-V-lysoBox0088"/>
    7273         <rotationref ref="R-V-lysoBox0087"/>
    7274       </physvol>
    7275     </assembly>

    ////      Part001 assembly of 8 PV-V-lysoBoxNNN referencing V-lysoBoxNNN     
    ////
    ////      They should be referencing a single V-lysoBox 
    ////

    7276     <volume name="worldVOL">
    7277       <materialref ref="G4_AIR"/>
    7278       <solidref ref="WorldBox"/>
    7279       <physvol name="PV-Part001">
    7280         <volumeref ref="Part001"/>
    7281         <positionref ref="center"/>
    7282         <rotationref ref="identity"/>
    7283       </physvol>
    7284       <physvol name="PV-Part">
    7285         <volumeref ref="Part"/>
    7286         <positionref ref="center"/>
    7287         <rotationref ref="identity"/>
    7288       </physvol>
    7289       <physvol name="PV-LV_Sphere">
    7290         <volumeref ref="LV_Sphere"/>
    7291         <positionref ref="center"/>
    7292         <rotationref ref="identity"/>
    7293       </physvol>
    7294     </volume>

    ////     Here the assembly volume soup are incorporated into PV-Part001 and PV-Part


    7295     <bordersurface name="ScintWrap002" surfaceproperty="ScintWrap">
    7296       <physvolref ref="av_2_impr_1_V-lysoBox001_pv_0"/>
    7297       <physvolref ref="av_1_impr_1_V-glassBox001_pv_0"/>
    7298     </bordersurface>
    7299     <bordersurface name="ScintWrap003" surfaceproperty="ScintWrap">
    7300       <physvolref ref="av_2_impr_1_V-lysoBox001_pv_0"/>
    7301       <physvolref ref="av_1_impr_1_V-glassBox002_pv_1"/>
    7302     </bordersurface>

    ////    ...........

    9335     <bordersurface name="ScintWrap512" surfaceproperty="ScintWrap">
    9336       <physvolref ref="av_2_impr_1_V-lysoBox008_pv_7"/>
    9337       <physvolref ref="av_1_impr_1_V-glassBox511_pv_510"/>
    9338     </bordersurface>
    9339     <bordersurface name="ScintWrap513" surfaceproperty="ScintWrap">
    9340       <physvolref ref="av_2_impr_1_V-lysoBox008_pv_7"/>
    9341       <physvolref ref="av_1_impr_1_V-glassBox512_pv_511"/>
    9342     </bordersurface>

    ////        512 ScintWrapNNN border surfaces between "distant cousin" PV
    ////
    ////        Current opticks only recognizes bordersurfaces between parent
    ////        and child. Not between very distant cousins structurally (within the volume tree)
    ////        that are presumably only related by your arrangement of transforms to make 
    ////        them close to each other. 
    ////

    9343   </structure>
    9344   <setup name="Default" version="1.0">
    9345     <world ref="worldVOL"/>
    9346   </setup>
    9347 </gdml>



Take a look at persisted geometry
-----------------------

------------

::

    epsilon:GNodeLib blyth$ pwd
    /Users/blyth/.opticks/GEOM/example_pet/GGeo/GNodeLib
    epsilon:GNodeLib blyth$ l
    total 408
     0 drwxr-xr-x  11 blyth  staff    352 Nov  4 20:32 .
     0 drwxr-xr-x  17 blyth  staff    544 Nov  4 20:29 ..
    96 -rw-r--r--   1 blyth  staff  45744 Nov  4 20:29 GTreePresent.txt
    24 -rw-r--r--   1 blyth  staff   8432 Nov  4 20:29 all_volume_nodeinfo.npy
    24 -rw-r--r--   1 blyth  staff   8432 Nov  4 20:29 all_volume_identity.npy
    24 -rw-r--r--   1 blyth  staff   8432 Nov  4 20:29 all_volume_center_extent.npy
    40 -rw-r--r--   1 blyth  staff  16784 Nov  4 20:29 all_volume_bbox.npy
    72 -rw-r--r--   1 blyth  staff  33488 Nov  4 20:29 all_volume_inverse_transforms.npy
    72 -rw-r--r--   1 blyth  staff  33488 Nov  4 20:29 all_volume_transforms.npy
    16 -rw-r--r--   1 blyth  staff   7291 Nov  4 20:29 all_volume_LVNames.txt
    40 -rw-r--r--   1 blyth  staff  17051 Nov  4 20:29 all_volume_PVNames.txt
    epsilon:GNodeLib blyth$ 


    epsilon:GNodeLib blyth$ wc -l *.txt
         522 GTreePresent.txt
         522 all_volume_LVNames.txt
         522 all_volume_PVNames.txt
        1566 total


Looks like AssemblyVolumes lead to peculiar pv names. 
That could easily mess up the RINDEX_NoRINDEX search for bordersurface:: 

    epsilon:GNodeLib blyth$ head -20 all_volume_PVNames.txt
    worldVOL_PV
    av_2_impr_1_V-lysoBox001_pv_0
    av_2_impr_1_V-lysoBox002_pv_1
    av_2_impr_1_V-lysoBox003_pv_2
    av_2_impr_1_V-lysoBox004_pv_3
    av_2_impr_1_V-lysoBox005_pv_4
    av_2_impr_1_V-lysoBox006_pv_5
    av_2_impr_1_V-lysoBox007_pv_6
    av_2_impr_1_V-lysoBox008_pv_7
    av_1_impr_1_V-glassBox001_pv_0
    av_1_impr_1_V-glassBox002_pv_1
    av_1_impr_1_V-glassBox003_pv_2
    av_1_impr_1_V-glassBox004_pv_3
    av_1_impr_1_V-glassBox005_pv_4
    av_1_impr_1_V-glassBox006_pv_5
    av_1_impr_1_V-glassBox007_pv_6
    av_1_impr_1_V-glassBox008_pv_7
    av_1_impr_1_V-glassBox009_pv_8
    av_1_impr_1_V-glassBox010_pv_9
    av_1_impr_1_V-glassBox011_pv_10

    epsilon:GNodeLib blyth$ tail -10 all_volume_PVNames.txt
    av_1_impr_1_V-glassBox504_pv_503
    av_1_impr_1_V-glassBox505_pv_504
    av_1_impr_1_V-glassBox506_pv_505
    av_1_impr_1_V-glassBox507_pv_506
    av_1_impr_1_V-glassBox508_pv_507
    av_1_impr_1_V-glassBox509_pv_508
    av_1_impr_1_V-glassBox510_pv_509
    av_1_impr_1_V-glassBox511_pv_510
    av_1_impr_1_V-glassBox512_pv_511
    PV-LV_Sphere
    epsilon:GNodeLib blyth$ 


g4-cls G4AssemblyVolume::

    034 // G4AssemblyVolume is a helper class to make the build process of geometry
     35 // easier. It allows to combine several volumes together in an arbitrary way
     36 // in 3D space and then work with the result as with a single logical volume
     37 // for placement.
     38 // The resulting objects are independent copies of each of the assembled
     39 // logical volumes. The placements are not, however, bound one to each other
     40 // when placement is done. They are seen as independent physical volumes in
     41 // space.
     ..
     60 class G4AssemblyVolume
     61 {
     62  public:  // with description
     63 
     64   G4AssemblyVolume();
     65   G4AssemblyVolume( G4LogicalVolume* volume,
     66                     G4ThreeVector& translation,
     67                     G4RotationMatrix* rotation);

    131 
    132   void MakeImprint( G4LogicalVolume* pMotherLV,
    133                     G4ThreeVector& translationInMother,
    134                     G4RotationMatrix* pRotationInMother,
    135                     G4int copyNumBase = 0,
    136                     G4bool surfCheck = false );
    137     //
    138     // Creates instance of an assembly volume inside the given mother volume.
    139 
    140   void MakeImprint( G4LogicalVolume* pMotherLV,
    141                     G4Transform3D&   transformation,
    142                     G4int copyNumBase = 0,
    143                     G4bool surfCheck = false );
    144     //


    220 void G4AssemblyVolume::MakeImprint( G4AssemblyVolume* pAssembly,
    221                                     G4LogicalVolume*  pMotherLV,
    222                                     G4Transform3D&    transformation,
    223                                     G4int copyNumBase,
    224                                     G4bool surfCheck )
    225 {
    226   unsigned int  numberOfDaughters;
    227 
    228   if( copyNumBase == 0 )
    229   { 
    230     numberOfDaughters = pMotherLV->GetNoDaughters();
    231   }
    232   else
    233   {
    234     numberOfDaughters = copyNumBase;
    235   }
    236 
    237   // We start from the first available index
    238   //
    239   numberOfDaughters++;
    240 
    241   ImprintsCountPlus();
    242  
    243   std::vector<G4AssemblyTriplet> triplets = pAssembly->fTriplets;
    244 
    245   for( unsigned int   i = 0; i < triplets.size(); i++ )
    246   {
    247     G4Transform3D Ta( *(triplets[i].GetRotation()),
    248                       triplets[i].GetTranslation() );
    249     if ( triplets[i].IsReflection() )  { Ta = Ta * G4ReflectZ3D(); }
    250 
    251     G4Transform3D Tfinal = transformation * Ta;
    252    
    253     if ( triplets[i].GetVolume() )
    254     {
    255       // Generate the unique name for the next PV instance
    256       // The name has format:
    257       //
    258       // av_WWW_impr_XXX_YYY_ZZZ
    259       // where the fields mean:
    260       // WWW - assembly volume instance number
    261       // XXX - assembly volume imprint number
    262       // YYY - the name of a log. volume we want to make a placement of
    263       // ZZZ - the log. volume index inside the assembly volume
    264       //
    265       std::stringstream pvName;
    266       pvName << "av_"
    267              << GetAssemblyID()
    268              << "_impr_"
    269              << GetImprintsCount()
    270              << "_"
    271              << triplets[i].GetVolume()->GetName().c_str()
    272              << "_pv_"
    273              << i
    274              << std::ends;
    275       
    276       // Generate a new physical volume instance inside a mother
    277       // (as we allow 3D transformation use G4ReflectionFactory to 
    278       //  take into account eventual reflection)
    279       //
    280       G4PhysicalVolumesPair pvPlaced
    281         = G4ReflectionFactory::Instance()->Place( Tfinal,
    282                                                   pvName.str().c_str(),
    283                                                   triplets[i].GetVolume(),
    284                                                   pMotherLV,
    285                                                   false,
    286                                                   numberOfDaughters + i,
    287                                                   surfCheck );
    288       
    289       // Register the physical volume created by us so we can delete it later
    290       //
    291       fPVStore.push_back( pvPlaced.first );





Look at NoRINDEX code
-----------------------

::

    epsilon:opticks blyth$ opticks-fl NoRINDEX
    ./cfg4/CBoundaryProcess.hh
    ./cfg4/DsG4OpBoundaryProcessStatus.h
    ./cfg4/CBoundaryProcess.cc
    ./cfg4/OpStatus.cc
    ./cfg4/DsG4OpBoundaryProcess.cc

     ## cfg4 is an old package that is no longer included in om-subs list 

    ./extg4/X4PhysicalVolume.cc
         X4PhysicalVolume::convertImplicitSurfaces_r 

    ./extg4/X4OpBoundaryProcessStatus.hh
    ./sysrap/SBnd.h

    ./ggeo/GSurfaceLib.hh
    ./ggeo/GSurfaceLib.cc
           GSurfaceLib::addImplicitBorderSurface_RINDEX_NoRINDEX( const char* pv1, const char* pv2 )

    ./u4/U4OpBoundaryProcessStatus.h
    ./u4/U4StepPoint.cc
           U4StepPoint::BoundaryFlag

    ./u4/InstrumentedG4OpBoundaryProcess.hh
    ./u4/InstrumentedG4OpBoundaryProcess.cc
    ./u4/U4Material.cc
    ./examples/Geant4/BoundaryStandalone/G4OpBoundaryProcess_MOCK.cc
    ./examples/Geant4/BoundaryStandalone/G4OpBoundaryProcess_MOCK.hh




X4PhysicalVolume::convertImplicitSurfaces_r
----------------------------------------------


::

     551 void X4PhysicalVolume::convertImplicitSurfaces_r(const G4VPhysicalVolume* const parent_pv, int depth)
     552 {
     553     const G4LogicalVolume* parent_lv = parent_pv->GetLogicalVolume() ;
     554     const G4Material* parent_mt = parent_lv->GetMaterial() ;
     555     const G4String& parent_mtName = parent_mt->GetName();
     556 
     557     G4MaterialPropertiesTable* parent_mpt = parent_mt->GetMaterialPropertiesTable();
     558     const G4MaterialPropertyVector* parent_rindex = parent_mpt ? parent_mpt->GetProperty(kRINDEX) : nullptr ;     // WHAT: cannot do this with const mpt 
     559 
     560     for (size_t i=0 ; i < size_t(parent_lv->GetNoDaughters()) ;i++ )  // G4LogicalVolume::GetNoDaughters returns 1042:G4int, 1062:size_t
     561     {
     562         const G4VPhysicalVolume* const daughter_pv = parent_lv->GetDaughter(i);
     563         const G4LogicalVolume* daughter_lv = daughter_pv->GetLogicalVolume() ;
     564         const G4Material* daughter_mt = daughter_lv->GetMaterial() ;
     565         G4MaterialPropertiesTable* daughter_mpt = daughter_mt->GetMaterialPropertiesTable();
     566         const G4MaterialPropertyVector* daughter_rindex = daughter_mpt ? daughter_mpt->GetProperty(kRINDEX) : nullptr ; // WHAT: cannot do this with const mpt 
     567         const G4String& daughter_mtName = daughter_mt->GetName();
     568 
     569         // naming order for outgoing photons, not ingoing volume traversal  
     570         bool RINDEX_NoRINDEX = daughter_rindex != nullptr && parent_rindex == nullptr ;
     571         bool NoRINDEX_RINDEX = daughter_rindex == nullptr && parent_rindex != nullptr ;
     572 
     573         //if(RINDEX_NoRINDEX || NoRINDEX_RINDEX)
     574         if(RINDEX_NoRINDEX)
     575         {
     576             const char* pv1 = X4::Name( daughter_pv ) ;
     577             const char* pv2 = X4::Name( parent_pv ) ;
     578             GBorderSurface* bs = m_slib->findBorderSurface(pv1, pv2);
     579 
     580             LOG(LEVEL)
     581                << " parent_mtName " << parent_mtName
     582                << " daughter_mtName " << daughter_mtName
     583                ;
     584 
     585             LOG(LEVEL)
     586                 << " RINDEX_NoRINDEX " << RINDEX_NoRINDEX
     587                 << " NoRINDEX_RINDEX " << NoRINDEX_RINDEX
     588                 << " pv1 " << std::setw(30) << pv1
     589                 << " pv2 " << std::setw(30) << pv2


