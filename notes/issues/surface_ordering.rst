surface_ordering
==================

This spawned from :doc:`direct_route_boundary_match`

strategy
---------

* want to not have to specify any ordering of surfaces via json, 
  as was done in the old route : ie want to just use the order from G4

* need to have old route and new one agree on the order ... using the 
  order in the G4DAE seems the logical choice : as it might well be (or could be)
  the original G4 ordering 


Workflow comparing old Assimp/G4DAE route with X4 direct route
----------------------------------------------------------------------

1. run the old route::

   op --gdml2gltf
   OPTICKS_RESOURCE_LAYOUT=104 OKTest -G --gltf 3  

   ## loads G4DAE assimp + the analytic description from python

* comparing the idx with ab-idx : see lots of zeros from old route  , FIXED see ab-idx-notes


2. run the new route, using the geocache written by the above for fixup::

   OPTICKS_RESOURCE_LAYOUT=104 OKX4Test  

   ## loads GDML to conjure up the G4 model, fixes omissions with G4DAE,
   ## then converts the G4 model directly to Opticks GGeo   


3. NumPy geometry comparisons with ab-l ab-ls ab-bnd ab-blib ab-part etc...
   see notes in ab-vi

   ab-surf
   ab-surf1



Material Ordering 
--------------------

* A is ordered into the preference order : actually there is good reason to 
  keep that order ... 

* A with sorting switching off (GMaterialLib) get something close
  to alphabetical : Probably ColladaLoader map mangling again.

* B X4MaterialTable::init is just following the G4 order, getting same order 
  as in the GDML 


How to get the direct route to follow the desired order without specifying it ?
Need to control the G4 order that is bases of.

::

    g4-cls G4MaterialTable   ## just a vector of G4Material


Added a g4dae_material_srcidx property to ColladaParser and 
grabbed that in AssimpGGeo and set it on GPropertyMap MetaKV of
the materials.  This allowing to sort the G4DAE loaded materials
by their original Geant4 ordering. 

Modulo the test materials (which should be stuffed on then end and not sorted, 
or set to an appropriate srcidx to make them stay in their place)
this gets the same ordering for A and B : the original G4 ordering.  

But its not the desired ordering ...



FIXED name mismatch 
---------------------

new way finds LV based on their names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    182 X4PhysicalVolume::convertSensors_r
    183 -----------------------------------
    184 
    185 Collect (in m_ggeo) LV names that match strings 
    186 specified by m_lvsdname "LV sensitive detector name"
    187 eg something like "Cathode,cathode"   
    188 

::

    191 void X4PhysicalVolume::convertSensors_r(const G4VPhysicalVolume* const pv, int depth)
    192 {   
    193     const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    194     const char* lvname = lv->GetName().c_str(); 
    195     bool is_lvsdname = BStr::Contains(lvname, m_lvsdname, ',' ) ;
    196     
    197     if(is_lvsdname)
    198     {   
    199         m_ggeo->addCathodeLV(lvname) ;
    200     }
    201     
    202     for (int i=0 ; i < lv->GetNoDaughters() ;i++ )
    203     {
    204         const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);
    205         convertSensors_r(child_pv, depth+1 );
    206     }
    207 }




::

    2018-08-04 20:57:35.432 FATAL [8860707] [X4PhysicalVolume::convertSurfaces@268] ]
    2018-08-04 20:57:35.432 FATAL [8860707] [X4PhysicalVolume::convertSensors@156] [
    2018-08-04 20:57:35.464 INFO  [8860707] [GGeoSensor::AddSensorSurfaces@45] GGeoSensor::AddSensorSurfaces i 0 sslv /dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98 index 80
    2018-08-04 20:57:35.464 FATAL [8860707] [*GGeoSensor::MakeOpticalSurface@87]  sslv /dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98 name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2018-08-04 20:57:35.464 ERROR [8860707] [GPropertyMap<float>::setStandardDomain@269]  setStandardDomain(NULL) -> default_domain  GDomain  low 60 high 820 step 20 length 39
    2018-08-04 20:57:35.464 INFO  [8860707] [GGeoSensor::AddSensorSurfaces@56]  gss GSS:: GPropertyMap<T>:: 80    skinsurface s: GOpticalSurface  type 0 model 1 finish 3 value     1/dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface k:ABSLENGTH EFFICIENCY GROUPVEL RAYLEIGH REEMISSIONPROB RINDEX
    2018-08-04 20:57:35.464 INFO  [8860707] [GSurfaceLib::add@379]  GSkinSurface /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2018-08-04 20:57:35.464 FATAL [8860707] [*GGeoSensor::MakeOpticalSurface@87]  sslv /dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98 name /dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    2018-08-04 20:57:35.465 INFO  [8860707] [GGeoSensor::AddSensorSurfaces@45] GGeoSensor::AddSensorSurfaces i 1 sslv /dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca0 index 81
    2018-08-04 20:57:35.465 FATAL [8860707] [*GGeoSensor::MakeOpticalSurface@87]  sslv /dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca0 name /dd/Geometry/PMT/lvPmtHemiCathodeSensorSurface
    2018-08-04 20:57:35.465 ERROR [8860707] [GPropertyMap<float>::setStandardDomain@269]  setStandardDomain(NULL) -> default_domain  GDomain  low 60 high 820 step 20 length 39
    2018-08-04 20:57:35.465 INFO  [8860707] [GGeoSensor::AddSensorSurfaces@56]  gss GSS:: GPropertyMap<T>:: 81    skinsurface s: GOpticalSurface  type 0 model 1 finish 3 value     1/dd/Geometry/PMT/lvPmtHemiCathodeSensorSurface k:ABSLENGTH EFFICIENCY GROUPVEL RAYLEIGH REEMISSIONPROB RINDEX
    2018-08-04 20:57:35.465 INFO  [8860707] [GSurfaceLib::add@379]  GSkinSurface /dd/Geometry/PMT/lvPmtHemiCathodeSensorSurface
    2018-08-04 20:57:35.465 FATAL [8860707] [*GGeoSensor::MakeOpticalSurface@87]  sslv /dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca0 name /dd/Geometry/PMT/lvPmtHemiCathodeSensorSurface
    2018-08-04 20:57:35.465 ERROR [8860707] [X4PhysicalVolume::convertSensors@169]  m_lvsdname PmtHemiCathode,HeadonPmtCathode num_clv 2 num_bds 8 num_sks0 34 num_sks1 36
    2018-08-04 20:57:35.465 FATAL [8860707] [X4PhysicalVolume::convertSensors@177] ]
    2018-08-04 20:57:35.465 ERROR [8860707] [GSurfaceLib::sort@475]  not sorting 
    2018-08-04 20:57:35.465 INFO  [8860707] [GPropertyLib::close@423] GPropertyLib::close type GSurfaceLib buf 48,2,39,4




following fixes reach : same surfaces and order, but a name difference for sensorsurface
-------------------------------------------------------------------------------------------

Also with live geometry must check the digest following something that changes geometry.
Check the dates of the cachemeta.json to see when the caches were last written
If the B date is not keeping up, there is probably a digest change.::

    epsilon:npy blyth$ ab-l
    A -rw-r--r-- 1 blyth staff 63 Aug 4 20:15 /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/104/cachemeta.json
    B -rw-r--r-- 1 blyth staff 53 Aug 4 20:46 /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/742ab212f7f2da665ed627411ebdb07d/1/cachemeta.json


ab-blib::

    perfectAbsorbSurface						perfectAbsorbSurface
    perfectSpecularSurface						perfectSpecularSurface
    perfectDiffuseSurface						perfectDiffuseSurface
    lvHeadonPmtCathodeSensorSurface				      |	/dd/Geometry/PMT/lvHeadonPmtCathodeSensorSurface
    lvPmtHemiCathodeSensorSurface				      |	/dd/Geometry/PMT/lvPmtHemiCathodeSensorSurface


assimp fork fix
-----------------

::

    epsilon:code blyth$ git commit -m "workaround loss of ordering of SkinSurface and BorderSurface in the map by changing the key to keep order "
    [master f2f90350] workaround loss of ordering of SkinSurface and BorderSurface in the map by changing the key to keep order
     1 file changed, 28 insertions(+), 5 deletions(-)
    epsilon:code blyth$ git push 
    warning: redirecting to https://github.com/simoncblyth/assimp.git/
    Counting objects: 4, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (4/4), done.
    Writing objects: 100% (4/4), 900 bytes | 900.00 KiB/s, done.
    Total 4 (delta 3), reused 0 (delta 0)
    remote: Resolving deltas: 100% (3/3), completed with 3 local objects.
    To http://github.com/simoncblyth/assimp.git
       caa04750..f2f90350  master -> master
    epsilon:code blyth$ 


Dumping default slib has sensor surfaces, but the 104 skipped em 
-------------------------------------------------------------------

* somehow matching cathode GMaterial on pointer not working, have to match by cathode material name (maybe standardization other object  ?)


::

    OKX4Test 

    2018-08-04 16:41:31.817 INFO  [8695989] [SLog::operator@21] OpticksHub::OpticksHub  DONE
    2018-08-04 16:41:31.817 ERROR [8695989] [CPropLib::init@68] CPropLib::init
    2018-08-04 16:41:31.817 INFO  [8695989] [CPropLib::init@70] GSurfaceLib numSurfaces 48 this 0x7fe0e85c7e80 basis 0x0 isClosed 1 hasDomain 1
    2018-08-04 16:41:31.817 INFO  [8695989] [GSurfaceLib::Summary@231] GSurfaceLib::dump NumSurfaces 48 NumFloat4 2
    2018-08-04 16:41:31.817 INFO  [8695989] [GSurfaceLib::dump@1040]  (index,type,finish,value) 
    2018-08-04 16:41:31.817 WARN  [8695989] [GSurfaceLib::dump@1047]                NearPoolCoverSurface (  0,  0,  3,100) (  0)               dielectric_metal                        ground value 100
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                NearDeadLinerSurface (  1,  0,  3, 20) (  1)               dielectric_metal                        ground value 20
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                 NearOWSLinerSurface (  2,  0,  3, 20) (  2)               dielectric_metal                        ground value 20
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]               NearIWSCurtainSurface (  3,  0,  3, 20) (  3)               dielectric_metal                        ground value 20
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                SSTWaterSurfaceNear1 (  4,  0,  3,100) (  4)               dielectric_metal                        ground value 100
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                       SSTOilSurface (  5,  0,  3,100) (  5)               dielectric_metal                        ground value 100
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]       lvPmtHemiCathodeSensorSurface (  6,  0,  3,100) (  6)               dielectric_metal                        ground value 100
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]     lvHeadonPmtCathodeSensorSurface (  7,  0,  3,100) (  7)               dielectric_metal                        ground value 100
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                        RSOilSurface (  8,  0,  3,100) (  8)               dielectric_metal                        ground value 100
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                    ESRAirSurfaceTop (  9,  0,  0,  0) (  9)               dielectric_metal                      polished value 0
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                    ESRAirSurfaceBot ( 10,  0,  0,  0) ( 10)               dielectric_metal                      polished value 0
    2018-08-04 16:41:31.818 WARN  [8695989] [GSurfaceLib::dump@1047]                  AdCableTraySurface ( 11,  0,  3,100) ( 11)               dielectric_metal                        ground value 100


    OPTICKS_RESOURCE_LAYOUT=104 OKX4Test 

    2018-08-04 16:45:31.574 INFO  [8698092] [SLog::operator@21] OpticksHub::OpticksHub  DONE
    2018-08-04 16:45:31.574 ERROR [8698092] [CPropLib::init@68] CPropLib::init
    2018-08-04 16:45:31.574 INFO  [8698092] [CPropLib::init@70] GSurfaceLib numSurfaces 46 this 0x7fec97fb14e0 basis 0x0 isClosed 1 hasDomain 1
    2018-08-04 16:45:31.574 INFO  [8698092] [GSurfaceLib::Summary@231] GSurfaceLib::dump NumSurfaces 46 NumFloat4 2
    2018-08-04 16:45:31.574 INFO  [8698092] [GSurfaceLib::dump@1040]  (index,type,finish,value) 
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]                    ESRAirSurfaceTop (  0,  0,  0,  0) (  0)               dielectric_metal                      polished value 0
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]                    ESRAirSurfaceBot (  1,  0,  0,  0) (  1)               dielectric_metal                      polished value 0
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]                       SSTOilSurface (  2,  0,  3,100) (  2)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]                SSTWaterSurfaceNear1 (  3,  0,  3,100) (  3)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]                SSTWaterSurfaceNear2 (  4,  0,  3,100) (  4)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]               NearIWSCurtainSurface (  5,  0,  3, 20) (  5)               dielectric_metal                        ground value 20
    2018-08-04 16:45:31.574 WARN  [8698092] [GSurfaceLib::dump@1047]                 NearOWSLinerSurface (  6,  0,  3, 20) (  6)               dielectric_metal                        ground value 20
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                NearDeadLinerSurface (  7,  0,  3, 20) (  7)               dielectric_metal                        ground value 20
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                NearPoolCoverSurface (  8,  0,  3,100) (  8)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                        RSOilSurface (  9,  0,  3,100) (  9)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                  AdCableTraySurface ( 10,  0,  3,100) ( 10)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                 PmtMtTopRingSurface ( 11,  0,  3,100) ( 11)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                PmtMtBaseRingSurface ( 12,  0,  3,100) ( 12)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                    PmtMtRib1Surface ( 13,  0,  3,100) ( 13)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                    PmtMtRib2Surface ( 14,  0,  3,100) ( 14)               dielectric_metal                        ground value 100
    2018-08-04 16:45:31.575 WARN  [8698092] [GSurfaceLib::dump@1047]                    PmtMtRib3Surface ( 15,  0,  3,100) ( 15)               dielectric_metal                        ground value 100


Running OKX4Test direct using fixup from the 104 geocache
------------------------------------------------------------

::

     
    OPTICKS_RESOURCE_LAYOUT=104 OKTest -G --gltf 3    ## create the 104 geocache : with G4DAE ordering unmangled 

    OPTICKS_RESOURCE_LAYOUT=104 OKX4Test   ## run direct, using fixup from 104 ... see lib additions are in correct order 


    2018-08-04 19:07:08.918 ERROR [8777288] [X4LogicalBorderSurfaceTable::init@32]  NumberOfBorderSurfaces 8
    2018-08-04 19:07:08.918 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface ESRAirSurfaceTop
    2018-08-04 19:07:08.918 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface ESRAirSurfaceBot
    2018-08-04 19:07:08.918 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface SSTOilSurface
    2018-08-04 19:07:08.918 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface SSTWaterSurfaceNear1
    2018-08-04 19:07:08.918 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface SSTWaterSurfaceNear2
    2018-08-04 19:07:08.918 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface NearIWSCurtainSurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface NearOWSLinerSurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@323]  GBorderSurface NearDeadLinerSurface
    2018-08-04 19:07:08.919 ERROR [8777288] [X4LogicalSkinSurfaceTable::init@32]  NumberOfSkinSurfaces num_src 34
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface NearPoolCoverSurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface RSOilSurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface AdCableTraySurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface PmtMtTopRingSurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface PmtMtBaseRingSurface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface PmtMtRib1Surface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface PmtMtRib2Surface
    2018-08-04 19:07:08.919 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface PmtMtRib3Surface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface LegInIWSTubSurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface TablePanelSurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface SupportRib1Surface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface SupportRib5Surface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface SlopeRib1Surface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface SlopeRib5Surface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface ADVertiCableTraySurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface ShortParCableTraySurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface NearInnInPiperSurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface NearInnOutPiperSurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface LegInOWSTubSurface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib6Surface
    2018-08-04 19:07:08.920 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib7Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib3Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib5Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib4Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib1Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib2Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib8Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface UnistrutRib9Surface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface TopShortCableTraySurface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface TopCornerCableTraySurface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface VertiCableTraySurface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface NearOutInPiperSurface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface NearOutOutPiperSurface
    2018-08-04 19:07:08.921 INFO  [8777288] [GSurfaceLib::add@379]  GSkinSurface LegInDeadTubSurface
    2018-08-04 19:07:08.921 INFO  [8777288] [X4PhysicalVolume::convertSurfaces@261] convertSurfaces num_lbs 8 num_sks 34




Try to run from the 104
-------------------------

::

    epsilon:boostrap blyth$ OPTICKS_RESOURCE_LAYOUT=104 OKX4Test 
    ...
    2018-08-04 16:10:22.279 INFO  [8673638] [CGDMLDetector::init@62] parse /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    G4GDML: Reading '/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml' done!
    2018-08-04 16:10:22.537 INFO  [8673638] [CDetector::setTop@81] .
    2018-08-04 16:10:22.662 INFO  [8673638] [CTraverser::Summary@105] CDetector::traverse numMaterials 36 numMaterialsWithoutMPT 36
    2018-08-04 16:10:22.663 WARN  [8673638] [CGDMLDetector::addMPT@118] CGDMLDetector::addMPT ALL G4 MATERIALS LACK MPT  FIXING USING G4DAE MATERIALS 
    2018-08-04 16:10:22.663 WARN  [8673638] [CPropLib::addConstProperty@330] CPropLib::addConstProperty OVERRIDE GdDopedLS.SCINTILLATIONYIELD from 11522 to 10
    2018-08-04 16:10:22.663 WARN  [8673638] [CPropLib::addConstProperty@330] CPropLib::addConstProperty OVERRIDE LiquidScintillator.SCINTILLATIONYIELD from 11522 to 10
    2018-08-04 16:10:22.664 FATAL [8673638] [*CPropLib::makeMaterialPropertiesTable@218] CPropLib::makeMaterialPropertiesTable material with SENSOR_MATERIAL name Bialkali but no sensor_surface 
    2018-08-04 16:10:22.664 FATAL [8673638] [*CPropLib::makeMaterialPropertiesTable@222] m_sensor_surface is obtained from slib at CPropLib::init  when Bialkai material is in the mlib  it is required for a sensor surface (with EFFICIENCY/detect) property  to be in the slib 
    Assertion failed: (surf), function makeMaterialPropertiesTable, file /Users/blyth/opticks/cfg4/CPropLib.cc, line 228.
    Abort trap: 6
    epsilon:~ blyth$ 



From ab-blib-notes
---------------------

4. surface count matching but ORDERING DIFFERS 

::

    epsilon:0 blyth$ diff -y $(ab-a-idpath)/GItemList/GSurfaceLib.txt $(ab-b-idpath)/GItemList/GSurfaceLib.txt

    NearPoolCoverSurface<
    NearDeadLinerSurface    NearDeadLinerSurface
    NearOWSLinerSurface     NearOWSLinerSurface
    NearIWSCurtainSurface   NearIWSCurtainSurface
    SSTWaterSurfaceNear1    SSTWaterSurfaceNear1
    SSTOilSurface           SSTOilSurface
    RSOilSurface<
    ESRAirSurfaceTop        ESRAirSurfaceTop
    ESRAirSurfaceBot        ESRAirSurfaceBot
    AdCableTraySurface<
    SSTWaterSurfaceNear2    SSTWaterSurfaceNear2
                           >NearPoolCoverSurface
                           >RSOilSurface
                           >AdCableTraySurface
    PmtMtTopRingSurface     PmtMtTopRingSurface
    PmtMtBaseRingSurface    PmtMtBaseRingSurface
    PmtMtRib1Surface        PmtMtRib1Surface


* switching off sorting in A in GSurfaceLib makes the ordering differ more 
* B order is that coming out of the G4 border and skin surface tables



::

    2018-08-04 14:04:19.628 ERROR [8603404] [X4LogicalBorderSurfaceTable::init@32]  NumberOfBorderSurfaces 8
    2018-08-04 14:04:19.628 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] NearDeadLinerSurface
    2018-08-04 14:04:19.628 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] NearOWSLinerSurface
    2018-08-04 14:04:19.628 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] NearIWSCurtainSurface
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] SSTWaterSurfaceNear1
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] SSTOilSurface
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] ESRAirSurfaceTop
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] ESRAirSurfaceBot
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalBorderSurfaceTable::init@38] SSTWaterSurfaceNear2


Huh B order doesnt follow the order in the G4DAE::

    153290       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
    153291         <physvolref ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468"/>
    153292         <physvolref ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0"/>
    153293       </bordersurface>
    153294       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot">
    153295         <physvolref ref="__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458"/>
    153296         <physvolref ref="__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08"/>
    153297       </bordersurface>
    153298       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface">
    153299         <physvolref ref="__dd__Geometry__AD__lvSST--pvOIL0xc241510"/>
    153300         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xc128d90"/>
    153301       </bordersurface>
    153302       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1">
    153303         <physvolref ref="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528"/>
    153304         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xc128d90"/>
    153305       </bordersurface>
    153306       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2">
    153307         <physvolref ref="__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8"/>
    153308         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xc128d90"/>
    153309       </bordersurface>
    153310       <bordersurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface">
    153311         <physvolref ref="__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498"/>
    153312         <physvolref ref="__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20"/>
    153313       </bordersurface>
    153314       <bordersurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface">
    153315         <physvolref ref="__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10"/>
    153316         <physvolref ref="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270"/>
    153317       </bordersurface>
    153318       <bordersurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface">
    153319         <physvolref ref="__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018"/>
    153320         <physvolref ref="__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270"/>
    153321       </bordersurface>


After some fixup of ColladaParser by changing the name, get the same order as in the G4DAE:: 

    ColladaParser::DumpExtraBorderSurface 
    bs 0 BS:000:__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop 
    BorderSurface::Summary
     nam __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop
     osn __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop
     pv1 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
     pv2 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0
     osp 0x0x0 
    bs 1 BS:001:__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot 
    BorderSurface::Summary
     nam __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot
     osn __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot
     pv1 __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
     pv2 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08
     osp 0x0x0 
    bs 2 BS:002:__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface 
    BorderSurface::Summary
     nam __dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface
     osn __dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface
     pv1 __dd__Geometry__AD__lvSST--pvOIL0xc241510
     pv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90
     osp 0x0x0 
    bs 3 BS:003:__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 
    BorderSurface::Summary
     nam __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1
     osn __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1
     pv1 __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528
     pv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90
     osp 0x0x0 



::

    153188       <skinsurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface">
    153189         <volumeref ref="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"/>
    153190       </skinsurface>
    153191       <skinsurface name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface">
    153192         <volumeref ref="__dd__Geometry__AdDetails__lvRadialShieldUnit0xc3d7ec0"/>
    153193       </skinsurface>
    153194       <skinsurface name="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface">
    153195         <volumeref ref="__dd__Geometry__AdDetails__lvAdVertiCableTray0xc3a27f0"/>
    153196       </skinsurface>
    153197       <skinsurface name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface" surfaceproperty="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface">
    153198         <volumeref ref="__dd__Geometry__PMT__lvPmtTopRing0xc3486f0"/>
    153199       </skinsurface>



    ss 0 SS:000:__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface 
    SkinSurface::Summary
     n   __dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface
     osn __dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface
     v    __dd__Geometry__PoolDetails__lvNearTopCover0xc137060
     os  0x0x0 
    ss 1 SS:001:__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface 
    SkinSurface::Summary
     n   __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface
     osn __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface
     v    __dd__Geometry__AdDetails__lvRadialShieldUnit0xc3d7ec0
     os  0x0x0 
    ss 2 SS:002:__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface 
    SkinSurface::Summary
     n   __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface
     osn __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface
     v    __dd__Geometry__AdDetails__lvAdVertiCableTray0xc3a27f0
     os  0x0x0 
    ss 3 SS:003:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface 
    SkinSurface::Summary
     n   __dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface
     osn __dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface
     v    __dd__Geometry__PMT__lvPmtTopRing0xc3486f0
     os  0x0x0 
    ss 4 SS:004:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface 
    SkinSurface::Summary
     n   __dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface
     osn __dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface
     v    __dd__Geometry__PMT__lvPmtBaseRing0xc00f400
     os  0x0x0 



::

    2018-08-04 15:36:26.180 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:000:__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:001:__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:002:__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:003:__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:004:__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:005:__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:006:__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface
    2018-08-04 15:36:26.181 INFO  [8654037] [GSurfaceLib::add@323]  GBorderSurface BS:007:__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface
    2018-08-04 15:36:26.182 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:000:__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface
    2018-08-04 15:36:26.182 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:001:__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface
    2018-08-04 15:36:26.182 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:002:__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface
    2018-08-04 15:36:26.182 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:003:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface
    2018-08-04 15:36:26.182 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:004:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface
    2018-08-04 15:36:26.182 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:005:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib1Surface
    2018-08-04 15:36:26.183 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:006:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib2Surface
    2018-08-04 15:36:26.183 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:007:__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib3Surface
    2018-08-04 15:36:26.183 INFO  [8654037] [GSurfaceLib::add@379]  GSkinSurface SS:008:__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInIWSTubSurface




Recompile assimp with dumping in ColladaParser suggests are loosing the order due to the map::

    ColladaParser::DumpExtraBorderSurface 
    ColladaParser::DumpExtraSkinSurface 
    ss 0 __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface 
    SkinSurface::Summary
     n   __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface
     osn __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface
     v    __dd__Geometry__AdDetails__lvAdVertiCableTray0xc3a27f0
     os  0x0x7f8cb964c218 
    OpticalSurface::Summary __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface 3 1 0 1 0x0x7f8cb964bc30 
    ExtraProperties::Summary
     REFLECTIVITY : REFLECTIVITY0xccef2e8 
     RINDEX : RINDEX0xc0d2610 
    ss 1 __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface 
    SkinSurface::Summary
     n   __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface
     osn __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface
     v    __dd__Geometry__AdDetails__lvRadialShieldUnit0xc3d7ec0
     os  0x0x7f8cb964af68 
    OpticalSurface::Summary __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface 3 1 0 1 0x0x7f8cb964ab00 
    ExtraProperties::Summary
     BACKSCATTERCONSTANT : BACKSCATTERCONSTANT0xc28d340 
     REFLECTIVITY : REFLECTIVITY0xc563328 
     SPECULARLOBECONSTANT : SPECULARLOBECONSTANT0xbfa85d0 
     SPECULARSPIKECONSTANT : SPECULARSPIKECONSTANT0xc03fc20 
    ss 2 __dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnInPiperSurface 
    SkinSurface::Summary
     n   __dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnInPiperSurface
     osn __dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnInPiperSurface
     v    __dd__Geometry__PoolDetails__lvInnInWaterPipeNearTub0xbf29660
     os  0x0x7f8cb964fa68 
    OpticalSurface::Summary __dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnInPiperSurface 3 1 0 1 0x0x7f8cb964faf0 
    ExtraProperties::Summary



    2018-08-04 14:04:19.629 ERROR [8603404] [X4LogicalSkinSurfaceTable::init@32]  NumberOfSkinSurfaces num_src 34
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] NearPoolCoverSurface
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] RSOilSurface
    2018-08-04 14:04:19.629 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] AdCableTraySurface
    2018-08-04 14:04:19.630 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] PmtMtTopRingSurface
    2018-08-04 14:04:19.630 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] PmtMtBaseRingSurface
    2018-08-04 14:04:19.630 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] PmtMtRib1Surface
    2018-08-04 14:04:19.630 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] PmtMtRib2Surface
    2018-08-04 14:04:19.630 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] PmtMtRib3Surface
    2018-08-04 14:04:19.630 INFO  [8603404] [X4LogicalSkinSurfaceTable::init@38] LegInIWSTubSurface


    153188       <skinsurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface">
    153189         <volumeref ref="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"/>
    153190       </skinsurface>
    153191       <skinsurface name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface">
    153192         <volumeref ref="__dd__Geometry__AdDetails__lvRadialShieldUnit0xc3d7ec0"/>
    153193       </skinsurface>
    153194       <skinsurface name="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface">
    153195         <volumeref ref="__dd__Geometry__AdDetails__lvAdVertiCableTray0xc3a27f0"/>
    153196       </skinsurface>
    153197       <skinsurface name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface" surfaceproperty="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface">
    153198         <volumeref ref="__dd__Geometry__PMT__lvPmtTopRing0xc3486f0"/>
    153199       </skinsurface>

    153287       <skinsurface name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInDeadTubSurface" surfaceproperty="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInDeadTubSurface">
    153288         <volumeref ref="__dd__Geometry__PoolDetails__lvLegInDeadTub0xce5bea8"/>
    153289       </skinsurface>


    153290       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
    153291         <physvolref ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468"/>
    153292         <physvolref ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0"/>
    153293       </bordersurface>
    153294       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot">
    153295         <physvolref ref="__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458"/>
    153296         <physvolref ref="__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08"/>
    153297       </bordersurface>
    153298       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface">
    153299         <physvolref ref="__dd__Geometry__AD__lvSST--pvOIL0xc241510"/>
    153300         <physvolref ref="__dd__Geometry__AD__lvADE--pvSST0xc128d90"/>
    153301       </bordersurface>












