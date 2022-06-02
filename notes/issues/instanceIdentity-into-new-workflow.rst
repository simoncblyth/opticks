instanceIdentity-into-new-workflow
====================================

Idea for better way in new workflow : U4InstanceIdentifier.h
----------------------------------------------------------------

Within Opticks U4 define pure virtual protocol API using 
only Geant4 and standard types in interface which
returns identity+sensor info from a G4PVPlacement.

* HMM: will be called for all Opticks factorized instances, only some of them will contain sensors 
  so keeping it general

::

     01 #pragma once
      2 /**
      3 U4InstanceIdentifier.h
      4 =======================
      5 
      6 Pure virtual protocol base used to interface Opticks geometry translation 
      7 with detector specific code. 
      8 
      9 getInstanceId
     10     method is called on the outer volume of every factorized instance during geometry translation, 
     11     the returned unsigned value is used by IAS_Builder to set the OptixInstance .instanceId 
     12     Within CSGOptiX/CSGOptiX7.cu:: __closesthit__ch *optixGetInstanceId()* is used to 
     13     passes the instanceId value into "quad2* prd" (per-ray-data) which is available 
     14     within qudarap/qsim.h methods. 
     15     
     16     The 32 bit unsigned returned by *getInstanceIdentity* may not use the top bit as ~0u 
     17     is reserved by OptiX to mean "not-an-instance". So this provides 31 bits of identity 
     18     information per instance.  
     19 
     20 **/
     21 class G4PVPlacement ;
     22 
     23 struct U4InstanceIdentifier
     24 {
     25     virtual unsigned getInstanceId(const G4PVPlacement* pv) = 0 ;
     26 };



In detector framework code implement this protocol 
and pass the pointer to the object that fulfils the 
protocol to the Opticks setGeometry translation, 
or nullptr if not needed. 

This avoids the back and forth between detector 
specific code and Opticks, because the Opticks
translation is able to use the detector specific
identity lookups within itself by calling the protocol
methods. 


HMM how to use that in gx
-----------------------------

* setGeometry(const G4VPhysicalVolume* world) and X4Geo::Translate needs additional "const U4InstanceIdentifier*" argument
* ii array (values for every instance) needs to be persisted within GGeo and passed into CSGFoundry SSim  
  so the other signatures do not change

gx::

     24 void G4CXOpticks::setGeometry(const char* gdmlpath)
     25 {
     26     const G4VPhysicalVolume* world = U4GDML::Read(gdmlpath);
     27     setGeometry(world);
     28 }
     29 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world)
     30 {
     31     wd = world ;
     32     GGeo* gg_ = X4Geo::Translate(wd) ;
     33     setGeometry(gg_);
     34 }
     35 void G4CXOpticks::setGeometry(const GGeo* gg_)
     36 {
     37     gg = gg_ ;
     38     CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ;
     39     setGeometry(fd_);
     40 }
     41 void G4CXOpticks::setGeometry(CSGFoundry* fd_)
     42 {
     43     fd = fd_ ;
     44     cx = CSGOptiX::Create(fd);
     45     qs = cx->sim ;
     46 }



Old way detector specific code
---------------------------------

Not doing the translation in one call, brings complications:

1. pass the world
2. do the instancing
3. return sensor placements 
4. for each placement set sensor index, category, efficiencies 

::

    epsilon:offline blyth$ jcv LSExpDetectorConstruction_Opticks
    2 files to edit
    ./Simulation/DetSimV2/DetSimOptions/include/LSExpDetectorConstruction_Opticks.hh
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc
    epsilon:offline blyth$ 


::

    084 G4Opticks* LSExpDetectorConstruction_Opticks::Setup(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd_, int opticksMode )  // static
     85 {
     86     if( opticksMode == 0 ) return nullptr ;
     87     LOG(info) << "[ WITH_G4OPTICKS opticksMode " << opticksMode  ;
     88 
     89     assert(world);
     90 
     91     // 1. pass geometry to Opticks, translate it to GPU and return sensor placements  
     92 
     93     G4Opticks* g4ok = new G4Opticks ;
    ...
    105     g4ok->setGeometry(world);
    106 
    107     const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;
    108     unsigned num_sensor = sensor_placements.size();
    109 
    110     // 2. use the placements to pass sensor data : efficiencies, categories, identifiers  
    111 
    112     const junoSD_PMT_v2* sd = dynamic_cast<const junoSD_PMT_v2*>(sd_) ;
    113     assert(sd) ;
    114 
    115     LOG(info) << "[ setSensorData num_sensor " << num_sensor ;
    116     for(unsigned i=0 ; i < num_sensor ; i++)
    117     {
    118         const G4PVPlacement* pv = sensor_placements[i] ; // i is 0-based unlike sensor_index
    119         unsigned sensor_index = 1 + i ; // 1-based 
    120         assert(pv);
    121         G4int copyNo = pv->GetCopyNo();
    122         int pmtid = copyNo ;
    123         int pmtcat = 0 ; // sd->getPMTCategory(pmtid); 
    124         float efficiency_1 = sd->getQuantumEfficiency(pmtid);
    125         float efficiency_2 = sd->getEfficiencyScale() ;
    126 
    127         g4ok->setSensorData( sensor_index, efficiency_1, efficiency_2, pmtcat, pmtid );
    128     }
    129     LOG(info) << "] setSensorData num_sensor " << num_sensor ;
    130 
    131     // 3. pass theta dependent efficiency tables for all sensor categories 
    132 
    133     PMTEfficiencyTable* pt = sd->getPMTEfficiencyTable();
    134     assert(pt);
    135 
    136     const std::vector<int>& shape = pt->getShape();
    137     const std::vector<float>& data = pt->getData();
    138 

