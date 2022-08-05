instanceIdentity-into-new-workflow
====================================


OptiX 7 Limits
---------------

::

    2022-06-14 19:10:50.195 ERROR [367071] [SGeo::SetLastUploadCFBase@30]  cfbase /tmp/blyth/opticks/GeoChain/BoxedSphere
    [ 4][       KNOBS]: All knobs on default.

    [ 4][  DISK CACHE]: Opened database: "/var/tmp/OptixCache_blyth/cache7.db"
    [ 4][  DISK CACHE]:     Cache data size: "70.4 MiB"
    Properties::dump
                          limitMaxTraceDepth :         31
               limitMaxTraversableGraphDepth :         16
                    limitMaxPrimitivesPerGas :  536870912  20000000
                     limitMaxInstancesPerIas :   16777216   1000000
                               rtcoreVersion :          0
                          limitMaxInstanceId :   16777215    ffffff
          limitNumBitsInstanceVisibilityMask :          8
                    limitMaxSbtRecordsPerGas :   16777216   1000000
                           limitMaxSbtOffset :   16777215    ffffff
    2022-06-14 19:10:50.258 INFO  [367071] [PIP::ExceptionFlags@136]  options STACK_OVERFLOW exceptionFlags 1



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


In detector framework code needs to implement this protocol 
and pass the pointer to the object that fulfils the 
protocol to the Opticks setGeometry translation, 
or nullptr if not needed. 

This avoids the back and forth between detector 
specific code and Opticks, because the Opticks
translation is able to use the detector specific
identity lookups within itself by calling the protocol
methods. 


HMM how to use the identifier in gx
--------------------------------------

* it needs to populate sensor instance identity arrays such that intersects can provide the identifier

* setGeometry(const G4VPhysicalVolume* world) and X4Geo::Translate needs additional "const U4InstanceIdentifier*" argument

* HMM: avoid duplication with::

    G4CXOpticks::setIdentifier(const U4InstanceIdentifier* id_ )

* BUT: if it really only makes sense with "G4CXOpticks::setGeometry(const G4VPhysicalVolume* world)" should add it there 

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


New way of sensor handling ? Going back to basics for simplicity
--------------------------------------------------------------------

Old way seems contorted and fragile ? Involving collecting surface indices
of surfaces with an efficiency. How to do it more directly ?

Go back to basics::

    1617 G4bool InstrumentedG4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
    1618 {
    1619   G4Step aStep = *pStep;
    1620 
    1621   aStep.AddTotalEnergyDeposit(thePhotonMomentum);
    1622 
    1623   G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
    1624   if (sd) return sd->Hit(&aStep);
    1625   else return false;
    1626 }


g4-cls G4LogicalVolume::

    285     G4VSensitiveDetector* GetSensitiveDetector() const;
    286       // Gets current SensitiveDetector.
    287     void SetSensitiveDetector(G4VSensitiveDetector *pSDetector);
    288       // Sets SensitiveDetector (can be 0).


For every logical volume in the U4Tree::convertNodes_r traverse check GetSensitiveDetector
and incorporate into stree/snode accordingly.  

HMM: maybe the reason for the contortions was that SD is set on more volumes than
just the sensitive ones.

Also SD not surviving GDML is a factor. So need to collect that info into the stree 
on the primary GDML writing pass and read it back in subsequent GDML running. 

TODO: check this is U4TreeTest.cc::

    epsilon:offline blyth$ jgr SetSensitiveDetector
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  body_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  inner1_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  face_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  face_interior_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/Hello8inchPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/Hello8inchPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/R12860TorusPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/R12860TorusPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/MCP20inchPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/MCP20inchPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);


Handling GDML dropping SD
---------------------------

* could arrange for GDML sidecar with the persisted stree 


Ideas on sensor_placements
----------------------------

Pure Geant4 code that traverses the volume tree can easily reproduce the sensor order.
So there is no need for Opticks API to provide that. 
Can just provide example code depending only on Geant4 that gets the sensor volumes in 
the same order that Opticks does.  

Then the API for accepting sensor info can just accept an array with first dimension 
the number of sensors. Hmm a higher level way would be to accept a vector of sensor struct.  



Old way sensor_placements
----------------------------

::

    0648 void G4Opticks::setGeometry(const GGeo* ggeo)
     649 {
     650     bool loaded = ggeo->isLoadedFromCache() ;
     651     unsigned num_sensor = ggeo->getNumSensorVolumes();
     652 
     653 
     654     if( loaded == false )
     655     {
     656         if(m_placement_outer_volume) LOG(error) << "CAUTION : m_placement_outer_volume TRUE " ;
     657         X4PhysicalVolume::GetSensorPlacements(ggeo, m_sensor_placements, m_placement_outer_volume);
     658         assert( num_sensor == m_sensor_placements.size() ) ;
     659     }
     660 

    2009 void X4PhysicalVolume::GetSensorPlacements(const GGeo* gg, std::vector<G4PVPlacement*>& placements, bool outer_volume ) // static
    2010 {
    2011     placements.clear();
    2012 
    2013     std::vector<void*> placements_ ;
    2014     gg->getSensorPlacements(placements_, outer_volume);
    2015 
    2016     for(unsigned i=0 ; i < placements_.size() ; i++)
    2017     {
    2018          G4PVPlacement* p = static_cast<G4PVPlacement*>(placements_[i]);
    2019          placements.push_back(p);
    2020     }
    2021 }

    1171 void GGeo::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const
    1172 {
    1173     m_nodelib->getSensorPlacements(placements, outer_volume);
    1174 }

    639 /**
    640 GNodeLib::getSensorPlacements
    641 ------------------------------
    642 
    643 TODO: eliminate the outer_volume kludge 
    644 
    645 When outer_volume = true the placements returned are not 
    646 those of the sensors themselves but rather those of the 
    647 outer volumes of the instances that contain the sensors.
    648 
    649 That is probably a kludge needed because it is the 
    650 CopyNo of the  outer volume that carries the sensorId
    651 for JUNO.  Need a way of getting that from the actual placed
    652 sensor volume in detector specific code, not here.
    653 
    654 **/
    655 
    656 void GNodeLib::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const
    657 {
    658     unsigned numSensorVolumes = getNumSensorVolumes();
    659     LOG(LEVEL) << "numSensorVolumes " << numSensorVolumes ;
    660     for(unsigned i=0 ; i < numSensorVolumes ; i++)
    661     {
    662         unsigned sensorIndex = 1 + i ; // 1-based
    663         const GVolume* sensor = getSensorVolume(sensorIndex) ;
    664         assert(sensor);
    665 
    666         void* origin = NULL ;
    667 
    668         if(outer_volume)
    669         {
    670             const GVolume* outer = sensor->getOuterVolume() ;
    671             assert(outer);
    672             origin = outer->getOriginNode() ;
    673             assert(origin);
    674         }
    675         else
    676         {
    677             origin = sensor->getOriginNode() ;
    678             assert(origin);
    679         }
    680 
    681         placements.push_back(origin);
    682     }
    683 }


    424 void GNodeLib::addVolume(const GVolume* volume)
    425 {
    ...
    461     bool is_sensor = volume->hasSensorIndex(); // volume with 1-based sensorIndex assigned
    462     if(is_sensor)
    463     {
    464         m_sensor_volumes.push_back(volume);
    465         m_sensor_identity.push_back(id);
    466         m_num_sensors += 1 ;
    467     }
    468 
    469 
    470 
    471     const void* origin = volume->getOriginNode() ;
    472     int origin_copyNumber = volume->getOriginCopyNumber() ;
    473 


    308 /**
    309 GVolume::setSensorIndex
    310 -------------------------
    311 
    312 sensorIndex is expected to be a 1-based contiguous index, with the 
    313 default value of SENSOR_UNSET (0)  meaning no sensor.
    314 
    315 This is canonically invoked from X4PhysicalVolume::convertNode during GVolume creation.
    316 
    317 * GNode::setSensorIndices duplicates the index to all faces of m_mesh triangulated geometry
    318 
    319 **/
    320 void GVolume::setSensorIndex(unsigned sensorIndex)
    321 {
    322     m_sensorIndex = sensorIndex ;
    323     setSensorIndices( m_sensorIndex );
    324 }
    325 unsigned GVolume::getSensorIndex() const
    326 {
    327     return m_sensorIndex ;
    328 }
    329 bool GVolume::hasSensorIndex() const
    330 {
    331     return m_sensorIndex != SENSOR_UNSET ;
    332 }


    1679 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recurs     ive_select )
    1680 {
    ....
    1838 
    1839     GVolume* volume = new GVolume(ndIdx, gtransform, mesh, origin_node, origin_copyNumber );
    1840     volume->setBoundary( boundary );   // must setBoundary before adding sensor volume 
    1841     volume->setCopyNumber(copyNumber);  // NB within instances this is changed by GInstancer::labelRepeats_r 
                                                 // when m_duplicate_outernode_copynumber is true
    ...
    1860     bool is_sensor = m_blib->isSensorBoundary(boundary) ; // this means that isurf/osurf has non-zero EFFICIENCY property 
    1861     unsigned sensorIndex = GVolume::SENSOR_UNSET ;
    1862     if(is_sensor)
    1863     {
    1864         sensorIndex = 1 + m_blib->getSensorCount() ;  // 1-based index
    1865         m_blib->countSensorBoundary(boundary);
    1866     }
    1867     volume->setSensorIndex(sensorIndex);   // must set to GVolume::SENSOR_UNSET for non-sensors, for sensor_indices array  
    1868 

    0654 /**
     655 GBndLib::isSensorBoundary
     656 --------------------------
     657 
     658 Canonically invoked from X4PhysicalVolume::convertNode 
     659 
     660 
     661 **/
     662 
     663 bool GBndLib::isSensorBoundary(unsigned boundary) const
     664 {
     665     const guint4& bnd = m_bnd[boundary];
     666     bool osur_sensor = m_slib->isSensorIndex(bnd[OSUR]);
     667     bool isur_sensor = m_slib->isSensorIndex(bnd[ISUR]);
     668     bool is_sensor = osur_sensor || isur_sensor ;
     669     return is_sensor ;
     670 }

    1040 /**
    1041 GPropertyLib::isSensorIndex
    1042 ----------------------------
    1043 
    1044 Checks for the presense of the index within m_sensor_indices, which 
    1045 is a pre-cache transient (non-persisted) vector of surface indices
    1046 from the GSurfaceLib subclass or material indices 
    1047 from GMaterialLib subclass.
    1048 
    1049 **/
    1050 
    1051 bool GPropertyLib::isSensorIndex(unsigned index) const
    1052 {
    1053     typedef std::vector<unsigned>::const_iterator UI ;
    1054     UI b = m_sensor_indices.begin();
    1055     UI e = m_sensor_indices.end();
    1056     UI i = std::find(b, e, index);
    1057     return i != e ;
    1058 }
    1059 
    1060 /**
    1061 GPropertyLib::addSensorIndex
    1062 ------------------------------
    1063 
    1064 Canonically invoked from GSurfaceLib::collectSensorIndices
    1065 based on finding non-zero EFFICIENCY property.
    1066 
    1067 **/
    1068 void GPropertyLib::addSensorIndex(unsigned index)
    1069 {
    1070     m_sensor_indices.push_back(index);
    1071 }


    0878 /**
     879 GSurfaceLib::collectSensorIndices
     880 ----------------------------------
     881 
     882 Loops over all surfaces collecting the 
     883 indices of surfaces having non-zero EFFICIENCY or detect
     884 properties.
     885 
     886 **/
     887 
     888 void GSurfaceLib::collectSensorIndices()
     889 {
     890     unsigned ni = getNumSurfaces();
     891     for(unsigned i=0 ; i < ni ; i++)
     892     {
     893         GPropertyMap<double>* surf = m_surfaces[i] ;
     894         bool is_sensor = surf->isSensor() ;
     895         if(is_sensor)
     896         {
     897             addSensorIndex(i);
     898             assert( isSensorIndex(i) == true ) ;
     899         }
     900     }
     901 }




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

