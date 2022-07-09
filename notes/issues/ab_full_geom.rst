ab_full_geom
==============

Objective
------------

Random aligned AB testing with input photons using full (or nearly full) geometry. 


A : gxs.sh using G4CXSimulateTest
----------------------------------------------- 

Initially can boot from same GDML used for B side.  
But converting geom every time is tedious, so can subsequently 
boot quickly using CFBASE. 

HMM maybe use "GEOM"_CFBaseFromGEOM to provide signal to use the faster route. 

geom_::

     19 #geom=hama_body_log
     20 geom=J000
     21 
     22 export GEOM=${GEOM:-$geom}
     23 
     24 reldir(){
     25    case $1 in
     26      J000) echo DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1 ;;
     27    esac
     28 }  
     29 
     30 if [ "$GEOM" == "J000" ]; then
     31     export J000_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml
     32     export J000_CFBASE=$HOME/.opticks/geocache/$(reldir $GEOM)/CSG_GGeo
     33 fi  


::

     24 struct G4CX_API G4CXOpticks
     25 {   
     26     static const plog::Severity LEVEL ;
     27     static std::string Desc();
     28         
     29     const G4VPhysicalVolume* wd ;
     30     const GGeo*             gg ;
     31     CSGFoundry* fd ;
     32     CSGOptiX*   cx ;  
     33     QSim*       qs ;
     34     
     35  
     36     G4CXOpticks();
     37     std::string desc() const ; 
     38 
     39     void setGeometry(); 
     40     void setGeometry(const char* gdmlpath);
     41     void setGeometry(const G4VPhysicalVolume* wd);
     42     void setGeometry(const GGeo* gg); 
     43     void setGeometry(CSGFoundry* fd);
     44     


* hmm getting current GDML will take some work as Opticks changes mean that 
  will not be able to run the current JUNO-Opticks combo

* so start with the last GDML available : SPath::Resolve("$SomeGDMLPath", NOOP)
  using that path via the SOpticksResource::GDMLPath() 


B : U4RecorderTest.sh using  U4VolumeMaker::PV GEOM
-----------------------------------------------------

::

   export GEOM=JV101
   export JV101_GDMLPath=/..../     # can use SPath tokens 

   u4
   ./U4RecorderTest.sh 


Whats needed
----------------

1. gdml
2. input photons starting in the water and targetting a single PMT instance 
   (need to get the instance transform and author the input photons within the transform frame)



