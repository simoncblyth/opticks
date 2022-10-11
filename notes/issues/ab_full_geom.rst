ab_full_geom
==============

Objective
------------

Random aligned AB testing with input photons using full (or nearly full) geometry. 



Updating saved geometry : with "GEOM=J004 ntds3"
--------------------------------------------------

New saving dir::

    /home/blyth/.opticks/GEOM/J004

    N[blyth@localhost ~]$ l /home/blyth/.opticks/GEOM/J004/
    total 41012
        0 drwxr-xr-x.  4 blyth blyth      109 Oct 11 22:50 .
    20504 -rw-rw-r--.  1 blyth blyth 20992917 Oct 11 22:50 origin.gdml
        4 -rw-rw-r--.  1 blyth blyth      190 Oct 11 22:50 origin_gdxml_report.txt
    20504 -rw-rw-r--.  1 blyth blyth 20994470 Oct 11 22:50 origin_raw.gdml
        0 drwxrwxr-x. 15 blyth blyth      273 Oct 11 22:50 GGeo
        0 drwxr-xr-x.  3 blyth blyth      190 Oct 11 22:50 CSGFoundry
        0 drwxr-xr-x.  3 blyth blyth       18 Oct 11 22:50 ..


Simplify saving::

    2022-10-11 22:50:10.198 INFO  [371266] [G4CXOpticks::setGeometry@265] ] CSGOptiX::Create 
    2022-10-11 22:50:10.198 INFO  [371266] [G4CXOpticks::setGeometry@267]  cx 0x15eadf1b0 qs 0x15e64ca20 QSim::Get 0x15e64ca20
    2022-10-11 22:50:10.198 INFO  [371266] [G4CXOpticks::setGeometry@272] [ G4CXOpticks__setGeometry_saveGeometry 
    2022-10-11 22:50:10.198 INFO  [371266] [G4CXOpticks::saveGeometry@473] [ /home/blyth/.opticks/GEOM/J004
    2022-10-11 22:50:15.232 INFO  [371266] [BFile::preparePath@837] created directory /home/blyth/.opticks/GEOM/J004/GGeo/GItemList
    2022-10-11 22:50:15.331 INFO  [371266] [BFile::preparePath@837] created directory /home/blyth/.opticks/GEOM/J004/GGeo/GNodeLib
    2022-10-11 22:50:15.620 INFO  [371266] [BFile::preparePath@837] created directory /home/blyth/.opticks/GEOM/J004/GGeo/GScintillatorLib/LS
    2022-10-11 22:50:15.622 INFO  [371266] [BFile::preparePath@837] created directory /home/blyth/.opticks/GEOM/J004/GGeo/GScintillatorLib/LS_ori
    2022-10-11 22:50:17.777 INFO  [371266] [U4GDML::write@152]  ekey U4GDML_GDXML_FIX_DISABLE U4GDML_GDXML_FIX_DISABLE 0 U4GDML_GDXML_FIX 1
    G4GDML: Writing '/home/blyth/.opticks/GEOM/J004/origin_raw.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/home/blyth/.opticks/GEOM/J004/origin_raw.gdml' done !
    2022-10-11 22:50:21.357 INFO  [371266] [U4GDML::write@163]  Apply GDXML::Fix  rawpath /home/blyth/.opticks/GEOM/J004/origin_raw.gdml dstpath /home/blyth/.opticks/GEOM/J004/origin.gdml
    2022-10-11 22:50:21.357 INFO  [371266] [G4CXOpticks::saveGeometry@479] ] /home/blyth/.opticks/GEOM/J004
    2022-10-11 22:50:21.357 INFO  [371266] [G4CXOpticks::setGeometry@274] ] G4CXOpticks__setGeometry_saveGeometry 
    2022-10-11 22:50:21.357 INFO  [371266] [G4CXOpticks::setGeometry@277] ] fd 0x15b5afba0






Old saving dir was /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks::

    2022-10-11 21:43:24.664 INFO  [356625] [G4CXOpticks::setGeometry@282] [ G4CXOpticks__setGeometry_saveGeometry 
    2022-10-11 21:43:24.664 INFO  [356625] [G4CXOpticks::saveGeometry@477] dir [$DefaultOutputDir
    2022-10-11 21:43:24.664 INFO  [356625] [G4CXOpticks::saveGeometry_@488] [ /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks
    2022-10-11 21:43:32.056 INFO  [356625] [U4GDML::write@152]  ekey U4GDML_GDXML_FIX_DISABLE U4GDML_GDXML_FIX_DISABLE 0 U4GDML_GDXML_FIX 1
    G4GDML: Writing '/tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks/origin_raw.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks/origin_raw.gdml' done !
    2022-10-11 21:43:35.456 INFO  [356625] [U4GDML::write@163]  Apply GDXML::Fix  rawpath /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks/origin_raw.gdml dstpath /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks/origin.gdml
    2022-10-11 21:43:35.456 INFO  [356625] [G4CXOpticks::saveGeometry_@494] ] /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks
    2022-10-11 21:43:35.456 INFO  [356625] [G4CXOpticks::setGeometry@284] ] G4CXOpticks__setGeometry_saveGeometry 


::

    epsilon:issues blyth$ GEOM=ntds3 SPathTest  
    2022-10-11 15:04:45.530 INFO  [26722014] [test_Resolve@204] 
                                                            $TMP :                                           /tmp/blyth/opticks
                                               $DefaultOutputDir :                      /tmp/blyth/opticks/GEOM/ntds3/SPathTest
                                                    $OPTICKS_TMP :                                           /tmp/blyth/opticks

::

    epsilon:sysrap blyth$ SOpticksResourceTest --ddod
    SOpticksResource::Desc_DefaultOutputDir() 
    SOpticksResource::Desc_DefaultOutputDir
     SPath::Resolve("$TMP/GEOM",NOOP) /tmp/blyth/opticks/GEOM
     SSys::getenvvar("GEOM") -
     SOpticksResource::ExecutableName() SOpticksResourceTest
     SOpticksResource::DefaultOutputDir() /tmp/blyth/opticks/GEOM/SOpticksResourceTest


This explains "/tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks" the ntds3 comes from SCRIPT envvar swapout of executable name "python" 

That is OK for an output directory, but not really for geometry saving. 




Test Environment : target Hama:0:1000 with storch_test "down" disc beam 
--------------------------------------------------------------------------


Test using bin/OPTICKS_INPUT_PHOTON.sh::

     24 vers=down
     26 path=storch_test/$vers/$(uname)/ph.npy
     27 
     28 if [ -n "$path" ]; then
     29     export OPTICKS_INPUT_PHOTON=$path
     30     export OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000
     31 

bin/GEOM_.sh::

     27 geom=J000
     28 
     29 export GEOM=${GEOM:-$geom}
     30 
     31 reldir(){
     32    case $1 in
     33      J000) echo DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1 ;;
     34    esac
     35 }
     36 
     37 if [ "$GEOM" == "J000" ]; then
     38     export J000_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml
     39     export J000_CFBaseFromGEOM=$HOME/.opticks/geocache/$(reldir $GEOM)/CSG_GGeo
     40     ## to force translation from GDML comment the _CFBaseFromGEOM export 
     41 fi


Issue 1 : B shows 2 extra BT 
-----------------------------------

* :doc:`ab_full_geom_extra_BT`


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



