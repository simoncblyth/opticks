OKX4Test_j1707_bad_gdml
==========================

Want to do a codegen survey on juno solids::

   op --j1707 --okx4 
   op --j1707 --okx4 --g4codegen


But truncated values in the GDML matrix::

     1 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
     2 <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
     3 
     4   <define>
     5     <matrix coldim="2" name="ABSLENGTH0x14819e0" values="1.3778e-06 864.473 1.3793e-06 889.942 1.3808e-06 940.585 1.3824e-06 1019.07 1.3839e-06 1147 1.3855e-"/>
     6     <matrix coldim="2" name="AlphaFASTTIMECONSTANT0x148e950" values="-1 1 1 1"/>
     7     <matrix coldim="2" name="AlphaSLOWTIMECONSTANT0x148ea60" values="-1 35 1 35"/>
     8     <matrix coldim="2" name="AlphaYIELDRATIO0x148eb70" values="-1 0.65 1 0.65"/>
     9     <matrix coldim="2" name="FASTCOMPONENT0x14874e0" values="1.55e-06 0 2.0664e-06 0.007298 2.06985e-06 0.007011 2.07331e-06 0.007932 2.07679e-06 0.008065 2.0802"/>

Prevents parsing the GDML::

    epsilon:x4gen blyth$ op --j1707 --okx4 

    === op-cmdline-binary-match : finds 1st argument with associated binary : --okx4
    400 -rwxr-xr-x  1 blyth  staff  203100 Jul 29 22:17 /usr/local/opticks/lib/OKX4Test
    proceeding.. : /usr/local/opticks/lib/OKX4Test --j1707 --okx4
    /usr/local/opticks/lib/OKX4Test --j1707 --okx4
    2018-07-29 22:52:57.965 ERROR [5603897] [BOpticksResource::init@74] layout : 1
    2018-07-29 22:52:57.965 INFO  [5603897] [SLog::operator@20] BOpticksResource::BOpticksResource  DONE
    2018-07-29 22:52:57.966 INFO  [5603897] [OpticksResource::assignDetectorName@412] OpticksResource::assignDetectorName m_detector juno1707
    2018-07-29 22:52:57.966 INFO  [5603897] [SLog::operator@20] OpticksResource::OpticksResource  DONE
    2018-07-29 22:52:57.966 INFO  [5603897] [SLog::operator@20] Opticks::Opticks  DONE
    2018-07-29 22:52:57.967 INFO  [5603897] [Opticks::dumpArgs@1213] Opticks::configure argc 3
      0 : /usr/local/opticks/lib/OKX4Test
      1 : --j1707
      2 : --okx4
    2018-07-29 22:52:57.967 INFO  [5603897] [OpticksHub::configure@234] OpticksHub::configure argc 3 argv[0] /usr/local/opticks/lib/OKX4Test m_gltf 0 is_tracer 0
    2018-07-29 22:52:57.968 INFO  [5603897] [OpticksHub::loadGeometry@391] OpticksHub::loadGeometry START
    2018-07-29 22:52:57.968 INFO  [5603897] [SLog::operator@20] GGeo::GGeo  DONE
    2018-07-29 22:52:57.968 INFO  [5603897] [OpticksGeometry::loadGeometry@87] OpticksGeometry::loadGeometry START 
    2018-07-29 22:52:57.968 ERROR [5603897] [OpticksGeometry::loadGeometryBase@119] OpticksGeometry::loadGeometryBase START 
    2018-07-29 22:52:57.968 INFO  [5603897] [GGeo::loadGeometry@531] GGeo::loadGeometry START loaded 1 gltf 0
    2018-07-29 22:52:57.968 ERROR [5603897] [GGeo::loadFromCache@660] GGeo::loadFromCache START
    2018-07-29 22:52:57.972 INFO  [5603897] [GMaterialLib::postLoadFromCache@72] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2018-07-29 22:52:57.972 INFO  [5603897] [GMaterialLib::replaceGROUPVEL@597] GMaterialLib::replaceGROUPVEL  ni 15
    ...
    018-07-29 22:52:59.072 WARN  [5603897] [CPropLib::init@72] CPropLib::init surface lib sensor_surface NULL 
    2018-07-29 22:52:59.072 INFO  [5603897] [CPropLib::initCheckConstants@110] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2018-07-29 22:52:59.073 INFO  [5603897] [CGDMLDetector::init@56] CGDMLDetector::init path /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml npath /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml
    G4GDML: Reading '/usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml'...
    G4GDML: Reading definitions...
    Evaluator : syntax error

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : InvalidExpression
          issued by : G4GDMLEvaluator::Evaluate()
    Error in expression: 1.3855e-
    *** Fatal Exception ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------



