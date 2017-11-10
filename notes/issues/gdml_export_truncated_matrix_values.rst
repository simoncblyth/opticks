gdml_export_truncated_matrix_values
=====================================

GDML Export of a test geometry has truncated matrix values attribute::

    simon:opticks blyth$ head -20  /tmp/blyth/opticks/CGeometry/CGeometry.gdml
    <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="">

      <define>
        <matrix coldim="2" name="ABSLENGTH0x10fd68d60" values="1.512e-06 0.0001 1.5498e-06 0.0001 1.58954e-06 333.755 1.63137e-06 529.476 1.67546e-06 587.911 1.72"/>
        <matrix coldim="2" name="GROUPVEL0x10fd69d50" values="1.512e-06 205.619 1.5498e-06 205.619 1.58954e-06 205.619 1.63137e-06 205.619 1.67546e-06 205.619 1."/>
        <matrix coldim="2" name="RAYLEIGH0x10fd69850" values="1.512e-06 1e+06 1.5498e-06 1e+06 1.58954e-06 1e+06 1.63137e-06 1e+06 1.67546e-06 1e+06 1.722e-06 1e"/>
        <matrix coldim="2" name="REEMISSIONPROB0x10fd69790" values="1.512e-06 0 1.5498e-06 0 1.58954e-06 0 1.63137e-06 0 1.67546e-06 0 1.722e-06 0 1.7712e-06 0 1.8233e"/>
        <matrix coldim="2" name="RINDEX0x10fd69580" values="1.512e-06 1.458 1.5498e-06 1.458 1.58954e-06 1.458 1.63137e-06 1.458 1.67546e-06 1.458 1.722e-06 1."/>
        <matrix coldim="2" name="ABSLENGTH0x10fd65df0" values="1.512e-06 1e+06 1.5498e-06 1e+06 1.58954e-06 1e+06 1.63137e-06 1e+06 1.67546e-06 1e+06 1.722e-06 1e"/>
        <matrix coldim="2" name="GROUPVEL0x10fd66ff0" values="1.512e-06 299.792 1.5498e-06 299.792 1.58954e-06 299.792 1.63137e-06 299.792 1.67546e-06 299.792 1."/>
        <matrix coldim="2" name="RAYLEIGH0x10fd66af0" values="1.512e-06 1e+06 1.5498e-06 1e+06 1.58954e-06 1e+06 1.63137e-06 1e+06 1.67546e-06 1e+06 1.722e-06 1e"/>
        <matrix coldim="2" name="REEMISSIONPROB0x10fd66a30" values="1.512e-06 0 1.5498e-06 0 1.58954e-06 0 1.63137e-06 0 1.67546e-06 0 1.722e-06 0 1.7712e-06 0 1.8233e"/>
        <matrix coldim="2" name="RINDEX0x10fd66820" values="1.512e-06 1 1.5498e-06 1 1.58954e-06 1 1.63137e-06 1 1.67546e-06 1 1.722e-06 1 1.7712e-06 1 1.8233e"/>
      </define>

      <materials>
        <isotope N="1" Z="1" name="H10x10fd665b0">
          <atom unit="g/mole" value="1.00782503081372"/>
        </isotope>
    simon:opticks blyth$ 


::


    simon:cfg4 blyth$ tboolean-;tboolean-media-g --okg4

    (lldb) target create "CTestDetectorTest"
    Current executable set to 'CTestDetectorTest' (x86_64).
    (lldb) settings set -- target.run-args  "--test" "--testconfig" "analytic=1_csgpath=/tmp/blyth/opticks/tboolean-media--_mode=PyCsgInBox_outerfirst=1_name=tboolean-media--" "--export" "--dbgsurf"
    (lldb) r
    Process 3412 launched: '/usr/local/opticks/lib/CTestDetectorTest' (x86_64)
    2017-11-10 12:27:07.398 INFO  [4111827] [main@47] CTestDetectorTest
    2017-11-10 12:27:07.401 INFO  [4111827] [Opticks::dumpArgs@811] Opticks::configure argc 6
      0 : CTestDetectorTest
      1 : --test
      2 : --testconfig
      3 : analytic=1_csgpath=/tmp/blyth/opticks/tboolean-media--_mode=PyCsgInBox_outerfirst=1_name=tboolean-media--
      4 : --export
      5 : --dbgsurf
    2017-11-10 12:27:07.401 INFO  [4111827] [OpticksHub::configure@170] OpticksHub::configure m_gltf 0
    2017-11-10 12:27:07.402 INFO  [4111827] [OpticksHub::loadGeometry@300] OpticksHub::loadGeometry START


    2017-11-10 12:27:08.710 INFO  [4111827] [CCheck::checkSurf@33] CCheck::checkSurf NumberOfBorderSurfaces 1
     checkSurfTraverse  depth 0 daughterCount 1 lv UNIVERSE_LV
     checkSurfTraverse  depth 1 daughterCount 0 lv box_lv0_
     daughter 0 pv box_pv0_ bsurf 0x10fd6be70 bsurf_v1 box_pv0_ bsurf_v2 UNIVERSE_PV
    2017-11-10 12:27:08.710 INFO  [4111827] [CDetector::export_gdml@344] export to /tmp/blyth/opticks/CGeometry/CGeometry.gdml
    G4GDML: Writing '/tmp/blyth/opticks/CGeometry/CGeometry.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/tmp/blyth/opticks/CGeometry/CGeometry.gdml' done !

