OKX4Test_j1808 : GDML 1042 booted, direct conversion to Opticks/GGeo geocache
=========================================================================================

After fixed :doc:`OpticksResourceTest_j1808_geokey_not_getting_thru_when_have_no_dae`.

    
Try to do a codegen survey on juno solids::

   op --j1808 --okx4 
   op --j1808 --okx4 --g4codegen

Sidestepping OpticksResource is the correct thing to do as want to start from pure G4::

   OKX4Test --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml  
   lldb OKX4Test --  --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml


rerun with codegen : to investigate PMT CSG structure
------------------------------------------------------------

::

    opticksdata-
    GLTF_ROOT=0 OKX4Test --gdmlpath $(opticksdata-j) --g4codegen




Old way : fails for lack of DAE
-----------------------------------

Not unsurprisingly::

    epsilon:opticks blyth$ op --j1808 --okx4 
    === op-cmdline-binary-match : finds 1st argument with associated binary : --okx4
    400 -rwxr-xr-x  1 blyth  staff  202572 Aug 29 22:29 /usr/local/opticks/lib/OKX4Test
    proceeding.. : /usr/local/opticks/lib/OKX4Test --j1808 --okx4
    /usr/local/opticks/lib/OKX4Test --j1808 --okx4
    PLOG::PLOG  instance 0x7fcb59c03fe0 this 0x7fcb59c03fe0 logpath /usr/local/opticks/lib/OKX4Test.log
    2018-08-30 12:57:06.373 INFO  [3975808] [SLog::SLog@12] Opticks::Opticks 
    2018-08-30 12:57:06.376 ERROR [3975808] [BOpticksResource::init@83] layout : 1
    2018-08-30 12:57:06.376 ERROR [3975808] [OpticksResource::init@250] OpticksResource::init
    2018-08-30 12:57:06.376 ERROR [3975808] [OpticksResource::readOpticksEnvironment@509]  inipath /usr/local/opticks/opticksdata/config/opticksdata.ini
    2018-08-30 12:57:06.377 ERROR [3975808] [OpticksResource::readEnvironment@572]  initial m_geokey OPTICKSDATA_DAEPATH_J1808
    2018-08-30 12:57:06.377 ERROR [3975808] [BOpticksResource::setupViaSrc@473]  srcpath /usr/local/opticks/opticksdata/export/juno1808/g4_00.dae srcdigest a181a603769c1f98ad927e7367c7aa51
    2018-08-30 12:57:06.377 INFO  [3975808] [BOpticksResource::getIdPathPath@164]  idpath /usr/local/opticks/geocache/juno1808/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1
    2018-08-30 12:57:06.377 INFO  [3975808] [BOpticksResource::getIdPathPath@164]  idpath /usr/local/opticks/geocache/juno1808/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1
    2018-08-30 12:57:06.377 INFO  [3975808] [OpticksResource::assignDetectorName@404] OpticksResource::assignDetectorName m_detector juno1707
    ...
    2018-08-30 12:57:06.381 ERROR [3975808] [GGeo::loadFromG4DAE@610] GGeo::loadFromG4DAE START
    2018-08-30 12:57:06.381 INFO  [3975808] [AssimpGGeo::load@143] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/juno1808/g4_00.dae query all ctrl  importVerbosity 0 loaderVerbosity 0
    2018-08-30 12:57:06.381 FATAL [3975808] [AssimpGGeo::load@155]  missing G4DAE path /usr/local/opticks/opticksdata/export/juno1808/g4_00.dae
    2018-08-30 12:57:06.381 FATAL [3975808] [GGeo::loadFromG4DAE@615] GGeo::loadFromG4DAE FAILED : probably you need to download opticksdata 
    Assertion failed: (rc == 0 && "G4DAE geometry file does not exist, try : opticksdata- ; opticksdata-- "), function loadFromG4DAE, file /Users/blyth/opticks/ggeo/GGeo.cc, line 619.
    /Users/blyth/opticks/bin/op.sh: line 876: 36905 Abort trap: 6           /usr/local/opticks/lib/OKX4Test --j1808 --okx4
    /Users/blyth/opticks/bin/op.sh RC 134



Direct way : problem with optical surfaces, warnings for no-mpt materials
------------------------------------------------------------------------------

Try a direct from GDML route in OKX4Test adding low level commandline option --gdmlpath to SAr,
so this sidesteps the OpticksResource machinery:: 

    OKX4Test --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml 

::

    epsilon:okg4 blyth$ OKX4Test --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml
    2018-08-30 13:32:55.743 INFO  [4000866] [main@74]  parsing /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml
    G4GDML: Reading '/usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml' done!
    2018-08-30 13:32:58.329 INFO  [4000866] [main@81] ///////////////////////////////// 
    2018-08-30 13:32:59.888 ERROR [4000866] [main@88]  SetKey OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
    2018-08-30 13:32:59.888 INFO  [4000866] [SLog::SLog@12] Opticks::Opticks 
    2018-08-30 13:32:59.889 ERROR [4000866] [BOpticksResource::init@83] layout : 1
    2018-08-30 13:32:59.889 ERROR [4000866] [OpticksResource::init@250] OpticksResource::init
    2018-08-30 13:32:59.890 ERROR [4000866] [OpticksResource::readOpticksEnvironment@509]  inipath /usr/local/opticks/opticksdata/config/opticksdata.ini
    2018-08-30 13:32:59.890 INFO  [4000866] [BOpticksResource::setupViaKey@392] BOpticksKey
                        spec  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
                     exename  : OKX4Test
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : 15cf540d9c315b7f5d0adc7c3907b922
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2018-08-30 13:32:59.890 INFO  [4000866] [BOpticksResource::setupViaKey@428]  idname OKX4Test_lWorld0x4bc2710_PV_g4live idfile g4ok.gltf srcdigest 15cf540d9c315b7f5d0adc7c3907b922 idpath /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1
    ...
    2018-08-30 13:32:59.890 INFO  [4000866] [OpticksResource::assignDetectorName@404] OpticksResource::assignDetectorName m_detector g4live
    2018-08-30 13:32:59.890 ERROR [4000866] [OpticksResource::init@272] OpticksResource::init DONE
    2018-08-30 13:32:59.890 INFO  [4000866] [SLog::operator@27] Opticks::Opticks  DONE
    2018-08-30 13:32:59.890 INFO  [4000866] [Opticks::dumpArgs@1276] Opticks::configure argc 6
      0 : OKX4Test
      1 : --gdmlpath
      2 : /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml
      3 : --tracer
      4 : --nogeocache
      5 : --xanalytic

    ...


7/17 materials have no properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

But it seems Geant4 10.4.2 GDML is not complete wrt the materials ? No it seems not an G4 level 
problem as some materials have mpt::

    2018-08-30 13:41:16.791 INFO  [4006633] [X4MaterialTable::init@59] . G4 nmat 17
    2018-08-30 13:41:16.791 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt LS
    2018-08-30 13:41:16.792 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Steel
    2018-08-30 13:41:16.792 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Tyvek
    2018-08-30 13:41:16.792 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Air
    2018-08-30 13:41:16.792 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt Scintillator
    2018-08-30 13:41:16.792 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt TiO2Coating
    2018-08-30 13:41:16.792 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt Adhesive
    2018-08-30 13:41:16.792 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt Aluminium
    2018-08-30 13:41:16.792 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Rock
    2018-08-30 13:41:16.792 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Acrylic
    2018-08-30 13:41:16.793 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt Copper
    2018-08-30 13:41:16.793 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Vacuum
    2018-08-30 13:41:16.793 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Pyrex
    2018-08-30 13:41:16.793 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt Water
    2018-08-30 13:41:16.793 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt Teflon
    2018-08-30 13:41:16.793 INFO  [4006633] [X4MaterialTable::init@72]  converting material with mpt vetoWater
    2018-08-30 13:41:16.793 WARN  [4006633] [X4MaterialTable::init@68] skip convert of material with no mpt Galactic
    2018-08-30 13:41:16.793 FATAL [4006633] [X4PhysicalVolume::convertMaterials@240] . num_materials 10


eg Scintillator has no property refs::

   310     <material name="Scintillator0x4bbd230" state="solid">
   311       <T unit="K" value="293.15"/>
   312       <MEE unit="eV" value="64.6844741120544"/>
   313       <D unit="g/cm3" value="1.032"/>
   314       <fraction n="0.0854556223030713" ref="Hydrogen0x4b5d220"/>
   315       <fraction n="0.914544377696929" ref="Carbon0x4b5cff0"/>
   316     </material>


UpperChimneyTyvekOpticalSurface trips glisur assert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

glisur assert::

    lldb OKX4Test --  --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00.gdml

    2018-08-30 13:45:08.757 FATAL [4008732] [X4PhysicalVolume::convertSurfaces@255] [
    2018-08-30 13:45:08.757 ERROR [4008732] [X4LogicalBorderSurfaceTable::init@32]  NumberOfBorderSurfaces 9
    2018-08-30 13:45:08.757 ERROR [4008732] [*X4OpticalSurface::Convert@84]  name UpperChimneyTyvekOpticalSurface type 0 model 1 finish 3 value 0.2 value_s 0.200000
    Assertion failed: (0), function Convert, file /Users/blyth/opticks/extg4/X4OpticalSurface.cc, line 56.
    Process 43274 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff7adfcb6e <+10>: jae    0x7fff7adfcb78            ; <+20>
        0x7fff7adfcb70 <+12>: movq   %rax, %rdi
        0x7fff7adfcb73 <+15>: jmp    0x7fff7adf3b00            ; cerror_nocancel
        0x7fff7adfcb78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7afc7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7ad581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106bcfef3 libExtG4.dylib`X4OpticalSurface::Convert(surf=0x00000001272915b0) at X4OpticalSurface.cc:56
        frame #5: 0x0000000106bcf5e3 libExtG4.dylib`X4LogicalBorderSurface::Convert(src=0x0000000128c7bf60) at X4LogicalBorderSurface.cc:25
        frame #6: 0x0000000106bcef67 libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=0x00007ffeefbfd278) at X4LogicalBorderSurfaceTable.cc:40
        frame #7: 0x0000000106bced64 libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd278, dst=0x000000011338ae70) at X4LogicalBorderSurfaceTable.cc:23
        frame #8: 0x0000000106bced1d libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd278, dst=0x000000011338ae70) at X4LogicalBorderSurfaceTable.cc:22
        frame #9: 0x0000000106bcecec libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=0x000000011338ae70) at X4LogicalBorderSurfaceTable.cc:15
        frame #10: 0x0000000106bdb1f3 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:260
        frame #11: 0x0000000106bdabc7 libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:128
        frame #12: 0x0000000106bda996 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x0000000113387d30, top=0x0000000128c7c450) at X4PhysicalVolume.cc:115
        frame #13: 0x0000000106bda385 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x0000000113387d30, top=0x0000000128c7c450) at X4PhysicalVolume.cc:109
        frame #14: 0x00000001000149ed OKX4Test`main(argc=3, argv=0x00007ffeefbfe970) at OKX4Test.cc:104
        frame #15: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 4
    frame #4: 0x0000000106bcfef3 libExtG4.dylib`X4OpticalSurface::Convert(surf=0x00000001272915b0) at X4OpticalSurface.cc:56
       53  	    G4OpticalSurfaceModel model = surf->GetModel(); 
       54  	    switch( model )
       55  	    {
    -> 56  	        case glisur             : assert(0) ; break ;   // original GEANT3 model
       57  	        case unified            :             break ;   // UNIFIED model
       58  	        case LUT                : assert(0) ; break ;   // Look-Up-Table model
       59  	        case dichroic           : assert(0) ; break ; 
    (lldb) 




torus negative startPhi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torus negative startPhi assert::

    Assertion failed: (startPhi == 0.f && deltaPhi == 360.f), function convertTorus, file /Users/blyth/opticks/extg4/X4Solid.cc, line 762.
    Process 43740 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff7adfcb6e <+10>: jae    0x7fff7adfcb78            ; <+20>
        0x7fff7adfcb70 <+12>: movq   %rax, %rdi
        0x7fff7adfcb73 <+15>: jmp    0x7fff7adf3b00            ; cerror_nocancel
        0x7fff7adfcb78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7afc7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7ad581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106bb483e libExtG4.dylib`X4Solid::convertTorus(this=0x0000000114ad2570) at X4Solid.cc:762
        frame #5: 0x0000000106bb122a libExtG4.dylib`X4Solid::init(this=0x0000000114ad2570) at X4Solid.cc:117
        frame #6: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2570, solid=0x0000000127a32670, top=false) at X4Solid.cc:73
        frame #7: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2570, solid=0x0000000127a32670, top=false) at X4Solid.cc:72
        frame #8: 0x0000000106bb153e libExtG4.dylib`X4Solid::convertDisplacedSolid(this=0x0000000114ad2510) at X4Solid.cc:204
        frame #9: 0x0000000106bb10a7 libExtG4.dylib`X4Solid::init(this=0x0000000114ad2510) at X4Solid.cc:96
        frame #10: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2510, solid=0x0000000127a327e0, top=false) at X4Solid.cc:73
        frame #11: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2510, solid=0x0000000127a327e0, top=false) at X4Solid.cc:72
        frame #12: 0x0000000106bb6284 libExtG4.dylib`X4Solid::convertBooleanSolid(this=0x0000000114ad2170) at X4Solid.cc:237
        frame #13: 0x0000000106bb1695 libExtG4.dylib`X4Solid::convertSubtractionSolid(this=0x0000000114ad2170) at X4Solid.cc:194
        frame #14: 0x0000000106bb10da libExtG4.dylib`X4Solid::init(this=0x0000000114ad2170) at X4Solid.cc:99
        frame #15: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2170, solid=0x0000000127a32740, top=false) at X4Solid.cc:73
        frame #16: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2170, solid=0x0000000127a32740, top=false) at X4Solid.cc:72
        frame #17: 0x0000000106bb153e libExtG4.dylib`X4Solid::convertDisplacedSolid(this=0x0000000114ad2110) at X4Solid.cc:204
        frame #18: 0x0000000106bb10a7 libExtG4.dylib`X4Solid::init(this=0x0000000114ad2110) at X4Solid.cc:96
        frame #19: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2110, solid=0x0000000127a329f0, top=false) at X4Solid.cc:73
        frame #20: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad2110, solid=0x0000000127a329f0, top=false) at X4Solid.cc:72
        frame #21: 0x0000000106bb6284 libExtG4.dylib`X4Solid::convertBooleanSolid(this=0x0000000114ad19f0) at X4Solid.cc:237
        frame #22: 0x0000000106bb1655 libExtG4.dylib`X4Solid::convertUnionSolid(this=0x0000000114ad19f0) at X4Solid.cc:186
        frame #23: 0x0000000106bb10b8 libExtG4.dylib`X4Solid::init(this=0x0000000114ad19f0) at X4Solid.cc:97
        frame #24: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad19f0, solid=0x0000000127a32920, top=false) at X4Solid.cc:73
        frame #25: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad19f0, solid=0x0000000127a32920, top=false) at X4Solid.cc:72
        frame #26: 0x0000000106bb623e libExtG4.dylib`X4Solid::convertBooleanSolid(this=0x0000000114ad1970) at X4Solid.cc:236
        frame #27: 0x0000000106bb1655 libExtG4.dylib`X4Solid::convertUnionSolid(this=0x0000000114ad1970) at X4Solid.cc:186
        frame #28: 0x0000000106bb10b8 libExtG4.dylib`X4Solid::init(this=0x0000000114ad1970) at X4Solid.cc:97
        frame #29: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad1970, solid=0x0000000127a32ca0, top=false) at X4Solid.cc:73
        frame #30: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x0000000114ad1970, solid=0x0000000127a32ca0, top=false) at X4Solid.cc:72
        frame #31: 0x0000000106bb623e libExtG4.dylib`X4Solid::convertBooleanSolid(this=0x00007ffeefbfb140) at X4Solid.cc:236
        frame #32: 0x0000000106bb1675 libExtG4.dylib`X4Solid::convertIntersectionSolid(this=0x00007ffeefbfb140) at X4Solid.cc:190
        frame #33: 0x0000000106bb10c9 libExtG4.dylib`X4Solid::init(this=0x00007ffeefbfb140) at X4Solid.cc:98
        frame #34: 0x0000000106bb0ec1 libExtG4.dylib`X4Solid::X4Solid(this=0x00007ffeefbfb140, solid=0x0000000127a32fa0, top=true) at X4Solid.cc:73
        frame #35: 0x0000000106bb0e3c libExtG4.dylib`X4Solid::X4Solid(this=0x00007ffeefbfb140, solid=0x0000000127a32fa0, top=true) at X4Solid.cc:72
        frame #36: 0x0000000106bb0ce0 libExtG4.dylib`X4Solid::Convert(solid=0x0000000127a32fa0, boundary=0x0000000000000000) at X4Solid.cc:58
        frame #37: 0x0000000106bdea4e libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfe0b0, lvIdx=18, soIdx=18, solid=0x0000000127a32fa0, lvname="PMT_20inch_inner1_log0x4cb3cc0") const at X4PhysicalVolume.cc:440
        frame #38: 0x0000000106bde411 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f5ecf0, depth=9) at X4PhysicalVolume.cc:431
        frame #39: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f5ef10, depth=8) at X4PhysicalVolume.cc:418
        frame #40: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f5f0e0, depth=7) at X4PhysicalVolume.cc:418
        frame #41: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110d19650, depth=6) at X4PhysicalVolume.cc:418
        frame #42: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129499520, depth=5) at X4PhysicalVolume.cc:418
        frame #43: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x00000001294996f0, depth=4) at X4PhysicalVolume.cc:418
        frame #44: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000127b191c0, depth=3) at X4PhysicalVolume.cc:418
        frame #45: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000127b19310, depth=2) at X4PhysicalVolume.cc:418
        frame #46: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000127b194a0, depth=1) at X4PhysicalVolume.cc:418
        frame #47: 0x0000000106bdde87 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000127b19bb0, depth=0) at X4PhysicalVolume.cc:418
        frame #48: 0x0000000106bdbb69 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:406
        frame #49: 0x0000000106bdabeb libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:131
        frame #50: 0x0000000106bda996 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x00000001153c5bf0, top=0x0000000127b19bb0) at X4PhysicalVolume.cc:115
        frame #51: 0x0000000106bda385 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x00000001153c5bf0, top=0x0000000127b19bb0) at X4PhysicalVolume.cc:109
        frame #52: 0x00000001000149ed OKX4Test`main(argc=3, argv=0x00007ffeefbfe970) at OKX4Test.cc:104
        frame #53: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) 

::

    (lldb) f 4
    frame #4: 0x0000000106bb483e libExtG4.dylib`X4Solid::convertTorus(this=0x0000000114ad2570) at X4Solid.cc:762
       759 	    float deltaPhi = solid->GetDPhi()/degree ; 
       760 	
       761 	    assert( rmin == 0.f ); // torus with rmin not yet handled 
    -> 762 	    assert( startPhi == 0.f && deltaPhi == 360.f ); 
       763 	
       764 	    float r = rmax ; 
       765 	    float R = rtor ; 
    (lldb) p startPhi
    (float) $0 = -0.00999999977
    (lldb) p deltaPhi
    (float) $1 = 360
    (lldb) 




PMT_20inch_inner1_log0x4cb3cc0 depth 4 CSG tree : needs balancing ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Balancing trees of this structure not implemented::

    2018-08-30 13:55:56.585 FATAL [4016543] [X4Solid::convertTorus@763]  changing torus -ve startPhi (degrees) to zero -0.01
    2018-08-30 13:55:56.585 FATAL [4016543] [*NTreeBalance<nnode>::create_balanced@59] balancing trees of this structure not implemented
    Assertion failed: (0), function create_balanced, file /Users/blyth/opticks/npy/NTreeBalance.cpp, line 60.
    Process 44200 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff7adfcb6e <+10>: jae    0x7fff7adfcb78            ; <+20>
        0x7fff7adfcb70 <+12>: movq   %rax, %rdi
        0x7fff7adfcb73 <+15>: jmp    0x7fff7adf3b00            ; cerror_nocancel
        0x7fff7adfcb78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7afc7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7ad581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010e386feb libNPY.dylib`NTreeBalance<nnode>::create_balanced(this=0x00000001154196b0) at NTreeBalance.cpp:60
        frame #5: 0x000000010e38909d libNPY.dylib`NTreeProcess<nnode>::init(this=0x00007ffeefbfaf18) at NTreeProcess.cpp:87
        frame #6: 0x000000010e389002 libNPY.dylib`NTreeProcess<nnode>::NTreeProcess(this=0x00007ffeefbfaf18, root_=0x0000000115419590) at NTreeProcess.cpp:78
        frame #7: 0x000000010e388f1d libNPY.dylib`NTreeProcess<nnode>::NTreeProcess(this=0x00007ffeefbfaf18, root_=0x0000000115419590) at NTreeProcess.cpp:77
        frame #8: 0x000000010e388ac7 libNPY.dylib`NTreeProcess<nnode>::Process(root_=0x0000000115419590, soIdx=18, lvIdx=18) at NTreeProcess.cpp:43
        frame #9: 0x0000000106bdea47 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfe0b0, lvIdx=18, soIdx=18, solid=0x0000000127b56120, lvname="PMT_20inch_inner1_log0x4cb3cc0") const at X4PhysicalVolume.cc:447
        frame #10: 0x0000000106bde3b1 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000112000980, depth=9) at X4PhysicalVolume.cc:431
        frame #11: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000112000bb0, depth=8) at X4PhysicalVolume.cc:418
        frame #12: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000112000d80, depth=7) at X4PhysicalVolume.cc:418
        frame #13: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000127d16430, depth=6) at X4PhysicalVolume.cc:418
        frame #14: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129d28dc0, depth=5) at X4PhysicalVolume.cc:418
        frame #15: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129d28fb0, depth=4) at X4PhysicalVolume.cc:418
        frame #16: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x00000001297ee9e0, depth=3) at X4PhysicalVolume.cc:418
        frame #17: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x00000001297eeb10, depth=2) at X4PhysicalVolume.cc:418
        frame #18: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x00000001297eec80, depth=1) at X4PhysicalVolume.cc:418
        frame #19: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x00000001297ef390, depth=0) at X4PhysicalVolume.cc:418
        frame #20: 0x0000000106bdbb09 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:406
        frame #21: 0x0000000106bdab8b libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:131
        frame #22: 0x0000000106bda936 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x00000001153b20c0, top=0x00000001297ef390) at X4PhysicalVolume.cc:115
        frame #23: 0x0000000106bda325 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x00000001153b20c0, top=0x00000001297ef390) at X4PhysicalVolume.cc:109
        frame #24: 0x00000001000149ed OKX4Test`main(argc=3, argv=0x00007ffeefbfe970) at OKX4Test.cc:104
        frame #25: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) 

::

    (lldb) f 9
    frame #9: 0x0000000106bdea47 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfe0b0, lvIdx=18, soIdx=18, solid=0x0000000127b56120, lvname="PMT_20inch_inner1_log0x4cb3cc0") const at X4PhysicalVolume.cc:447
       444 	         X4CSG::GenerateTest( solid, m_g4codegendir , lvIdx ) ; 
       445 	     }
       446 	
    -> 447 	     nnode* root = NTreeProcess<nnode>::Process(raw, soIdx, lvIdx);  // balances deep trees
       448 	     root->other = raw ; 
       449 	
       450 	     const NSceneConfig* config = NULL ; 
    (lldb) 


    (lldb) f 8
    frame #8: 0x000000010e388ac7 libNPY.dylib`NTreeProcess<nnode>::Process(root_=0x0000000115419590, soIdx=18, lvIdx=18) at NTreeProcess.cpp:43
       40  	 
       41  	    unsigned height0 = root_->maxdepth(); 
       42  	
    -> 43  	    NTreeProcess<T> proc(root_); 
       44  	
       45  	    assert( height0 == proc.balancer->height0 ); 
       46  	
    (lldb) p height0
    (unsigned int) $0 = 4

    (lldb) f 5
    frame #5: 0x000000010e38909d libNPY.dylib`NTreeProcess<nnode>::init(this=0x00007ffeefbfaf18) at NTreeProcess.cpp:87
       84  	    if(balancer->height0 > MaxHeight0 )
       85  	    {
       86  	        positiver = new NTreePositive<T>(root) ; 
    -> 87  	        balanced = balancer->create_balanced() ;  
       88  	        result = balanced ; 
       89  	    }
       90  	    else
    (lldb) p MaxHeight0
    (unsigned int) $1 = 3
    (lldb) p balancer->height0
    (unsigned int) $2 = 4
    (lldb) 


Height 4 means (5 levels) so in does need balancing, or a rethink to simplify. Need to see the tree 
to see how to proceed.


::
 
     30 template <typename T>
     31 T* NTreeBalance<T>::create_balanced()
     32 {
     33     assert( is_positive_form() && " must positivize the tree before balancing ");
     34 
     35     unsigned op_mask = operators();
     36     unsigned hop_mask = operators(2);  // operators above the bileaf operators
     37 
     38     OpticksCSG_t op = CSG_MonoOperator(op_mask) ;
     39     OpticksCSG_t hop = CSG_MonoOperator(hop_mask) ;
     40 
     41     T* balanced = NULL ;
     42 
     43     if( op == CSG_INTERSECTION || op == CSG_UNION )
     44     {
     45         std::vector<T*> prims ;
     46         subtrees( prims, 0 );    // subdepth 0 
     47         //LOG(info) << " prims " << prims.size() ; 
     48         balanced = NTreeBuilder<T>::CommonTree(prims, op );
     49     }
     50     else if( hop == CSG_INTERSECTION || hop == CSG_UNION )
     51     {
     52         std::vector<T*> bileafs ;
     53         subtrees( bileafs, 1 );  // subdepth 1
     54         //LOG(info) << " bileafs " << bileafs.size() ; 
     55         balanced = NTreeBuilder<T>::BileafTree(bileafs, hop );
     56     }
     57     else
     58     {
     59         LOG(fatal) << "balancing trees of this structure not implemented" ;
     60         assert(0);
     61         balanced = root ;
     62     }
     63     return balanced ;
     64 }



::

      3 typedef enum {
      4     CSG_ZERO=0,
      5     CSG_UNION=1,
      6     CSG_INTERSECTION=2,
      7     CSG_DIFFERENCE=3,
      8     CSG_PARTLIST=4,
      9 

::

    (lldb) p op_mask
    (unsigned int) $5 = 6
    (lldb) p hop_mask         ## means both UNION and INTERSECTION above bileaf
    (unsigned int) $6 = 6
    (lldb) 

::

     03 
      4 #include "OpticksCSG.h"
      5 
      6 typedef enum {
      7 
      8    CSGMASK_UNION        = 0x1 << CSG_UNION ,               ## 2 
      9    CSGMASK_INTERSECTION = 0x1 << CSG_INTERSECTION ,        ## 4
     10    CSGMASK_DIFFERENCE   = 0x1 << CSG_DIFFERENCE,
     11    CSGMASK_CYLINDER     = 0x1 << CSG_CYLINDER,
     12    CSGMASK_DISC         = 0x1 << CSG_DISC,
     13    CSGMASK_CONE         = 0x1 << CSG_CONE,
     14    CSGMASK_ZSPHERE      = 0x1 << CSG_ZSPHERE,
     15    CSGMASK_BOX3         = 0x1 << CSG_BOX3
     16 
     17 } OpticksCSGMask_t ;
     18 




5 primitives in tree : would be much better for that to be 4 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    2018-08-30 14:18:01.708 FATAL [4065914] [X4Solid::convertTorus@763]  changing torus -ve startPhi (degrees) to zero -0.01
    2018-08-30 14:18:01.708 INFO  [4065914] [*NTreeProcess<nnode>::Process@39] before
    NTreeAnalyse height 4 count 9
                                  in    

                          un          cy

          un                  cy        

      sp          di                    

              cy      to                


    2018-08-30 14:18:01.708 FATAL [4065914] [*NTreeBalance<nnode>::create_balanced@59] balancing trees of this structure not implemented
    Assertion failed: (0), function create_balanced, file /Users/blyth/opticks/npy/NTreeBalance.cpp, line 60.




Skip the assert to proceed::

    2018-08-30 14:24:59.398 FATAL [4073597] [X4Solid::convertPolyconePrimitives@864]  !z_ascending  z1 -15.8745 z2 -75.8735
    Assertion failed: (z_ascending), function convertPolyconePrimitives, file /Users/blyth/opticks/extg4/X4Solid.cc, line 868.
    Process 52776 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff7adfcb6e <+10>: jae    0x7fff7adfcb78            ; <+20>
        0x7fff7adfcb70 <+12>: movq   %rax, %rdi
        0x7fff7adfcb73 <+15>: jmp    0x7fff7adf3b00            ; cerror_nocancel
        0x7fff7adfcb78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7afc7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7ad581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106bba3b0 libExtG4.dylib`X4Solid::convertPolyconePrimitives(this=0x00007ffeefbfb720, zp=size=1, prims=size=1) at X4Solid.cc:868
        frame #5: 0x0000000106bb3a8c libExtG4.dylib`X4Solid::convertPolycone(this=0x00007ffeefbfb720) at X4Solid.cc:938
        frame #6: 0x0000000106bb1008 libExtG4.dylib`X4Solid::init(this=0x00007ffeefbfb720) at X4Solid.cc:111
        frame #7: 0x0000000106bb0d11 libExtG4.dylib`X4Solid::X4Solid(this=0x00007ffeefbfb720, solid=0x00000001279f8720, top=true) at X4Solid.cc:73
        frame #8: 0x0000000106bb0c8c libExtG4.dylib`X4Solid::X4Solid(this=0x00007ffeefbfb720, solid=0x00000001279f8720, top=true) at X4Solid.cc:72
        frame #9: 0x0000000106bb0b30 libExtG4.dylib`X4Solid::Convert(solid=0x00000001279f8720, boundary=0x0000000000000000) at X4Solid.cc:58
        frame #10: 0x0000000106bde9ee libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfe0b0, lvIdx=26, soIdx=26, solid=0x00000001279f8720, lvname="PMT_3inch_cntr_log0x510bd20") const at X4PhysicalVolume.cc:440
        frame #11: 0x0000000106bde3b1 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110e5f8c0, depth=7) at X4PhysicalVolume.cc:431
        frame #12: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x000000012923c710, depth=6) at X4PhysicalVolume.cc:418
        frame #13: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129ed46c0, depth=5) at X4PhysicalVolume.cc:418
        frame #14: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129ed4890, depth=4) at X4PhysicalVolume.cc:418
        frame #15: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f56a40, depth=3) at X4PhysicalVolume.cc:418
        frame #16: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f35120, depth=2) at X4PhysicalVolume.cc:418
        frame #17: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f3b750, depth=1) at X4PhysicalVolume.cc:418
        frame #18: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f35210, depth=0) at X4PhysicalVolume.cc:418
        frame #19: 0x0000000106bdbb09 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:406
        frame #20: 0x0000000106bdab8b libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:131
        frame #21: 0x0000000106bda936 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x0000000114e2fdb0, top=0x0000000110f35210) at X4PhysicalVolume.cc:115
        frame #22: 0x0000000106bda325 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x0000000114e2fdb0, top=0x0000000110f35210) at X4PhysicalVolume.cc:109
        frame #23: 0x00000001000149ed OKX4Test`main(argc=3, argv=0x00007ffeefbfe970) at OKX4Test.cc:104
        frame #24: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) 



polycone z-ascending assert : fixed with a swap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


z-ascending assert::

     846 void X4Solid::convertPolyconePrimitives( const std::vector<zplane>& zp,  std::vector<nnode*>& prims )
     847 {
     848     for( unsigned i=1 ; i < zp.size() ; i++ )
     849     {
     850         const zplane& zp1 = zp[i-1] ;
     851         const zplane& zp2 = zp[i] ;
     852         double r1 = zp1.rmax ;
     853         double r2 = zp2.rmax ;
     854         double z1 = zp1.z ;
     855         double z2 = zp2.z ;
     856 
     857         if( z1 == z2 )
     858         {
     859             //LOG(warning) << " skipping z2 == z1 zp " ; 
     860             continue ;
     861         }
     862 
     863         bool z_ascending = z2 > z1 ;
     864         if(!z_ascending) LOG(fatal) << " !z_ascending "
     865                                     << " z1 " << z1
     866                                     << " z2 " << z2
     867                                     ;
     868         assert(z_ascending);
     869 
     870         nnode* n = NULL ;
     871         if( r2 == r1 )
     872         {
     873             n = new ncylinder(make_cylinder(r2, z1, z2));
     874             n->label = BStr::concat( m_name, i-1, "_zp_cylinder" );
     875         }
     876         else
     877         {
     878             n = new ncone(make_cone(r1,z1,r2,z2)) ;
     879             n->label = BStr::concat<unsigned>(m_name, i-1 , "_zp_cone" ) ;
     880         }
     881         prims.push_back(n);
     882     }   // over pairs of planes
     883 }


::

    (lldb) p zp1
    (const X4Solid::zplane) $0 = (rmin = 0, rmax = 29.998999999999999, z = -15.8745078663875)
    (lldb) p zp2
    (const X4Solid::zplane) $1 = (rmin = 0, rmax = 29.998999999999999, z = -75.873507866387498)
    (lldb) p zp
    (const std::__1::vector<X4Solid::zplane, std::__1::allocator<X4Solid::zplane> >) $2 = size=2 {
      [0] = (rmin = 0, rmax = 29.998999999999999, z = -15.8745078663875)
      [1] = (rmin = 0, rmax = 29.998999999999999, z = -75.873507866387498)
    }


Found that my swap fix had a bug::

     931     if( zp.size() == 2 && zp[0].z > zp[1].z )  // Aug 2018 FIX: was [0] [0] 
     932     {
     933         LOG(warning) << "Polycone swap misordered pair of zplanes for " << m_name ;
     934         std::reverse( std::begin(zp), std::end(zp) ) ;
     935     }


Torus deltaPhi 356 ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7adfcb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7afc7080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7ad581ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106bb47a6 libExtG4.dylib`X4Solid::convertTorus(this=0x00007ffeefbfb720) at X4Solid.cc:771
        frame #5: 0x0000000106bb107a libExtG4.dylib`X4Solid::init(this=0x00007ffeefbfb720) at X4Solid.cc:117
        frame #6: 0x0000000106bb0d11 libExtG4.dylib`X4Solid::X4Solid(this=0x00007ffeefbfb720, solid=0x0000000127a38bc0, top=true) at X4Solid.cc:73
        frame #7: 0x0000000106bb0c8c libExtG4.dylib`X4Solid::X4Solid(this=0x00007ffeefbfb720, solid=0x0000000127a38bc0, top=true) at X4Solid.cc:72
        frame #8: 0x0000000106bb0b30 libExtG4.dylib`X4Solid::Convert(solid=0x0000000127a38bc0, boundary=0x0000000000000000) at X4Solid.cc:58
        frame #9: 0x0000000106bde9ee libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfe0b0, lvIdx=32, soIdx=32, solid=0x0000000127a38bc0, lvname="lvacSurftube0x5b3c020") const at X4PhysicalVolume.cc:440
        frame #10: 0x0000000106bde3b1 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000110f029a0, depth=7) at X4PhysicalVolume.cc:431
        frame #11: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f1e170, depth=6) at X4PhysicalVolume.cc:418
        frame #12: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f1e350, depth=5) at X4PhysicalVolume.cc:418
        frame #13: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f1e540, depth=4) at X4PhysicalVolume.cc:418
        frame #14: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f9c790, depth=3) at X4PhysicalVolume.cc:418
        frame #15: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f9c8a0, depth=2) at X4PhysicalVolume.cc:418
        frame #16: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f9ca10, depth=1) at X4PhysicalVolume.cc:418
        frame #17: 0x0000000106bdde27 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfe0b0, pv=0x0000000129f9d120, depth=0) at X4PhysicalVolume.cc:418
        frame #18: 0x0000000106bdbb09 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:406
        frame #19: 0x0000000106bdab8b libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe0b0) at X4PhysicalVolume.cc:131
        frame #20: 0x0000000106bda936 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x000000011442df20, top=0x0000000129f9d120) at X4PhysicalVolume.cc:115
        frame #21: 0x0000000106bda325 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe0b0, ggeo=0x000000011442df20, top=0x0000000129f9d120) at X4PhysicalVolume.cc:109
        frame #22: 0x00000001000149ed OKX4Test`main(argc=3, argv=0x00007ffeefbfe970) at OKX4Test.cc:104
        frame #23: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x0000000106bb47a6 libExtG4.dylib`X4Solid::convertTorus(this=0x00007ffeefbfb720) at X4Solid.cc:771
       768 	
       769 	
       770 	    assert( rmin == 0.f ); // torus with rmin not yet handled 
    -> 771 	    assert( startPhi == 0.f && deltaPhi == 360.f ); 
       772 	
       773 	    float r = rmax ; 
       774 	    float R = rtor ; 
    (lldb) p rmin
    (float) $0 = 0
    (lldb) p rmax
    (float) $1 = 8
    (lldb) p rtor
    (float) $2 = 17836
    (lldb) p startPhi
    (float) $3 = 0
    (lldb) p deltaPhi
    (float) $4 = 356
    (lldb) 



Default gensteps expecting GdLS : this was redherring : real problem was the skipped materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    2018-08-30 14:42:15.264 INFO  [4083817] [OpticksHub::adoptGeometry@463] OpticksHub::adoptGeometry DONE
    2018-08-30 14:42:15.264 INFO  [4083817] [OpticksHub::configureGeometryTri@558] OpticksHub::configureGeometryTri restrict_mesh -1 nmm 6
    2018-08-30 14:42:15.265 ERROR [4083817] [*OpticksGen::makeInputGensteps@185]  code 4096 srctype TORCH
    2018-08-30 14:42:15.265 INFO  [4083817] [*Opticks::makeSimpleTorchStep@1972] Opticks::makeSimpleTorchStep config  cfg NULL
    2018-08-30 14:42:15.266 INFO  [4083817] [OpticksGen::targetGenstep@306] OpticksGen::targetGenstep setting frame 3153 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 6711.2002,-16634.5000,23439.8496,1.0000
    Process 54167 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
        frame #0: 0x00007fff7afba220 libsystem_platform.dylib`_platform_strncmp + 320
    libsystem_platform.dylib`_platform_strncmp:
    ->  0x7fff7afba220 <+320>: movzbq (%rdi,%rcx), %rax
        0x7fff7afba225 <+325>: movzbq (%rsi,%rcx), %r8
        0x7fff7afba22a <+330>: subq   %r8, %rax
        0x7fff7afba22d <+333>: jne    0x7fff7afba23d            ; <+349>
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x00007fff7afba220 libsystem_platform.dylib`_platform_strncmp + 320
        frame #1: 0x000000010dbed2a2 libGGeo.dylib`GBndLib::getMaterialLine(this=0x00000001298a7080, shortname_="GdDopedLS") at GBndLib.cc:639
        frame #2: 0x000000010c952b52 libOpticksGeo.dylib`OpticksGen::setMaterialLine(this=0x000000029eea9290, gs=0x000000029ef9b330) at OpticksGen.cc:336
        frame #3: 0x000000010c951de4 libOpticksGeo.dylib`OpticksGen::makeTorchstep(this=0x000000029eea9290) at OpticksGen.cc:365
        frame #4: 0x000000010c951a44 libOpticksGeo.dylib`OpticksGen::makeInputGensteps(this=0x000000029eea9290, code=4096) at OpticksGen.cc:198
        frame #5: 0x000000010c951457 libOpticksGeo.dylib`OpticksGen::initFromGensteps(this=0x000000029eea9290) at OpticksGen.cc:172
        frame #6: 0x000000010c950ad0 libOpticksGeo.dylib`OpticksGen::init(this=0x000000029eea9290) at OpticksGen.cc:104
        frame #7: 0x000000010c950976 libOpticksGeo.dylib`OpticksGen::OpticksGen(this=0x000000029eea9290, hub=0x000000029eea55e0) at OpticksGen.cc:48
        frame #8: 0x000000010c950afd libOpticksGeo.dylib`OpticksGen::OpticksGen(this=0x000000029eea9290, hub=0x000000029eea55e0) at OpticksGen.cc:47
        frame #9: 0x000000010c94a6d8 libOpticksGeo.dylib`OpticksHub::init(this=0x000000029eea55e0) at OpticksHub.cc:187
        frame #10: 0x000000010c94a41a libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000029eea55e0, ok=0x00000001149f2a00) at OpticksHub.cc:156
        frame #11: 0x000000010c94a82d libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000029eea55e0, ok=0x00000001149f2a00) at OpticksHub.cc:155
        frame #12: 0x0000000100109d74 libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe048, argc=3, argv=0x00007ffeefbfe970, argforced=0x0000000000000000) at OKMgr.cc:44
        frame #13: 0x000000010010a1bb libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe048, argc=3, argv=0x00007ffeefbfe970, argforced=0x0000000000000000) at OKMgr.cc:52
        frame #14: 0x0000000100014a81 OKX4Test`main(argc=3, argv=0x00007ffeefbfe970) at OKX4Test.cc:118
        frame #15: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) 


Caused by boundary issue::

    (lldb) f 2
    frame #2: 0x000000010c952b52 libOpticksGeo.dylib`OpticksGen::setMaterialLine(this=0x000000029eea9290, gs=0x000000029ef9b330) at OpticksGen.cc:336
       333 	      LOG(fatal) << "NULL material from GenstepNPY, probably missed material in torch config" ;
       334 	   assert(material);
       335 	
    -> 336 	   unsigned int matline = m_blib->getMaterialLine(material);
       337 	   gs->setMaterialLine(matline);  
       338 	
       339 	   LOG(debug) << "OpticksGen::setMaterialLine"
    (lldb) p material
    (const char *) $0 = 0x000000029ef71740 "GdDopedLS"
    (lldb) p m_blib
    (GBndLib *) $1 = 0x00000001298a7080
    (lldb) f 1
    frame #1: 0x000000010dbed2a2 libGGeo.dylib`GBndLib::getMaterialLine(this=0x00000001298a7080, shortname_="GdDopedLS") at GBndLib.cc:639
       636 	        const char* omat = m_mlib->getName(bnd[OMAT]);
       637 	        const char* imat = m_mlib->getName(bnd[IMAT]);
       638 	
    -> 639 	        if(strncmp(imat, shortname_, strlen(shortname_))==0)
       640 	        { 
       641 	            line = getLine(i, IMAT);  
       642 	            break ;
    (lldb) p imat
    (const char *) $2 = 0x0000000000000000
    (lldb) p omat
    (const char *) $3 = 0x0000000000000000
    (lldb) p bnd
    (const guint4) $4 = (x = 4294967295, y = 4294967295, z = 4294967295, w = 4294967295)
    (lldb) 



Hmm not all materials, but several are missing proper indices in the bnd ?
Is this non-mpt matererials ?::

    (lldb) p m_bnd
    (std::__1::vector<guint4, std::__1::allocator<guint4> >) $7 = size=28 {
      [0] = (x = 4294967295, y = 4294967295, z = 4294967295, w = 4294967295)
      [1] = (x = 4294967295, y = 4294967295, z = 4294967295, w = 4)
      [2] = (x = 4, y = 4294967295, z = 4294967295, w = 3)
      [3] = (x = 3, y = 4294967295, z = 4294967295, w = 3)
      [4] = (x = 3, y = 4294967295, z = 4294967295, w = 0)
      [5] = (x = 3, y = 4294967295, z = 4294967295, w = 1)
      [6] = (x = 3, y = 4294967295, z = 4294967295, w = 2)
      [7] = (x = 3, y = 4294967295, z = 4294967295, w = 4294967295)
      [8] = (x = 4, y = 4294967295, z = 4294967295, w = 2)
      [9] = (x = 2, y = 4294967295, z = 4294967295, w = 9)
      [10] = (x = 9, y = 8, z = 4294967295, w = 2)
      [11] = (x = 2, y = 4294967295, z = 4294967295, w = 8)
      [12] = (x = 8, y = 4294967295, z = 4294967295, w = 5)
      [13] = (x = 5, y = 4294967295, z = 4294967295, w = 0)
      [14] = (x = 8, y = 4294967295, z = 4294967295, w = 1)
      [15] = (x = 8, y = 4294967295, z = 4294967295, w = 4294967295)
      [16] = (x = 8, y = 4294967295, z = 4294967295, w = 8)
      [17] = (x = 8, y = 4294967295, z = 4294967295, w = 7)
      [18] = (x = 7, y = 4294967295, z = 4294967295, w = 7)
      [19] = (x = 7, y = 3, z = 1, w = 6)
      [20] = (x = 7, y = 4294967295, z = 2, w = 6)
      [21] = (x = 7, y = 6, z = 4, w = 6)
      [22] = (x = 7, y = 4294967295, z = 5, w = 6)
      [23] = (x = 8, y = 4294967295, z = 4294967295, w = 0)
      [24] = (x = 8, y = 9, z = 9, w = 1)
      [25] = (x = 8, y = 10, z = 10, w = 4294967295)
      [26] = (x = 4294967295, y = 10, z = 10, w = 6)
      [27] = (x = 9, y = 4294967295, z = 4294967295, w = 8)
    }
    (lldb) 


Allowing materials not to have mpt, gets thru to the viz.


