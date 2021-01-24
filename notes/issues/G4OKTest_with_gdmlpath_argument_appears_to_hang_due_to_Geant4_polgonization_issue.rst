G4OKTest_with_gdmlpath_argument_appears_to_hang_due_to_Geant4_polgonization_issue
===================================================================================


Solution
----------

The solution to this is to identify the lvIdx of the solids with polygonization problem
and skip them with "--x4polyskip 211,232" see geocache.

Because of such problems it is best to not have a workflow that converts geometry everytime
you want to run some tests. Instead create a geocache for your geometry and set envvar OPTICKS_KEY
so all opticks executables will use that geometry.

Study the geocache-create bash function and adapt it for your gdmlpath.

::

    epsilon:~ blyth$ t geocache-create
    geocache-create () 
    { 
        geocache-dx1 $*
    }
    epsilon:~ blyth$ t geocache-dx1
    geocache-dx1 () 
    { 
        geocache-dx1- --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $*
    }
    epsilon:~ blyth$ t geocache-dx1-
    geocache-dx1- () 
    { 
        opticksaux-;
        geocache-create- --gdmlpath $(opticksaux-dx1) --x4polyskip 211,232 --geocenter --noviz $*
    }
    epsilon:~ blyth$ 




Issue : looks like a hung conversion
---------------------------------------


* debug by running in debugger and looking at backtrace whilst hung 

::

    epsilon:g4ok blyth$ lldb_ -- G4OKTest --gdmlpath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml
    (lldb) target create "G4OKTest"
    Current executable set to 'G4OKTest' (x86_64).
    (lldb) settings set -- target.run-args  "--gdmlpath" "/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml"
    (lldb) r
    Process 76383 launched: '/usr/local/opticks/lib/G4OKTest' (x86_64)
    2021-01-23 10:14:15.084 INFO  [2452388] [G4Opticks::G4Opticks@305] ctor : DISABLE FPE detection : as it breaks OptiX launches

      C4FPEDetection::InvalidOperationDetection_Disable       NOT IMPLEMENTED 
    2021-01-23 10:14:15.109 INFO  [2452388] [CGDML::read@71]  resolved path_ /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml as path /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml
    G4GDML: Reading '/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    ...
    2021-01-23 10:14:18.416 ERROR [2452388] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx  85 num_prim  3 num_coincidence  2 num_nudge  1 
    2021-01-23 10:14:27.609 ERROR [2452388] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx 130 num_prim  3 num_coincidence  2 num_nudge  2 
    2021-01-23 10:14:27.612 ERROR [2452388] [*X4Solid::convertSphere_@445]  convertSphere_duplicate_py_segment_omission 1 has_deltaPhi 1 startPhi 0 deltaPhi 180 name UpperAcrylicHemisphere0xc0b2ac00x3ea4550
    2021-01-23 10:14:27.612 ERROR [2452388] [*X4Solid::convertSphere_@445]  convertSphere_duplicate_py_segment_omission 1 has_deltaPhi 1 startPhi 0 deltaPhi 180 name LowerAcrylicHemisphere0xc0b2be80x3ea4940
    2021-01-23 10:14:32.226 ERROR [2452388] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx 143 num_prim  2 num_coincidence  1 num_nudge  1 
    2021-01-23 10:14:32.229 ERROR [2452388] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx 144 num_prim  3 num_coincidence  2 num_nudge  2 
    2021-01-23 10:14:32.232 ERROR [2452388] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx 145 num_prim  6 num_coincidence  5 num_nudge  5 
    2021-01-23 10:14:32.239 ERROR [2452388] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx 145 num_prim  6 num_coincidence  5 num_nudge  5 
    Process 76383 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
        frame #0: 0x0000000105ad5de8 libG4graphics_reps.dylib`CLHEP::Hep3Vector::~Hep3Vector(this=0x0000000117b5e000) at ThreeVector.icc:128
       125 	inline Hep3Vector::Hep3Vector(const Hep3Vector & p)
       126 	: dx(p.dx), dy(p.dy), dz(p.dz) {}
       127 	
    -> 128 	inline Hep3Vector::~Hep3Vector() {}
       129 	
       130 	inline Hep3Vector & Hep3Vector::operator = (const Hep3Vector & p) {
       131 	  dx = p.dx;
    Target 0: (G4OKTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
      * frame #0: 0x0000000105ad5de8 libG4graphics_reps.dylib`CLHEP::Hep3Vector::~Hep3Vector(this=0x0000000117b5e000) at ThreeVector.icc:128
        frame #1: 0x0000000105b226fe libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer() [inlined] std::__1::allocator<ExtEdge>::destroy(this=0x00007ffeefbf2e40, __p=0x0000000117b5dfe8) at memory:1838
        frame #2: 0x0000000105b226f5 libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer() [inlined] void std::__1::allocator_traits<std::__1::allocator<ExtEdge> >::__destroy<ExtEdge>(__a=0x00007ffeefbf2e40, __p=0x0000000117b5dfe8) at memory:1706
        frame #3: 0x0000000105b226df libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer() [inlined] void std::__1::allocator_traits<std::__1::allocator<ExtEdge> >::destroy<ExtEdge>(__a=0x00007ffeefbf2e40, __p=0x0000000117b5dfe8) at memory:1574
        frame #4: 0x0000000105b226c3 libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer() [inlined] std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::__destruct_at_end(this=0x00007ffeefbf1180, __new_last=0x0000000117b5deb0) at __split_buffer:296
        frame #5: 0x0000000105b2264a libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer() [inlined] std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::__destruct_at_end(this=0x00007ffeefbf1180, __new_last=0x0000000117b5deb0) at __split_buffer:141
        frame #6: 0x0000000105b22620 libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer() [inlined] std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::clear(this=0x00007ffeefbf1180) at __split_buffer:86
        frame #7: 0x0000000105b22607 libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer(this=0x00007ffeefbf1180) at __split_buffer:341
        frame #8: 0x0000000105b221f5 libG4graphics_reps.dylib`std::__1::__split_buffer<ExtEdge, std::__1::allocator<ExtEdge>&>::~__split_buffer(this=0x00007ffeefbf1180) at __split_buffer:340
        frame #9: 0x0000000105b219f8 libG4graphics_reps.dylib`void std::__1::vector<ExtEdge, std::__1::allocator<ExtEdge> >::__push_back_slow_path<ExtEdge>(this=0x00007ffeefbf2e30 size=17, __x=0x00007ffeefbf15c0) at vector:1575
        frame #10: 0x0000000105b0b63a libG4graphics_reps.dylib`BooleanProcessor::takePolyhedron(HepPolyhedron const&, double, double, double) [inlined] std::__1::vector<ExtEdge, std::__1::allocator<ExtEdge> >::push_back(this=0x00007ffeefbf2e30 size=17, __x=0x00007ffeefbf15c0) at vector:1611
        frame #11: 0x0000000105b0b45d libG4graphics_reps.dylib`BooleanProcessor::takePolyhedron(this=0x00007ffeefbf2e18, p=0x00007ffeefbf2f48, dx=0, dy=0, dz=0) at BooleanProcessor.src:488
        frame #12: 0x0000000105b18803 libG4graphics_reps.dylib`BooleanProcessor::execute(this=0x00007ffeefbf2e18, op=2, a=0x00007ffeefbf2f48, b=0x0000000117b5c368, err=0x00007ffeefbf2e0c) at BooleanProcessor.src:2022
        frame #13: 0x0000000105b1b517 libG4graphics_reps.dylib`HepPolyhedronProcessor::execute1(this=0x00007ffeefbf35b0, a_poly=0x0000000117b5a8f0, a_is=size=12) at HepPolyhedronProcessor.src:171
        frame #14: 0x0000000105b1cd85 libG4graphics_reps.dylib`HepPolyhedron_exec::visit(this=0x00007ffeefbf34e8, a_is=size=12) at HepPolyhedronProcessor.src:131
        frame #15: 0x0000000105b1d15f libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=11, a_is=size=12) at HepPolyhedronProcessor.src:94
        frame #16: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=10, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #17: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=9, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #18: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=8, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #19: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=7, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #20: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=6, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #21: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=5, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #22: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=4, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #23: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=3, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #24: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=2, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #25: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=1, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #26: 0x0000000105b1d18a libG4graphics_reps.dylib`HEPVis::bijection_visitor::visit(this=0x00007ffeefbf34e8, a_level=0, a_is=size=12) at HepPolyhedronProcessor.src:96
        frame #27: 0x0000000105b1b305 libG4graphics_reps.dylib`HEPVis::bijection_visitor::visitx(this=0x00007ffeefbf34e8) at HepPolyhedronProcessor.src:84
        frame #28: 0x0000000105b1b113 libG4graphics_reps.dylib`HepPolyhedronProcessor::execute(this=0x00007ffeefbf35b0, a_poly=0x0000000117b5a8f0) at HepPolyhedronProcessor.src:147
        frame #29: 0x00000001055426cd libG4geometry.dylib`G4SubtractionSolid::CreatePolyhedron(this=0x0000000114d00030) const at G4SubtractionSolid.cc:591
        frame #30: 0x00000001003bf67d libExtG4.dylib`X4Mesh::polygonize(this=0x00007ffeefbf49b0) at X4Mesh.cc:167
        frame #31: 0x00000001003be90f libExtG4.dylib`X4Mesh::init(this=0x00007ffeefbf49b0) at X4Mesh.cc:132
        frame #32: 0x00000001003be8e2 libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbf49b0, solid=0x0000000114d00030) at X4Mesh.cc:122
        frame #33: 0x00000001003be85d libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbf49b0, solid=0x0000000114d00030) at X4Mesh.cc:121
        frame #34: 0x00000001003be80c libExtG4.dylib`X4Mesh::Convert(solid=0x0000000114d00030) at X4Mesh.cc:106
        frame #35: 0x00000001003e5ec9 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=0x00007ffeefbfd618, lvIdx=211, soIdx=211, solid=0x0000000114d00030, lvname="/dd/Geometry/Pool/lvNearPoolIWS0xc28bc600x3efba00", balance_deep_tree=true) const at X4PhysicalVolume.cc:654
        frame #36: 0x00000001003e30c1 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114cd7700, depth=7) at X4PhysicalVolume.cc:526
        frame #37: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114cd9e30, depth=6) at X4PhysicalVolume.cc:510
        frame #38: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114fe7390, depth=5) at X4PhysicalVolume.cc:510
        frame #39: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114fe80d0, depth=4) at X4PhysicalVolume.cc:510
        frame #40: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114fe96f0, depth=3) at X4PhysicalVolume.cc:510
        frame #41: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114fea490, depth=2) at X4PhysicalVolume.cc:510
        frame #42: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x0000000114fea4e0, depth=1) at X4PhysicalVolume.cc:510
        frame #43: 0x00000001003e2ab7 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=0x00007ffeefbfd618, pv=0x000000010e57de80, depth=0) at X4PhysicalVolume.cc:510
        frame #44: 0x00000001003e0e28 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=0x00007ffeefbfd618) at X4PhysicalVolume.cc:467
        frame #45: 0x00000001003e0257 libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfd618) at X4PhysicalVolume.cc:194
        frame #46: 0x00000001003dff25 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfd618, ggeo=0x0000000113d56c40, top=0x000000010e57de80) at X4PhysicalVolume.cc:177
        frame #47: 0x00000001003df1e5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfd618, ggeo=0x0000000113d56c40, top=0x000000010e57de80) at X4PhysicalVolume.cc:168
        frame #48: 0x00000001000e3184 libG4OK.dylib`G4Opticks::translateGeometry(this=0x000000010e45d810, top=0x000000010e57de80) at G4Opticks.cc:787
        frame #49: 0x00000001000e2844 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010e45d810, world=0x000000010e57de80) at G4Opticks.cc:447
        frame #50: 0x00000001000e2655 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010e45d810, gdmlpath="/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml") at G4Opticks.cc:433
        frame #51: 0x000000010000f5d2 G4OKTest`G4OKTest::initGeometry(this=0x00007ffeefbfe838) at G4OKTest.cc:190
        frame #52: 0x000000010000f022 G4OKTest`G4OKTest::init(this=0x00007ffeefbfe838) at G4OKTest.cc:144
        frame #53: 0x000000010000edc9 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe838, argc=3, argv=0x00007ffeefbfe8a8) at G4OKTest.cc:114
        frame #54: 0x000000010000f083 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe838, argc=3, argv=0x00007ffeefbfe8a8) at G4OKTest.cc:113
        frame #55: 0x0000000100011e08 G4OKTest`main(argc=3, argv=0x00007ffeefbfe8a8) at G4OKTest.cc:377
        frame #56: 0x00007fff66b64015 libdyld.dylib`start + 1
        frame #57: 0x00007fff66b64015 libdyld.dylib`start + 1
    (lldb) ^D
    epsilon:g4ok blyth$ 

