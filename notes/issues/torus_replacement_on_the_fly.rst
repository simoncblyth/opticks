torus_replacement_on_the_fly
=============================

context
---------

* :doc:`quartic_solve_optix_600_misaligned_address_exception`


thoughts
-----------

As only have TITAN RTX for a few more days, and want to 
make some full geometry benchmarks with and without OptiX_600 
RTX better to:

1. just skip the torus guide tube (--csgskiplv ??)
2. on-the-fly change PMT neck to use hyperboloid OR polycone


geocache-j1808 
-----------------

:: 

   geocache-j1808 () 
    { 
        type $FUNCNAME;
        opticksdata-;
        OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22
    }


okg4/tests/OKX4Test.cc
-------------------------

1. pure G4 parses GDML
2. X4PhysicalVolume traverses the G4 tree and populates a GGeo 
3. wraps the GGeo in OKMgr and visualizes


csgskiplv : added int list handling  
-----------------------------------------

::

    [blyth@localhost opticks]$ opticks-findl csgskiplv
    ./ana/geocache.bash
    ./ggeo/GInstancer.cc
    ./optickscore/OpticksCfg.cc
    ./ggeo/GInstancer.hh
    ./optickscore/OpticksCfg.hh

    [blyth@localhost optickscore]$ opticks-f getCSGSkipLV
    ./ggeo/GGeo.cc:    m_instancer->setCSGSkipLV(m_ok->getCSGSkipLV()) ;  
    ./optickscore/Opticks.hh:       int   getCSGSkipLV() const ;
    ./optickscore/Opticks.cc:int Opticks::getCSGSkipLV() const 
    ./optickscore/Opticks.cc:   return m_cfg->getCSGSkipLV();
    ./optickscore/OpticksCfg.cc:int OpticksCfg<Listener>::getCSGSkipLV() const 
    ./optickscore/OpticksCfg.hh:     int          getCSGSkipLV() const ;  


finding lv with torus 
-----------------------

::

    2019-04-15 15:59:04.897 INFO  [220717] [X4PhysicalVolume::convertSolid@500]  ] 39
    2019-04-15 15:59:04.898 INFO  [220717] [X4PhysicalVolume::dumpTorusLV@557]  num_afflicted 6
     lvIdx ( 18 19 20 21 32 33  ) 
    18 PMT_20inch_inner1_log0x4cb3cc0
    19 PMT_20inch_inner2_log0x4c9a6e0
    20 PMT_20inch_body_log0x4cb3aa0
    21 PMT_20inch_log0x4cb3bb0
    32 lvacSurftube0x5b3c020
    33 lSurftube0x5b3ac50



need to be able to skip more than one lv
--------------------------------------------

::

    --csgskiplv 22,32,33

    ## 22: lMaskVirtual0x4c803b0 a misused polycone (actually a tubs) 
           used for technical G4 performance reasons to contain the PMT : but that obscures it so skip

    ## 32,33 : lvacSurftube0x5b3c020, lSurftube0x5b3ac50


::

    2019-04-15 16:47:13.683 INFO  [306341] [GInstancer::dump@625] GGeo::prepareVolumes
    2019-04-15 16:47:13.683 INFO  [306341] [GInstancer::dumpMeshset@569]  numRepeats 5 numRidx 6 (slot 0 for global non-instanced) 
     ridx 1 ms 5 ( 23 24 25 26 27  ) 
     ridx 2 ms 6 ( 17 18 19 20 21 22  ) 
     ridx 3 ms 4 ( 4 5 6 7  ) 
     ridx 4 ms 1 ( 15  ) 
     ridx 5 ms 1 ( 16  ) 
    2019-04-15 16:47:13.683 INFO  [306341] [GInstancer::dumpCSGSkips@601] 
     lvIdx 22 skip total : 20046 nodeIdx ( 63555 63561 63567 63573 63579 63585 63591 63597 63603 63609 63615 63621 63627 63633 63639 63645 63651 63657 63663 63669  ...  ) 


* hmm the skipping not working for globals 32 and 33 (guide tube)



for fast cycle need to write out GDML for a single PMT ? 
----------------------------------------------------------

* hmm investigate g4codegen, G4 code would do just as well as GDML



after add traverseGlobals to GInstancer
---------------------------------------------

::

    2019-04-15 17:19:53.344 INFO  [357943] [GInstancer::dump@651] GGeo::prepareVolumes
    2019-04-15 17:19:53.344 INFO  [357943] [GInstancer::dumpMeshset@595]  numRepeats 5 numRidx 6 (slot 0 for global non-instanced) 
     ridx 0 ms 23 ( 0 1 2 3 8 9 10 11 12 13 14 28 29 30 31 32 33 34 35 36 37 38 39  ) 
     ridx 1 ms 5 ( 23 24 25 26 27  ) 
     ridx 2 ms 6 ( 17 18 19 20 21 22  ) 
     ridx 3 ms 4 ( 4 5 6 7  ) 
     ridx 4 ms 1 ( 15  ) 
     ridx 5 ms 1 ( 16  ) 
    2019-04-15 17:19:53.345 INFO  [357943] [GInstancer::dumpCSGSkips@627] 
     lvIdx 22 skip total : 20046 nodeIdx ( 63555 63561 63567 63573 63579 63585 63591 63597 63603 63609 63615 63621 63627 63633 63639 63645 63651 63657 63663 63669  ...  ) 
     lvIdx 32 skip total : 1 nodeIdx ( 352854  ) 
     lvIdx 33 skip total : 1 nodeIdx ( 352853  ) 
    2019-04-15 17:19:53.345 INFO  [357943] [GGeo::prepare@683] prepareVertexColors


segv after getting global volumes skipped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ## with gdb --args OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --csgskiplv 22,32,33   
    ## no segv with --csgskiplv 22
    ## still segv with --csgskiplv 22,32
    
 
    (gdb) bt
    #0  0x00007fffe20cf3b1 in __strlen_sse2 () from /lib64/libc.so.6
    #1  0x00007fffe20cf0be in strdup () from /lib64/libc.so.6
    #2  0x00007ffff7548583 in RBuf::RBuf (this=0x18850bb30, num_items_=0, num_bytes_=1062129359, num_elements_=1057458056, ptr_=0x3f4ececf3f078788, name_=0x3f7afafb3f4ececf <Address 0x3f7afafb3f4ececf out of bounds>)
        at /home/blyth/opticks/oglrap/RBuf.cc:26
    #3  0x00007ffff756ffa2 in Renderer::setDrawable (this=0x18636f530, drawable=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:286
    #4  0x00007ffff756f9a2 in Renderer::upload (this=0x18636f530, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:257
    #5  0x00007ffff75618db in Scene::uploadGeometryGlobal (this=0x186369d90, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Scene.cc:553
    #6  0x00007ffff756216f in Scene::uploadGeometry (this=0x186369d90) at /home/blyth/opticks/oglrap/Scene.cc:634
    #7  0x00007ffff757871f in OpticksViz::uploadGeometry (this=0x186368af0) at /home/blyth/opticks/oglrap/OpticksViz.cc:326
    #8  0x00007ffff757790b in OpticksViz::init (this=0x186368af0) at /home/blyth/opticks/oglrap/OpticksViz.cc:141
    #9  0x00007ffff75774f1 in OpticksViz::OpticksViz (this=0x186368af0, hub=0x18634b540, idx=0x186367620, immediate=true) at /home/blyth/opticks/oglrap/OpticksViz.cc:98
    #10 0x00007ffff79cb92e in OKMgr::OKMgr (this=0x7fffffffcb60, argc=6, argv=0x7fffffffda18, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:49
    #11 0x000000000040521f in main (argc=6, argv=0x7fffffffda18) at /home/blyth/opticks/okg4/tests/OKX4Test.cc:121
    (gdb) f 5
    #5  0x00007ffff75618db in Scene::uploadGeometryGlobal (this=0x186369d90, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Scene.cc:553
    553             m_global_renderer->upload(mm);  
    (gdb) f 4
    #4  0x00007ffff756f9a2 in Renderer::upload (this=0x18636f530, mm=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:257
    257     setDrawable(mm);
    (gdb) f 3
    #3  0x00007ffff756ffa2 in Renderer::setDrawable (this=0x18636f530, drawable=0x1121b0470) at /home/blyth/opticks/oglrap/Renderer.cc:286
    286     m_cbuf = MAKE_RBUF(m_drawable->getColorsBuffer());
    (gdb) f 2
    #2  0x00007ffff7548583 in RBuf::RBuf (this=0x18850bb30, num_items_=0, num_bytes_=1062129359, num_elements_=1057458056, ptr_=0x3f4ececf3f078788, name_=0x3f7afafb3f4ececf <Address 0x3f7afafb3f4ececf out of bounds>)
        at /home/blyth/opticks/oglrap/RBuf.cc:26
    26      debug_index(-1)
    (gdb) f 6
    #6  0x00007ffff756216f in Scene::uploadGeometry (this=0x186369d90) at /home/blyth/opticks/oglrap/Scene.cc:634
    634            uploadGeometryGlobal(mm);
    (gdb) f 7
    #7  0x00007ffff757871f in OpticksViz::uploadGeometry (this=0x186368af0) at /home/blyth/opticks/oglrap/OpticksViz.cc:326
    326     m_scene->uploadGeometry();
    (gdb) 


::

    275 void Renderer::setDrawable(GDrawable* drawable) // CPU side buffer setup
    276 {
    277     assert(drawable);
    278     m_drawable = drawable ;
    279 
    280     NSlice* islice = drawable->getInstanceSlice();
    281     NSlice* fslice = drawable->getFaceSlice();
    282 
    283     //  nvert: vertices, normals, colors
    284     m_vbuf = MAKE_RBUF(m_drawable->getVerticesBuffer());
    285     m_nbuf = MAKE_RBUF(m_drawable->getNormalsBuffer());
    286     m_cbuf = MAKE_RBUF(m_drawable->getColorsBuffer());
    287 

::
 
     12 
     13 #define MAKE_RBUF(buf) ((buf) ? new RBuf((buf)->getNumItems(), (buf)->getNumBytes(), (buf)->getNumElements(), (buf)->getPointer(), (buf)->getName() ) : NULL )
     14 
     15 
     16 struct OGLRAP_API RBuf
     17 {   
     18     static char* Owner ;
     19     static const unsigned UNSET ;
     20     
     21     unsigned id ; 
     22     
     23     unsigned num_items ;
     24     unsigned num_bytes ;
     25     unsigned num_elements ;
     26     int      query_count ;
     27     void*       ptr ;
     28     const char* name ;
     29     
     30     bool     gpu_resident ;
     31     unsigned max_dump ;
     32     int      debug_index ;
     33 
     34     unsigned item_bytes() const ;
     35     bool isUploaded() const  ;
     36     
     37     void* getPointer() const { return ptr ; } ;
     38     unsigned getBufferId() const { return id ; } ;
     39     unsigned getNumItems() const { return num_items ; } ;
     40     unsigned getNumBytes() const { return num_bytes ; } ;
     41     unsigned getNumElements() const { return num_elements ; } ;
     42     
     43     RBuf(unsigned num_items_, unsigned num_bytes_, unsigned num_elements_, void* ptr_, const char* name_=NULL) ;
     44 


::

     27 GBuffer::GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem, const char* name)
     28     :    
     29     m_nbytes(nbytes),     // total number of bytes 
     30     m_pointer(pointer),   // pointer to the bytes
     31     m_itemsize(itemsize), // sizeof each item, eg sizeof(gfloat3) = 3*4 = 12
     32     m_nelem(nelem),       // number of elements for each item, eg 2 or 3 for floats per vertex or 16 for a 4x4 matrix
     33     m_name(name ? strdup(name) : NULL),
     34     m_buffer_id(-1),       // OpenGL buffer Id, set by Renderer on uploading to GPU 
     35     m_buffer_target(0),
     36     m_bufspec(NULL)
     37 {


::

     837 void GMesh::setColors(gfloat3* colors)
     838 {
     839     m_colors = colors ;
     840     m_colors_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_colors, sizeof(gfloat3), 3 , "colors") ;
     841     assert(sizeof(gfloat3) == sizeof(float)*3);
     842 }
     843 void GMesh::setColorsBuffer(GBuffer* buffer)
     844 {
     845     m_colors_buffer = buffer ;
     846     if(!buffer) return ;
     847 
     848     m_colors = (gfloat3*)buffer->getPointer();
     849     unsigned int numBytes = buffer->getNumBytes();
     850     unsigned int num_colors = numBytes/sizeof(gfloat3);
     851 
     852     assert( m_num_vertices == num_colors );  // must load vertices before colors
     853 }


::

    [blyth@localhost ggeo]$ grep setColors *.*
    GBBoxMesh.cc:      setColors(  new gfloat3[NUM_VERTICES]);
    GMaker.cc:    mesh->setColors(  new gfloat3[nvert]);
    GMesh.cc:        setColors(  new gfloat3[numVertices]);
    GMesh.cc:    if(strcmp(name, colors_) == 0)       setColorsBuffer(buffer) ; 
    GMesh.cc:void GMesh::setColors(gfloat3* colors)
    GMesh.cc:void GMesh::setColorsBuffer(GBuffer* buffer)
    GMesh.cc:        setColors(new gfloat3[m_num_vertices]);
    GMeshFixer.cc:    m_dst->setColors( (gfloat3*)dd_colors );
    GMesh.hh:      void setColorsBuffer(GBuffer* buffer);
    GMesh.hh:      void setColors(gfloat3* colors);
    GMeshMaker.cc:    mesh->setColors(  new gfloat3[nvert]);
    GMeshMaker.cc:    mesh->setColors(  new gfloat3[nvert]);
    GMeshMaker.cc:    mesh->setColors(  new gfloat3[nvert]);
    [blyth@localhost ggeo]$ 


::

    [blyth@localhost ggeo]$ grep colorizer *.*
    GColorizer.hh:Canonical m_colorizer instances are residents of GGeo and GScene, 
    GGeo.cc:   m_colorizer(NULL),
    GGeo.cc:    return m_colorizer ; 
    GGeo.cc:   m_colorizer = new GColorizer( m_nodelib, m_geolib, m_bndlib, colors, style ); // colorizer needs full tree, so pre-cache only 
    GGeo.cc:    m_colorizer->writeVertexColors();
    GGeo.hh:        GColorizer*                   m_colorizer ; 
    GScene.cc:    m_colorizer(new GColorizer(m_nodelib, m_geolib, m_tri_bndlib, ggeo->getColors(), GColorizer::PSYCHEDELIC_NODE )),   // GColorizer::SURFACE_INDEX
    GScene.cc:    m_colorizer->writeVertexColors();
    GScene.hh:        GColorizer*   m_colorizer ; 
    [blyth@localhost ggeo]$ 



Looks like skipping global volumes with csgskiplv causes inconsistencies : so need to do it earlier ?
--------------------------------------------------------------------------------------------------------

* edit the GDML opticksdata-jv2  used by geocache-j1808-v2
* replace rather than remove in hope of keeping indices the same

::


    [blyth@localhost juno1808]$ diff g4_00.gdml  g4_00_v2.gdml
    782,783c782,788
    <     <torus aunit="deg" deltaphi="356" lunit="mm" name="svacSurftube0x5b3bf50" rmax="8" rmin="0" rtor="17836" startphi="-268"/>
    <     <torus aunit="deg" deltaphi="356" lunit="mm" name="sSurftube0x5b3ab80" rmax="10" rmin="0" rtor="17836" startphi="-268"/>
    ---
    > 
    >     <!-- kludge replace the guide tube torus with small box of same names : see notes/issues/torus_replacement_on_the_fly.rst  -->
    >     <!--torus aunit="deg" deltaphi="356" lunit="mm" name="svacSurftube0x5b3bf50" rmax="8" rmin="0" rtor="17836" startphi="-268"/-->
    >     <box lunit="mm" name="svacSurftube0x5b3bf50" x="8" y="8" z="8"/>
    >     <!--torus aunit="deg" deltaphi="356" lunit="mm" name="sSurftube0x5b3ab80" rmax="10" rmin="0" rtor="17836" startphi="-268"/-->
    >     <box lunit="mm" name="sSurftube0x5b3ab80" x="10" y="10" z="10"/>
    > 
    [blyth@localhost juno1808]$ 



That leaves the four instanced PMT volumes which have torii
--------------------------------------------------------------

::

    2019-04-15 19:47:53.534 INFO  [157792] [X4PhysicalVolume::dumpTorusLV@560]  num_afflicted 4
     lvIdx ( 18 19 20 21  ) 
    18 PMT_20inch_inner1_log0x4cb3cc0
    19 PMT_20inch_inner2_log0x4c9a6e0
    20 PMT_20inch_body_log0x4cb3aa0
    21 PMT_20inch_log0x4cb3bb0
    2019-04-15 19:47:53.534 INFO  [157792] [X4PhysicalVolume::convertSolids@422] ]




Comparing lvIdx 18 : PMT_20inch_inner1_log0x4cb3cc0 
------------------------------------------------------

Compare the original GDML with the GDML snippet written by X4GDMLParser into the generated x018.cc,
by eye they look to be a perfect match. The GDML was read by OKX4Test and the G4VSolid trees 
written out again as per-lvIdx GDML snippets.   


::

   [blyth@localhost issues]$ opticksdata-
   [blyth@localhost issues]$ opticksdata-jv2
   /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v2.gdml


   vi /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v2.gdml


   665     <ellipsoid ax="249" by="249" cz="179" lunit="mm" name="PMT_20inch_inner_solid_1_Ellipsoid0x4c91130" zcut1="-179" zcut2="179"/>
   666     <tube aunit="deg" deltaphi="360" lunit="mm" name="PMT_20inch_inner_solid_2_Tube0x4c91210" rmax="75.95124689239" rmin="0" startphi="0" z="47.5650199027483"/>
   667     <torus aunit="deg" deltaphi="360" lunit="mm" name="PMT_20inch_inner_solid_2_Torus0x4c91340" rmax="52.01" rmin="0" rtor="97" startphi="-0.00999999999999938"/>
   668     <subtraction name="PMT_20inch_inner_solid_part20x4cb2d80">
   669       <first ref="PMT_20inch_inner_solid_2_Tube0x4c91210"/>
   670       <second ref="PMT_20inch_inner_solid_2_Torus0x4c91340"/>
   671       <position name="PMT_20inch_inner_solid_part20x4cb2d80_pos" unit="mm" x="0" y="0" z="-23.7725099513741"/>
   672     </subtraction>
   673     <union name="PMT_20inch_inner_solid_1_20x4cb30f0">
   674       <first ref="PMT_20inch_inner_solid_1_Ellipsoid0x4c91130"/>
   675       <second ref="PMT_20inch_inner_solid_part20x4cb2d80"/>
   676       <position name="PMT_20inch_inner_solid_1_20x4cb30f0_pos" unit="mm" x="0" y="0" z="-195.227490048626"/>
   677     </union>
   678     <tube aunit="deg" deltaphi="360" lunit="mm" name="PMT_20inch_inner_solid_3_EndTube0x4cb2fc0" rmax="45.01" rmin="0" startphi="0" z="115.02"/>
   679     <union name="PMT_20inch_inner_solid0x4cb32e0">
   680       <first ref="PMT_20inch_inner_solid_1_20x4cb30f0"/>
   681       <second ref="PMT_20inch_inner_solid_3_EndTube0x4cb2fc0"/>
   682       <position name="PMT_20inch_inner_solid0x4cb32e0_pos" unit="mm" x="0" y="0" z="-276.5"/>
   683     </union>
   684     <tube aunit="deg" deltaphi="360" lunit="mm" name="Inner_Separator0x4cb3530" rmax="254.000000001" rmin="0" startphi="0" z="184.000000002"/>
   685     <intersection name="PMT_20inch_inner1_solid0x4cb3610">
   686       <first ref="PMT_20inch_inner_solid0x4cb32e0"/>
   687       <second ref="Inner_Separator0x4cb3530"/>
   688       <position name="PMT_20inch_inner1_solid0x4cb3610_pos" unit="mm" x="0" y="0" z="91.999999999"/>
   689     </intersection>



/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/g4codegen/tests/x018.cc::

     27 // gdml from X4GDMLParser::ToString(G4VSolid*)  
     28 const std::string gdml = R"( 
     29 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
     30 <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">
     31 
     32   <solids>
     33     <ellipsoid ax="249" by="249" cz="179" lunit="mm" name="PMT_20inch_inner_solid_1_Ellipsoid0x4c91130" zcut1="-179" zcut2="179"/>
     34     <tube aunit="deg" deltaphi="360" lunit="mm" name="PMT_20inch_inner_solid_2_Tube0x4c91210" rmax="75.95124689239" rmin="0" startphi="0" z="47.5650199027483"/>
     35     <torus aunit="deg" deltaphi="360" lunit="mm" name="PMT_20inch_inner_solid_2_Torus0x4c91340" rmax="52.01" rmin="0" rtor="97" startphi="-0.00999999999999938"/>
     36     <subtraction name="PMT_20inch_inner_solid_part20x4cb2d80">
     37       <first ref="PMT_20inch_inner_solid_2_Tube0x4c91210"/>
     38       <second ref="PMT_20inch_inner_solid_2_Torus0x4c91340"/>
     39       <position name="PMT_20inch_inner_solid_part20x4cb2d80_pos" unit="mm" x="0" y="0" z="-23.7725099513741"/>
     40     </subtraction>
     41     <union name="PMT_20inch_inner_solid_1_20x4cb30f0">
     42       <first ref="PMT_20inch_inner_solid_1_Ellipsoid0x4c91130"/>
     43       <second ref="PMT_20inch_inner_solid_part20x4cb2d80"/>
     44       <position name="PMT_20inch_inner_solid_1_20x4cb30f0_pos" unit="mm" x="0" y="0" z="-195.227490048626"/>
     45     </union>
     46     <tube aunit="deg" deltaphi="360" lunit="mm" name="PMT_20inch_inner_solid_3_EndTube0x4cb2fc0" rmax="45.01" rmin="0" startphi="0" z="115.02"/>
     47     <union name="PMT_20inch_inner_solid0x4cb32e0">
     48       <first ref="PMT_20inch_inner_solid_1_20x4cb30f0"/>
     49       <second ref="PMT_20inch_inner_solid_3_EndTube0x4cb2fc0"/>
     50       <position name="PMT_20inch_inner_solid0x4cb32e0_pos" unit="mm" x="0" y="0" z="-276.5"/>
     51     </union>
     52     <tube aunit="deg" deltaphi="360" lunit="mm" name="Inner_Separator0x4cb3530" rmax="254.000000001" rmin="0" startphi="0" z="184.000000002"/>
     53     <intersection name="PMT_20inch_inner1_solid0x4cb3610">
     54       <first ref="PMT_20inch_inner_solid0x4cb32e0"/>
     55       <second ref="Inner_Separator0x4cb3530"/>
     56       <position name="PMT_20inch_inner1_solid0x4cb3610_pos" unit="mm" x="0" y="0" z="91.999999999"/>
     57     </intersection>
     58   </solids>
     59 
     60 </gdml>
     61 
     62 )" ;









::

    2019-04-15 19:47:53.233 INFO  [157792] [X4PhysicalVolume::convertSolid@466]  [ 18 PMT_20inch_inner1_log0x4cb3cc0
    2019-04-15 19:47:53.234 FATAL [157792] [X4Solid::convertTorus@778]  changing torus -ve startPhi (degrees) to zero -0.01
    2019-04-15 19:47:53.234 INFO  [157792] [X4PhysicalVolume::convertSolid@472] [--g4codegen] lvIdx 18 soIdx 18 lvname PMT_20inch_inner1_log0x4cb3cc0
    // start portion generated by nnode::to_g4code 
    G4VSolid* make_solid()
    { 
        G4VSolid* d = new G4Ellipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
        G4VSolid* g = new G4Tubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510, 0.000000, CLHEP::twopi) ; // 4
        G4VSolid* i = new G4Torus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000, -0.000175, CLHEP::twopi) ; // 4
        
        G4ThreeVector A(0.000000,0.000000,-23.772510);
        G4VSolid* f = new G4SubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
        
        G4ThreeVector B(0.000000,0.000000,-195.227490);
        G4VSolid* c = new G4UnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2
        G4VSolid* k = new G4Tubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000, 0.000000, CLHEP::twopi) ; // 2
        
        G4ThreeVector C(0.000000,0.000000,-276.500000);
        G4VSolid* b = new G4UnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
        G4VSolid* m = new G4Tubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4ThreeVector D(0.000000,0.000000,92.000000);
        G4VSolid* a = new G4IntersectionSolid("PMT_20inch_inner1_solid0x4cb3610", b, m, NULL, D) ; // 0
        return a ; 
    } 
    // end portion generated by nnode::to_g4code 
    2019-04-15 19:47:53.234 FATAL [157792] [X4Solid::convertTorus@778]  changing torus -ve startPhi (degrees) to zero -0.01
    2019-04-15 19:47:53.235 INFO  [157792] [NTreeBalance<T>::create_balanced@40] op_mask union intersection 
    2019-04-15 19:47:53.235 INFO  [157792] [NTreeBalance<T>::create_balanced@41] hop_mask union intersection 
    2019-04-15 19:47:53.235 FATAL [157792] [NTreeBalance<T>::create_balanced@84] balancing trees of this structure not implemented
    2019-04-15 19:47:53.247 INFO  [157792] [NTreeProcess<T>::Process@39] before
    NTreeAnalyse height 4 count 9
                                  in    

                          un          cy

          un                  cy        

      sp          di                    

              cy      to                


    2019-04-15 19:47:53.247 INFO  [157792] [NTreeBalance<T>::create_balanced@40] op_mask union intersection 
    2019-04-15 19:47:53.247 INFO  [157792] [NTreeBalance<T>::create_balanced@41] hop_mask union intersection 
    2019-04-15 19:47:53.247 FATAL [157792] [NTreeBalance<T>::create_balanced@84] balancing trees of this structure not implemented
    2019-04-15 19:47:53.247 INFO  [157792] [NTreeProcess<T>::Process@54] after
    NTreeAnalyse height 4 count 9
                                  in    

                          un          cy

          un                  cy        

      sp          in                    

              cy     !to                





Did the solid previously in ana/x019.cc ana/x018_torus_hyperboloid_plt.py  how to incorporate it ?
----------------------------------------------------------------------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/commits/22c2fa9360ec637682c77b779cdc9f1e244d5a1d

* have some G4 geometry code, so can get that running and use G4 GDML writing that to GDML : then manually edit it in  
* start with the unmodified tree : so can check for GDML matching source

::

    In [1]: run x018_torus_hyperboloid_plt.py
    x.f  f :     SubtractionSolid : array([   0.     , -195.22749]) : [g :                 Tubs : None : [75.951247, 23.78251] , i :                Torus : array([  0.     , -23.77251]) : [52.01, 97.0] , array([  0.     , -23.77251])] 
    Hyp r0:44.99 zf:32.02685120893359 stereo(radians):0.9521509204164084  
    hyp halfZLen  50.49140514041275
    /home/blyth/anaconda2/lib/python2.7/site-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      warnings.warn(message, mplDeprecation, stacklevel=1)
    pt  Ellipse(xy=(0.0, 0.0), width=498.0, height=358.0, angle=0.0)
    pt  Rectangle(xy=(-75.9512, -219.01), width=151.902, height=47.565, angle=0)
    pt  Circle(xy=(-97, -219), radius=52.01)
    pt  Circle(xy=(97, -219), radius=52.01)
    pt  Rectangle(xy=(-45.01, -334.01), width=90.02, height=115.02, angle=0)
    pt  Rectangle(xy=(-254, 0), width=508, height=184, angle=0)

    In [2]: 




Something analogous to X4CSG::Serialize that writes GDML
----------------------------------------------------------

* implemented this in X4GDMLParser

* wrap the G4VSolid into a Geant4 jacket, with names for the volumes passed as arguments

::

     28 void X4CSG::Serialize( const G4VSolid* solid, const char* csgpath ) // static
     29 {
     30     X4CSG xcsg(solid);
     31     std::cerr << xcsg.save(csgpath) << std::endl ;   // NB only stderr emission to be captured by bash 
     32     xcsg.dumpTestMain();
     33 }


G4GDML
----------

* hmm actually just want to write the solid, can G4 GDML be persuaded/tricked into doing that 
* hacked GDML writing of solids in X4GDMLParser X4GDMLWriter, notes in review. 
* :doc:`G4GDML_review`



Making sense of 18,19,20,21
------------------------------

::

    18 PMT_20inch_inner1_log0x4cb3cc0
    19 PMT_20inch_inner2_log0x4c9a6e0
    20 PMT_20inch_body_log0x4cb3aa0
    21 PMT_20inch_log0x4cb3bb0
 



Compare 18 and 19 : 18 intersects with the separator cylinder, 19 subtracts it 
---------------------------------------------------------------------------------------

The solids are identical : just a very few name changes and flipping from an intersection to a subtraction.
So the constituent solids are mostly the same between 18 and 19.

* it would be cute if it were not such an expensive way of modelling the cathode and the rest 

::

    geocache-tcd
    [blyth@localhost tests]$ diff -y x018.cc x019.cc 
    ... too wide to appear here 

::


    [blyth@localhost tests]$ diff x018.cc x019.cc 
    53c53
    <     <intersection name="PMT_20inch_inner1_solid0x4cb3610">
    ---
    >     <subtraction name="PMT_20inch_inner2_solid0x4cb3870">
    56,57c56,57
    <       <position name="PMT_20inch_inner1_solid0x4cb3610_pos" unit="mm" x="0" y="0" z="91.999999999"/>
    <     </intersection>
    ---
    >       <position name="PMT_20inch_inner2_solid0x4cb3870_pos" unit="mm" x="0" y="0" z="91.999999999"/>
    >     </subtraction>
    63c63
    < // LV=18
    ---
    > // LV=19
    83c83
    <     G4VSolid* a = new G4IntersectionSolid("PMT_20inch_inner1_solid0x4cb3610", b, m, NULL, D) ; // 0
    ---
    >     G4VSolid* a = new G4SubtractionSolid("PMT_20inch_inner2_solid0x4cb3870", b, m, NULL, D) ; // 0
    [blyth@localhost tests]$ 



Compare 20 and 21 : many more differences in all names, numbers but CSG structure is the same
------------------------------------------------------------------------------------------------

* dimensions of 21 are very slightly enlarged over 20

::

    geocache-tcd
    vimdiff x020.cc x021.cc  // clearer in macOS Terminal than on linux


Volumes
-------------

::

    18 PMT_20inch_inner1_log0x4cb3cc0
    19 PMT_20inch_inner2_log0x4c9a6e0
    20 PMT_20inch_body_log0x4cb3aa0
    21 PMT_20inch_log0x4cb3bb0

::

      1447     <volume name="PMT_20inch_inner1_log0x4cb3cc0">
      1448       <materialref ref="Vacuum0x4b9b630"/>
      1449       <solidref ref="PMT_20inch_inner1_solid0x4cb3610"/>            // 18 : intersection to give the cap
      1450     </volume>

      1451     <volume name="PMT_20inch_inner2_log0x4c9a6e0">
      1452       <materialref ref="Vacuum0x4b9b630"/>
      1453       <solidref ref="PMT_20inch_inner2_solid0x4cb3870"/>            // 19 : subtraction to give the remainder
      1454     </volume>

      1455     <volume name="PMT_20inch_body_log0x4cb3aa0">               // 20    
      1456       <materialref ref="Pyrex0x4bae2a0"/>
      1457       <solidref ref="PMT_20inch_body_solid0x4c90e50"/>               
      1458       <physvol name="PMT_20inch_inner1_phys0x4c9a870">             // 18 : cap (cathode) vacuum
      1459         <volumeref ref="PMT_20inch_inner1_log0x4cb3cc0"/>
      1460       </physvol>
      1461       <physvol name="PMT_20inch_inner2_phys0x4c9a920">             // 19 : remainder vacuum
      1462         <volumeref ref="PMT_20inch_inner2_log0x4c9a6e0"/>
      1463       </physvol>
      1464     </volume>

      1465     <volume name="PMT_20inch_log0x4cb3bb0">                    // 21   
      1466       <materialref ref="Pyrex0x4bae2a0"/>
      1467       <solidref ref="PMT_20inch_pmt_solid0x4c81b40"/>
      1468       <physvol name="PMT_20inch_body_phys0x4c9a7f0">             // 20 inside (very slightly smaller dimension) : outer coating attempt 
      1469         <volumeref ref="PMT_20inch_body_log0x4cb3aa0"/>
      1470       </physvol>
      1471     </volume>



ana/x018.py x019.py x020.py x021.py xplt.py
-------------------------------------------------

Manually translated the generated g4code into python (in a style that 
can be generated if necessary)::

     09 class x021(X):
     10     """ 
     11     // LV=21 
     12     // start portion generated by nnode::to_g4code 
     13     G4VSolid* make_solid()  
     14     { 
     15         G4VSolid* c = new G4Ellipsoid("PMT_20inch_pmt_solid_1_Ellipsoid0x4c3bc00", 254.001000, 254.001000, 184.001000, -184.001000, 184.001000) ; // 2
     16         G4VSolid* f = new G4Tubs("PMT_20inch_pmt_solid_2_Tube0x4c3bc90", 0.000000, 77.976532, 21.496235, 0.000000, CLHEP::twopi) ; // 3
     17         G4VSolid* h = new G4Torus("PMT_20inch_pmt_solid_2_Torus0x4c84bd0", 0.000000, 47.009000, 97.000000, -0.000175, CLHEP::twopi) ; // 3
     18         
     19         G4ThreeVector A(0.000000,0.000000,-21.486235);
     20         G4VSolid* e = new G4SubtractionSolid("PMT_20inch_pmt_solid_part20x4c84c70", f, h, NULL, A) ; // 2
     21            
     22         G4ThreeVector B(0.000000,0.000000,-197.513765);
     23         G4VSolid* b = new G4UnionSolid("PMT_20inch_pmt_solid_1_20x4c84f90", c, e, NULL, B) ; // 1
     24         G4VSolid* j = new G4Tubs("PMT_20inch_pmt_solid_3_EndTube0x4c84e60", 0.000000, 50.011000, 60.010500, 0.000000, CLHEP::twopi) ; // 1
     25            
     26         G4ThreeVector C(0.000000,0.000000,-279.000500);
     27         G4VSolid* a = new G4UnionSolid("PMT_20inch_pmt_solid0x4c81b40", b, j, NULL, C) ; // 0
     28         return a ; 
     29     }       
     30     // end portion generated by nnode::to_g4code 
     31     """     
     32     def __init__(self):
     33         c = Ellipsoid( "c", [254.001000, 184.001000] )
     34         f = Tubs(     "f", [77.976532, 21.496235] )
     35         h = Torus(    "h", [47.009000, 97.000000] )
     36 
     37         A = np.array( [0.000000,-21.486235] )
     38         e = SubtractionSolid( "e" , [f, h, A] )
     39 
     40         B = np.array( [0.000000,-197.513765])
     41         b = UnionSolid( "b", [c, e, B] )
     42 
     43         j = Tubs( "j", [50.011000, 60.010500] )
     44         C = np.array( [0.000000,-279.000500] )
     45         a = UnionSolid( "a", [b, j, C] )
     46 
     47         self.root = a




Cutting Ellipsoid
---------------------

::

     g4-cls G4Ellipsoid

     37 //   A G4Ellipsoid is an ellipsoidal solid, optionally cut at a given z.
     38 //
     39 //   Member Data:
     40 //
     41 //      xSemiAxis       semi-axis, x
     42 //      ySemiAxis       semi-axis, y
     43 //      zSemiAxis       semi-axis, z
     44 //      zBottomCut      lower cut plane level, z (solid lies above this plane)
     45 //      zTopCut         upper cut plane level, z (solid lies below this plane)
     46 



How to rationalize : starting in ana/x018.py x019.py x020.py 
---------------------------------------------------------------

::

    18 : cap : single ellipsoid only with zBottomCut
    19 : rest :  ellipsoid with zTopCut (=zBottomCut above) + (polycone) + tubs
    20 : ellipsoid + (polycone) + tubs
    21 : ellipsoid + (polycone) + tubs

* where the polycone replaces cylinder-torus


Maths to calculate the cons to replace the Subtraction Solid (tubs - torus)
------------------------------------------------------------------------------

Are generalizing the initial imp of ana/x018_torus_hyperboloid_plt.py into ana/shape.py

* have replaced the tubs-torus bileaf with cons
* used tree surgery on a copy

* surgery is applied to x018 x019 in ana/shape.py removing root level intersect/subtraction

  * 18 : ellipsoid becomes root with zrange upper half
  * 19 : root.left becomes root and ellipsoid zrange lower half 
  * z-cuts just need to select upper and lower halfs of the ellipsoid 


Getting the post-op solids into the main GDML file
----------------------------------------------------

Remaining:

* simple python model missing ellipsoid z-cuts and a matplotlib presentation of these

* given rationalized python trees,  need to then generate 
  corresponding G4 code (will need to propagate the original names in, 
  so can do that with generated x018.py etc)

* running the G4 code to make a rationalized solid can then 
  be converted to GDML snippets for manual inclusion 
  into the opticksdata-jv2 GDML 

Better way, do the surgery at NNode level : avoids some steps.

* actually implementing the rationalization tree surgery of shape.py on nnode trees in C++
  rather than in python would allow the G4VSolid trees to be reconstructed live 
  (already have all the G4VSolid parameters for the g4codegen so can just turn around 
  and recreate the G4 objects) from the nnode trees after their surgery. 
  Then the G4VSolids can be directly written out to GDML using X4GDMLParser 
  for inclusion into the full GDML file.

  * :google:`C++11 dynamic arguments to method call` 

  * this way is not easy because of C++ lack of dynamism
  * workarounds (variadic templates) not applicable when cannot change the target code
    G4VSolid ctors
  * so are back to generating strings for the code, which has the advantage
    of being simple and fully flexible at the cost of having to compile that code 
  * hmm but need to change parameter, so need a type signature for each 

G4VSolid type signatures
--------------------------

Actually not so difficult, only need this for a very small subset 
of G4VSolids used in JUNO PMT so can just use an if statement.

NNode.hpp::

        const char*  g4code ; 
    +    const char*  g4name ; 
    +    std::map<std::string, double>* g4args ; 
     

::


    G4Sphere(const G4String& pName,
                   G4double pRmin, G4double pRmax,
                   G4double pSPhi, G4double pDPhi,
                   G4double pSTheta, G4double pDTheta);


   G4Ellipsoid(const G4String& pName,
                      G4double  pxSemiAxis,
                      G4double  pySemiAxis,
                      G4double  pzSemiAxis,
                      G4double  pzBottomCut=0,
                      G4double  pzTopCut=0);

    G4Cons(const G4String& pName,
                 G4double pRmin1, G4double pRmax1,
                 G4double pRmin2, G4double pRmax2,
                 G4double pDz,
                 G4double pSPhi, G4double pDPhi);

    G4Tubs( const G4String& pName,
                  G4double pRMin,
                  G4double pRMax,
                  G4double pDz,
                  G4double pSPhi,
                  G4double pDPhi );


 


