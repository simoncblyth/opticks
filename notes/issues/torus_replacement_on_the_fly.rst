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



::


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
* :doc:`G4GDML_review`





