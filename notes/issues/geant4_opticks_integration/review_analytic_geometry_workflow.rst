Review Analytic Geometry Workflow
===================================


Changes needed
------------------

* current approach assumes simple 1-1, boolean CSG tree breaks that 


Test Geometry
--------------

configured eg in tboolean-
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    131 tboolean-box-sphere()
    132 {
    133     local operation=${1:-difference}
    134     local material=$(tboolean-material)
    135     local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    136     local test_config=(
    137                  mode=BoxInBox
    138                  analytic=1
    139 
    140                  node=box          parameters=0,0,0,1000          boundary=Rock//perfectAbsorbSurface/Vacuum
    141 
    142                  node=$operation   parameters=0,0,0,300           boundary=Vacuum///$material
    143                  node=box          parameters=0,0,0,$inscribe     boundary=Vacuum///$material
    144                  node=sphere       parameters=0,0,0,200           boundary=Vacuum///$material
    145                )
    146 
    147      echo "$(join _ ${test_config[@]})" 
    148 }


config string parsed with GGeoTestConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   GGeoTestConfig::GGeoTestConfig(const char* config) 

* GGeoTestConfig provides by index API access to node properties, parameters, codes, shapes, transforms etc..

::

    simon:opticks blyth$ opticks-lfind GGeoTestConfig 
    ./bin/oks.bash
    ./cfg4/cfg4.bash
    ./ggeo/ggeodev.bash

    ./ggeo/CMakeLists.txt

    ./ggeo/GGeo.cc

    ./ggeo/GGeoTestConfig.cc
    ./ggeo/GGeoTestConfig.hh

    ./ggeo/GGeoTest.cc
    ./ggeo/GGeoTest.hh
    ./ggeo/tests/GGeoTestTest.cc

    ./cfg4/CGeometry.cc

    ./cfg4/CTestDetector.cc
    ./cfg4/CTestDetector.hh
    ./cfg4/tests/CTestDetectorTest.cc



GGeo::modifyGeometry : steering of geometry mods in test mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     743 void GGeo::modifyGeometry(const char* config)
     744 {
     745     // NB only invoked with test option : "ggv --test" 
     746     GGeoTestConfig* gtc = new GGeoTestConfig(config);
     747 
     748     LOG(trace) << "GGeo::modifyGeometry"
     749               << " config [" << ( config ? config : "" ) << "]" ;
     750 
     751     assert(m_geotest == NULL);
     752 
     753     m_geotest = new GGeoTest(m_opticks, gtc, this);
     754     m_geotest->modifyGeometry();
     755 }



GGeoTest::createBoxInBox act on config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* convert config into vector of GSolid using GMaker to 
  do the low level conversion


::

    233 GMergedMesh* GGeoTest::createBoxInBox()
    234 {
    235     std::vector<GSolid*> solids ;
    236     unsigned int n = m_config->getNumElements();
    237     
    238     for(unsigned int i=0 ; i < n ; i++)
    239     {
    240         std::string node = m_config->getNodeString(i);
    241         char nodecode = m_config->getNode(i) ;
    242         const char* spec = m_config->getBoundary(i);
    243         glm::vec4 param = m_config->getParameters(i);
    244         glm::mat4 trans = m_config->getTransform(i);
    245         unsigned int boundary = m_bndlib->addBoundary(spec);
    ...
    258         if(nodecode == 'U') LOG(fatal) << "GGeoTest::createBoxInBox configured node not implemented " << node ;
    259         assert(nodecode != 'U');
    260         
    261         GSolid* solid = m_maker->make(i, nodecode, param, spec );
    262         solids.push_back(solid);
    263         
    264         // TODO: handle csg tree nodes, that break the 1-to-1 
    265     }   


GMaker::make tesselated vertices etc.. in associated GMesh, analytic desc in associated GParts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    104 GSolid* GMaker::make(unsigned int /*index*/, char shapecode, glm::vec4& param, const char* spec )
    105 {
    106     // invoked from eg GGeoTest::createBoxInBox while looping over configured shape/boundary/param entries
    107     // hmm for generality a boolean shape needs to reference two others, the prior two? 
    108     // hmm this is too soon to do booleans, need the basis solids first 
    109     // unless handle booleans by setting constituent flag 
    110 
    111      GSolid* solid = NULL ;
    112      switch(shapecode)
    113      {
    114          case 'B': solid = makeBox(param); break;
    115          case 'M': solid = makePrism(param, spec); break;
    116          case 'S': solid = makeSubdivSphere(param, 3, "I") ; break; // I:icosahedron O:octahedron HO:hemi-octahedron C:cube 
    117          case 'Z': solid = makeZSphere(param) ; break;
    118          case 'L': solid = makeZSphereIntersect(param, spec) ; break;   // composite handled by adding child node
    119          case 'I': solid = makeBox(param); break ;    // boolean intersect
    120          case 'J': solid = makeBox(param); break ;    // boolean union
    121          case 'K': solid = makeBox(param); break ;    // boolean difference
    122      }
    123      assert(solid);
    124 
    125      OpticksShape_t shapeflag = GMaker::NodeFlag(shapecode) ;
    126      solid->setShapeFlag( shapeflag );
    127 
    128      // TODO: most parts alread hooked up above, do this uniformly
    129      GParts* pts = solid->getParts();
    130      if(pts == NULL)
    131      {
    132          pts = GParts::make(shapecode, param, spec);
    133          solid->setParts(pts);
    134      }
    135      assert(pts);
    136 
    137      unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing
    138      solid->setBoundaryAll(boundary);   // All loops over immediate children, needed for composite
    139      pts->setBoundaryAll(boundary);
    140      pts->enlargeBBoxAll(0.01f );
    141 
    142      return solid ;
    143 }


GParts provides control of float parts (n_part,4,4) and uint prim (n_prim,4) buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Prim buffer is derived from parts buffer by `GParts::makePrimBuffer`
* prim/part buffer structure tied to oxrap/cu/hemi-pmt.cu


hemi-pmt.cu
~~~~~~~~~~~~~


::

    1250 RT_PROGRAM void intersect(int primIdx)
    1251 {
    1252   const uint4& prim    = primBuffer[primIdx];
    1253 
    1254   unsigned partOffset  = prim.x ;
    1255   unsigned numParts    = prim.y ;
    1256   unsigned primFlags   = prim.w ;   // <-- hmm not good to keep flags up here, use flags in partBuffer for simplicity

    ////
    ////   primBuffer acts as index to the partBuffer
    ////   current intersect_boolean hardcoded to 
    ////   handle operations between two basis parts only 
    ////
    ////   TODO: work out tree generalization approach
    ////         (only small csg trees are envisaged with a couple of boolean ops)
    ////

    1257 
    1258   uint4 identity = identityBuffer[instance_index] ;
    1259 
    1260   // for analytic test geometry (PMT too?) the identityBuffer  
    1261   // is composed of placeholder zeros
    1262 
    1263   if(primFlags & SHAPE_BOOLEAN)
    1264   {
    1265       quad q1 ;
    1266       q1.f = partBuffer[4*(partOffset+0)+1];
    1267       identity.z = q1.u.z ;        // replace placeholder zero ? with test analytic geometry boundary
    1268 
    1269       intersect_boolean( prim, identity );
    1270       //intersect_boolean_only_first( prim, identity );
    1271 
    1272   }
    1273   else
    1274   {
    1275       for(unsigned int p=0 ; p < numParts ; p++)
    1276       {
    1277           unsigned int partIdx = partOffset + p ;
    1278 
    1279           quad q0, q1, q2, q3 ;
    1280 
    1281           q0.f = partBuffer[4*partIdx+0];
    1282           q1.f = partBuffer[4*partIdx+1];
    1283           q2.f = partBuffer[4*partIdx+2] ;
    1284           q3.f = partBuffer[4*partIdx+3];



intersect_boolean.h
~~~~~~~~~~~~~~~~~~~~~~

Issues

* hardcodes two basis shapes, no transforms

* does the OptiX reporting, preventing recursion


* partIdx needs to be able to point at an "operation" node, 
  not just a basis shape node, and find flags there
  identifying the type of intersect 




::

     33 static __device__
     34 void intersect_boolean( const uint4& prim, const uint4& identity )
     35 {
     36    // hmm to work with boolean CSG tree primitives this
     37    // needs to have the same signature as intersect_part 
     38    // ie with deferring the reporting to OptiX to the caller
     39 
     40     unsigned primFlags  = prim.w ;
     41 
     42 
     43     // TODO: pass "operation" enum from CPU side, instead of wishy-washy flags   
     44     enum { INTERSECT, UNION, DIFFERENCE  };
     45     int bop = primFlags & SHAPE_INTERSECTION ?
     46                                                   INTERSECT
     47                                              :
     48                                                   ( primFlags & SHAPE_DIFFERENCE ? DIFFERENCE : UNION )
     49                                              ;
     50 
     51     unsigned a_partIdx = prim.x + 1 ;
     52     unsigned b_partIdx = prim.x + 2 ;
     53 

     ////  hmm the partIdx needs to be able to point at an "operation" node, 
     ////  not just a basis shape node




tree generalization 
~~~~~~~~~~~~~~~~~~~~~~

* aiming to allow multiple small csg trees (for diffrent solids) 
  to reside in a single parts buffer with the prim buffer being used to index 
  into the into the relevant indices holding the csg tree 

* so view the prim buffer just as the high level splitting up index into
  the part buffer (not as a provider of flags) :

  Keeping everything other than the splitting info in the part buffer is needed to allow 
  intersects based on just a partIdx that could be pointing at a basis shape node 
  or an operation node 

* recursion generaly best avoided in CUDA, so use stack to handle the tree 

* see oxrap/cu/bintree.py for prototype of CSG binary tree serialization/deserialization
  using breadth first ordering in order to work with simpler pointerless index 
  navigation  


From csg- with tree root at index i=0, with valid indices 0 through numParts − 1, 
then the i-th element has children at 

* 2i + 1 
* 2i + 2 
* children can navigate to parent at index floor((i − 1) ∕ 2).

::
 
     // i: 0..n-1
     //  2i+1,2i+2, floor((i − 1) ∕ 2)
  
     0
     1        2
     3   4    5     6     
     7 8 9 10 11 12 13 14 
     
     // hmm storing multiple trees in one array requires offsets
     // after the 1st 



recursion to iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://blog.moertel.com/tags/recursion-to-iteration%20series.html

 

