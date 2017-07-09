CSG Intersect Comparisons
============================


Approaches ?
-----------------


* Extend cfg4/CMaker to work with NCSG geometry

* cfg4 needs updating to NCSG rather than old commandline geometry



cfg4 review
------------

::

     13 CMaker
     14 ======
     15 
     16 CMaker is a constitent of CTestDetector used
     17 to convert GCSG geometry into G4 geometry in 
     18 G4VPhysicalVolume* CTestDetector::Construct(). 
     19 
     20 CMaker::makeSolid handles some boolean intersection
     21 and union combinations via recursive calls to itself.
     22 
     23 CMaker only handles the geometrical shapes.
     24 Material assignments are done elsewhere, 
     25 at a higher level eg by CTestDetector.
     26 


     36 CDetector
     37 ===========
     38   
     39 *CDetector* is the base class of *CGDMLDetector* and *CTestDetector*, 
     40 it is a *G4VUserDetectorConstruction* providing detector geometry to Geant4.
     41 The *CDetector* instance is a constituent of *CGeometry* which is 
     42 instanciated in *CGeometry::init*. 
     43   

     57 CTestDetector
     58 =================
     59 
     60 *CTestDetector* is a :doc:`CDetector` subclass that
     61 constructs simple Geant4 detector test geometries based on commandline specifications
     62 parsed and represented by an instance of :doc:`../ggeo/GGeoTestConfig`.
     63 





makeSolid
-------------


::

    070 G4VSolid* CMaker::makeSolid(OpticksCSG_t type, const glm::vec4& param)
     71 {
     72     G4VSolid* solid = NULL ;
     73     switch(type)
     74     {
     75         case CSG_BOX:   solid = makeBox(param);break;
     76         case CSG_SPHERE:solid = makeSphere(param);break;
     77         case CSG_UNION:
     78         case CSG_INTERSECTION:
     79         case CSG_DIFFERENCE:
     80         case CSG_ZSPHERE:
     81         case CSG_ZLENS:
     82         case CSG_PMT:
     83         case CSG_PRISM:
     84         case CSG_TUBS:
     85         case CSG_PARTLIST:
     86         case CSG_CYLINDER:
     87         case CSG_DISC:
     88         case CSG_CONE:
     89         case CSG_MULTICONE:
     90         case CSG_BOX3:
     91         case CSG_PLANE:
     92         case CSG_SLAB:
     93         case CSG_TRAPEZOID:
     94         case CSG_ZERO:
     95         case CSG_UNDEFINED:
     96         case CSG_FLAGPARTLIST:
     97         case CSG_FLAGNODETREE:
     98         case CSG_FLAGINVISIBLE:
     99         case CSG_CONVEXPOLYHEDRON:
    100                          solid = NULL ; break ;
    101 
    102     }
    103     return solid ;
    104 }


::

     59 CTestDetector::CTestDetector(OpticksHub* hub, GGeoTestConfig* config, OpticksQuery* query)
     60   :
     61   CDetector(hub, query),
     62   m_config(config),
     63   m_maker(NULL)
     64 {
     65     init();
     66 }



::

    delta:cfg4 blyth$ grep makeSolid *.*
    CMaker.cc://G4VSolid* CMaker::makeSolid(char shapecode, const glm::vec4& param)
    CMaker.cc:G4VSolid* CMaker::makeSolid(OpticksCSG_t type, const glm::vec4& param)
    CMaker.cc:G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
    CMaker.cc:           << "CMaker::makeSolid (GCSG)  "
    CMaker.cc:       G4VSolid* asol = makeSolid(csg, a );
    CMaker.cc:       G4VSolid* bsol = makeSolid(csg, b );
    CMaker.cc:       G4VSolid* isol = makeSolid(csg, i );
    CMaker.cc:       G4VSolid* jsol = makeSolid(csg, j );
    CMaker.cc:       G4VSolid* ksol = makeSolid(csg, k );
    CMaker.cc:       LOG(info) << "CMaker::makeSolid csg Sphere"
    CMaker.cc:       LOG(info) << "CMaker::makeSolid"
    CMaker.cc:       LOG(warning) << "CMaker::makeSolid implementation missing " ; 
    CMaker.hh:CMaker::makeSolid handles some boolean intersection
    CMaker.hh:        //G4VSolid* makeSolid(char shapecode, const glm::vec4& param);
    CMaker.hh:        G4VSolid* makeSolid(OpticksCSG_t type, const glm::vec4& param);
    CMaker.hh:        G4VSolid* makeSolid(GCSG* csg, unsigned int i); 
    CTestDetector.cc:        G4VSolid* solid = m_maker->makeSolid(type, param);  
    CTestDetector.cc:    G4VSolid* solid = m_maker->makeSolid(csg, i );
    delta:cfg4 blyth$ 



This was the primordial CSG approach before NCSG::


    107 G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
    108 {
    109    // hmm this is somewhat specialized to known structure of DYB PMT
    110    //  eg intersections are limited to 3 ?
    111 
    112     unsigned int nc = csg->getNumChildren(index);
    113     unsigned int fc = csg->getFirstChildIndex(index);
    114     unsigned int lc = csg->getLastChildIndex(index);
    115     unsigned int tc = csg->getTypeCode(index);
    116     const char* tn = csg->getTypeName(index);
    117 

