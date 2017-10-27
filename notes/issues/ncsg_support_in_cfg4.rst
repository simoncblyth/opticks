ncsg_support_in_cfg4 : Push CSG node tree support thru to cfg4
=================================================================

Objective
----------

Direct CSG intersect comparisons between Opticks and G4 using Opticks 
defined test geometries, eg invoked by the tboolean- tests 
using --okg4 to switch on bi-simulation.


Overview
----------

* creation of Geant4 geometries from the NCSG/GParts node tree description
* comparisons of GPU and CPU propagations using CSG node tree geometries

* tpmt-t tconcentric-t were primary users of cfg4 comparison funcs
  using GCSG translation : but GCSG translation to G4 geometry was
  very limited ... OpticksCSG supports many more primitives  



Approach
-------------------------------------------------------

* review GCSG usage in cfg4 
* decide what level to operate (NCSG/GParts/..) ? 
* start with test geometry scope only, not full structure
* implement the conversion
* new versions of tpmt-t tconcentric-t 


review GCSG, ggeo created, used in cfg4
------------------------------------------

GCSG:

* primordial CSG approach, used to describe manual/detdesc analytic PMT
* is referred to in past tense, as regarded as almost dead code, new dev should not use it.
* keeping alive to enable comparisons with new approaches only, until the new approaches can take over
* very limited, sphere/tubs/boolean, to what was needed for DYB PMT


::

    simon:cfg4 blyth$ grep GCSG *.*
    CMaker.cc:#include "GCSG.hh"
    CMaker.cc:G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
    CMaker.cc:           << "CMaker::makeSolid (GCSG)  "
    CMaker.hh:class GCSG ; 
    CMaker.hh:to convert GCSG geometry into G4 geometry in 
    CMaker.hh:        G4VSolid* makeSolid(GCSG* csg, unsigned int i);  // ancient CSG 
    CTestDetector.cc:#include "GCSG.hh"
    CTestDetector.cc:    GCSG* csg = pmt ? pmt->getCSG() : NULL ;
    CTestDetector.cc:G4LogicalVolume* CTestDetector::makeLV(GCSG* csg, unsigned int i)
    CTestDetector.hh:class GCSG ; 
    CTestDetector.hh:    G4LogicalVolume* makeLV(GCSG* csg, unsigned int i);
    cfg4.bash:     Constitent of CTestDetector used to convert GCSG geometry 
    simon:cfg4 blyth$ 


::

     78 G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
     79 {
     80    // hmm this is somewhat specialized to known structure of DYB PMT
     81    //  eg intersections are limited to 3 ?
     82 
     83     unsigned int nc = csg->getNumChildren(index);
     84     unsigned int fc = csg->getFirstChildIndex(index);
     85     unsigned int lc = csg->getLastChildIndex(index);
     86     unsigned int tc = csg->getTypeCode(index);
     87     const char* tn = csg->getTypeName(index);
     88 



::

    105 G4VPhysicalVolume* CTestDetector::makeDetector()
    106 {
    107    // analagous to ggeo-/GGeoTest::CreateBoxInBox
    108    // but need to translate from a surface based geometry spec into a volume based one
    109    //
    110    // creates Russian doll geometry layer by layer, starting from the outermost 
    111    // hooking up mother volume to prior 
    112    //
    113     GMergedMesh* mm = m_ggeo->getMergedMesh(0);
    114     unsigned numSolidsMesh = mm->getNumSolids();
    115     unsigned int numSolidsConfig = m_config->getNumElements();
    116 
    117     bool is_pib = isPmtInBox() ;
    118     bool is_bib = isBoxInBox() ;
    119     // CsgInBox not yet handled
    120 
    121     LOG(info)  << "CTestDetector::makeDetector"
    122                << " PmtInBox " << is_pib
    123                << " BoxInBox " << is_bib
    124                << " numSolidsMesh " << numSolidsMesh
    125                << " numSolidsConfig " << numSolidsConfig
    126               ;
    127 
    128     assert( ( is_pib || is_bib ) && "CTestDetector::makeDetector mode not recognized");
    129 





NCSG
------

Huh, made start already.

::

    294 G4VSolid* CMaker::makeSolid(NCSG* csg)
    295 {
    296     nnode* root_ = csg->getRoot();
    297 
    298     G4VSolid* root = makeSolid_r(root_);
    299 
    300     return root  ;
    301 }
    302 
    303 G4VSolid* CMaker::makeSolid_r(const nnode* node)
    304 {
    305     // hmm rmin/rmax is handled as a CSG subtraction
    306     // so could collapse some operators into primitives





