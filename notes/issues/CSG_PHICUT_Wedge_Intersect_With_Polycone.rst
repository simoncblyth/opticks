CSG_PHICUT_Wedge_Intersect_With_Polycone.rst
===============================================



Tools::

    u4/tests/U4SolidMakerTest.sh

    g4cx/tests/G4CX_U4TreeCreateCSGFoundryTest.sh    ## convert Geant4 geom to Opticks depending on GEOM envvar

    NOXGEOM=1 cxr_min.sh

    NOXGEOM=1 ssst.sh

    NOXGEOM=1 EYE=0,0,1 UP=0,1,0 cxr_min.sh





* FIXED : ssst.sh example solid appears outside world box
* ISSUE : whacky view dependent geometry with cxr_min.sh





::

    (ok) A[blyth@localhost sysrap]$ opticks-f CSG_SEGMENT
    ./cfg4/CMaker.cc:    else if(node->type == CSG_TRAPEZOID || node->type == CSG_SEGMENT || node->type == CSG_CONVEXPOLYHEDRON)

    ./npy/NCSG.cpp:       case CSG_SEGMENT:  
    ./npy/NConvexPolyhedron.cpp:        // for CSG_SEGMENT two planes include the origin, so allow zero
    ./npy/NConvexPolyhedron.cpp:        // for CSG_SEGMENT two planes include the origin, so allow zero
    ./npy/NPrimitives.hpp: CSG_SEGMENT              segment             nconvexpolyhedron

    ./sysrap/OpticksCSG.h:        CSG_SEGMENT=114,
    ./sysrap/OpticksCSG.h:static const char* CSG_SEGMENT_       = "segment" ;
    ./sysrap/OpticksCSG.h:        else if(strcmp(nodename, CSG_SEGMENT_) == 0)        tc = CSG_SEGMENT ;
    ./sysrap/OpticksCSG.h:            case CSG_SEGMENT:       s = CSG_SEGMENT_       ; break ;
    ./sysrap/OpticksCSG.h:        return (type == CSG_TRAPEZOID || type == CSG_CONVEXPOLYHEDRON || type == CSG_SEGMENT ) ;

    ./sysrap/tests/OpticksCSGTest.cc:                CSG_SEGMENT,
    ./sysrap/tests/OpticksCSG_test.cc:                CSG_SEGMENT,
    ./u4/U4Polycone.h:3. HMM: use boolean intersection with a special segment (CSG_SEGMENT)
    (ok) A[blyth@localhost opticks]$ 



::

    csg_intersect_leaf_phicut.h

    (ok) A[blyth@localhost opticks]$ opticks-fl phicut
    ./CSG/csg_geochain.sh
    ./CSG/csg_intersect_leaf_phicut.h
    ./CSG/csg_intersect_leaf.h
    ./CSG/phicut.sh
    ./CSG/tests/CMakeLists.txt
    ./CSG/tests/CSGClassifyTest.cc
    ./CSG/tests/CSGIntersectSolidTest.py
    ./CSG/tests/CSGNodeScanTest.cc
    ./CSG/tests/intersect_leaf_phicut_test.cc
    ./CSG/CMakeLists.txt
    ./GeoChain/translate.sh
    ./extg4/X4Solid.cc
    ./extg4/X4SolidMaker.cc
    ./npy/NCSG.cpp
    ./npy/NNode.cpp
    ./npy/NPhiCut.cpp
    ./npy/NPhiCut.hpp
    ./npy/tests/NPhiCutTest.cc
    ./sysrap/OpticksCSG.h
    ./sysrap/OpticksCSG.py
    ./sysrap/SPhiCut.cc
    ./u4/U4Solid.h
    ./u4/U4SolidMaker.cc
    (ok) A[blyth@localhost opticks]$ 



PHICUT::

    (ok) A[blyth@localhost sysrap]$ opticks-f PHICUT
    ./CSG/CSGNode.cc:    else if( tc == CSG_PHICUT )
    ./CSG/CSGNode.cc:    nd.setTypecode(CSG_PHICUT);
    ./CSG/csg_intersect_leaf.h:    CSG_PHICUT 
    ./CSG/csg_intersect_leaf.h:        case CSG_PHICUT:           distance = distance_leaf_phicut(            local_position, node->q0 )           ; break ;
    ./CSG/csg_intersect_leaf.h:        case CSG_PHICUT:           intersect_leaf_phicut(           valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    ./npy/NCSG.cpp:       case CSG_PHICUT:         node = nphicut::Create(p0)        ; break ; 
    ./npy/NNode.cpp:        case CSG_PHICUT:          { nphicut* n       = (nphicut*)node        ; c = new nphicut(*n)       ; } break ; 
    ./npy/NPhiCut.cpp:    nnode::Init(n,CSG_PHICUT) ; 
    ./npy/NPhiCut.cpp:    nnode::Init(n,CSG_PHICUT) ; 
    ./sysrap/OpticksCSG.h:        CSG_PHICUT=121,
    ./sysrap/OpticksCSG.h:static const char* CSG_PHICUT_        = "phicut" ;
    ./sysrap/OpticksCSG.h:        else if(strcmp(nodename, CSG_PHICUT_) == 0)         tc = CSG_PHICUT ;
    ./sysrap/OpticksCSG.h:            case CSG_PHICUT:        s = CSG_PHICUT_        ; break ;
    ./sysrap/OpticksCSG.h:        return  type == CSG_PHICUT || type == CSG_THETACUT || type == CSG_INFCYLINDER  || type == CSG_PLANE || type == CSG_SLAB ;
    ./sysrap/OpticksCSG.py:    PHICUT = 121
    ./sysrap/sn.h:    else if( typecode == CSG_PHICUT )
    ./sysrap/tests/OpticksCSGTest.cc:                CSG_PHICUT, 
    ./sysrap/tests/OpticksCSG_test.cc:                CSG_PHICUT, 
    (ok) A[blyth@localhost opticks]$ 





example solid failing to appear in cxr_min.sh and outside world in ssst.sh::

    g4cx/tests/G4CX_U4TreeCreateCSGFoundryTest.sh
    NOXGEOM=1 cxr_min.sh
    NOXGEOM=1 ssst.sh


    NOXGEOM=1 EYE=0,0,1 UP=0,1,0 cxr_min.sh




::

    026-01-12 16:39:39.317 INFO  [730855] [U4VolumeMaker::Local@930]  name LocalLProfileSectorPolycone name_after_PREFIX LProfileSectorPolycone 
        bb [-15168.568, 15283.102,   -28.000,   146.760, 21561.000,    28.000] extent 7657.66


::

    (ok) A[blyth@localhost tests]$ ./CSGFoundryLoadTest.sh
                       BASH_SOURCE : ./CSGFoundryLoadTest.sh
                              name : CSGFoundryLoadTest
                               bin : CSGFoundryLoadTest
                            script : CSGFoundryLoadTest.py
                               PWD : /home/blyth/opticks/CSG/tests
                              GEOM : LocalLProfileSectorPolycone
                              TEST : descPrimRange
                              LVID : 
    [CSGFoundryLoadTest::descPrimRange 
    2026-01-12 16:28:13.306 INFO  [729498] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 17b94274d71d339e3600d1309c7dd930 dynamic 17b94274d71d339e3600d1309c7dd930
    [CSGFoundry.descBase 
     CFBase       /home/blyth/.opticks/GEOM/LocalLProfileSectorPolycone
     OriginCFBase /home/blyth/.opticks/GEOM/LocalLProfileSectorPolycone
    ]CSGFoundry.descBase 
    [CSGFoundry::descPrimRange num_solid  1
    [CSGFoundry::descPrimRange solidIdx  0 so.primOffset     0 so.numPrim     2
    [CSGPrim::Desc numPrim 2 mg_subs 0 EXTENT_DIFF :    200.000 [CSGPrim__DescRange_EXTENT_DIFF]
     [CSGPrim__DescRange_CE_ZMIN_ZMAX] CE_ZMIN :      0.000 CE_ZMAX :      0.000
       1 :  ; ce = np.array([     0.000,     0.000,     0.000, 21561.000])  ; bb = np.array([-21561.000,-21561.000,   -28.000, 21561.000, 21561.000,    28.000])  # CSGPrim::descRangeNumPy  lvid   0 ridx    0 pidx     1 so[example]
       0 :  ; ce = np.array([     0.000,     0.000,     0.000, 11486.496])  ; bb = np.array([-11486.496,-11486.496,-11486.496, 11486.496, 11486.496, 11486.496])  # CSGPrim::descRangeNumPy  lvid   1 ridx    0 pidx     0 so[G4_AIR_solid]
     numPrim 2 mg_subs 0 EXTENT_DIFF :    200.000 [CSGPrim__DescRange_EXTENT_DIFF]
     [CSGPrim__DescRange_CE_ZMIN_ZMAX] CE_ZMIN :      0.000 CE_ZMAX :      0.000
    ]CSGPrim::Desc
    ]CSGFoundry::descPrimRange solidIdx  0
    [CSGFoundry::descPrimRange num_solid  1
    [CSGFoundry.descBase 
     CFBase       /home/blyth/.opticks/GEOM/LocalLProfileSectorPolycone
     OriginCFBase /home/blyth/.opticks/GEOM/LocalLProfileSectorPolycone
    ]CSGFoundry.descBase 
    ]CSGFoundryLoadTest::descPrimRange
    (ok) A[blyth@localhost tests]$ 



::

    EMFCoil::BuildLProfileSectorPolycone i  0 z           -28 rMin      21505 rMax      21561
    EMFCoil::BuildLProfileSectorPolycone i  1 z           -24 rMin      21505 rMax      21561
    EMFCoil::BuildLProfileSectorPolycone i  2 z           -24 rMin      21557 rMax      21561
    EMFCoil::BuildLProfileSectorPolycone i  3 z            28 rMin      21557 rMax      21561


Issues:

1. bbox does not account for the phicut
2. 






CSGClassifyTest I
---------------------


Consider INTERSECTION setween an ordinary bounded A and an unbounded B (eg phicut or thetacut)
 
* assume that find a way to special case reclassify "B Miss" into "B Exit" 
  for rays going in the appropriate range of directions 

  * currently complement miss flips the signbits of isect.xyz 
    but only signbit of isect.x is read as the signal for complemented miss

  * rationalize the signalling:
 
    1. isect.x signbit for complement miss
    2. isect.y signbit for unbounded miss that can be promoted to unbounded exit

  * hmm this is only applicable when start "inside" B as can only EXIT_B when start inside
  
* as the "otherside" of B is at infinity the comparison will always be "A Closer"


::
 
              .
                           => MISS         /
                             2:B_MISS     /
                               2         /  B : unbounded
                              /         /
                             /         /
                            /         /
                           /         /
                          /         /
              +----------1-A_EXIT--/-------------+
              | A       /         / . . . . . . .|  
              |        /         / . . . . . . . |  
              |       0         / . . .  0--->---1----------- 2:B_MISS  
              |                / . . . . . . . . A_EXIT  
              |               + . 0 . . . . . . .|       (A_EXIT, B_MISS) => RETURN_MISS
              |                \ / . . . . . . . |                
              |                 1 . . . . . . . .|       (A_EXIT, B_EXIT) => A_Closer => RETURN_A  
              |                / \ . . . . . . . |  
              +---------------2---\--------------+
                                   \
                       1:B_EXIT     \
                       2:A_EXIT      \
                       1,2:B_Closer   \
                       => RETURN_B





