Union Splitting
====================

Mesh mending
-------------

* :google:`mesh remove inner faces`
* http://www.grasshopper3d.com/m/discussion?id=2985220%3ATopic%3A742890

  looking for any mesh face that is connected to more other faces than it has edges







Visualisations
---------------

::

    ggv --jdyb --zexplode --zexplodeconfig -5564.975,100. -O   

            # offset the split by 10cm to make obvious

    ggv --jdyb --zexplode --zexplodeconfig -5564.975,100. --meshversion _v0 -O 

            # meshversion _v0 is the openmeshtest- fixed mesh  

    ggv --jdyb --zexplode --zexplodeconfig -5564.975,100. --meshversion _v0  --geocenter

            # attempt to propagate in surgery applied geometry, 
            # failing to viz, torch targetting problem ?   


oav fix 
---------

iav mesh fix works after opening up the pairing criteria::

    ggv --ldyb -G --noinstanced
    ggv --ldyb 

Using openmesh option can run find a failure to pair::

    ggv --jdyb --openmesh   ## working iav fix that has been integrated
    ggv --ldyb --openmesh   ## not-working oav 

::

    simon:env blyth$ ggv --ldyb --openmesh 
    [2015-10-02 20:55:16.507820] [0x000007fff7448031] [info]    idpath /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.932b2e7ad32b2012f96141b06cbdd4ee.dae
    GMesh::Summary idx 0 vx 148 fc 288 n (null) sn (null) 
    center_extent -18079.596 -799699.438  -7062.645   2047.355 
    GMesh::Summary
     a   2047.355      0.000      0.000 -18079.596 
     b      0.000   2047.355      0.000 -799699.438 
     c      0.000      0.000   2047.355  -7062.645 
     d      0.000      0.000      0.000      1.000 
    ncomp: 2 
    [2015-10-02 20:55:16.512663] [0x000007fff7448031] [info]    ws nface 288 nvert 148 nedge 432 V - E + F = 4 (should be 2 for Euler Polyhedra) 
    [2015-10-02 20:55:16.512678] [0x000007fff7448031] [info]    wa nface 96 nvert 50 nedge 144 V - E + F = 2 (should be 2 for Euler Polyhedra) 
    [2015-10-02 20:55:16.512691] [0x000007fff7448031] [info]    wb nface 192 nvert 98 nedge 288 V - E + F = 2 (should be 2 for Euler Polyhedra) 
    [2015-10-02 20:55:16.512697] [0x000007fff7448031] [info]    ws.dumpBounds
             bb.max vec3  -16042.206 -797662.062  -5015.290  
             bb.min vec3  -20116.986 -801736.812  -9110.000  
    bb.max - bb.min vec3    4074.780   4074.750   4094.710  
    [2015-10-02 20:55:16.512755] [0x000007fff7448031] [info]    wb.dumpBounds
             bb.max vec3  -16042.206 -797662.062  -5015.290  
             bb.min vec3  -20116.986 -801736.812  -5172.910  
    bb.max - bb.min vec3    4074.780   4074.750    157.620  
    [2015-10-02 20:55:16.512790] [0x000007fff7448031] [info]    wa.dumpBounds
             bb.max vec3  -16082.016 -797702.000  -5173.000  
             bb.min vec3  -20076.891 -801696.875  -9110.000  
    bb.max - bb.min vec3    3994.875   3994.875   3937.000  
    [2015-10-02 20:55:16.512813] [0x000007fff7448031] [info]    write /tmp/comp0.off
    [2015-10-02 20:55:16.513178] [0x000007fff7448031] [info]    write /tmp/comp1.off
    [2015-10-02 20:55:16.516775] [0x000007fff7448031] [info]    MWrap<MeshT>::labelSpatialPairs fposprop centroid fpropname paired npair 0
    [2015-10-02 20:55:16.516987] [0x000007fff7448031] [info]    deleteFaces paired 0
    [2015-10-02 20:55:16.517278] [0x000007fff7448031] [info]    deleteFaces paired 0
    [2015-10-02 20:55:16.517291] [0x000007fff7448031] [warning] collectBoundaryLoop : No boundary found


Relaxing the spatial pairing requirements in x and y succeeds to find 24 pairings::

    [2015-10-02 20:59:54.039020] [0x000007fff7448031] [info]    write /tmp/comp1.off
    [2015-10-02 20:59:54.043260] [0x000007fff7448031] [info]    MWrap<MeshT>::labelSpatialPairs fposprop centroid fpropname paired npair 24
    [2015-10-02 20:59:54.043816] [0x000007fff7448031] [info]    deleteFaces paired 24
    [2015-10-02 20:59:54.044330] [0x000007fff7448031] [info]    deleteFaces paired 24
    [2015-10-02 20:59:54.044377] [0x000007fff7448031] [info]    findBoundaryVertexMapping
     (  3-> 73) ap           -16595.504 -801040.312 -5173.000 bpc           -16565.965 -801067.125 -5172.910 dpc               29.539 -26.812 0.090 dpcn     39.893
     (  5-> 74) ap           -16299.028 -800610.500 -5173.000 bpc           -16263.560 -800628.750 -5172.910 dpc               35.469 -18.250 0.090 dpcn     39.889
     (  7-> 75) ap           -16123.893 -800118.688 -5173.000 bpc           -16084.920 -800127.062 -5172.910 dpc               38.973 -8.375 0.090 dpcn     39.862
     (  9-> 76) ap           -16082.016 -799598.250 -5173.000 bpc           -16042.206 -799596.250 -5172.910 dpc               39.810 2.000 0.090 dpcn     39.860
     ( 11-> 77) ap           -16176.266 -799084.750 -5173.000 bpc           -16138.346 -799072.438 -5172.910 dpc               37.920 12.312 0.090 dpcn     39.869
     ( 13-> 78) ap           -16400.213 -798613.062 -5173.000 bpc           -16366.766 -798591.375 -5172.910 dpc               33.447 21.688 0.090 dpcn     39.863
     ( 15-> 79) ap           -16738.602 -798215.500 -5173.000 bpc           -16711.928 -798185.812 -5172.910 dpc               26.674 29.688 0.090 dpcn     39.911
     ( 17-> 80) ap           -17168.363 -797919.000 -5173.000 bpc           -17150.281 -797883.438 -5172.910 dpc               18.082 35.562 0.090 dpcn     39.896
     ( 19-> 81) ap           -17660.217 -797743.875 -5173.000 bpc           -17651.977 -797704.750 -5172.910 dpc                8.240 39.125 0.090 dpcn     39.983
     ( 21-> 82) ap           -18180.639 -797702.000 -5173.000 bpc           -18182.799 -797662.062 -5172.910 dpc               -2.160 39.938 0.090 dpcn     39.996
     ( 23-> 83) ap           -18694.164 -797796.250 -5173.000 bpc           -18706.602 -797758.188 -5172.910 dpc              -12.438 38.062 0.090 dpcn     40.043
     ( 25-> 84) ap           -19165.801 -798020.188 -5173.000 bpc           -19187.670 -797986.625 -5172.910 dpc              -21.869 33.562 0.090 dpcn     40.059
     ( 27-> 85) ap           -19563.402 -798358.562 -5173.000 bpc           -19593.227 -798331.750 -5172.910 dpc              -29.824 26.812 0.090 dpcn     40.105
     ( 29-> 86) ap           -19859.877 -798788.375 -5173.000 bpc           -19895.625 -798770.125 -5172.910 dpc              -35.748 18.250 0.090 dpcn     40.137
     ( 31-> 87) ap           -20035.014 -799280.188 -5173.000 bpc           -20074.271 -799271.812 -5172.910 dpc              -39.258 8.375 0.090 dpcn     40.141
     ( 33-> 88) ap           -20076.891 -799800.625 -5173.000 bpc           -20116.986 -799802.688 -5172.910 dpc              -40.096 -2.062 0.090 dpcn     40.149
     ( 35-> 89) ap           -19982.641 -800314.125 -5173.000 bpc           -20020.844 -800326.438 -5172.910 dpc              -38.203 -12.312 0.090 dpcn     40.138
     ( 37-> 90) ap           -19758.693 -800785.812 -5173.000 bpc           -19792.416 -800807.500 -5172.910 dpc              -33.723 -21.688 0.090 dpcn     40.095
     ( 39-> 91) ap           -19420.305 -801183.375 -5173.000 bpc           -19447.262 -801213.062 -5172.910 dpc              -26.957 -29.688 0.090 dpcn     40.100
     ( 41-> 92) ap           -18990.543 -801479.875 -5173.000 bpc           -19008.910 -801515.500 -5172.910 dpc              -18.367 -35.625 0.090 dpcn     40.081
     ( 43-> 93) ap           -18498.689 -801655.000 -5173.000 bpc           -18507.215 -801694.125 -5172.910 dpc               -8.525 -39.125 0.090 dpcn     40.043
     ( 45-> 94) ap           -17978.268 -801696.875 -5173.000 bpc           -17976.385 -801736.812 -5172.910 dpc                1.883 -39.938 0.090 dpcn     39.982
     ( 47-> 95) ap           -17464.742 -801602.625 -5173.000 bpc           -17452.588 -801640.688 -5172.910 dpc               12.154 -38.062 0.090 dpcn     39.956
     (  0-> 72) ap           -16993.105 -801378.688 -5173.000 bpc           -16971.520 -801412.250 -5172.910 dpc               21.586 -33.562 0.090 dpcn     39.905
    [2015-10-02 20:59:54.045240] [0x000007fff7448031] [info]    createWithWeldedBoundary 24
    (3->73)(3->122)
    (5->74)(5->123)
    (7->75)(7->124)
    (9->76)(9->125)






Check Topology of Meshes
------------------------

The below approx 10 percent of the ~250 Dyb meshes have issues, either:

* are topologically multiple meshes, see t value below
* cause OpenMesh error output, eg "complex edge/vertex" (possibly a winding order problem) 

Mostly are small bits of geometry with small extent x (mm), not in optically active parts of 
geometry. Issues with large important meshes used by optically active nodes are highlighted, 
these must be fixed.::

    [2015-Oct-02 13:06:32.202216]:info: App::checkGeometry  nso 12230 nme 249
         9 (v   24 f   36 )(t    3 oe    0) : x  5871.000 : n    18 : n*v    432 :                 near_span_hbeam0xc2a27d8 : 2359,2360,2432,2433,2434, 
        10 (v   24 f   36 )(t    3 oe    0) : x  1000.060 : n     2 : n*v     48 :           near_side_short_hbeam0xc2b1ea8 : 2361,2362, 
        11 (v   16 f   24 )(t    2 oe    0) : x   596.000 : n   162 : n*v   2592 :     near_thwart_long_angle_iron0xc21e058 : 2363,2364,2365,2366,2367, 
        21 (v   16 f   24 )(t    2 oe    0) : x 22000.250 : n     1 : n*v     16 :             near_hall_top_dwarf0xc0316c8 : 2, 

     ** 24 (v  148 f  288 )(t    2 oe    0) : x  1587.245 : n     2 : n*v    296 :                             iav0xc346f90 : 3158,4818, 

        25 (v  168 f  384 )(t    1 oe 1632) : x   150.000 : n     2 : n*v    336 :                       IavTopHub0xc405968 : 3161,4821, 
        26 (v  168 f  384 )(t    1 oe 1632) : x   150.000 : n     4 : n*v    672 :                 CtrGdsOflBotClp0xbf5dec0 : 3162,3166,4822,4826, 
        29 (v  264 f  576 )(t    2 oe 1632) : x   150.000 : n     2 : n*v    528 :                       OcrGdsPrt0xc352518 : 3165,4825, 

     ** 42 (v  148 f  288 )(t    2 oe    0) : x  2047.355 : n     2 : n*v    296 :                             oav0xc2ed7c8 : 3156,4816, 

        54 (v  100 f  192 )(t    2 oe    0) : x    82.501 : n    12 : n*v   1200 :                 headon-pmt-assy0xbf55198 : 4351,4358,4365,4372,4379, 
        63 (v   33 f   62 )(t    2 oe  136) : x  1125.000 : n    16 : n*v    528 :                       SstBotRib0xc26c4c0 : 4431,4432,4433,4434,4435, 
        75 (v  240 f  576 )(t    1 oe 3264) : x   125.000 : n     2 : n*v    480 :                       OavTopHub0xc2c9030 : 4505,6165, 
        77 (v  240 f  576 )(t    1 oe 3264) : x   112.500 : n     6 : n*v   1440 :                 CtrLsoOflTopClp0xc178498 : 4507,4513,4519,6167,6173, 
        81 (v  168 f  384 )(t    1 oe 1632) : x    98.000 : n     2 : n*v    336 :                    OcrGdsLsoPrt0xc104978 : 4511,6171, 
        82 (v   98 f  188 )(t    2 oe    0) : x   247.488 : n     2 : n*v    196 :                  OcrGdsInLsoOfl0xc26f450 : 4516,6176, 
        84 (v   98 f  188 )(t    2 oe    0) : x   247.488 : n     2 : n*v    196 :                  OcrGdsLsoInOil0xc540738 : 4514,6174, 
        85 (v  168 f  384 )(t    1 oe 1632) : x   105.357 : n     2 : n*v    336 :                    OcrCalLsoPrt0xc1076b0 : 4517,6177, 
        86 (v   98 f  188 )(t    2 oe    0) : x   247.488 : n     2 : n*v    196 :                       OcrCalLso0xc103c18 : 4520,6180, 
       105 (v  629 f 1242 )(t    4 oe    0) : x   102.303 : n     6 : n*v   3774 :                 led-source-assy0xc3061d0 : 4540,4628,4710,6200,6288, 
       112 (v  357 f  698 )(t    4 oe    0) : x   102.303 : n     6 : n*v   2142 :                     source-assy0xc2d5d78 : 4551,4639,4721,6211,6299, 
       132 (v  296 f  576 )(t    4 oe    0) : x   102.303 : n     6 : n*v   1776 :             amcco60-source-assy0xc0b1df8 : 4566,4654,4736,6226,6314, 
       140 (v  192 f  384 )(t    2 oe    0) : x   920.021 : n     2 : n*v    384 :                       LsoOflTnk0xc17d928 : 4606,6266, 
       141 (v  288 f  576 )(t    3 oe    0) : x   910.028 : n     2 : n*v    576 :                          LsoOfl0xc348ac0 : 4607,6267, 
       142 (v  776 f 1552 )(t    5 oe    0) : x   660.041 : n     2 : n*v   1552 :                       GdsOflTnk0xc3d5160 : 4608,6268, 
       143 (v  100 f  192 )(t    2 oe    0) : x   650.000 : n     2 : n*v    200 :                          GdsOfl0xbf73918 : 4609,6269, 
       144 (v  172 f  336 )(t    2 oe    0) : x   924.000 : n     2 : n*v    344 :                  OflTnkCnrSpace0xc3d3d30 : 4605,6265, 
       145 (v  366 f  720 )(t    3 oe    0) : x  1015.000 : n     2 : n*v    732 :                 OflTnkContainer0xc17cf50 : 4604,6264, 



Where to do mesh fixing ?
---------------------------

* easiest to do just after creation in AssimpGGeo to avoid
  having to chase down and swap pointers with replacement GMesh 



Mesh Surgery implemented in openmeshtest-
--------------------------------------------

* converted NPY mesh into OpenMesh by 1st removing duplicate vertices

* divide the split union mesh into two Euler polyhedrons corresponding 
  to the connected mesh components of the original

* identify back to back faces between the two components and delete them  

* combine the two now open component meshes by finding vertices around the open 
  boundary and aligning those with the other, then adding new faces to 
  weld together the pieces

* save the mesh into NPY format in the "--jdyb" cache under postfix "_v0" 



G4DAE/G4 triangulation code quickly goes down rabbit hole
-----------------------------------------------------------

g4dae/src/G4DAEWriteSolids.cc::

    164 G4String G4DAEWriteSolids::
    165 GeometryWrite(xercesc::DOMElement* solidsElement, const G4VSolid* const solid, const G4String& matSymbol )
    166 {
    167    const G4String& geoId = GenerateName(solid->GetName(),solid);
    168 
    169    xercesc::DOMElement* geometryElement = NewElementTwoAtt("geometry", "name", geoId, "id", geoId);
    170    xercesc::DOMElement* meshElement = NewElement("mesh");
    171 
    172    G4bool recPoly = GetRecreatePoly();
    173    G4DAEPolyhedron poly(solid, matSymbol, recPoly );  // recPoly=true  always creates a new poly, even when one exists already   
    174 
    175    G4int nvert = poly.GetNoVertices() ;
    176    G4int nface = poly.GetNoFacets() ;
    177    G4int ntexl = poly.GetNoTexels() ;

g4dae/src/G4DAEPolyhedron.cc::

     08 G4DAEPolyhedron::G4DAEPolyhedron( const G4VSolid* const solid, const G4String& matSymbol, G4bool create )
      9 {
     10     fStart = "\n" ;
     11     fBefItem  = "\t\t\t\t" ;
     12     fAftItem  = "\n" ;
     13     fEnd   = "" ;
     14 
     15 
     16     G4Polyhedron* pPolyhedron ;
     17 
     18     //  visualization/management/src/G4VSceneHandler.cc
     19 
     20     G4int noofsides = 24 ;
     21     G4Polyhedron::SetNumberOfRotationSteps (noofsides);
     22     std::stringstream coutbuf;
     23     std::stringstream cerrbuf;
     24     {
     25        cout_redirect out(coutbuf.rdbuf());
     26        cerr_redirect err(cerrbuf.rdbuf());
     27        if( create ){
     28            AddMeta( "create", "1" );
     29            pPolyhedron = solid->CreatePolyhedron ();  // always create a new poly   
     30        } else {
     31            AddMeta( "create", "0" );
     32            pPolyhedron = solid->GetPolyhedron ();     // if poly created already and no parameter change just provide that one 
     33        }
     34     }


CreatePolyhedron::

    simon:geant4.10.00.p01 blyth$ find . -name '*.hh' -exec grep -H CreatePolyhedron {} \;
    ./source/geometry/management/include/G4ReflectedSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./source/geometry/management/include/G4VSolid.hh:    virtual G4Polyhedron* CreatePolyhedron () const;
    ./source/geometry/solids/Boolean/include/G4DisplacedSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./source/geometry/solids/Boolean/include/G4IntersectionSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./source/geometry/solids/Boolean/include/G4SubtractionSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./source/geometry/solids/Boolean/include/G4UnionSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./source/geometry/solids/CSG/include/G4Box.hh://                     and SendPolyhedronTo() to CreatePolyhedron()
    ./source/geometry/solids/CSG/include/G4Box.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Cons.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/CSG/include/G4CutTubs.hh:    G4Polyhedron*       CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Orb.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/CSG/include/G4OTubs.hh:    G4Polyhedron*       CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Para.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Sphere.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/CSG/include/G4Torus.hh:    G4Polyhedron*       CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Trap.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Trd.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./source/geometry/solids/CSG/include/G4Tubs.hh:// 22.07.96 J.Allison: Changed SendPolyhedronTo to CreatePolyhedron
    ./source/geometry/solids/CSG/include/G4Tubs.hh:    G4Polyhedron*       CreatePolyhedron   () const;
    ./source/geometry/solids/specific/include/G4Ellipsoid.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4EllipticalCone.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4EllipticalTube.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4GenericPolycone.hh:  G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4GenericTrap.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4Hype.hh:  G4Polyhedron* CreatePolyhedron   () const;
    ./source/geometry/solids/specific/include/G4Paraboloid.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4Polycone.hh:  G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4Polyhedra.hh:  G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4TessellatedSolid.hh:    virtual G4Polyhedron* CreatePolyhedron () const;
    ./source/geometry/solids/specific/include/G4Tet.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./source/geometry/solids/specific/include/G4TwistedTubs.hh:  G4Polyhedron   *CreatePolyhedron   () const;
    ./source/geometry/solids/specific/include/G4UGenericPolycone.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4UPolycone.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4UPolyhedra.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./source/geometry/solids/specific/include/G4VCSGfaceted.hh:    virtual G4Polyhedron* CreatePolyhedron() const = 0;
    ./source/geometry/solids/specific/include/G4VTwistedFaceted.hh:  virtual G4Polyhedron   *CreatePolyhedron   () const ;
    ./source/geometry/solids/usolids/include/G4USolid.hh:    G4Polyhedron* CreatePolyhedron() const;


source/geometry/solids/Boolean/src/G4UnionSolid.cc::

    487 G4Polyhedron*
    488 G4UnionSolid::CreatePolyhedron () const
    489 {
    490   HepPolyhedronProcessor processor;
    491   // Stack components and components of components recursively
    492   // See G4BooleanSolid::StackPolyhedron
    493   G4Polyhedron* top = StackPolyhedron(processor, this);
    494   G4Polyhedron* result = new G4Polyhedron(*top);
    495   if (processor.execute(*result)) { return result; }
    496   else { return 0; }
    497 }


source/graphics_reps/src/HepPolyhedronProcessor.src::

    139 bool HepPolyhedronProcessor::execute(HepPolyhedron& a_poly) {
    140   //{for(unsigned int index=0;index<5;index++) {
    141   //  printf("debug : bijection : %d\n",index);
    142   //  HEPVis::bijection_dump bd(index);
    143   //  bd.visitx();
    144   //}}
    145 
    146   HepPolyhedron_exec e(m_ops.size(),*this,a_poly);
    147   if(!e.visitx()) return true;
    148   //std::cerr << "HepPolyhedronProcessor::execute :"
    149   //          << " all shifts and combinatory tried."
    150   //          << " Boolean operations failed."
    151   //          << std::endl;
    152   return false;
    153 }
    ...
    121 class HepPolyhedron_exec : public HEPVis::bijection_visitor {
    122 public:
    123   HepPolyhedron_exec(unsigned int a_number,
    124        HepPolyhedronProcessor& a_proc,
    125        HepPolyhedron& a_poly)
    126   : HEPVis::bijection_visitor(a_number)
    127   ,m_proc(a_proc)
    128   ,m_poly(a_poly)
    129   {}
    130   virtual bool visit(const is_t& a_is) {
    131     if(m_proc.execute1(m_poly,a_is)==true) return false; //stop
    132     return true;//continue
    133   }
    134 private:
    135   HepPolyhedronProcessor& m_proc;
    136   HepPolyhedron& m_poly;
    137 };
    ...
    155 bool HepPolyhedronProcessor::execute1(
    156  HepPolyhedron& a_poly
    157 ,const std::vector<unsigned int>& a_is
    158 ) {
    159   HepPolyhedron result(a_poly);
    160   unsigned int number = m_ops.size();
    161   int num_shift = BooleanProcessor::get_num_shift();
    162   for(int ishift=0;ishift<num_shift;ishift++) {
    163     BooleanProcessor::set_shift(ishift);
    164 
    165     result = a_poly;
    166     bool done = true;
    167     for(unsigned int index=0;index<number;index++) {
    168       BooleanProcessor processor; //take a fresh one.
    169       const op_t& elem = m_ops[a_is[index]];
    170       int err;
    171       result = processor.execute(elem.first,result,elem.second,err);
    172       if(err) {
    173         done = false;
    174         break;
    175       }
    176     }
    177     if(done) {
    178       a_poly = result;
    179       return true;
    180     }
    181   }
    182 
    183   //std::cerr << "HepPolyhedronProcessor::execute :"
    184   //          << " all shifts tried. Boolean operations failed."
    185   //          << std::endl;
    186 
    187   //a_poly = result;
    188   return false;
    189 }
      


::

    simon:geant4.10.00.p01 blyth$ find . -name '*.cc' -exec grep -H BooleanProcessor {} \;
    ./source/graphics_reps/src/HepPolyhedron.cc:#include "BooleanProcessor.src"
    ./source/graphics_reps/src/HepPolyhedron.cc:  BooleanProcessor processor;
    ./source/graphics_reps/src/HepPolyhedron.cc:  BooleanProcessor processor;
    ./source/graphics_reps/src/HepPolyhedron.cc:  BooleanProcessor processor;
    ./source/graphics_reps/src/HepPolyhedron.cc://       since there is no BooleanProcessor.h
    ./source/visualization/OpenGL/src/G4OpenGLImmediateWtViewer.cc:  // BooleanProcessor is up to it, abandon this and use generic
    ./source/visualization/OpenGL/src/G4OpenGLSceneHandler.cc:  // when the BooleanProcessor is up to it, abandon this and use
    ./source/visualization/OpenGL/src/G4OpenGLSceneHandler.cc:  // But...if not, when the BooleanProcessor is up to it...
    ./source/visualization/OpenGL/src/G4OpenGLViewer.cc:  // BooleanProcessor is up to it, abandon this and use generic
    simon:geant4.10.00.p01 blyth$ 


source/graphics_reps/src/BooleanProcessor.src::

    ... scary code ...
 


::

   source/graphics_reps/include/G4Polyhedron.hh
   source/graphics_reps/src/G4Polyhedron.cc
   source/graphics_reps/include/HepPolyhedron.h
   source/graphics_reps/src/HepPolyhedron.cc
   source/graphics_reps/include/HepPolyhedronProcessor.h
   source/graphics_reps/src/HepPolyhedronProcessor.src



Idea mesh scanning to identify internal faces
-----------------------------------------------

* handle meshes one by one (only ~250 distinct meshes so performance is not an issue)
  construct single mesh OptiX geometries

* use uniform spherical OptiX rays shot from inside the mesh and 
  collect indices of faces giving frontside intersections, should
  always get backside intersection so long as emission point is really inside
  
  * define origin as the barycenter of all vertices, or center of bounding box
  * avoid pathological faces by emitting not just from one point but 
    from axis aligned line segments 

* for development (visualization etc..) would be good to do this within ggv 
  but for production use probably better to be a pre-step ?


ExplodeZVertices makes it obvious that have two closed meshes, not interior faces of one 
-------------------------------------------------------------------------------------------

App::loadGeometry::

   // for --jdyb --idyb --kdyb testing : making the cleave obvious
    m_mesh0->explodeZVertices(1000.f, -(5564.950f + 5565.000f)/2.f ); 

    simon:issues blyth$ ggv --jdyb -O



Single face eyeballing
------------------------

Allows to jump into difficult to navigate to positions targetting a single face, works post-cache::

    udp.py --pickface 100,3158,0


Using wireframe view (B) with normal and face plane indicators (Q) its
plain that there are back to back inner faces with normals pointing up and down.

Comparing the afflicted jdyb with OK kdyb::

  ggv --jdyb -O 
  ggv --kdyb -O

  ggv --jdyb --torchconfig "radius=1500;zenith_azimuth=1,0,1,0"


Numerical view
----------------

Last triplet normal, together with z makes is possible to see whats what numerically, 
faces 264-287

::

    udp.py --pickface 264,288,3158,0    # plucks all downward normal interior faces
    udp.py --pickface 48,72,3158,0      # plucks all upward normal interior faces


    In [1]: 72 - 48 
    Out[1]: 24

    In [2]: 288 - 264
    Out[2]: 24



::

    simon:nrmvec blyth$ ggv --jdyb --loader

     i  48 f   96   97   98 : -18079.453 -799699.438  -5565.000    -17232.102 -801009.250  -5565.000     -16921.973 -800745.312  -5565.000   :       0.000      0.000      1.000 
     i  49 f   96   98   99 : -18079.453 -799699.438  -5565.000    -16921.973 -800745.312  -5565.000     -16690.721 -800410.062  -5565.000   :       0.000      0.000      1.000 
     i  50 f   96   99  100 : -18079.453 -799699.438  -5565.000    -16690.721 -800410.062  -5565.000     -16554.107 -800026.438  -5565.000   :       0.000      0.000      1.000 
     i  51 f   96  100  101 : -18079.453 -799699.438  -5565.000    -16554.107 -800026.438  -5565.000     -16521.451 -799620.500  -5565.000   :       0.000      0.000      1.000 
     i  52 f  102  101  103 : -18079.453 -799699.438  -5565.000    -16521.451 -799620.500  -5565.000     -16594.969 -799219.938  -5565.000   :      -0.000      0.000      1.000 
     i  53 f  102  103  104 : -18079.453 -799699.438  -5565.000    -16594.969 -799219.938  -5565.000     -16769.646 -798852.062  -5565.000   :      -0.000      0.000      1.000 
     i  54 f  102  104  105 : -18079.453 -799699.438  -5565.000    -16769.646 -798852.062  -5565.000     -17033.592 -798541.938  -5565.000   :      -0.000      0.000      1.000 
     i  55 f  106  105  107 : -18079.453 -799699.438  -5565.000    -17033.592 -798541.938  -5565.000     -17368.803 -798310.688  -5565.000   :      -0.000      0.000      1.000 
     i  56 f  106  107  108 : -18079.453 -799699.438  -5565.000    -17368.803 -798310.688  -5565.000     -17752.447 -798174.062  -5565.000   :      -0.000      0.000      1.000 
     i  57 f  106  108  109 : -18079.453 -799699.438  -5565.000    -17752.447 -798174.062  -5565.000     -18158.377 -798141.438  -5565.000   :      -0.000      0.000      1.000 
     i  58 f  106  109  110 : -18079.453 -799699.438  -5565.000    -18158.377 -798141.438  -5565.000     -18558.926 -798214.938  -5565.000   :      -0.000     -0.000      1.000 
     i  59 f  111  110  112 : -18079.453 -799699.438  -5565.000    -18558.926 -798214.938  -5565.000     -18926.805 -798389.625  -5565.000   :      -0.000     -0.000      1.000 
     i  60 f  111  112  113 : -18079.453 -799699.438  -5565.000    -18926.805 -798389.625  -5565.000     -19236.934 -798653.562  -5565.000   :      -0.000     -0.000      1.000 
     i  61 f  111  113  114 : -18079.453 -799699.438  -5565.000    -19236.934 -798653.562  -5565.000     -19468.186 -798988.812  -5565.000   :      -0.000     -0.000      1.000 
     i  62 f  115  114  116 : -18079.453 -799699.438  -5565.000    -19468.186 -798988.812  -5565.000     -19604.799 -799372.438  -5565.000   :      -0.000     -0.000      1.000 
     i  63 f  115  116  117 : -18079.453 -799699.438  -5565.000    -19604.799 -799372.438  -5565.000     -19637.455 -799778.375  -5565.000   :       0.000     -0.000      1.000 
     i  64 f  115  117  118 : -18079.453 -799699.438  -5565.000    -19637.455 -799778.375  -5565.000     -19563.938 -800178.938  -5565.000   :       0.000     -0.000      1.000 
     i  65 f  115  118  119 : -18079.453 -799699.438  -5565.000    -19563.938 -800178.938  -5565.000     -19389.260 -800546.812  -5565.000   :       0.000     -0.000      1.000 
     i  66 f  120  119  121 : -18079.453 -799699.438  -5565.000    -19389.260 -800546.812  -5565.000     -19125.314 -800856.938  -5565.000   :       0.000     -0.000      1.000 
     i  67 f  120  121  122 : -18079.453 -799699.438  -5565.000    -19125.314 -800856.938  -5565.000     -18790.104 -801088.188  -5565.000   :       0.000     -0.000      1.000 
     i  68 f  120  122  123 : -18079.453 -799699.438  -5565.000    -18790.104 -801088.188  -5565.000     -18406.459 -801224.812  -5565.000   :       0.000     -0.000      1.000 
     i  69 f  120  123  124 : -18079.453 -799699.438  -5565.000    -18406.459 -801224.812  -5565.000     -18000.529 -801257.438  -5565.000   :       0.000      0.000      1.000 
     i  70 f   96  124  125 : -18079.453 -799699.438  -5565.000    -18000.529 -801257.438  -5565.000     -17599.980 -801183.938  -5565.000   :       0.000      0.000      1.000 
     i  71 f   96  125   97 : -18079.453 -799699.438  -5565.000    -17599.980 -801183.938  -5565.000     -17232.102 -801009.250  -5565.000   :       0.000      0.000      1.000 

     ...

     i 264 f  452  453  454 : -17229.393 -801013.562  -5564.950    -18079.461 -799699.562  -5564.950     -16918.270 -800748.750  -5564.950   :      -0.000      0.000     -1.000 
     i 265 f  454  453  455 : -16918.270 -800748.750  -5564.950    -18079.461 -799699.562  -5564.950     -16686.277 -800412.500  -5564.950   :      -0.000      0.000     -1.000 
     i 266 f  455  453  456 : -16686.277 -800412.500  -5564.950    -18079.461 -799699.562  -5564.950     -16549.230 -800027.625  -5564.950   :      -0.000      0.000     -1.000 
     i 267 f  456  453  457 : -16549.230 -800027.625  -5564.950    -18079.461 -799699.562  -5564.950     -16516.463 -799620.375  -5564.950   :      -0.000      0.000     -1.000 
     i 268 f  457  458  459 : -16516.463 -799620.375  -5564.950    -18079.461 -799699.562  -5564.950     -16590.217 -799218.562  -5564.950   :       0.000     -0.000     -1.000 
     i 269 f  459  458  460 : -16590.217 -799218.562  -5564.950    -18079.461 -799699.562  -5564.950     -16765.453 -798849.500  -5564.950   :       0.000     -0.000     -1.000 
     i 270 f  460  458  461 : -16765.453 -798849.500  -5564.950    -18079.461 -799699.562  -5564.950     -17030.244 -798538.375  -5564.950   :       0.000     -0.000     -1.000 
     i 271 f  461  458  462 : -17030.244 -798538.375  -5564.950    -18079.461 -799699.562  -5564.950     -17366.531 -798306.375  -5564.950   :       0.000     -0.000     -1.000 
     i 272 f  462  463  464 : -17366.531 -798306.375  -5564.950    -18079.461 -799699.562  -5564.950     -17751.410 -798169.312  -5564.950   :       0.000     -0.000     -1.000 
     i 273 f  464  463  465 : -17751.410 -798169.312  -5564.950    -18079.461 -799699.562  -5564.950     -18158.637 -798136.562  -5564.950   :       0.000     -0.000     -1.000 
     i 274 f  465  463  466 : -18158.637 -798136.562  -5564.950    -18079.461 -799699.562  -5564.950     -18560.475 -798210.312  -5564.950   :       0.000      0.000     -1.000 
     i 275 f  466  467  468 : -18560.475 -798210.312  -5564.950    -18079.461 -799699.562  -5564.950     -18929.527 -798385.562  -5564.950   :       0.000      0.000     -1.000 
     i 276 f  468  467  469 : -18929.527 -798385.562  -5564.950    -18079.461 -799699.562  -5564.950     -19240.654 -798650.312  -5564.950   :       0.000      0.000     -1.000 
     i 277 f  469  467  470 : -19240.654 -798650.312  -5564.950    -18079.461 -799699.562  -5564.950     -19472.643 -798986.625  -5564.950   :       0.000      0.000     -1.000 
     i 278 f  470  471  472 : -19472.643 -798986.625  -5564.950    -18079.461 -799699.562  -5564.950     -19609.691 -799371.500  -5564.950   :       0.000      0.000     -1.000 
     i 279 f  472  471  473 : -19609.691 -799371.500  -5564.950    -18079.461 -799699.562  -5564.950     -19642.455 -799778.750  -5564.950   :       0.000      0.000     -1.000 
     i 280 f  473  471  474 : -19642.455 -799778.750  -5564.950    -18079.461 -799699.562  -5564.950     -19568.709 -800180.562  -5564.950   :       0.000      0.000     -1.000 
     i 281 f  474  475  476 : -19568.709 -800180.562  -5564.950    -18079.461 -799699.562  -5564.950     -19393.465 -800549.625  -5564.950   :       0.000      0.000     -1.000 
     i 282 f  476  475  477 : -19393.465 -800549.625  -5564.950    -18079.461 -799699.562  -5564.950     -19128.682 -800860.750  -5564.950   :       0.000      0.000     -1.000 
     i 283 f  477  475  478 : -19128.682 -800860.750  -5564.950    -18079.461 -799699.562  -5564.950     -18792.389 -801092.750  -5564.950   :       0.000      0.000     -1.000 
     i 284 f  478  475  479 : -18792.389 -801092.750  -5564.950    -18079.461 -799699.562  -5564.950     -18407.510 -801229.812  -5564.950   :       0.000      0.000     -1.000 
     i 285 f  479  480  481 : -18407.510 -801229.812  -5564.950    -18079.461 -799699.562  -5564.950     -18000.281 -801262.562  -5564.950   :       0.000      0.000     -1.000 
     i 286 f  481  453  482 : -18000.281 -801262.562  -5564.950    -18079.461 -799699.562  -5564.950     -17598.449 -801188.812  -5564.950   :      -0.000      0.000     -1.000 
     i 287 f  482  453  452 : -17598.449 -801188.812  -5564.950    -18079.461 -799699.562  -5564.950     -17229.393 -801013.562  -5564.950   :      -0.000      0.000     -1.000 
    [2015-09-25 20:13:43.616253] [0x000007fff7448031] [info]    GGeo::dumpVolume nsolid 12230 nvert483 nface 288
    [



many upwards going photons think their m1 is Ac when actually Gd
---------------------------------------------------------------------------

* investigating using a torch emitter from middle of IAV

::

   3150 : nf    0 nv    0 id   3150 pid   3149 : __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10       __dd__Geometry__Pool__lvNearPoolOWS0xbf93840 
   3151 : nf    0 nv    0 id   3151 pid   3150 : __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20   __dd__Geometry__Pool__lvNearPoolCurtain0xc2ceef0 
   3152 : nf    0 nv    0 id   3152 pid   3151 : __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498       __dd__Geometry__Pool__lvNearPoolIWS0xc28bc60 
   3153 : nf   96 nv  157 id   3153 pid   3152 : __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528                 __dd__Geometry__AD__lvADE0xc2a78c0 
   3154 : nf   96 nv  157 id   3154 pid   3153 : __dd__Geometry__AD__lvADE--pvSST0xc128d90                 __dd__Geometry__AD__lvSST0xc234cd0 
   3155 : nf   96 nv  157 id   3155 pid   3154 : __dd__Geometry__AD__lvSST--pvOIL0xc241510                 __dd__Geometry__AD__lvOIL0xbf5e0b8 
   3156 : nf  288 nv  481 id   3156 pid   3155 : __dd__Geometry__AD__lvOIL--pvOAV0xbf8f638                 __dd__Geometry__AD__lvOAV0xbf1c760 
   3157 : nf  332 nv  678 id   3157 pid   3156 : __dd__Geometry__AD__lvOAV--pvLSO0xbf8e120                 __dd__Geometry__AD__lvLSO0xc403e40 

   3158 : nf  288 nv  483 id   3158 pid   3157 :    __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348                 __dd__Geometry__AD__lvIAV0xc404ee8 
   3159 : nf  288 nv  617 id   3159 pid   3158 :       __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00                 __dd__Geometry__AD__lvGDS0xbf6cbb8 
   3160 : nf   92 nv  211 id   3160 pid   3158 :       __dd__Geometry__AD__lvIAV--pvOcrGdsInIAV0xbf6b0e0         __dd__Geometry__AdDetails__lvOcrGdsInIav0xbf6dd58 

   3161 : nf  384 nv  632 id   3161 pid   3157 :    __dd__Geometry__AD__lvLSO--pvIavTopHub0xc34e6e8    __dd__Geometry__AdDetails__lvIavTopHub0xc129d88 
   3162 : nf  384 nv  636 id   3162 pid   3157 :    __dd__Geometry__AD__lvLSO--pvCtrGdsOflBotClp0xc2ce2a8 __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0 
   3163 : nf  192 nv  336 id   3163 pid   3157 : __dd__Geometry__AD__lvLSO--pvCtrGdsOflTfbInLso0xc2ca538 __dd__Geometry__AdDetails__lvCtrGdsOflTfbInLso0xbfa0728 
   3164 : nf   96 nv  157 id   3164 pid   3157 : __dd__Geometry__AD__lvLSO--pvCtrGdsOflInLso0xbf74250 __dd__Geometry__AdDetails__lvCtrGdsOflInLso0xc28cc88 
   3165 : nf  576 nv 1189 id   3165 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsPrt0xbf6d0d0    __dd__Geometry__AdDetails__lvOcrGdsPrt0xc352630 
   3166 : nf  384 nv  636 id   3166 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsBotClp0xbfa1610 __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0 
   3167 : nf  192 nv  488 id   3167 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsTfbInLso0xbfa1818 __dd__Geometry__AdDetails__lvOcrGdsTfbInLso0xc3529c0 
   3168 : nf   92 nv  210 id   3168 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsInLso0xbf6d280  __dd__Geometry__AdDetails__lvOcrGdsInLso0xc353990 
   3169 : nf   12 nv   24 id   3169 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs--OavBotRibRot0xbf5af90    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 
   3170 : nf   12 nv   24 id   3170 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs..1--OavBotRibRot0xc3531c0    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 
   3171 : nf   12 nv   24 id   3171 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs..2--OavBotRibRot0xc353e30    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 
   3172 : nf   12 nv   24 id   3172 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs..3--OavBotRibRot0xc541230    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 


Problem remains with only 2 volumes, 3158 and 3159::

    see ~/env/bin/ggv.sh
    export GGEOVIEW_QUERY="range:3158:3160" 
       # just 2 volumes (python style range) __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348, __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00  

    ggv --idyb --torchconfig="radius=0;zenith_azimuth=0,1,0,1"


Isolate issue to single volume : 3158
--------------------------------------

Single volume 3158 messing up all by itself ::

    ggv --jdyb --torchconfig "radius=0;zenith_azimuth=0,1,0,1"   
         

OpenGL Eyeballing
~~~~~~~~~~~~~~~~~~~ 
  
* flickery underside of top lid
* __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348  => /dd/Geometry/AD/lvLSO#pvIAV

* union of tubs and polycone seems to fail in this case, with the "internal" 
  tubs/polycone transition acting as an effective boundary to OptiX rayTrace 
  intersection tests (there is no corresponding GBoundary : so m1/m2/su will be wonky)

  side view in orthographic mode makes this very apparent, with a clear disc
  of photon intersections at the top of the cylinder with another disc on the polycone
  surface   

* looking up from inside (with flipped normals) can see a featureless but flickery surface
  in wireframe its apparent that the "spokes" are doubled up 


NumPy Look at faces/vertices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jump into geocache for 1 volume geometry::

    delta:ggeoview blyth$ cd $(ggv --jdyb --idp)
    delta:ggeoview blyth$ cd $(ggv --kdyb --idp)

Check mergedmesh 0::

    In [1]: n = np.load("GMergedMesh/0/nodeinfo.npy")

    In [3]: n[n[:,0]>0]
    Out[3]: array([[ 288,  483, 3158, 3157]], dtype=uint32)

    In [4]: f = np.load("GMergedMesh/0/indices.npy")

    In [4]: (f.min(), f.max())
    Out[4]: (0, 482)

    In [8]: v = np.load("GMergedMesh/0/vertices.npy")

    In [9]: v.shape
    Out[9]: (483, 3)

    In [19]: cuf = count_unique(f[:,0])   # hub vertices should be apparent by appearing in more faces 

    In [20]: cuf[cuf[:,1]>4]
    Out[20]: 
    array([[ 96,   6],
           [127,   6],
           [421,   6],
           [453,   6]])    # expected more, but the many repeated vertices explains why only 6 


    In [24]: v[[96,127,421,453]]
    Out[24]: 
    array([[ -18079.453, -799699.438,   -5565.   ],                 
           [ -18079.453, -799699.438,   -8650.   ],
           [ -18079.461, -799699.562,   -5475.51 ],
           [ -18079.461, -799699.562,   -5564.95 ]], dtype=float32)

    In [26]: v[[96,127,421,453]][:,2] + 8650
    Out[26]: array([ 3085.  ,     0.  ,  3174.49,  3085.05], dtype=float32)    ## OOPS 2 layers of Z only 0.05 different from each other

    In [29]: cnv = count_unique(v[:,2])     # unique z values

    In [30]: cnv
    Out[30]: 
    array([[-8650.  ,    79.  ],    # base
           [-5565.  ,    78.  ],    # squealer-
           [-5564.95,    79.  ],    # squealer+
           [-5549.95,   168.  ],    
           [-5475.51,    79.  ]])


    In [31]: cnv[:,0]
    Out[31]: array([-8650.  , -5565.  , -5564.95, -5549.95, -5475.51])

    In [32]: cnv[:,0] + 8650
    Out[32]: array([    0.  ,  3085.  ,  3085.05,  3100.05,  3174.49])    

    ##
    ##                        observed from         expected from
    ##                        vertices              detdesc parameter calc below
    ##        
    ##     IavBrlHeight         3085. 
    ##     IavLidFlgThickness     15.
    ##     IavHeight            3174.49  (+0.05)    3174.44     
    ##     
    ##
    ##     presumably Geant4 triangulation did the 0.05 nudge for visualization reasons ?
    ##
    ##     Pragmatic approach: need code to identify and heal afflicted meshes...
    ##     (G4 triangulation code is not smth I am motivated to get into)
    ## 
    ##   :google:`mesh remove internal faces`
    ##
    ##  hmm some circle fitting would be useful here ... 
    ##       http://stackoverflow.com/questions/26574945/how-to-find-the-center-of-circle-using-the-least-square-fit-in-python
    ##         http://autotrace.sourceforge.net/WSCG98.pdf
    ##
    ##   will need scipy py27-scipy 
    ##   maybe not   http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    ## 

::

    In [37]: p0 = v[v[:,2] == -8650.]

    In [41]: p1 = v[v[:,2] == -5565. ]

    In [42]: p2 = v[v[:,2] == (-5565.+.05) ]

    In [43]: p3 = v[v[:,2] == -5549.95]

    In [44]: p4 = v[v[:,2] == -5475.51]


    In [57]: p0   # half of the 79 are duplicated ?
    Out[57]: 
    array([[ -17232.102, -801009.25 ,   -8650.   ],
           [ -16921.973, -800745.312,   -8650.   ],
           [ -16921.973, -800745.312,   -8650.   ],
           [ -16690.721, -800410.062,   -8650.   ],
           [ -16690.721, -800410.062,   -8650.   ],
           [ -16554.107, -800026.438,   -8650.   ],
           [ -16554.107, -800026.438,   -8650.   ],
            ...

    In [59]: p1   # again 1st half are duplicated other than 1st 
    Out[59]: 
    array([[ -17232.102, -801009.25 ,   -5565.   ],
           [ -16921.973, -800745.312,   -5565.   ],
           [ -16921.973, -800745.312,   -5565.   ],
           [ -16690.721, -800410.062,   -5565.   ],
           [ -16690.721, -800410.062,   -5565.   ],
           [ -16554.107, -800026.438,   -5565.   ],






    In [39]: plt.plot( p0[:,0], p0[:,1] )
    Out[39]: [<matplotlib.lines.Line2D at 0x11143acd0>]

    In [40]: plt.show()


Some but not all the spokes line up::

    In [47]: plt.plot(p1[:,0], p1[:,1], p2[:,0], p2[:,1] )
    Out[47]: 
    [<matplotlib.lines.Line2D at 0x10fa8a390>,
     <matplotlib.lines.Line2D at 0x10fa8a610>]

    In [48]: plt.show()

Flange and top::

    In [49]: plt.plot(p3[:,0], p3[:,1], p4[:,0], p4[:,1] )
    Out[49]: 
    [<matplotlib.lines.Line2D at 0x113b5a550>,
     <matplotlib.lines.Line2D at 0x113b5a7d0>]

All together::

    In [55]: plt.plot(p0[:,0], p0[:,1], p1[:,0], p1[:,1], p2[:,0], p2[:,1], p3[:,0], p3[:,1], p4[:,0], p4[:,1] )


dybgaudi/Detector/XmlDetDesc/DDDB/AD/IAV.xml::

     01 <?xml version="1.0" encoding="UTF-8"?>
      2 <!-- Warning: this is a generated file.  Any modifications may be lost. -->
      3 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd" [
      4   <!ENTITY ADParameters SYSTEM "parameters.xml">
      5   <!ENTITY AdDetailParameters SYSTEM "../AdDetails/parameters.xml">
      6   <!ENTITY OverflowParameters SYSTEM "../OverflowTanks/parameters.xml">
      7   <!ENTITY CalibrationBoxParameters SYSTEM "../CalibrationBox/parameters.xml">
      8   <!ENTITY HandWrittenPhysVols SYSTEM "../AdDetails/IAVPhysVols.xml">
      9 ${DD_AD_IAV_EE}
     10  ]>
     11 <DDDB>
     12 &ADParameters;
     13 &AdDetailParameters;
     14 &OverflowParameters;
     15 &CalibrationBoxParameters;
     16 ${DD_AD_IAV_TOP}
     17 <logvol name="lvIAV" material="Acrylic">
     18   <union name="iav">
     19     <tubs name="iav_cyl"
     20           sizeZ="IavBrlHeight"
     21           outerRadius="IavBrlOutRadius"
     22           />
     23     <polycone name="iav_polycone">
     24       <zplane z="IavBrlHeight"
     25               outerRadius="IavLidRadius"
     26               />
     27       <zplane z="IavBrlHeight+IavLidFlgThickness"
     28               outerRadius="IavLidRadius"
     29               />
     30       <zplane z="IavBrlHeight+IavLidFlgThickness"
     31               outerRadius="IavLidConBotRadius"
     32               />
     33       <zplane z="IavHeight"
     34               outerRadius="IavLidConTopRadius"
     35               />
     36     </polycone>
     //
     //
     //     ARGHH : IS THIS THE CAUSE ???????? 
     //                   POLYCONE WITH TWO ZPLANES AT SAME Z 
     // 
     //
     37     <posXYZ z="-(IavBrlHeight)/2"/>
     38   </union>
     39   <physvol name="pvGDS" logvol="/dd/Geometry/AD/lvGDS">
     40     <posXYZ z="IavBotThickness-IavBrlHeight/2+GdsBrlHeight/2" />
     41   </physvol>
     42   &HandWrittenPhysVols;
     43   ${DD_AD_IAV_PV}
     44 </logvol>
     45 </DDDB>





dybgaudi/Detector/XmlDetDesc/DDDB/AD/parameters.xml::

    149 <!-- Iav barrel thickness -->
    150 <parameter name="IavBrlThickness" value="10*mm"/>
    ...
    153 <!-- Iav bottom thickness -->
    154 <parameter name="IavBotThickness" value="15*mm"/>
    ...
    158 <parameter name="IavBrlHeight" value="3085*mm"/>
    159 <!-- Iav barrel outer radius -->
    160 <parameter name="IavBrlOutRadius" value="1560*mm"/>
    161 <!-- Iav barrel outer radius -->
    162 <parameter name="ADiavRadius" value="IavBrlOutRadius"/>
    163 <!-- Iav lid radius -->
    164 <parameter name="IavLidRadius" value="1565*mm"/>
    165 <!-- Iav lid thickness -->
    166 <parameter name="IavLidThickness" value="15*mm"/>
    167 <!-- Iav lid flange thickness -->
    168 <parameter name="IavLidFlgThickness" value="15*mm"/>
    169 <!-- Iav lid cone inside radius -->
    170 <parameter name="IavLidConInrRadius" value="1520*mm"/>
    171 <!-- Iav lid conical angle -->
    172 <parameter name="IavLidConAngle" value="3.*degree"/>
    173 <!-- Iav lid cone bottom radius -->
    174 <parameter name="IavLidConBotRadius" value="IavLidConInrRadius+IavLidFlgThickness*tan(IavLidConAngle/2.)"/>
    ///
    ///       1520 + 15*tan(3deg/2.)
    ///
    175 <!-- Iav lid cone top radius -->
    176 <parameter name="IavLidConTopRadius" value="100*mm"/>
    177 <!-- Iav lid cone height -->
    178 <parameter name="IavLidConHeight" value="(IavLidConBotRadius-IavLidConTopRadius)*tan(IavLidConAngle)"/>
    ///
    ///          (1520 + 15*tan(1.5deg) - 100)*tan(3deg)
    ///
    /// In [16]: (1520. + 15.*math.tan( math.pi*1.5/180. ) - 100.)*math.tan(math.pi*3./180. )
    /// Out[16]: 74.43963177188732

    ...
    189 <!-- Iav height to the top of the cone -->
    190 <parameter name="IavHeight" value="IavBrlHeight+IavLidFlgThickness+IavLidConHeight"/>
    ///
    /// In [17]: 3085. + 15. + (1520. + 15.*math.tan( math.pi*1.5/180. ) - 100.)*math.tan(math.pi*3./180. )
    /// Out[17]: 3174.4396317718874
    ///     
    ///
    191 <!-- Iav lid height from barrel top the cone top -->
    192 <parameter name="IavLidHeight" value="IavHeight-IavBrlHeight"/>
    ///
    ///
    ///


    ...
    217 <!-- Gds cone top radius -->
    218 <parameter name="GdsConTopRadius" value="75*mm"/>
    219 <!-- Gds cone bottom radius (same as IAV lid cone inner radius -->
    220 <parameter name="GdsConBotRadius" value="IavLidConInrRadius"/>
    221 <!-- Gds barrel radius -->
    222 <parameter name="GdsBrlRadius" value="IavBrlOutRadius-IavBrlThickness"/>
    223 <!-- Gds barrel height -->
    224 <parameter name="GdsBrlHeight" value="IavBrlHeight-IavBotThickness"/>
    225 <!-- Gds cone height -->
    226 <parameter name="GdsConHeight" value="(GdsConBotRadius-GdsConTopRadius)*tan(IavLidConAngle)"/>
    227 <!-- Gds total height (till the bot of IAV hub) -->
    228 <parameter name="GdsHeight" value="GdsBrlHeight+IavLidFlgThickness+IavLidConHeight"/>



dybgaudi/Detector/XmlDetDesc/DDDB/AD/parameters.xml::

    058 <parameter name="OavThickness" value="18*mm"/>
     59 <!-- Oav barrel height -->
     60 <parameter name="OavBrlHeight" value="3982*mm"/>
     61 <!-- Oav barrel outer radius -->
     62 <parameter name="OavBrlOutRadius" value="2000*mm"/>
     63 <!-- Oav barrel flange thickness -->
     64 <parameter name="OavBrlFlgThickness" value="45*mm"/>
     65 <!-- Oav barrel flange radius -->
     66 <parameter name="OavBrlFlgRadius" value="2040*mm"/>
     67 <!-- Oav lid flange thickness -->
     68 <parameter name="OavLidFlgThickness" value="39*mm"/>
     69 <!-- Oav lid flange width -->
     70 <parameter name="OavLidFlgWidth" value="110*mm"/>
     71 <!-- Oav lid conical angle -->
     72 <parameter name="OavLidConAngle" value="3.*degree"/>
     73 <!-- Oav conical lid bottom radius -->
     74 <parameter name="OavLidConBotRadius" value="OavBrlFlgRadius-OavLidFlgWidth"/>
     75 <!-- Oav conical lid top radius -->
     76 <parameter name="OavLidConTopRadius" value="125*mm"/>
     77 <!-- Oav cone height from the turning point -->
     78 <parameter name="OavLidConHeight" value="(OavLidConBotRadius-OavLidConTopRadius)*tan(OavLidConAngle)"/>
     79 <!-- Oav height to the top of the cone -->
     80 <parameter name="OavHeight" value="OavBrlHeight+OavThickness/cos(OavLidConAngle)+OavLidConHeight"/>
     81 <!-- Oav lid height from barrel top to the cone top -->
     82 <parameter name="OavLidHeight" value="OavHeight-OavBrlHeight"/>
     83 <!-- Oav bottom rib height -->
     84 <parameter name="OavBotRibHeight" value="197*mm"/>
    ...
    109 <!-- Lso barrel radius -->
    110 <parameter name="LsoBrlRadius" value="OavBrlOutRadius - OavThickness"/>
    111 <!-- Lso barrel height -->
    112 <parameter name="LsoBrlHeight" value="OavBrlHeight-OavThickness"/>
    113 <!-- Lso cone bottom radius -->
    114 <parameter name="LsoConBotRadius" value="OavLidConBotRadius"/>
    115 <!-- Lso cone top radius (same as the OAV lid top) -->
    116 <parameter name="LsoConTopRadius" value="OavLidConTopRadius"/>
    117 <!--
    118     The tip of LSO (with thickness of OAV lid flange) so LSO is filled to the very top of its container: OAV
    119 -->
    120 <parameter name="LsoConTopTipRadius" value="50*mm"/>
    121 <!-- Lso cone height -->
    122 <parameter name="LsoConHeight" value="(LsoConBotRadius-LsoConTopRadius)*tan(OavLidConAngle)"/>
    123 <!-- Lso total height (till the bot of hub, or the very top of OAV) -->
    124 <parameter name="LsoHeight" value="LsoBrlHeight+OavThickness/cos(OavLidConAngle)+OavLidConHeight"/>
    125 <!-- The 1th corner z pos of LSO -->
    ...


Next volume : 3159, same structure acting OK
-----------------------------------------------
 
::

    ggv --kdyb --torchconfig "radius=0;zenith_azimuth=0,1,0,1"     # volume 3159

Single volume 3159 : uniform all Gd 1st intersection

* __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00  == /dd/Geometry/AD/lvIAV#pvGDS

* in this case the union seems to work with no photons "seeing" the virtual 
  tubs/polycone boundary : again use orthographic side view and rotate 
  around, clearly only one boundary being intersected

* looking up from inside (with flipped normals) can see up to the top little cylindrical snout



Check at detdesc level 
--------------------------

Below detdesc xml generated by 

http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Detector/XmlDetDesc/python/XmlDetDescGen/AD/gen.py







dybgaudi/Detector/XmlDetDesc/DDDB/AD/LSO.xml::

     01 <?xml version="1.0" encoding="UTF-8"?>
      2 <!-- Warning: this is a generated file.  Any modifications may be lost. -->
      3 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd" [
      4   <!ENTITY ADParameters SYSTEM "parameters.xml">
      5   <!ENTITY AdDetailParameters SYSTEM "../AdDetails/parameters.xml">
      6   <!ENTITY OverflowParameters SYSTEM "../OverflowTanks/parameters.xml">
      7   <!ENTITY CalibrationBoxParameters SYSTEM "../CalibrationBox/parameters.xml">
      8   <!ENTITY HandWrittenPhysVols SYSTEM "../AdDetails/LSOPhysVols.xml">
      9 ${DD_AD_LSO_EE}
     10  ]>
     11 <DDDB>
     12 &ADParameters;
     13 &AdDetailParameters;
     14 &OverflowParameters;
     15 &CalibrationBoxParameters;
     16 ${DD_AD_LSO_TOP}
     17 <logvol name="lvLSO" material="LiquidScintillator">
     18   <union name="lso">
     19     <tubs name="lso_cyl"
     20           sizeZ="LsoBrlHeight"
     21           outerRadius="LsoBrlRadius"
     22           />
     23     <polycone name="lso_polycone">
     24       <zplane z="LsoBrlHeight"
     25               outerRadius="LsoConBotRadius"
     26               />
     27       <zplane z="LsoBrlHeight+LsoConHeight"
     28               outerRadius="LsoConTopRadius"
     29               />
     30       <zplane z="LsoBrlHeight+LsoConHeight"
     31               outerRadius="LsoConTopTipRadius"
     32               />
     33       <zplane z="LsoHeight"
     34               outerRadius="LsoConTopTipRadius"
     35               />
     36     </polycone>
     37     <posXYZ z="-(LsoBrlHeight)/2"/>
     38   </union>
     39   <physvol name="pvIAV" logvol="/dd/Geometry/AD/lvIAV">
     40     <posXYZ z="OavBotRibHeight+IavBotVitHeight+IavBotRibHeight-LsoBrlHeight/2+IavBrlHeight/2" />
     41   </physvol>
     42   &HandWrittenPhysVols;
     43   ${DD_AD_LSO_PV}
     44 </logvol>
     45 </DDDB>




dybgaudi/Detector/XmlDetDesc/DDDB/AD/GDS.xml::

     01 <?xml version="1.0" encoding="UTF-8"?>
      2 <!-- Warning: this is a generated file.  Any modifications may be lost. -->
      3 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd" [
      4   <!ENTITY ADParameters SYSTEM "parameters.xml">
      5   <!ENTITY AdDetailParameters SYSTEM "../AdDetails/parameters.xml">
      6   <!ENTITY OverflowParameters SYSTEM "../OverflowTanks/parameters.xml">
      7   <!ENTITY CalibrationBoxParameters SYSTEM "../CalibrationBox/parameters.xml">
      8   <!ENTITY HandWrittenPhysVols SYSTEM "../AdDetails/GDSPhysVols.xml">
      9 ${DD_AD_GDS_EE}
     10  ]>
     11 <DDDB>
     12 &ADParameters;
     13 &AdDetailParameters;
     14 &OverflowParameters;
     15 &CalibrationBoxParameters;
     16 ${DD_AD_GDS_TOP}
     17 <logvol name="lvGDS" material="GdDopedLS">
     18   <union name="gds">
     19     <tubs name="gds_cyl"
     20           sizeZ="GdsBrlHeight"
     21           outerRadius="GdsBrlRadius"
     22           />
     23     <polycone name="gds_polycone">
     24       <zplane z="GdsBrlHeight"
     25               outerRadius="GdsConBotRadius"
     26               />
     27       <zplane z="GdsBrlHeight+GdsConHeight"
     28               outerRadius="GdsConTopRadius"
     29               />
     30       <zplane z="GdsHeight"
     31               outerRadius="GdsConTopRadius"
     32               />
     33     </polycone>
     34     <posXYZ z="-(GdsBrlHeight)/2"/>
     35   </union>
     36   &HandWrittenPhysVols;
     37   ${DD_AD_GDS_PV}
     38 </logvol>
     39 </DDDB>




     * polycons : 
     * https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html






~                                                                                                                                      
~                                                                                                                                      


