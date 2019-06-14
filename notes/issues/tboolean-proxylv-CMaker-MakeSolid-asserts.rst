tboolean-proxylv-CMaker-MakeSolid-asserts
======================================================

Context
---------

* :doc:`tboolean-with-proxylv-bringing-in-basis-solids`


Overview
----------

* after implementing back translation from ncylinder to G4Polycone for non z-symmetric cylinders 
   the asserts of 22,26,27,37,38 are fixed leaving just the "temple" 16 (sFasteners)


assertive six : dirty half-dozen : 5 of them (at least) with non-z-symmetric ncylinder to G4Tubs asserts 
---------------------------------------------------------------------------------------------------------


00                       Upper_LS_tube0x5b2e9f0 ce0 0.0000,0.0000,0.0000,1750.0000 ce1 0.0000,0.0000,0.0000,1750.0000  0
 1                    Upper_Steel_tube0x5b2eb10 ce0 0.0000,0.0000,0.0000,1750.0000 ce1 0.0000,0.0000,0.0000,1750.0000  1
 2                    Upper_Tyvek_tube0x5b2ec30 ce0 0.0000,0.0000,0.0000,1750.0000 ce1 0.0000,0.0000,0.0000,1750.0000  2
 3                       Upper_Chimney0x5b2e8e0 ce0 0.0000,0.0000,0.0000,1750.0000 ce1 0.0000,0.0000,0.0000,1750.0000  3
 4                                sBar0x5b34ab0 ce0 0.0000,0.0000,0.0000,3430.0000 ce1 0.0000,0.0000,0.0000,3430.0000  4
 5                                sBar0x5b34920 ce0 0.0000,0.0000,0.0000,3430.0000 ce1 0.0000,0.0000,0.0000,3430.0000  5
 6                         sModuleTape0x5b34790 ce0 0.0000,0.0000,0.0000,3430.0000 ce1 0.0000,0.0000,0.0000,3430.0000  6
 7                             sModule0x5b34600 ce0 0.0000,0.0000,0.0000,3430.6001 ce1 0.0000,0.0000,0.0000,3430.6001  7
 8                              sPlane0x5b34470 ce0 0.0000,0.0000,0.0000,3430.6001 ce1 0.0000,0.0000,0.0000,3430.6001  8
 9                               sWall0x5b342e0 ce0 0.0000,0.0000,0.0000,3430.6001 ce1 0.0000,0.0000,0.0000,3430.6001  9
10                              sAirTT0x5b34000 ce0 0.0000,0.0000,0.0000,24000.0000 ce1 0.0000,0.0000,0.0000,24000.0000 10
11                            sExpHall0x4bcd390 ce0 0.0000,0.0000,0.0000,24000.0000 ce1 0.0000,0.0000,0.0000,24000.0000 11
12                            sTopRock0x4bccfc0 ce0 0.0000,0.0000,0.0000,27000.0000 ce1 0.0000,0.0000,0.0000,27000.0000 12
13                             sTarget0x4bd4340 ce0 0.0000,0.0000,60.0000,17760.0000 ce1 0.0000,0.0000,0.0000,17760.0000 13
14                            sAcrylic0x4bd3cd0 ce0 0.0000,0.0000,0.0000,17820.0000 ce1 0.0000,0.0000,0.0000,17820.0000 14
15                              sStrut0x4bd4b80 ce0 0.0000,0.0000,0.0000,600.0000 ce1 0.0000,0.0000,0.0000,600.0000 15
16                         *sFasteners0x4c01080* ce0 0.0000,0.0000,-92.5000,150.0000 ce1 0.0000,0.0000,0.0000,150.0000 16
17                               sMask0x4ca38d0 ce0 0.0000,0.0000,-78.9500,274.9500 ce1 0.0000,0.0000,0.0000,274.9500 17
18             PMT_20inch_inner1_solid0x4cb3610 ce0 0.0000,0.0000,89.5000,249.0000 ce1 0.0000,0.0000,0.0000,249.0000 18
19             PMT_20inch_inner2_solid0x4cb3870 ce0 0.0000,0.0000,-167.0050,249.0000 ce1 0.0000,0.0000,0.0000,249.0000 19
20               PMT_20inch_body_solid0x4c90e50 ce0 0.0000,0.0000,-77.5050,261.5050 ce1 0.0000,0.0000,0.0000,261.5050 20
21                PMT_20inch_pmt_solid0x4c81b40 ce0 0.0000,0.0000,-77.5050,261.5060 ce1 0.0000,0.0000,-0.0000,261.5060 21
22                       *sMask_virtual0x4c36e10* ce0 0.0000,0.0000,-79.0000,275.0500 ce1 0.0000,0.0000,0.0000,275.0500 22
23   PMT_3inch_inner1_solid_ell_helper0x510ae30 ce0 0.0000,0.0000,14.5216,38.0000 ce1 0.0000,0.0000,0.0000,38.0000 23
24   PMT_3inch_inner2_solid_ell_helper0x510af10 ce0 0.0000,0.0000,-4.4157,38.0000 ce1 0.0000,0.0000,0.0000,38.0000 24
25 PMT_3inch_body_solid_ell_ell_helper0x510ada0 ce0 0.0000,0.0000,4.0627,40.0000 ce1 0.0000,0.0000,0.0000,40.0000 25
26                *PMT_3inch_cntr_solid0x510afa0* ce0 0.0000,0.0000,-45.8740,29.9995 ce1 0.0000,0.0000,0.0000,29.9995 26
27                 *PMT_3inch_pmt_solid0x510aae0* ce0 0.0000,0.0000,-17.9373,57.9383 ce1 0.0000,0.0000,0.0000,57.9383 27
28                     sChimneyAcrylic0x5b310c0 ce0 0.0000,0.0000,0.0000,520.0000 ce1 0.0000,0.0000,0.0000,520.0000 28
29                          sChimneyLS0x5b312e0 ce0 0.0000,0.0000,0.0000,1965.0000 ce1 0.0000,0.0000,0.0000,1965.0000 29
30                       sChimneySteel0x5b314f0 ce0 0.0000,0.0000,0.0000,1665.0000 ce1 0.0000,0.0000,0.0000,1665.0000 30
31                          sWaterTube0x5b30eb0 ce0 0.0000,0.0000,0.0000,1965.0000 ce1 0.0000,0.0000,0.0000,1965.0000 31
32                        svacSurftube0x5b3bf50 ce0 0.0000,0.0000,0.0000,4.0000 ce1 0.0000,0.0000,0.0000,4.0000 32
33                           sSurftube0x5b3ab80 ce0 0.0000,0.0000,0.0000,5.0000 ce1 0.0000,0.0000,0.0000,5.0000 33
34                         sInnerWater0x4bd3660 ce0 0.0000,0.0000,850.0000,20900.0000 ce1 0.0000,0.0000,0.0000,20900.0000 34
35                      sReflectorInCD0x4bd3040 ce0 0.0000,0.0000,849.0000,20901.0000 ce1 0.0000,0.0000,0.0000,20901.0000 35
36                     sOuterWaterPool0x4bd2960 ce0 0.0000,0.0000,0.0000,21750.0000 ce1 0.0000,0.0000,0.0000,21750.0000 36
37                        *sPoolLining0x4bd1eb0* ce0 0.0000,0.0000,-1.5000,21753.0000 ce1 0.0000,0.0000,0.0000,21753.0000 37
38                        *sBottomRock0x4bcd770* ce0 0.0000,0.0000,-1500.0000,24750.0000 ce1 0.0000,0.0000,0.0000,24750.0000 38
39                              sWorld0x4bc2350 ce0 0.0000,0.0000,0.0000,60000.0000 ce1 0.0000,0.0000,0.0000,60000.0000 39





Investigate g4codegen fail for x016 sFasteners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`g4codegen_review`



Revived x4gen-- for easy access to what is special about this 6 : 5 at least using polycone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* for x016 the g4codegen failed 
*  *polycone is implicated* for the five

x022::

    . <solids>
        <polycone aunit="deg" deltaphi="360" lunit="mm" name="sMask_virtual0x4c36e10" startphi="0">
          <zplane rmax="264.05" rmin="0" z="-354.05"/>
          <zplane rmax="264.05" rmin="0" z="196.05"/>
        </polycone>
      </solids>

    // start portion generated by nnode::to_g4code 
    G4VSolid* make_solid()
    { 
        double A[]= { -354.050000, 196.050000 }  ; 
        double B[]= { 0.000000, 0.000000 }  ; 
        double C[]= { 264.050000, 264.050000 }  ; 
        G4VSolid* a = new G4Polycone("sMask_virtual0x4c36e10", 0.000000, CLHEP::twopi, 2, A, B, C) ; // 0
        return a ; 
    } 
    // end portion generated by nnode::to_g4code 


g4-;g4-cls G4Polycone::

     41 //   G4Polycone( const G4String& name, 
     42 //               G4double phiStart,     // initial phi starting angle
     43 //               G4double phiTotal,     // total phi angle
     44 //               G4int numZPlanes,      // number of z planes
     45 //               const G4double zPlane[],  // position of z planes
     46 //               const G4double rInner[],  // tangent distance to inner surface
     47 //               const G4double rOuter[])  // tangent distance to outer surface
     48 //

x026::

      <solids>
        <polycone aunit="deg" deltaphi="360" lunit="mm" name="PMT_3inch_cntr_solid0x510afa0" startphi="0">
          <zplane rmax="29.999" rmin="0" z="-15.874508"/>
          <zplane rmax="29.999" rmin="0" z="-75.873508"/>
        </polycone>
      </solids>

x027::

      <solids>
        <polycone aunit="deg" deltaphi="360" lunit="mm" name="PMT_3inch_pmt_solid_cyl0x510a370" startphi="0">
          <zplane rmax="30.001" rmin="0" z="-15.874508"/>
          <zplane rmax="30.001" rmin="0" z="-75.875508"/>
        </polycone>
        <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="PMT_3inch_pmt_solid_sph0x510a990" rmax="40.001" rmin="0" startphi="0" starttheta="0"/>
        <union name="PMT_3inch_pmt_solid0x510aae0">
          <first ref="PMT_3inch_pmt_solid_cyl0x510a370"/>
          <second ref="PMT_3inch_pmt_solid_sph0x510a990"/>
        </union>
      </solids>

x037::

      <solids>
        <polycone aunit="deg" deltaphi="360" lunit="mm" name="sPoolLining0x4bd1eb0" startphi="0">
          <zplane rmax="21753" rmin="0" z="-21753"/>
          <zplane rmax="21753" rmin="0" z="21750"/>
        </polycone>
      </solids>

x038::

      <solids>
        <polycone aunit="deg" deltaphi="360" lunit="mm" name="sBottomRock0x4bcd770" startphi="0">
          <zplane rmax="24750" rmin="0" z="-24750"/>
          <zplane rmax="24750" rmin="0" z="21750"/>
        </polycone>
      </solids>


The above are all the solids in the geometry with polycone, and they all trip the symmetric cylinder assert::


    [blyth@localhost tests]$ pwd
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/g4codegen/tests

    [blyth@localhost tests]$ grep -l polycone *.gdml
    x022.gdml
    x026.gdml
    x027.gdml
    x037.gdml
    x038.gdml


* it appears that polycone is being used to allow definition of non-z-symmetric cylinders



X4Solid::convertPolycone from G4VSolid to nnode 
--------------------------------------------------

* G4Polycone becomes a union of ncylinder
* G4Tubs becomes an ncylinder


::

    1019 void X4Solid::convertPolycone()
    1020 { 
    ....
    1067     std::vector<nnode*> prims ;
    1068     convertPolyconePrimitives( zp, prims );
    1069 
    1070     //LOG(info) << "pre-UnionTree" ; 
    1071     nnode* cn = NTreeBuilder<nnode>::UnionTree(prims) ;



CMaker::ConvertPrimitive : back translation from nnode to G4VSolid : CONFIRMED FIXED
----------------------------------------------------------------------------------------

* nnode has no POLYCONE its using ncylinder 

* the back translation sees ncylinder and yields only G4Tubs

* HENCE : THE PROBLEM IS THAT THE nnode MODEL DOESNT DISTINGUISH BETWEEN 
  ncylinder from G4Polycone and ncylinder from G4Tubs, with G4Tubs being
  symetrically restricted and G4Polycone not

* SOLUTION : dont assert, branch to create a G4Polycone when the ncylinder is
  not symmetric and hence G4Tubs cannot be used  

::

    308 G4VSolid* CMaker::ConvertPrimitive(const nnode* node) // static
    309 {
    ...
    419     else if(node->type == CSG_CYLINDER)
    420     {
    421         ncylinder* n = (ncylinder*)node ;
    422  
    423         float z1 = n->z1() ;
    424         float z2 = n->z2() ;
    425         assert( z2 > z1 && z2 == -z1 );
    426         float hz = fabs(z1) ;
    427  
    428         double innerRadius = 0. ;
    429         double outerRadius = n->radius() ;
    430         double zHalfLength = hz ;  // hmm will need transforms for nudged ?
    431         double startPhi = 0. ;
    432         double deltaPhi = twopi ;
    433  
    434         G4Tubs* tb = new G4Tubs( name, innerRadius, outerRadius, zHalfLength, startPhi, deltaPhi );
    435         result = tb ;
    436     }



ISSUE 1 : CMaker::ConvertPrimitive asserts for PROXYLV 22,26,27,37,38 : expecting symmetrically disposed cylinder 
-------------------------------------------------------------------------------------------------------------------

* hmm seems my fix of baking in the z-shift changes to NCSG and GMesh cannot be translated to Geant4 ?
* solution is to use placement : but the details are kinda painful as have three geometry models to juggle 
 

::

    PROXYLV=22 tboolean.sh -D


    (gdb) bt
    #0  0x00007fffe2019207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe201a8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2012026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20120d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5936b40) at /home/blyth/opticks/cfg4/CMaker.cc:425
    #5  0x00007fffefdd701b in CMaker::MakeSolid_r (node=0x5936b40, depth=0) at /home/blyth/opticks/cfg4/CMaker.cc:117
    #6  0x00007fffefdd6d7d in CMaker::MakeSolid (root=0x5936b40) at /home/blyth/opticks/cfg4/CMaker.cc:84
    #7  0x00007fffefdd6c76 in CMaker::MakeSolid (csg=0x592c100) at /home/blyth/opticks/cfg4/CMaker.cc:75
    #8  0x00007fffefddbc7c in CTestDetector::makeChildVolume (this=0x60ca540, csg=0x592c100, lvn=0x59411e0 "cylinder_lv0_", pvn=0x59411a0 "cylinder_pv0_", mother=0x611aff0) at /home/blyth/opticks/cfg4/CTestDetector.cc:156
    #9  0x00007fffefddc6c4 in CTestDetector::makeDetector_NCSG (this=0x60ca540) at /home/blyth/opticks/cfg4/CTestDetector.cc:237
    #10 0x00007fffefddbade in CTestDetector::makeDetector (this=0x60ca540) at /home/blyth/opticks/cfg4/CTestDetector.cc:95
    #11 0x00007fffefddb95c in CTestDetector::init (this=0x60ca540) at /home/blyth/opticks/cfg4/CTestDetector.cc:78
    #12 0x00007fffefddb7b6 in CTestDetector::CTestDetector (this=0x60ca540, hub=0x6b8d80, query=0x0, sd=0x60c7ee0) at /home/blyth/opticks/cfg4/CTestDetector.cc:64
    #13 0x00007fffefd78ada in CGeometry::init (this=0x60ca490) at /home/blyth/opticks/cfg4/CGeometry.cc:70
    #14 0x00007fffefd789d2 in CGeometry::CGeometry (this=0x60ca490, hub=0x6b8d80, sd=0x60c7ee0) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #15 0x00007fffefde9747 in CG4::CG4 (this=0x5ee7cd0, hub=0x6b8d80) at /home/blyth/opticks/cfg4/CG4.cc:121
    #16 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc40, argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #17 0x0000000000403998 in main (argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 
    (gdb) f 4
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5936b40) at /home/blyth/opticks/cfg4/CMaker.cc:425
    425         assert( z2 > z1 && z2 == -z1 ); 
    (gdb) l
    420     {
    421         ncylinder* n = (ncylinder*)node ; 
    422 
    423         float z1 = n->z1() ; 
    424         float z2 = n->z2() ;
    425         assert( z2 > z1 && z2 == -z1 ); 
    426         float hz = fabs(z1) ;
    427 
    428         double innerRadius = 0. ;
    429         double outerRadius = n->radius() ;
    (gdb) p z2
    $1 = 196.050003
    (gdb) p z1
    $2 = -354.049988
    (gdb) 

::

    [blyth@localhost opticks]$ GMeshLibTest | egrep ^22
    22                       sMask_virtual0x4c36e10 ce0 0.0000,0.0000,-79.0000,275.0500 ce1 0.0000,0.0000,0.0000,275.0500 22


    In [16]: (-275.0500 - 79.0, 275.0500 - 79.0)      ## hmm z-shifting is an Opticks capability that Geant4 doesnt have, hence the assert by CMaker 
    Out[16]: (-354.05, 196.05)



::

    PROXYLV=26 tboolean.sh -D


    2019-06-13 17:12:06.243 INFO  [369860] [NCSGList::createUniverse@258]  outer volume isContainer (ie auto scaled)  universe will be scaled/delted a bit from there 
    2019-06-13 17:12:06.247 FATAL [369860] [CMaker::ConvertPrimitive@394]  loosing offset of CSG_BOX  center 0.0000,0.0000,0.0000
    OKG4Test: /home/blyth/opticks/cfg4/CMaker.cc:425: static G4VSolid* CMaker::ConvertPrimitive(const nnode*): Assertion `z2 > z1 && z2 == -z1' failed.
    
    (gdb) bt
    #0  0x00007fffe2019207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe201a8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2012026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20120d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5978b60) at /home/blyth/opticks/cfg4/CMaker.cc:425
    #5  0x00007fffefdd701b in CMaker::MakeSolid_r (node=0x5978b60, depth=0) at /home/blyth/opticks/cfg4/CMaker.cc:117
    #6  0x00007fffefdd6d7d in CMaker::MakeSolid (root=0x5978b60) at /home/blyth/opticks/cfg4/CMaker.cc:84
    #7  0x00007fffefdd6c76 in CMaker::MakeSolid (csg=0x5975cc0) at /home/blyth/opticks/cfg4/CMaker.cc:75
    #8  0x00007fffefddbc7c in CTestDetector::makeChildVolume (this=0x60bfd60, csg=0x5975cc0, lvn=0x5a77650 "cylinder_lv0_", pvn=0x5a77610 "cylinder_pv0_", mother=0x61101a0) at /home/blyth/opticks/cfg4/CTestDetector.cc:156
    #9  0x00007fffefddc6c4 in CTestDetector::makeDetector_NCSG (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:237
    #10 0x00007fffefddbade in CTestDetector::makeDetector (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:95
    #11 0x00007fffefddb95c in CTestDetector::init (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:78
    #12 0x00007fffefddb7b6 in CTestDetector::CTestDetector (this=0x60bfd60, hub=0x6b8d80, query=0x0, sd=0x60bd700) at /home/blyth/opticks/cfg4/CTestDetector.cc:64
    #13 0x00007fffefd78ada in CGeometry::init (this=0x60bfcb0) at /home/blyth/opticks/cfg4/CGeometry.cc:70
    #14 0x00007fffefd789d2 in CGeometry::CGeometry (this=0x60bfcb0, hub=0x6b8d80, sd=0x60bd700) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #15 0x00007fffefde9747 in CG4::CG4 (this=0x5edd4f0, hub=0x6b8d80) at /home/blyth/opticks/cfg4/CG4.cc:121
    #16 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc40, argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #17 0x0000000000403998 in main (argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) f 4
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5978b60) at /home/blyth/opticks/cfg4/CMaker.cc:425
    425         assert( z2 > z1 && z2 == -z1 ); 
    (gdb) l
    420     {
    421         ncylinder* n = (ncylinder*)node ; 
    422 
    423         float z1 = n->z1() ; 
    424         float z2 = n->z2() ;
    425         assert( z2 > z1 && z2 == -z1 ); 
    426         float hz = fabs(z1) ;
    427 
    428         double innerRadius = 0. ;
    429         double outerRadius = n->radius() ;
    (gdb) p z2
    $2 = -15.8745079
    (gdb) p z1
    $3 = -75.8735046

::

    [blyth@localhost opticks]$ GMeshLibTest | egrep ^26
    26                PMT_3inch_cntr_solid0x510afa0 ce0 0.0000,0.0000,-45.8740,29.9995 ce1 0.0000,0.0000,0.0000,29.9995 26

    In [1]: (-29.9995-45.8740, 29.9995-45.8740)
    Out[1]: (-75.8735, -15.874500000000001)


    PROXYLV=37 tboolean.sh -D

    (gdb) bt
    #0  0x00007fffe2019207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe201a8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2012026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20120d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5a27fe0) at /home/blyth/opticks/cfg4/CMaker.cc:425
    #5  0x00007fffefdd701b in CMaker::MakeSolid_r (node=0x5a27fe0, depth=0) at /home/blyth/opticks/cfg4/CMaker.cc:117
    #6  0x00007fffefdd6d7d in CMaker::MakeSolid (root=0x5a27fe0) at /home/blyth/opticks/cfg4/CMaker.cc:84
    #7  0x00007fffefdd6c76 in CMaker::MakeSolid (csg=0x5a2c4b0) at /home/blyth/opticks/cfg4/CMaker.cc:75
    #8  0x00007fffefddbc7c in CTestDetector::makeChildVolume (this=0x60bfd60, csg=0x5a2c4b0, lvn=0x5a77650 "cylinder_lv0_", pvn=0x5a77610 "cylinder_pv0_", mother=0x6110810) at /home/blyth/opticks/cfg4/CTestDetector.cc:156
    #9  0x00007fffefddc6c4 in CTestDetector::makeDetector_NCSG (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:237
    #10 0x00007fffefddbade in CTestDetector::makeDetector (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:95
    #11 0x00007fffefddb95c in CTestDetector::init (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:78
    #12 0x00007fffefddb7b6 in CTestDetector::CTestDetector (this=0x60bfd60, hub=0x6b8d80, query=0x0, sd=0x60bd700) at /home/blyth/opticks/cfg4/CTestDetector.cc:64
    #13 0x00007fffefd78ada in CGeometry::init (this=0x60bfcb0) at /home/blyth/opticks/cfg4/CGeometry.cc:70
    #14 0x00007fffefd789d2 in CGeometry::CGeometry (this=0x60bfcb0, hub=0x6b8d80, sd=0x60bd700) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #15 0x00007fffefde9747 in CG4::CG4 (this=0x5edd4f0, hub=0x6b8d80) at /home/blyth/opticks/cfg4/CG4.cc:121
    #16 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc40, argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #17 0x0000000000403998 in main (argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) f 4
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5a27fe0) at /home/blyth/opticks/cfg4/CMaker.cc:425
    425         assert( z2 > z1 && z2 == -z1 ); 
    (gdb) l
    420     {
    421         ncylinder* n = (ncylinder*)node ; 
    422 
    423         float z1 = n->z1() ; 
    424         float z2 = n->z2() ;
    425         assert( z2 > z1 && z2 == -z1 ); 
    426         float hz = fabs(z1) ;
    427 
    428         double innerRadius = 0. ;
    429         double outerRadius = n->radius() ;
    (gdb) p z1
    $1 = -21753
    (gdb) p z2
    $2 = 21750
    (gdb) 

::

    37                         sPoolLining0x4bd1eb0 ce0 0.0000,0.0000,-1.5000,21753.0000 ce1 0.0000,0.0000,0.0000,21753.0000 37

    In [4]: (-21753-1.5,21753-1.5)
    Out[4]: (-21754.5, 21751.5)


::

    PROXYLV=38 tboolean.sh

    (gdb) bt
    #0  0x00007fffe2019207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe201a8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2012026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20120d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5a2fdd0) at /home/blyth/opticks/cfg4/CMaker.cc:425
    #5  0x00007fffefdd701b in CMaker::MakeSolid_r (node=0x5a2fdd0, depth=0) at /home/blyth/opticks/cfg4/CMaker.cc:117
    #6  0x00007fffefdd6d7d in CMaker::MakeSolid (root=0x5a2fdd0) at /home/blyth/opticks/cfg4/CMaker.cc:84
    #7  0x00007fffefdd6c76 in CMaker::MakeSolid (csg=0x5a33cc0) at /home/blyth/opticks/cfg4/CMaker.cc:75
    #8  0x00007fffefddbc7c in CTestDetector::makeChildVolume (this=0x60bfd60, csg=0x5a33cc0, lvn=0x5a77650 "cylinder_lv0_", pvn=0x5a77610 "cylinder_pv0_", mother=0x6110810) at /home/blyth/opticks/cfg4/CTestDetector.cc:156
    #9  0x00007fffefddc6c4 in CTestDetector::makeDetector_NCSG (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:237
    #10 0x00007fffefddbade in CTestDetector::makeDetector (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:95
    #11 0x00007fffefddb95c in CTestDetector::init (this=0x60bfd60) at /home/blyth/opticks/cfg4/CTestDetector.cc:78
    #12 0x00007fffefddb7b6 in CTestDetector::CTestDetector (this=0x60bfd60, hub=0x6b8d80, query=0x0, sd=0x60bd700) at /home/blyth/opticks/cfg4/CTestDetector.cc:64
    #13 0x00007fffefd78ada in CGeometry::init (this=0x60bfcb0) at /home/blyth/opticks/cfg4/CGeometry.cc:70
    #14 0x00007fffefd789d2 in CGeometry::CGeometry (this=0x60bfcb0, hub=0x6b8d80, sd=0x60bd700) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #15 0x00007fffefde9747 in CG4::CG4 (this=0x5edd4f0, hub=0x6b8d80) at /home/blyth/opticks/cfg4/CG4.cc:121
    #16 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc40, argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #17 0x0000000000403998 in main (argc=32, argv=0x7fffffffcf78) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 
    
    38                         sBottomRock0x4bcd770 ce0 0.0000,0.0000,-1500.0000,24750.0000 ce1 0.0000,0.0000,0.0000,24750.0000 38
    
    (gdb) f 4
    #4  0x00007fffefdd8c57 in CMaker::ConvertPrimitive (node=0x5a2fdd0) at /home/blyth/opticks/cfg4/CMaker.cc:425
    425         assert( z2 > z1 && z2 == -z1 ); 
    (gdb) l
    420     {
    421         ncylinder* n = (ncylinder*)node ; 
    422 
    423         float z1 = n->z1() ; 
    424         float z2 = n->z2() ;
    425         assert( z2 > z1 && z2 == -z1 ); 
    426         float hz = fabs(z1) ;
    427 
    428         double innerRadius = 0. ;
    429         double outerRadius = n->radius() ;
    (gdb) p z1
    $1 = -24750
    (gdb) p z2
    $2 = 21750
    (gdb) 





ISSUE 2 : CMaker left transform assert for PROXYLV 16
--------------------------------------------------------

* :doc:`x016`



