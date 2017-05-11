GDML-glTF Geometry Route Transform Debug
===========================================


Issue : PMTs pointing in reverse direction !
------------------------------------------------

::

    tgltf-
    tgltf--

See:

* npy-/tests/NGLMTest.cc:test_axisAngle
* dev/csg/sc_transform_check.py 

Permuting axes (X,Y,Z)->(Y,Z,X) leads to much more reasonable interpretation 
of the txf transforms.  This is suggestive that a PMT orienting 
transform (to adjust from model frame with +Z in PMT pointing direction)
is being applied after PMT ring rotatations. 

::

     76         glm::mat4 trs2(1.f) ;
     77         trs2[0] = trs[1] ;  //  Y->X
     78         trs2[1] = trs[2] ;  //  Z->Y
     79         trs2[2] = trs[0] ;  //  X->Z
     80         trs2[3] = trs[3] ;
     81 
     82         //  ( X,Y,Z ) -> ( Y,Z,X )
     83         


Take axes for a spin::

    In [28]: from glm import rotate

    In [30]: rot = rotate([1,1,1,360./3.] )

    In [31]: rot
    Out[31]: 
    array([[-0.,  1., -0.,  0.],
           [-0., -0.,  1.,  0.],
           [ 1., -0., -0.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float32)


    In [32]: rot = rotate([1,1,1,-360./3.] )

    In [33]: rot
    Out[33]: 
    array([[-0., -0.,  1.,  0.],      // Z->X
           [ 1., -0., -0.,  0.],      // X->Y
           [-0.,  1., -0.,  0.],      // Y->Z
           [ 0.,  0.,  0.,  1.]], dtype=float32)



::

    * txf: 8,24,4,4
    ( 0, 0) {    0.0000    0.0000    1.0000} 1.7017 (  {   -0.13    0.99    0.00    0.00} {   -0.99   -0.13    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 1) {    0.0000    0.0000    1.0000} 1.9635 (  {   -0.38    0.92    0.00    0.00} {   -0.92   -0.38    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 2) {    0.0000    0.0000    1.0000} 2.2253 (  {   -0.61    0.79    0.00    0.00} {   -0.79   -0.61    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 3) {    0.0000    0.0000    1.0000} 2.4871 (  {   -0.79    0.61    0.00    0.00} {   -0.61   -0.79    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 4) {    0.0000    0.0000    1.0000} 2.7489 (  {   -0.92    0.38    0.00    0.00} {   -0.38   -0.92    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 5) {    0.0000    0.0000    1.0000} 3.0107 (  {   -0.99    0.13    0.00    0.00} {   -0.13   -0.99    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 6) {   -0.0000   -0.0000   -1.0000} 3.0107 (  {   -0.99   -0.13    0.00    0.00} {    0.13   -0.99    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 7) {   -0.0000   -0.0000   -1.0000} 2.7489 (  {   -0.92   -0.38    0.00    0.00} {    0.38   -0.92    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 8) {   -0.0000   -0.0000   -1.0000} 2.4871 (  {   -0.79   -0.61    0.00    0.00} {    0.61   -0.79    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0, 9) {   -0.0000   -0.0000   -1.0000} 2.2253 (  {   -0.61   -0.79    0.00    0.00} {    0.79   -0.61    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,10) {   -0.0000   -0.0000   -1.0000} 1.9635 (  {   -0.38   -0.92    0.00    0.00} {    0.92   -0.38    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,11) {   -0.0000   -0.0000   -1.0000} 1.7017 (  {   -0.13   -0.99    0.00    0.00} {    0.99   -0.13    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,12) {    0.0000    0.0000   -1.0000} 1.4399 (  {    0.13   -0.99   -0.00    0.00} {    0.99    0.13    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,13) {    0.0000    0.0000   -1.0000} 1.1781 (  {    0.38   -0.92   -0.00    0.00} {    0.92    0.38    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,14) {    0.0000    0.0000   -1.0000} 0.9163 (  {    0.61   -0.79   -0.00    0.00} {    0.79    0.61    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,15) {    0.0000    0.0000   -1.0000} 0.6545 (  {    0.79   -0.61   -0.00    0.00} {    0.61    0.79    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,16) {    0.0000    0.0000   -1.0000} 0.3927 (  {    0.92   -0.38   -0.00    0.00} {    0.38    0.92    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,17) {    0.0000    0.0000   -1.0000} 0.1309 (  {    0.99   -0.13   -0.00    0.00} {    0.13    0.99    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,18) {    0.0000    0.0000    1.0000} 0.1309 (  {    0.99    0.13   -0.00    0.00} {   -0.13    0.99    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,19) {    0.0000    0.0000    1.0000} 0.3927 (  {    0.92    0.38   -0.00    0.00} {   -0.38    0.92    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,20) {    0.0000    0.0000    1.0000} 0.6545 (  {    0.79    0.61   -0.00    0.00} {   -0.61    0.79    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,21) {    0.0000    0.0000    1.0000} 0.9163 (  {    0.61    0.79   -0.00    0.00} {   -0.79    0.61    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,22) {    0.0000    0.0000    1.0000} 1.1781 (  {    0.38    0.92   -0.00    0.00} {   -0.92    0.38    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
    ( 0,23) {    0.0000    0.0000    1.0000} 1.4399 (  {    0.13    0.99   -0.00    0.00} {   -0.99    0.13    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )


Y-Z swap::

    | 1 0 0 0 |
    | 0 0 1 0 |
    | 0 1 0 0 |
    | 0 0 0 1 |


    
GDML/glTF route
----------------

opticks.ana.pmt.gdml:GDML
    parse GDML input file into wrapped element object model, no structural manipulations : just wrapping 

opticks.ana.pmt.treebase:Tree
    restructures stripped LV/PV/LV/... volume tree into homogenous node tree (LV,PV)/(LV,PV)/...


GDML Stage
~~~~~~~~~~~~

::

    191
    Position mm -2304.61358026 303.408133816 1750.0 
    Rotation deg -90.0 -82.5 -90.0 

    <position xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmt0xc110bd8_pos" unit="mm" x="-2304.61358026342" y="303.408133815512" z="1750"/>
            
    <rotation xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" name="/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmt0xc110bd8_rot" unit="deg" x="-90" y="-82.4999999999999" z="-90"/>
          
    [[    0.        -0.         1.         0.    ]
     [    0.1305     0.9914    -0.         0.    ]
     [   -0.9914     0.1305     0.         0.    ]
     [-2304.6135   303.4081  1750.         1.    ]]



In [24]: pmts[191].transform    ## with modified zyx order in glm.py:rotate_three_axis 
Out[24]: 
array([[    0.    ,    -0.1305,     0.9914,     0.    ],
       [    0.    ,    -0.9914,    -0.1305,     0.    ],
       [    1.    ,     0.    ,     0.    ,     0.    ],
       [-2304.6135,   303.4081,  1750.    ,     1.    ]], dtype=float32)

In [2]: pmts[191].transform     ## with the longstanding xyz order  
Out[2]: 
array([[    0.    ,    -0.    ,     1.    ,     0.    ],
       [    0.1305,     0.9914,    -0.    ,     0.    ],
       [   -0.9914,     0.1305,     0.    ,     0.    ],
       [-2304.6135,   303.4081,  1750.    ,     1.    ]], dtype=float32)



    In [18]: eulerAngleXYZ([-90.0,-82.5,-90.0])
    Out[18]: 
    array([[-0.    ,  0.    ,  1.    ,  0.    ],
           [ 0.1305,  0.9914,  0.    ,  0.    ],
           [-0.9914,  0.1305, -0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)


    In [17]: eulerAngleXYZ([90.0,-82.5,90.0])
    Out[17]: 
    array([[-0.    , -0.    ,  1.    ,  0.    ],
           [-0.1305,  0.9914, -0.    ,  0.    ],
           [-0.9914, -0.1305, -0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=float32)




Probably the 3-axis rotation interpretation I am using to 
convert this into a transform : doesnt match the GDML intention ?


::

     00                      Rotation deg 90.0 -82.5 90.0  Position mm -2304.61358026 -303.408133816 -1750.0  
      1                      Rotation deg 90.0 -67.5 90.0  Position mm -2147.55797332 -889.547638533 -1750.0  
      2                      Rotation deg 90.0 -52.5 90.0  Position mm -1844.14983951 -1415.06594173 -1750.0  
      3                      Rotation deg 90.0 -37.5 90.0  Position mm -1415.06594173 -1844.14983951 -1750.0  
      4                      Rotation deg 90.0 -22.5 90.0  Position mm -889.547638533 -2147.55797332 -1750.0  
      5                       Rotation deg 90.0 -7.5 90.0  Position mm -303.408133816 -2304.61358026 -1750.0  
      6                        Rotation deg 90.0 7.5 90.0  Position mm 303.408133816 -2304.61358026 -1750.0  
      7                       Rotation deg 90.0 22.5 90.0  Position mm 889.547638533 -2147.55797332 -1750.0  
      8                       Rotation deg 90.0 37.5 90.0  Position mm 1415.06594173 -1844.14983951 -1750.0  
      9                       Rotation deg 90.0 52.5 90.0  Position mm 1844.14983951 -1415.06594173 -1750.0  
     10                       Rotation deg 90.0 67.5 90.0  Position mm 2147.55797332 -889.547638533 -1750.0  
     11                       Rotation deg 90.0 82.5 90.0  Position mm 2304.61358026 -303.408133816 -1750.0  
     12                     Rotation deg -90.0 82.5 -90.0  Position mm 2304.61358026 303.408133816 -1750.0  
     13                     Rotation deg -90.0 67.5 -90.0  Position mm 2147.55797332 889.547638533 -1750.0  
     14                     Rotation deg -90.0 52.5 -90.0  Position mm 1844.14983951 1415.06594173 -1750.0  
     15                     Rotation deg -90.0 37.5 -90.0  Position mm 1415.06594173 1844.14983951 -1750.0  
     16                     Rotation deg -90.0 22.5 -90.0  Position mm 889.547638533 2147.55797332 -1750.0  
     17                      Rotation deg -90.0 7.5 -90.0  Position mm 303.408133816 2304.61358026 -1750.0  
     18                     Rotation deg -90.0 -7.5 -90.0  Position mm -303.408133816 2304.61358026 -1750.0  
     19                    Rotation deg -90.0 -22.5 -90.0  Position mm -889.547638533 2147.55797332 -1750.0  
     20                    Rotation deg -90.0 -37.5 -90.0  Position mm -1415.06594173 1844.14983951 -1750.0  
     21                    Rotation deg -90.0 -52.5 -90.0  Position mm -1844.14983951 1415.06594173 -1750.0  
     22                    Rotation deg -90.0 -67.5 -90.0  Position mm -2147.55797332 889.547638533 -1750.0  
     23                    Rotation deg -90.0 -82.5 -90.0  Position mm -2304.61358026 303.408133816 -1750.0  
     24                      Rotation deg 90.0 -82.5 90.0  Position mm -2304.61358026 -303.408133816 -1250.0  
     25                      Rotation deg 90.0 -67.5 90.0  Position mm -2147.55797332 -889.547638533 -1250.0  
     26                      Rotation deg 90.0 -52.5 90.0  Position mm -1844.14983951 -1415.06594173 -1250.0  
     27                      Rotation deg 90.0 -37.5 90.0  Position mm -1415.06594173 -1844.14983951 -1250.0  
     28                      Rotation deg 90.0 -22.5 90.0  Position mm -889.547638533 -2147.55797332 -1250.0  
     29                       Rotation deg 90.0 -7.5 90.0  Position mm -303.408133816 -2304.61358026 -1250.0  
     30                        Rotation deg 90.0 7.5 90.0  Position mm 303.408133816 -2304.61358026 -1250.0  
     31                       Rotation deg 90.0 22.5 90.0  Position mm 889.547638533 -2147.55797332 -1250.0  
     32                       Rotation deg 90.0 37.5 90.0  Position mm 1415.06594173 -1844.14983951 -1250.0  
     33                       Rotation deg 90.0 52.5 90.0  Position mm 1844.14983951 -1415.06594173 -1250.0  
     34                       Rotation deg 90.0 67.5 90.0  Position mm 2147.55797332 -889.547638533 -1250.0  
     35                       Rotation deg 90.0 82.5 90.0  Position mm 2304.61358026 -303.408133816 -1250.0  
     36                     Rotation deg -90.0 82.5 -90.0  Position mm 2304.61358026 303.408133816 -1250.0  
     37                     Rotation deg -90.0 67.5 -90.0  Position mm 2147.55797332 889.547638533 -1250.0  
     38                     Rotation deg -90.0 52.5 -90.0  Position mm 1844.14983951 1415.06594173 -1250.0  



/usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include/G4GDMLWriteDefine.hh::

     36 // History:
     37 // - Created.                                  Zoltan Torzsok, November 2007
     38 // -------------------------------------------------------------------------
     39 
     40 #ifndef _G4GDMLWRITEDEFINE_INCLUDED_
     41 #define _G4GDMLWRITEDEFINE_INCLUDED_
     42 
     43 #include "G4Types.hh"
     44 #include "G4ThreeVector.hh"
     45 #include "G4RotationMatrix.hh"
     46 
     47 #include "G4GDMLWrite.hh"
     48 
     49 class G4GDMLWriteDefine : public G4GDMLWrite
     50 {
     51 
     52   public:
     53 
     54     G4ThreeVector GetAngles(const G4RotationMatrix&);
     55     void ScaleWrite(xercesc::DOMElement* element,
     56                     const G4String& name, const G4ThreeVector& scl)
     57          { Scale_vectorWrite(element,"scale",name,scl); }
     58     void RotationWrite(xercesc::DOMElement* element,
     59                     const G4String& name, const G4ThreeVector& rot)
     60          { Rotation_vectorWrite(element,"rotation",name,rot); }
     61     void PositionWrite(xercesc::DOMElement* element,
     62                     const G4String& name, const G4ThreeVector& pos)
     63          { Position_vectorWrite(element,"position",name,pos); }
     64     void FirstrotationWrite(xercesc::DOMElement* element,
     65                     const G4String& name, const G4ThreeVector& rot)
     66          { Rotation_vectorWrite(element,"firstrotation",name,rot); }
     67     void FirstpositionWrite(xercesc::DOMElement* element,
     68                     const G4String& name, const G4ThreeVector& pos)


::

    simon:gdml blyth$ find . -type f -exec grep -H Rotation_vectorWrite {} \;
    ./include/G4GDMLWriteDefine.hh:         { Rotation_vectorWrite(element,"rotation",name,rot); }
    ./include/G4GDMLWriteDefine.hh:         { Rotation_vectorWrite(element,"firstrotation",name,rot); }
    ./include/G4GDMLWriteDefine.hh:    void Rotation_vectorWrite(xercesc::DOMElement*, const G4String&,
    ./src/G4GDMLWriteDefine.cc:Rotation_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
    simon:gdml blyth$ 


::

    097 void G4GDMLWriteDefine::
     98 Rotation_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
     99                      const G4String& name, const G4ThreeVector& rot)
    100 {
    101    const G4double x = (std::fabs(rot.x()) < kAngularPrecision) ? 0.0 : rot.x();
    102    const G4double y = (std::fabs(rot.y()) < kAngularPrecision) ? 0.0 : rot.y();
    103    const G4double z = (std::fabs(rot.z()) < kAngularPrecision) ? 0.0 : rot.z();
    104 
    105    xercesc::DOMElement* rotationElement = NewElement(tag);
    106    rotationElement->setAttributeNode(NewAttribute("name",name));
    107    rotationElement->setAttributeNode(NewAttribute("x",x/degree));
    108    rotationElement->setAttributeNode(NewAttribute("y",y/degree));
    109    rotationElement->setAttributeNode(NewAttribute("z",z/degree));
    110    rotationElement->setAttributeNode(NewAttribute("unit","deg"));
    111    element->appendChild(rotationElement);
    112 }


::

     51 G4ThreeVector G4GDMLWriteDefine::GetAngles(const G4RotationMatrix& mtx)
     52 {
     53    G4double x,y,z;
     54    G4RotationMatrix mat = mtx;
     55    mat.rectify();   // Rectify matrix from possible roundoff errors
     56 
     57    // Direction of rotation given by left-hand rule; clockwise rotation
     58 
     59    static const G4double kMatrixPrecision = 10E-10;
     60    const G4double cosb = std::sqrt(mtx.xx()*mtx.xx()+mtx.yx()*mtx.yx());
     ..                                       r11^2 + r21^2
     61 
     62    if (cosb > kMatrixPrecision)
     63    {
     64       x = std::atan2(mtx.zy(),mtx.zz());   
     ..                         r32      r33   
     65       y = std::atan2(-mtx.zx(),cosb);
     ..                        -r31 
     66       z = std::atan2(mtx.yx(),mtx.xx());
     ..                         r21     r11
     67    }
     68    else
     69    {
     70       x = std::atan2(-mtx.yz(),mtx.yy());
     71       y = std::atan2(-mtx.zx(),cosb);
     72       z = 0.0;
     73    }
     74 
     75    return G4ThreeVector(x,y,z);
     76 }



Decomposing Euler Angles

* http://nghiaho.com/?page_id=846


::

    simon:gdml blyth$ find . -type f -exec grep -H GetAngles {} \;
    ./include/G4GDMLWriteDefine.hh:    G4ThreeVector GetAngles(const G4RotationMatrix&);
    ./src/G4GDMLWriteDefine.cc:G4ThreeVector G4GDMLWriteDefine::GetAngles(const G4RotationMatrix& mtx)
    ./src/G4GDMLWriteParamvol.cc:   Angles=GetAngles(paramvol->GetObjectRotationValue());
    ./src/G4GDMLWriteParamvol.cc:                   GetAngles(paramvol->GetObjectRotationValue()));
    ./src/G4GDMLWriteSolids.cc:      G4ThreeVector rot = GetAngles(rotm);
    ./src/G4GDMLWriteSolids.cc:         firstrot += GetAngles(disp->GetObjectRotation());
    ./src/G4GDMLWriteSolids.cc:         rot += GetAngles(disp->GetObjectRotation());
    ./src/G4GDMLWriteStructure.cc:   const G4ThreeVector rot = GetAngles(rotate.getRotation());
    simon:gdml blyth$ 



::

    107 void G4GDMLWriteStructure::PhysvolWrite(xercesc::DOMElement* volumeElement,
    108                                         const G4VPhysicalVolume* const physvol,
    109                                         const G4Transform3D& T,
    110                                         const G4String& ModuleName)
    111 {
    112    HepGeom::Scale3D scale;
    113    HepGeom::Rotate3D rotate;
    114    HepGeom::Translate3D translate;
    115 
    116    T.getDecomposition(scale,rotate,translate);
    117 
    118    const G4ThreeVector scl(scale(0,0),scale(1,1),scale(2,2));
    119    const G4ThreeVector rot = GetAngles(rotate.getRotation());
    120    const G4ThreeVector pos = T.getTranslation();
    121 
    122    const G4String name = GenerateName(physvol->GetName(),physvol);
    123    const G4int copynumber = physvol->GetCopyNo();
    124 
    125    xercesc::DOMElement* physvolElement = NewElement("physvol");
    126    physvolElement->setAttributeNode(NewAttribute("name",name));
    127    if (copynumber) physvolElement->setAttributeNode(NewAttribute("copynumber", copynumber));
    128 
    129    volumeElement->appendChild(physvolElement);
    130 



GDML Manual Unhelpful
~~~~~~~~~~~~~~~~~~~~~~~~


3.2.5 Rotations

Rotations are usually defined in the beginning of the GDML file (in the define
section). Once defined, they can be referenced in place where rotations are
expected. Positive rotations are expected to be right-handed. A rotation can be
defined as in the following example:

::

   <rotation name="RotateZ" z=="30" unit="deg"/>


GLM Euler
-----------

/usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtx/euler_angles.inl

Translated  eulerAngleX, eulerAngleY, eulerAngleZ into my glm.py 





GDML/glTF Tracing Transforms
--------------------------------






 
