
Surface Debug : lacking hits due to surface/volume model mismatch 
====================================================================

Not getting any SURFACE_DETECT despite photons obviously traversing PMTs
because the requisite boundaries have no associated surfaces.

Surface flags::

   SURFACE_DETECT
   SURFACE_ABSORB
   SURFACE_DREFLECT
   SURFACE_SREFLECT 


generate.cu::

    281 
    282         command = propagate_to_boundary( p, s, rng );
    283         if(command == BREAK)    break ;           // BULK_ABSORB
    284         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    285         // PASS : survivors will go on to pick up one of the below flags, 
    286         
    287         
    288         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    289         {
    290             command = propagate_at_surface(p, s, rng);
    291             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    292             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    293         }   
    294         else
    295         {
    296             propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    297             // tacit CONTINUE
    298         }   


state.h::

     19 __device__ void fill_state( State& s, int boundary, int sensor, float wavelength )
     20 {
     21     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     22     // >0 outward going photon
     23     // <0 inward going photon
     24         
     25     int line = boundary > 0 ? (boundary - 1)*6 : (-boundary - 1)*6  ;
     26             
     27     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     28     //      
     29     int m1_line = boundary > 0 ? line + 0 : line + 1 ;   // inner-material / outer-material
     30     int m2_line = boundary > 0 ? line + 1 : line + 0 ;   // outer-material / inner-material
     31     int su_line = boundary > 0 ? line + 2 : line + 3 ;   // inner-surface  / outer-surface
     32             
     33     s.material1 = wavelength_lookup( wavelength, m1_line );
     34     s.material2 = wavelength_lookup( wavelength, m2_line ) ;
     35     s.surface   = wavelength_lookup( wavelength, su_line );
     36         
     37     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     38         
     39     s.index.x = optical_buffer[m1_line].x ;
     40     s.index.y = optical_buffer[m2_line].x ;
     41     s.index.z = optical_buffer[su_line].x ;
     42     s.index.w = sensor  ;
     43         
     44 }



Chroma Solution to same issue
--------------------------------

Addressed this with G4DAEChroma by adding "fake" surfaces 

* env/geant4/geometry/surfaces/surfaces_roundtrip.rst
* http://simoncblyth.bitbucket.org/env/notes/geant4/geometry/surfaces/surfaces_roundtrip/


::

    delta:env blyth$ hg shortlog | grep sensitive
    ef4ab750f29e | 2014-10-10 20:15:22 +0800 | simoncblyth: debug the tranlation of sensitive materials into surfaces for G4 to Chroma model translation
    4b938256ea50 | 2014-10-10 14:02:43 +0800 | simoncblyth: add extra SkinSurface and OpticalSurface objects to DAE level geometry in order to be transformed into sensitive surfaces needed for chroma SURFACE_DETECT
    d41b1d971f68 | 2014-10-09 21:02:37 +0800 | simoncblyth: working out how to reconcile the G4 and Chroma models regards sensitive detectors, in order to get Chroma to come up with photon hit data
    b192e176d992 | 2009-06-23 18:31:51 +0800 | simoncblyth: tg-quickstart 1st checkin of OfflineDB project ... untouched other than exclusions of sensitive {{{.ini}}} files
    3b26e5c356f4 | 2008-08-21 13:00:42 +0800 | simoncblyth: improved access control to sensitive variables
    101ef1dc0491 | 2007-12-24 08:16:45 +0800 | thho: sensitive skin opacity setting
    b04e2e8719ab | 2007-07-27 12:00:47 +0800 | simoncblyth: sensitive skin testing
    delta:env blyth$ 


* https://bitbucket.org/simoncblyth/env/commits/4b938256ea50
  
  * g4daenode.py:add_sensitive_surfaces

  * https://bitbucket.org/simoncblyth/env/src/tip/geant4/geometry/collada/g4daenode.py


* env/geant4/geometry/collada/g4daenode.py 

::

     395     @classmethod
     396     def add_sensitive_surfaces(cls, matid='__dd__Materials__Bialkali', qeprop='EFFICIENCY'):
     397         """
     398         Chroma expects sensitive detectors to have an Optical Surface 
     399         with channel_id associated.  
     400         Whereas Geant4 just has sensitive LV.
     401 
     402         This attempts to bridge from Geant4 to Chroma model 
     403         by creation of "fake" chroma skinsurfaces.  
     404 
     405         Effectively sensitive materials are translated 
     406         into sensitive surfaces 
     407 
     408         :: 
     409 
     410             In [57]: DAENode.orig.materials['__dd__Materials__Bialkali0xc2f2428'].extra
     411             Out[57]: <MaterialProperties keys=['RINDEX', 'EFFICIENCY', 'ABSLENGTH'] >
     412 
     413 
     414         #. Different efficiency for different cathodes ?
     415 
     416         """
     417         log.info("add_sensitive_surfaces matid %s qeprop %s " % (matid, qeprop))
     418         sensitive_material = cls.materialsearch(matid)
     419         assert sensitive_material
     420 
     421         if sensitive_material.extra is None:
     422             log.warn("sensitive_material.extra not available cannot sensitize ")
     423             return
     424 
     425         efficiency = sensitive_material.extra.properties[qeprop]
     426         assert not efficiency is None
     427 
     428         cls.sensitize(matid=matid)
     429 
     430         # follow convention used in G4DAE exports of using same names for 
     431         # the SkinSurface and the OpticalSurface it refers too
     432 
     433         for node in cls.sensitive_nodes:
     434             ssid = cls.sensitive_surface_id(node)
     435             volumeref = node.lv.id
     436 
     437             surf = OpticalSurface.sensitive(name=ssid, properties={qeprop:efficiency})
     438             cls.add_extra_opticalsurface(surf)
     439 
     440             skin = SkinSurface.sensitive(name=ssid, surfaceproperty=surf, volumeref=volumeref )
     441             cls.add_extra_skinsurface(skin)
     442         pass




How to do this with GGeo ?
---------------------------

Which level to add the fake cathode surfaces at ?

* AssimpGGeo::convertMaterials, creates and adds to GGeo instances of
  GOpticalSurface, GSkinSurface, GBorderSurface, GMaterial 
  based on the properties that the assimp "materials" have 

* AssimpGGeo::convertStructureVisit pulls GBoundary into existance 
  based on boundary identity combining imat/omat/isur/osur


::

    603     GSolid* solid = new GSolid(nodeIndex, gtransform, mesh, NULL, NULL ); // boundary and sensor start NULL
    604     solid->setLevelTransform(ltransform);
    605 
    606     const char* lv   = node->getName(0);
    607     const char* pv   = node->getName(1);
    608     const char* pv_p   = pnode->getName(1);
    609 
    610     gg->countMeshUsage(msi, nodeIndex, lv, pv);
    611 
    612     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
    613     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
    614     GSkinSurface*   sks = gg->findSkinSurface(lv);
    615 


Avoiding interference with this structure means would need to 
add the surfaces prior to AssimpGGeo::convertStructure

Approach using AssimpGGeo::convertSensors
--------------------------------------------

2 sensor skin surfaces are added::

   lvPmtHemiCathodeSensorSurface
   lvHeadonPmtCathodeSensorSurface

But only one shows up in boundarylib (may be due to identity digest not including the name)::

    ggv --blib

    boundary : index 21 x6 126 e554f1b518cd18fae063073e9147b70d Bialkali/Vacuum/-/lvPmtHemiCathodeSensorSurface 


Running does not yet yield any SURFACE_DETECT, but getting lots of SURFACE_SREFLECT::

    288         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    289         {
    290             command = propagate_at_surface(p, s, rng);
    291             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    292             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    293         }


    402 __device__ int
    403 propagate_at_surface(Photon &p, State &s, curandState &rng)
    404 {
    405 
    406     float u = curand_uniform(&rng);
    407 
    408     if( u < s.surface.y )   // absorb   
    409     {
    410         s.flag = SURFACE_ABSORB ;
    411         return BREAK ;
    412     }
    413     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    414     {
    415         s.flag = SURFACE_DETECT ;
    416         return BREAK ;
    417     }
    418     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    419     {
    420         s.flag = SURFACE_DREFLECT ;
    421         propagate_at_diffuse_reflector(p, s, rng);
    422         return CONTINUE;
    423     }
    424     else
    425     {
    426         s.flag = SURFACE_SREFLECT ;
    427         propagate_at_specular_reflector(p, s, rng );
    428         return CONTINUE;
    429     }
    430 }


Hmm setting efficiency to 1.0 still getting nothing other than SURFACE_SREFLECT

::

    delta:env blyth$ ggv --blib 126 127 128 129 130 131
    [2015-10-06 13:22:01.832171] [0x000007fff7448031] [warning] GBoundaryLib::setWavelengthBuffer didnt see 54, numBoundary: 57

    boundary : index  0 x6   0 019d50af046b6733287e43af2e8f7fa2 Vacuum/Vacuum/-/- 
    ...
    boundary : index 21 x6 126 31ec4ad900fe9b40be261fa11af380b7 Bialkali/Vacuum/-/lvPmtHemiCathodeSensorSurface 
    GBoundaryLib.dumpWavelengthBuffer 126 
    GBoundaryLib::dumpWavelengthBuffer wline 126 numSub 57 domainLength 39 numQuad 6 

     126 |  21/  0 __dd__Materials__Bialkali0xc2f2428 
               1.458           1.458           1.458           1.458           1.458           1.458           1.458           1.458
            1000.000        1000.000        1000.000        1077.339        1736.682        1393.428         821.650         529.476
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000

    GBoundaryLib.dumpWavelengthBuffer 127 
    GBoundaryLib::dumpWavelengthBuffer wline 127 numSub 57 domainLength 39 numQuad 6 

     127 |  21/  1 __dd__Materials__Vacuum0xbf9fcc0 
               1.000           1.000           1.000           1.000           1.000           1.000           1.000           1.000
        10000000.000    10000000.000    10000000.000    10000000.000    10000000.000    10000000.000    10000000.000    10000000.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000

    GBoundaryLib.dumpWavelengthBuffer 128 
    GBoundaryLib::dumpWavelengthBuffer wline 128 numSub 57 domainLength 39 numQuad 6 

     128 |  21/  2 - 
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000

    GBoundaryLib.dumpWavelengthBuffer 129 
    GBoundaryLib::dumpWavelengthBuffer wline 129 numSub 57 domainLength 39 numQuad 6 

     129 |  21/  3 __dd__Geometry__PMT__lvPmtHemiCathodeSensorSurface 
               1.000           1.000           1.000           1.000           1.000           1.000           1.000           1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000

    GBoundaryLib.dumpWavelengthBuffer 130 
    GBoundaryLib::dumpWavelengthBuffer wline 130 numSub 57 domainLength 39 numQuad 6 

     130 |  21/  4 - 
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000

    GBoundaryLib.dumpWavelengthBuffer 131 
    GBoundaryLib::dumpWavelengthBuffer wline 131 numSub 57 domainLength 39 numQuad 6 

     131 |  21/  5 - 
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
              -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000          -1.000
    delta:env blyth$ 


Hmm is inner/outer surface swapped somewhere ? Dont think so.  


Possibly a problem with PMT normals ?
---------------------------------------

Suspect issue with PMT front face normals. The Q normal view shows no normals coming out the front of PMTs 

::

    43 (v  482 f  960 )  (t    1 oe    0) : x    98.143 : n   672 : n*v 323904 :                         pmt-hemi-cathode : 3201,3207,3213,3219,3225, 
    44 (v  242 f  480 )  (t    1 oe    0) : x    98.143 : n   672 : n*v 162624 :                             pmt-hemi-bot : 3202,3208,3214,3220,3226, 
    45 (v   50 f   96 )  (t    1 oe    0) : x    83.000 : n   672 : n*v  33600 :                          pmt-hemi-dynode : 3203,3209,3215,3221,3227, 
    46 (v  338 f  672 )  (t    1 oe    0) : x   146.252 : n   672 : n*v 227136 :                             pmt-hemi-vac : 3200,3206,3212,3218,3224, 
    47 (v  362 f  720 )  (t    1 oe    0) : x   149.997 : n   672 : n*v 243264 :                                 pmt-hemi : 3199,3205,3211,3217,3223, 


Wow 960 faces for the cathode ? 

Add *mdyb* for checking pmt-hemi-cathode geometry, its a flikering mess and a cats cradle of normals::

    ggv --mdyb -G --noinstanced

    ggv --mdyb -O 
    udp.py --target 3201

    ggv --mdyb --torchconfig="pos_target=3201;pos_offset=500,0,0"

    ggv --mdyb --torchconfig "pos_target=3201;pos_offset=800,0,0;radius=100"

       # hmm dont see photons that miss

    ggv --mdyb --torchconfig "pos_target=3201;pos_offset=0,1000,0;radius=100;direction=0,-1,0" --geocenter

       # targetting the beam is not easier as can only see the photons when they hit 

    ggv --mdyb --torchconfig "pos_target=3154;radius=3000;direction=0,0,-1" 

       # added SST but do not see records that just get absorbed either
       # that means are propagating in a lump of Steel

    ggv --mdyb --torchconfig "pos_target=3154;radius=3000;direction=0,0,-1" --save

       # export GGEOVIEW_QUERY="range:3201:3202,range:3153:3154"   # 2 volumes : first pmt-hemi-cathode and ADE  
       # change envelope volume to ADE much better, as photons get somewhere in IwsWater/IwsWater  


    ggv --mdyb --torchconfig "frame=3201;source=0,0,1000;target=0,0,0;radius=300;" --save

       # head on beam strarting 1m out in front of PMT cathode
       #
       # using reworked the Torch configuration to be frame based with source and target positions
       # specified in the identified frame 
       #
       # note effect of material inconsistency, photons destined to hit the cathode
       # think they are in a vacuum, hence they lead ahead of those destined to hit ADE envelope


Hmm would be easiest to target the PMT using its own frame, hmm view targetting did something similar ?



Detdesc dive
--------------

Looks like need to replace the cathode with something simpler ?


G5:/home/blyth/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-pmt.xml::

    118   <!-- The Photo Cathode -->
    119   <!-- use if limit photocathode to a face on diameter gt 167mm. -->
    120   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    121     <union name="pmt-hemi-cathode">
    122       <sphere name="pmt-hemi-cathode-face"
    123           outerRadius="PmtHemiFaceROCvac"
    124           innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    125           deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    ///          
    ///                  PmtHemiFaceROC-PmtHemiGlassThickness : 131. - 3. = 128.         
    ///                                                       128. - 0.05 = 127.95 
    ///
    126       <sphere name="pmt-hemi-cathode-belly"
    127           outerRadius="PmtHemiBellyROCvac"
    128           innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"
    129           startThetaAngle="PmtHemiBellyCathodeAngleStart"
    130           deltaThetaAngle="PmtHemiBellyCathodeAngleDelta"/>
    ///
    ///         
    ///
    ///
    131       <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
    ///
    ///             56. - 17. = 39.
    ///    
    132     </union>
    133   </logvol>


G5:/home/blyth/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-parameters.xml::

    010 <!-- Radius of curvature of face of PMT, HM catalog -->
     11 <parameter name="PmtHemiFaceROC" value="131*mm"/>
     12 
     13 <!-- Radius of curvature of top and bottom belly parts, average of Tak Pui's numbers -->
     14 <parameter name="PmtHemiBellyROC" value="102*mm"/>
     15 
     16 <!-- Offset of face hemisphere -->
     17 <!-- <parameter name="PmtHemiFaceOff" value="60*mm"/> -->
     18 <!-- Shrink offset from Tak Pui's numbers to better fit G4dyb hitz vs. hity -->
     19 <parameter name="PmtHemiFaceOff" value="56*mm"/>
     20 
     21 <!-- Offset of top/bottom belly hemispheres, average of Tak Pui's numbers -->
     22 <!-- <parameter name="PmtHemiBellyOff" value="17*mm"/> -->
     23 <!-- Shrink offset from Tak Pui's numbers to better fit G4dyb hitz vs. hity -->
     24 <parameter name="PmtHemiBellyOff" value="13*mm"/>
     25 
     26 <!-- Radius of cylindrical glass base, HM catalog -->
     27 <parameter name="PmtHemiGlassBaseRadius" value="42.25*mm"/>
     28 
     29 <!-- Radius of opaque "dynode" -->
     30 <parameter name="PmtHemiDynodeRadius" value="27.5*mm"/>

     44 <!-- Thickness of the glass, from GLG4sim numbers -->
     45 <parameter name="PmtHemiGlassThickness" value="3*mm"/>
     46 
     47 <!-- Thickness of the photo cathode, this is a made up number -->
     48 <parameter name="PmtHemiCathodeThickness" value="0.05*mm"/>
     ..
     68 
     69 <!-- Radius of curvature of vacuum side of face of PMT, HM catalog -->
     70 <parameter name="PmtHemiFaceROCvac" value="PmtHemiFaceROC-PmtHemiGlassThickness"/>
     //                                                 131. - 3. = 128. 
     //
     72 <!-- Radius of curvature of vacuum side of top and bottom belly parts, average of Tak Pui's numbers -->
     73 <parameter name="PmtHemiBellyROCvac" value="PmtHemiBellyROC-PmtHemiGlassThickness"/>
     //                                                   102.-3. = 99.
     74 
     75 
     76 <!-- 
     77      a = PmtHemiFaceROCvac
     78      b = PmtHemiBellyROCvac
     79      d = (PmtHemiFaceOff-PmtHemiBellyOff)
     80 
     81      y = PmtHemiFaceTopOff = distance from center of top belly hemi to
     82      z location of interface between top and face hemis.
     83 
     84  -->
     85 
     86 <parameter name="PmtHemiFaceTopOff" value="(PmtHemiFaceROCvac^2-PmtHemiBellyROCvac^2-(PmtHemiFaceOff-PmtHemiBellyOff)^2)/(2*(PmtHemiFaceOff-PmtHemiBellyOff))"/>
     //                                                (128.*128.- 99.*99. - (56.-13.)*(56.-13.))/(2.*(56.-13.))
     //
     //       In [1]: (128.*128.- 99.*99. - (56.-13.)*(56.-13.))/(2.*(56.-13.))
     //       Out[1]: 55.04651162790697
     //   
     //     
     //
     87 
     88 <!-- Angular extent of photocathode on face 
     89      acos((y+b)/a)
     90 -->
     91 <parameter name="PmtHemiFaceCathodeAngle" value="0.5*degree+radian*acos((PmtHemiFaceTopOff+(PmtHemiFaceOff-PmtHemiBellyOff))/PmtHemiFaceROCvac)"/>
     //
     //          math.acos((55.0465+(56.-13.))/128.)
     //
     // In [8]: 0.5+math.acos((55.0465+(56.-13.))/128.)*180./math.pi
     // Out[8]: 40.50500580674586
     //
     92 
     93 <!-- Start angle for photocathode on belly 
     94      acos(y/b)
     95 -->
     96 <parameter name="PmtHemiBellyCathodeAngleStart" value="-0.5*degree+radian*acos(PmtHemiFaceTopOff/PmtHemiBellyROCvac)"/>
     97 
     98 <!-- Stop angle for photocathode on belly 
     99      asin(PC diameter / 2 / a)
    100 -->
    101 <!-- 
    102 <parameter name="PmtHemiBellyCathodeAngleDelta" value="radian*asin(0.5*PmtHemiCathodeDiameter/PmtHemiBellyROCvac)-PmtHemiBellyCathodeAngleStart"/>
    103  -->
    104 <parameter name="PmtHemiBellyCathodeAngleDelta" value="PmtHemiBellyIntAngle-PmtHemiBellyCathodeAngleStart"/>
    105 
    106 <!-- Angle where belly spheres intersect -->
    107 <parameter name="PmtHemiBellyIntAngle" value="acos(PmtHemiBellyOff/PmtHemiBellyROCvac)*radian"/>
    108 




Sphere Sphere Intersection
----------------------------

* http://mathworld.wolfram.com/Sphere-SphereIntersection.html


How to try some simple replacement cathode ?
-----------------------------------------------

* adding analytic spheres to OptiX at the positions corresponding 
  to front face of cathode : would allow a simple geometry check  

::

    OGeo::makeGeometryInstance(GMergedMesh* mergedmesh)


Five volumes within repeated PMT instance::

    [2015-Oct-07 12:26:54.103163]:info: GGeo::dumpNodeInfo mmindex 1 solids 5
        720    362   3199   3155 lv            __dd__Geometry__PMT__lvPmtHemi0xc133740 pv __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtA.......--pvAdPmtUnit--pvAdPmt0xc2a6b40  
        672    338   3200   3199 lv      __dd__Geometry__PMT__lvPmtHemiVacuum0xc2c7cc8 pv __dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xc1340e8  
        960    482   3201   3200 lv     __dd__Geometry__PMT__lvPmtHemiCathode0xc2cdca0 pv __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xc02c380  
        480    242   3202   3200 lv      __dd__Geometry__PMT__lvPmtHemiBottom0xc12ad60 pv __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiBottom0xc21de78  
         96     50   3203   3200 lv      __dd__Geometry__PMT__lvPmtHemiDynode0xc02b280 pv __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiDynode0xc04ad28  


Note identity relative transform for 1st three::

    In [5]: n = np.load("nodeinfo.npy")

    In [6]: n
    Out[6]: 
    array([[ 720,  362, 3199, 3155],
           [ 672,  338, 3200, 3199],
           [ 960,  482, 3201, 3200],
           [ 480,  242, 3202, 3200],
           [  96,   50, 3203, 3200]], dtype=uint32)



    In [1]: t = np.load("transforms.npy")

    In [4]: t.reshape(-1,4,4)
    Out[4]: 
    array([[[  1. ,   0. ,   0. ,   0. ],
            [  0. ,   1. ,   0. ,   0. ],
            [  0. ,   0. ,   1. ,   0. ],
            [  0. ,   0. ,   0. ,   1. ]],

           [[  1. ,   0. ,   0. ,   0. ],
            [  0. ,   1. ,   0. ,   0. ],
            [  0. ,   0. ,   1. ,   0. ],
            [  0. ,   0. ,   0. ,   1. ]],

           [[  1. ,   0. ,   0. ,   0. ],
            [  0. ,   1. ,   0. ,   0. ],
            [  0. ,   0. ,   1. ,   0. ],
            [  0. ,   0. ,   0. ,   1. ]],

           [[  1. ,   0. ,   0. ,   0. ],
            [  0. ,   1. ,   0. ,   0. ],
            [  0. ,   0. ,   1. ,   0. ],
            [  0. ,   0. ,  69. ,   1. ]],

           [[  1. ,   0. ,   0. ,   0. ],
            [  0. ,   1. ,   0. ,   0. ],
            [  0. ,   0. ,   1. ,   0. ],
            [  0. ,   0. , -81.5,   1. ]]], dtype=float32)




Analytic OptiX geometry 
------------------------

Per triangle buffers with boundaries, nodes and sensors are used by TriangleMesh to set attributes based on primIdx::

    In [1]: b = np.load("boundaries.npy")

    In [2]: b
    Out[2]: 
    array([[11],
           [11],
           [11],
           ..., 
           [12],
           [12],
           [12]], dtype=int32)

    In [3]: b.shape
    Out[3]: (434816, 1)

    In [4]: n = np.load("nodes.npy")

    In [5]: n.shape
    Out[5]: (434816, 1)

    In [6]: n
    Out[6]: 
    array([[ 3153],
           [ 3153],
           [ 3153],
           ..., 
           [12220],
           [12220],
           [12220]], dtype=int32)

    In [7]: s = np.load("sensors.npy")

    In [8]: s.shape
    Out[8]: (434816, 1)

    In [9]: s
    Out[9]: 
    array([[3154],
           [3154],
           [3154],
           ..., 


Whats the equivalent for instanced analytic geometry ? 

In the full analytic treatment might have 10-20 primitives per instance
arranged into a CSG tree of boolean operations and transforms.
Although there are only 5 volumes there are multiple primitives (spheres, cones, boxes)
inside each.

On top of identifying the primitive also have the instance index.

So need an analytic index::

    instance_index*numPrim + prim_index



Triangulated Case
-------------------

OGeo.cc::

    283     optix::Geometry geometry = m_context->createGeometry();
    284     RayTraceConfig* cfg = RayTraceConfig::getInstance();
    285     geometry->setIntersectionProgram(cfg->createProgram("TriangleMesh.cu.ptx", "mesh_intersect"));
    286     geometry->setBoundingBoxProgram(cfg->createProgram("TriangleMesh.cu.ptx", "mesh_bounds"));
    ...
    296     geometry->setPrimitiveCount(numFaces);


Gross structure of geometry communicated to OptiX by returning
bounding boxes from the *BoundingBoxProgram* for each primIdx. 
The range of primIdx is specified by *setPrimitiveCount* 

When a ray intersects with a bbox the associated *primIdx* is 
used to invoke the *IntersectionProgram* which 
reports the parametric t with *rtPotentialIntersection(t)*

cu/TriangleMesh.cu::

    96 RT_PROGRAM void mesh_bounds (int primIdx, float result[6])
    34 RT_PROGRAM void mesh_intersect(int primIdx)


With instanced geometry::

    166 optix::Group OGeo::makeRepeatedGroup(GMergedMesh* mm, unsigned int limit)
    167 {
    168     GBuffer* tbuf = mm->getITransformsBuffer();
    169     unsigned int numTransforms = limit > 0 ? std::min(tbuf->getNumItems(), limit) : tbuf->getNumItems() ;
    170     assert(tbuf && numTransforms > 0);
    171 
    172     LOG(info) << "OGeo::makeRepeatedGroup numTransforms " << numTransforms ;
    173 
    174     float* tptr = (float*)tbuf->getPointer();
    175 
    176     optix::Group group = m_context->createGroup();
    177     group->setChildCount(numTransforms);
    178 
    179     optix::GeometryInstance gi = makeGeometryInstance(mm);
    180     optix::GeometryGroup repeated = m_context->createGeometryGroup();
    181     repeated->addChild(gi);
    182     repeated->setAcceleration( makeAcceleration() );
    ///
    ///   can an id be planted in GeometryGroup  ?
    ///   seems not but can with GeometryInstance according to docs, 
    ///   so need to adjust to having a GeometryInstance for every xform
    ///   to plant an instance index
    ///
    183 
    184     bool transpose = true ;
    185     for(unsigned int i=0 ; i<numTransforms ; i++)
    186     {
    187         optix::Transform xform = m_context->createTransform();
    188         group->setChild(i, xform);
    189         xform->setChild(repeated);
    190         const float* tdata = tptr + 16*i ;
    191         optix::Matrix4x4 m(tdata) ;
    192         xform->setMatrix(transpose, m.getData(), 0);
    193         //dump("OGeo::makeRepeatedGroup", m.getData());
    194     }
    195     return group ;
    196 }
        

Hmm how with instanced geometry to know which instance was 
landed on, because all the geometry info lives within the
instance island ? 

* https://devtalk.nvidia.com/default/topic/541450/?comment=3791463

::

    Is there an easy way to know withing the closest hit or any hit program which
    object was hit? I'd prefer to use a single material for all but can encode the
    information into the material. I do however wish to use the same hit program to
    be flexible with the number of objects.


    (Detlef Roettger)
    This should be pretty easy, if there aren't any other circumstances involved.

    - If you have exactly one Transform node per object, 
      let your Material have a variable rtDeclareVariable(unsigned int, objectID, , );
    - In your given node hierarchy you load the Geometry only once, 
      but assign different Material parameters (=> objectID) per hierarchy path 
      to each GeometryInstance (i.e. per Transform) to identify the Geometry as you wish.
    - To report back to the ray generation program that you hit a specific object, 
      add a member unsigned int objectID; to your custom PerRayData payload.
    - Inside the closest-hit program write the objectID from the material 
      into the objectID of the current ray payload. 
    - Inside the ray generation program add a switch-case which writes 
      the other PerRayData results you generated into the output 
      buffer you select with the PerRayData objectID.

