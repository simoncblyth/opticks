Analytic Geometry
===================


Principal
----------

Triangle meshes are a convenient initial step to the GPU 
as all geometry can be treated with the same code.
Special treatment of important geometry (PMTs) however
is expected to have large performance gains.

Ray intersection with CSG solids boils down to 
analytic solving quadratic/cubic polynomials. There is 
a technique to handle union intersections by applying boolean operations
to intersection segments of the sub volumes. 


How to proceed ?
------------------

* on revisiting G4DAE include GDML G4 CSG model description together
  with the triangulated COLLADA 


detdesc PMT is involved
------------------------

Complicated assemblies of CSG solids. Implementing analytic is non-trivial.

G5:/home/blyth/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/geometry.xml::

     08   <catalog name="PMT">
     09 
     10     <logvolref href="hemi-pmt.xml#lvPmtHemiFrame"/>
     11     <logvolref href="hemi-pmt.xml#lvPmtHemi"/>
     12     <logvolref href="hemi-pmt.xml#lvPmtHemiwPmtHolder"/>
     13     <logvolref href="hemi-pmt.xml#lvAdPmtCollar"/>
     14     <logvolref href="hemi-pmt.xml#lvPmtHemiCathode"/>
     15     <logvolref href="hemi-pmt.xml#lvPmtHemiVacuum"/>
     16     <logvolref href="hemi-pmt.xml#lvPmtHemiBottom"/>
     ..

dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-pmt.xml::

     37   <!-- The PMT glass -->
     38   <logvol name="lvPmtHemi" material="Pyrex">
     39     <union name="pmt-hemi">
     40       <intersection name="pmt-hemi-glass-bulb">
     41     <sphere name="pmt-hemi-face-glass"
     42         outerRadius="PmtHemiFaceROC"/>
     43 
     44     <sphere name="pmt-hemi-top-glass"
     45         outerRadius="PmtHemiBellyROC"/>
     46     <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
     47 
     48     <sphere name="pmt-hemi-bot-glass"
     49         outerRadius="PmtHemiBellyROC"/>
     50     <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
     51 
     52       </intersection>
     53       <tubs name="pmt-hemi-base"
     54         sizeZ="PmtHemiGlassBaseLength"
     55         outerRadius="PmtHemiGlassBaseRadius"/>
     56       <posXYZ z="-0.5*PmtHemiGlassBaseLength"/>
     57     </union>
     58 
     59     <physvol name="pvPmtHemiVacuum"
     60          logvol="/dd/Geometry/PMT/lvPmtHemiVacuum"/>
     61 
     62   </logvol>


::

    118   <!-- The Photo Cathode -->
    119   <!-- use if limit photocathode to a face on diameter gt 167mm. -->
    120   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    121     <union name="pmt-hemi-cathode">
    122       <sphere name="pmt-hemi-cathode-face"
    123           outerRadius="PmtHemiFaceROCvac"
    124           innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    125           deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    126       <sphere name="pmt-hemi-cathode-belly"
    127           outerRadius="PmtHemiBellyROCvac"
    128           innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"
    129           startThetaAngle="PmtHemiBellyCathodeAngleStart"
    130           deltaThetaAngle="PmtHemiBellyCathodeAngleDelta"/>
    131       <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
    132     </union>
    133   </logvol>









