
Analytic Scene
==================

Looking uptree from lvPmtHemi see many nodes 
without geometry and just physvol/logvol striping, 
does this need a separate "placement" tree ? Prepping 
for OptiX/OpenGL instancing ? 

* TODO: expand ddbase.py to handle this 


Checking users of lvPmtHemi AdPmts/geometry.xml uses paramphysvol to instance it around the rings etc..::

    simon:pmt blyth$ pmt-dfind lvPmtHemi

    ./AdPmts/geometry.xml:  <physvol name="pvAdPmtUnit" logvol="/dd/Geometry/PMT/lvPmtHemiwPmtHolder">


    simon:DDDB blyth$ pmt-dfind lvPmtHemiwPmtHolder
    ./AdPmts/geometry.xml:  <physvol name="pvAdPmtUnit" logvol="/dd/Geometry/PMT/lvPmtHemiwPmtHolder">
    ./PMT/geometry.xml:    <logvolref href="hemi-pmt.xml#lvPmtHemiwPmtHolder"/>
    ./PMT/hemi-pmt.xml:  <logvol name="lvPmtHemiwPmtHolder">
    simon:DDDB blyth$ 

AdPmts/geometry.xml::


     33   <logvol name="lvAdPmtRing">
     34     <paramphysvol number="AdPmtNperRing">
     35       <physvol name="pvAdPmtInRing:1" logvol="/dd/Geometry/AdPmts/lvAdPmtUnit" />
     36       <posXYZ/>
     37       <rotXYZ rotZ="AdPmtAngularSep" />
     38     </paramphysvol>
     39   </logvol>
     40 
     41   <logvol name="lvAdPmtArrayZero">
     42     <paramphysvol number="AdPmtNrings">
     43       <physvol name="pvAdPmtRingInCyl:1" logvol="/dd/Geometry/AdPmts/lvAdPmtRing"/>
     44       <posXYZ z="AdPmtZsep"/>
     45     </paramphysvol>
     46   </logvol>




Detdesc generation
--------------------

Detdesc generation complicates things greatly... Take a look at GDML

/usr/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/python/XmlDetDescGen



Multi File Detdesc Parsing with ddbase.py ?
-----------------------------------------------

* https://wiki.bnl.gov/dayabay/index.php?title=Tutorial:DetDesc/XML_Files


Detdesc using catalog to handle cross-file ? Always ? 

::

    simon:PMT blyth$ pmt-dfind hemi-pmt.xml
    ./PMT/geometry.xml:    <logvolref href="hemi-pmt.xml#lvPmtHemiFrame"/>
    ./PMT/geometry.xml:    <logvolref href="hemi-pmt.xml#lvPmtHemi"/>
    ./PMT/geometry.xml:    <logvolref href="hemi-pmt.xml#lvPmtHemiwPmtHolder"/>
    ./PMT/geometry.xml:    <logvolref href="hemi-pmt.xml#lvAdPmtCollar"/>
    ...

     06 <DDDB>
      7 
      8   <catalog name="PMT">
      9 
     10     <logvolref href="hemi-pmt.xml#lvPmtHemiFrame"/>
     11     <logvolref href="hemi-pmt.xml#lvPmtHemi"/>
     12     <logvolref href="hemi-pmt.xml#lvPmtHemiwPmtHolder"/>
     13     <logvolref href="hemi-pmt.xml#lvAdPmtCollar"/>
     14     <logvolref href="hemi-pmt.xml#lvPmtHemiCathode"/>
     15     <logvolref href="hemi-pmt.xml#lvPmtHemiVacuum"/>
     16     <logvolref href="hemi-pmt.xml#lvPmtHemiBottom"/>



     

