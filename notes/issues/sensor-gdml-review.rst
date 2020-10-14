sensor-gdml-review
======================

Parent
--------

* :doc:`sensor-review`


GDML Level : opticksaux-jv5-vi
-------------------------------


opticksaux-jv5-cd::

    epsilon:juno1808 blyth$ grep -n EFFICIENCY g4_00_v5.gdml
    50:    <matrix coldim="2" name="EFFICIENCY0x4b9feb0" values="1.55e-06 0.002214 1.77143e-06 0.002214 1.7971e-06 0.003426 1.82353e-06 0.005284 1.85075e-06 0.007921 1.87879e-06 0.011425 1.90769e-06 0.015808 1.9375e-06 0.021143 1.96825e-06 0.026877 2e-06 0.033344 2.03279e-06 0.040519 2.06667e-06 0.048834 2.10169e-06 0.057679 2.13793e-06 0.067843 2.17544e-06 0.079047 2.21429e-06 0.091286 2.25454e-06 0.104205 2.2963e-06 0.119611 2.33962e-06 0.135205 2.38462e-06 0.154528 2.43137e-06 0.17464 2.48e-06 0.194504 2.53061e-06 0.210267 2.58333e-06 0.223053 2.6383e-06 0.234931 2.69565e-06 0.248108 2.75556e-06 0.26528 2.81818e-06 0.281478 2.88372e-06 0.293765 2.95238e-06 0.30198 3.02439e-06 0.302932 3.1e-06 0.303274 3.17949e-06 0.299854 3.26316e-06 0.285137 3.35135e-06 0.270132 3.44444e-06 0.252713 3.54286e-06 0.227767 3.64706e-06 0.192104 3.75758e-06 0.143197 3.875e-06 0.063755 4e-06 0.015229 4.13333e-06 0.007972 1.55e-05 1e-06"/>
    63:    <matrix coldim="2" name="EFFICIENCY0x4ba1f40" values="1.55e-06 1e-05 1.737e-06 0.00159 1.769e-06 0.00255 1.791e-06 0.00355 1.808e-06 0.00469 1.825e-06 0.00605 1.844e-06 0.00774 1.864e-06 0.01003 1.884e-06 0.01325 1.904e-06 0.01718 1.923e-06 0.02059 1.947e-06 0.02608 1.978e-06 0.03229 2.008e-06 0.0396 2.041e-06 0.0479 2.069e-06 0.0548 2.104e-06 0.06387 2.141e-06 0.07797 2.174e-06 0.09129 2.211e-06 0.10541 2.251e-06 0.12003 2.303e-06 0.13668 2.361e-06 0.15564 2.41e-06 0.17078 2.462e-06 0.19267 2.522e-06 0.21437 2.595e-06 0.23089 2.675e-06 0.24073 2.771e-06 0.24868 2.857e-06 0.24983 2.954e-06 0.24753 3.04e-06 0.24185 3.147e-06 0.23304 3.248e-06 0.22351 3.355e-06 0.20848 3.482e-06 0.19001 3.594e-06 0.1692 3.661e-06 0.14451 3.744e-06 0.12059 3.78e-06 0.09924 3.831e-06 0.07906 3.868e-06 0.06154 3.912e-06 0.04971 3.956e-06 0.0396 4.002e-06 0.03126 4.043e-06 0.02525 4.09e-06 0.01894 4.122e-06 0.01516 4.161e-06 0.01185 4.194e-06 0.00893 4.222e-06 0.0067 4.251e-06 0.00521 4.286e-06 0.004 4.315e-06 0.00307 4.363e-06 0.00229 4.394e-06 0.00181 4.437e-06 0.00137 6.2e-06 1e-05 1.033e-05 1e-05 1.55e-05 1e-05"/>
    709:      <property name="EFFICIENCY" ref="EFFICIENCY0x4b9feb0"/>
    839:      <property name="EFFICIENCY" ref="EFFICIENCY0x4ba1f40"/>
    epsilon:juno1808 blyth$ 


* matrix element in define block at head
* reference to the matrix from opticalsurface within the solids

::

   706     <opticalsurface finish="0" model="0" name="Photocathode_opsurf" type="0" value="1">
   707       <property name="RINDEX" ref="RINDEX0x4b9f4f0"/>
   708       <property name="REFLECTIVITY" ref="REFLECTIVITY0x4b9fd70"/>
   709       <property name="EFFICIENCY" ref="EFFICIENCY0x4b9feb0"/>
   710       <property name="GROUPVEL" ref="GROUPVEL0x4b9fac0"/>
   711       <property name="KINDEX" ref="KINDEX0x4b9fc30"/>
   712       <property name="THICKNESS" ref="THICKNESS0x4ba0250"/>
   713     </opticalsurface>

::

    epsilon:juno1808 blyth$ grep -n Photocathode_opsurf g4_00_v5.gdml
    706:    <opticalsurface finish="0" model="0" name="Photocathode_opsurf" type="0" value="1">
    836:    <opticalsurface finish="0" model="0" name="Photocathode_opsurf_3inch" type="0" value="1">
    289595:    <bordersurface name="PMT_20inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf">
    289603:    <bordersurface name="PMT_20inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf">
    289607:    <bordersurface name="PMT_3inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf_3inch">
    289615:    <bordersurface name="PMT_3inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf_3inch">


Four bordersurface elements amongst the structure volumes have @surfaceproperty referencing the opticalsurface::

    289595*    <bordersurface name="PMT_20inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf">
    289596       <physvolref ref="PMT_20inch_inner1_phys0x4c9a870"/>
    289597       <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    289598     </bordersurface>

    289603*    <bordersurface name="PMT_20inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf">
    289604       <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    289605       <physvolref ref="PMT_20inch_inner1_phys0x4c9a870"/>
    289606     </bordersurface>
    ///////                      note same pv pair in opposite order to above 

    289607*    <bordersurface name="PMT_3inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf_3inch">
    289608       <physvolref ref="PMT_3inch_inner1_phys0x510beb0"/>
    289609       <physvolref ref="PMT_3inch_body_phys0x510be30"/>
    289610     </bordersurface>

    289615*    <bordersurface name="PMT_3inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf_3inch">
    289616       <physvolref ref="PMT_3inch_body_phys0x510be30"/>
    289617       <physvolref ref="PMT_3inch_inner1_phys0x510beb0"/>
    289618     </bordersurface>
    /////////////////          again same pv pair in opposite order


Note no referencing to the logsurf, they come into play for the pv pairs they are associated with::

    epsilon:juno1808 blyth$ grep -n logsurf g4_00_v5.gdml
    289595:    <bordersurface name="PMT_20inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf">
    289599:    <bordersurface name="PMT_20inch_mirror_logsurf1" surfaceproperty="Mirror_opsurf">
    289603:    <bordersurface name="PMT_20inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf">
    289607:    <bordersurface name="PMT_3inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf_3inch">
    289611:    <bordersurface name="PMT_3inch_absorb_logsurf1" surfaceproperty="Absorb_opsurf">
    289615:    <bordersurface name="PMT_3inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf_3inch">
    289619:    <bordersurface name="PMT_3inch_absorb_logsurf3" surfaceproperty="Absorb_opsurf">

Find the pv::

    epsilon:juno1808 blyth$ grep -n PMT_20inch_body_phys g4_00_v5.gdml
    1555:      <physvol name="PMT_20inch_body_phys0x4c9a7f0">
    289597:      <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    289601:      <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    289604:      <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    epsilon:juno1808 blyth$ 

Look for the pv pairs::

    289599     <bordersurface name="PMT_20inch_mirror_logsurf1" surfaceproperty="Mirror_opsurf">
    289600       <physvolref ref="PMT_20inch_inner2_phys0x4c9a920"/>
    289601       <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    289602     </bordersurface>

    289603*    <bordersurface name="PMT_20inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf">
    289604       <physvolref ref="PMT_20inch_body_phys0x4c9a7f0"/>
    289605       <physvolref ref="PMT_20inch_inner1_phys0x4c9a870"/>
    289606     </bordersurface>
    ///////                      note same pv in opposite order to above 


Below two Pyrex/Vacuum borders are the bordersurface pairs:: 

    1552     <volume name="PMT_20inch_log0x4cb3bb0">
    1553       <materialref ref="Pyrex0x4bae2a0"/>
    1554       <solidref ref="PMT_20inch_pmt_solid0x4c81b40"/>
    1555       <physvol name="PMT_20inch_body_phys0x4c9a7f0">           ##
    1556         <volumeref ref="PMT_20inch_body_log0x4cb3aa0"/>

          1542     <volume name="PMT_20inch_body_log0x4cb3aa0">
          1543       <materialref ref="Pyrex0x4bae2a0"/>
          1544       <solidref ref="PMT_20inch_body_solid0x4c90e50"/>
          1545       <physvol name="PMT_20inch_inner1_phys0x4c9a870">   ##
          1546         <volumeref ref="PMT_20inch_inner1_log0x4cb3cc0"/>

              1534     <volume name="PMT_20inch_inner1_log0x4cb3cc0">
              1535       <materialref ref="Vacuum0x4b9b630"/>
              1536       <solidref ref="PMT_20inch_inner1_solid0x4cb3610"/>
              1537     </volume>

          1547       </physvol>

          1548       <physvol name="PMT_20inch_inner2_phys0x4c9a920">    ##
          1549         <volumeref ref="PMT_20inch_inner2_log0x4c9a6e0"/>

              1538     <volume name="PMT_20inch_inner2_log0x4c9a6e0">
              1539       <materialref ref="Vacuum0x4b9b630"/>
              1540       <solidref ref="PMT_20inch_inner2_solid0x4cb3870"/>
              1541     </volume>

          1550       </physvol>
          1551     </volume>

    1557       </physvol>
    1558     </volume>


::


    epsilon:juno1808 blyth$ grep -n PMT_20inch_log0x4cb3bb0 g4_00_v5.gdml
    1552:    <volume name="PMT_20inch_log0x4cb3bb0">
    1566:        <volumeref ref="PMT_20inch_log0x4cb3bb0"/>

    001559     <volume name="lMaskVirtual0x4c803b0">
      1560       <materialref ref="Water0x4bb9ba0"/>
      1561       <solidref ref="sMask_virtual0x4c36e10"/>
      1562       <physvol name="pMask0x4c3bf20">
      1563         <volumeref ref="lMask0x4ca3960"/>
      1564       </physvol>
      1565       <physvol name="PMT_20inch_log_phys0x4ca16b0">
      1566         <volumeref ref="PMT_20inch_log0x4cb3bb0"/>
      1567       </physvol>
      1568     </volume>


Thence find the 20k of those::

    epsilon:juno1808 blyth$ grep -n lMaskVirtual0x4c803b0 g4_00_v5.gdml | wc -l
       20047



DYB GDML : opticksaux-dx0-vi
--------------------------------------

::

    grep -n EFFICIENCY g4_00_CGeometry_export_v0.gdml

    11:    <matrix coldim="2" name="EFFICIENCY0x1e132a0" values="1.512e-06 0 1.5498e-06 0 1.58954e-06 0 1.63137e-06 0 1.67546e-06 0 1.722e-06 0 1.7712e-06 0 1.8233e-06 0 1.87855e-06 0 1.93725e-06 0 1.99974e-06 0 2.0664e-06 0 2.13766e-06 0 2.214e-06 0 2.296e-06 0 2.38431e-06 0 2.47968e-06 0 2.583e-06 0 2.69531e-06 0 2.81782e-06 0 2.952e-06 0 3.0996e-06 0 3.26274e-06 0 3.44401e-06 0 3.64659e-06 0 3.87451e-06 0 4.13281e-06 0 4.42801e-06 0 4.76862e-06 0 5.16601e-06 0 5.63564e-06 0 6.19921e-06 0 6.88801e-06 0 7.74901e-06 0 8.85601e-06 0 1.0332e-05 0 1.23984e-05 0 1.5498e-05 0 2.0664e-05 0"/>
    77:    <matrix coldim="2" name="EFFICIENCY0x1d79780" values="1.512e-06 0.0001 1.5498e-06 0.0001 1.58954e-06 0.000440306 1.63137e-06 0.000782349 1.67546e-06 0.00112439 1.722e-06 0.00146644 1.7712e-06 0.00180848 1.8233e-06 0.00272834 1.87855e-06 0.00438339 1.93725e-06 0.00692303 1.99974e-06 0.00998793 2.0664e-06 0.0190265 2.13766e-06 0.027468 2.214e-06 0.0460445 2.296e-06 0.0652553 2.38431e-06 0.0849149 2.47968e-06 0.104962 2.583e-06 0.139298 2.69531e-06 0.170217 2.81782e-06 0.19469 2.952e-06 0.214631 3.0996e-06 0.225015 3.26274e-06 0.24 3.44401e-06 0.235045 3.64659e-06 0.21478 3.87451e-06 0.154862 4.13281e-06 0.031507 4.42801e-06 0.00478915 4.76862e-06 0.00242326 5.16601e-06 0.000850572 5.63564e-06 0.000475524 6.19921e-06 0.000100476 6.88801e-06 7.50165e-05 7.74901e-06 5.00012e-05 8.85601e-06 2.49859e-05 1.0332e-05 0 1.23984e-05 0 1.5498e-05 0 2.0664e-05 0"/>    

    ...


dx0 has lots of zero effciciency matrix with refs to them from 1 material and 42 opticalsurface::


    epsilon:DayaBay_VGDX_20140414-1300 blyth$ grep -n name=\"EFFICIENCY\" g4_00_CGeometry_export_v1.gdml | head -3
    650:      <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>
    1253:      <property name="EFFICIENCY" ref="EFFICIENCY0x1e132a0"/>
    1631:      <property name="EFFICIENCY" ref="EFFICIENCY0x1f05bf0"/>
    epsilon:DayaBay_VGDX_20140414-1300 blyth$ grep -n name=\"EFFICIENCY\" g4_00_CGeometry_export_v1.gdml | wc -l 
          43


Revive analytic/gdml.py and parse the GDML to compare the EFFICIENCY::

    In [7]: run ../analytic/gdml.py                                                                                                                                                                      
    [2020-10-14 15:51:46,816] p26399 {/Users/blyth/opticks/ana/key.py:109} INFO - ppos 4
    [2020-10-14 15:51:46,816] p26399 {/Users/blyth/opticks/analytic/gdml.py:1239} INFO - parsing gdmlpath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v0.gdml 
    [2020-10-14 15:51:46,968] p26399 {/Users/blyth/opticks/analytic/gdml.py:1451} INFO - vv 249 (number of logical volumes) vvs 249 (number of lv with associated solid) 
    [2020-10-14 15:51:47,929] p26399 {/Users/blyth/opticks/analytic/gdml.py:1549} INFO - define_matrix_values startswith:EFFICIENCY scale:1000000.0 
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1e132a0                        
     2   (39, 2)  11132079ca7e28f352f025375e56d037 EFFICIENCY0x1d79780                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f05bf0                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f04370                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f02b20                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f07410                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f003f0                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f08cc0                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1f0a540                        
     2   (39, 2)  fc7cb1f6ce6d6a002b2b20fd24a3ca3f EFFICIENCY0x1e25430                    
     ...


From the digests there are two types of efficiency::

    In [8]: a = mv["EFFICIENCY0x1e132a0"]   ## all zero  
    In [9]: b = mv["EFFICIENCY0x1d79780"]   ## has non-zero efficiency, referenced from Bialkali material

    In [12]: a.T                                                                                                                                                                                         
    Out[12]: 
    array([[ 1.512,  1.55 ,  1.59 ,  1.631,  1.675,  1.722,  1.771,  1.823,  1.879,  1.937,  2.   ,  2.066,  2.138,  2.214,  2.296,  2.384,  2.48 ,  2.583,  2.695,  2.818,  2.952,  3.1  ,  3.263,
             3.444,  3.647,  3.875,  4.133,  4.428,  4.769,  5.166,  5.636,  6.199,  6.888,  7.749,  8.856, 10.332, 12.398, 15.498, 20.664],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
             0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]], dtype=float32)

    In [13]: b.T                                                                                                                                                                                         
    Out[13]: 
    array([[ 1.512,  1.55 ,  1.59 ,  1.631,  1.675,  1.722,  1.771,  1.823,  1.879,  1.937,  2.   ,  2.066,  2.138,  2.214,  2.296,  2.384,  2.48 ,  2.583,  2.695,  2.818,  2.952,  3.1  ,  3.263,
             3.444,  3.647,  3.875,  4.133,  4.428,  4.769,  5.166,  5.636,  6.199,  6.888,  7.749,  8.856, 10.332, 12.398, 15.498, 20.664],
           [ 0.   ,  0.   ,  0.   ,  0.001,  0.001,  0.001,  0.002,  0.003,  0.004,  0.007,  0.01 ,  0.019,  0.027,  0.046,  0.065,  0.085,  0.105,  0.139,  0.17 ,  0.195,  0.215,  0.225,  0.24 ,
             0.235,  0.215,  0.155,  0.032,  0.005,  0.002,  0.001,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]], dtype=float32)


::

     0648     <material name="/dd/Materials/Bialkali0x3e5d3e0" state="solid">
      649       <property name="RINDEX" ref="RINDEX0x1d78d30"/>
      650       <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>
      651       <property name="GROUPVEL" ref="GROUPVEL0x1d7a220"/>
      652       <property name="RAYLEIGH" ref="RAYLEIGH0x1d79be0"/>
      653       <property name="ABSLENGTH" ref="ABSLENGTH0x1d798c0"/>
      654       <property name="REEMISSIONPROB" ref="REEMISSIONPROB0x1d79f00"/>
      655       <P unit="pascal" value="101324.946686941"/>
      656       <MEE unit="eV" value="252.140155582489"/>
      657       <D unit="g/cm3" value="0.0999999473841014"/>
      658       <fraction n="0.375" ref="/dd/Materials/Sodium0x3e5bdc0"/>
      659       <fraction n="0.1875" ref="/dd/Materials/Potassium0x3e5c980"/>
      660       <fraction n="0.1875" ref="/dd/Materials/Cesium0x3e5cd50"/>
      661       <fraction n="0.25" ref="/dd/Materials/Antimony0x3e5cf90"/>
      662     </material>

     1251     <opticalsurface finish="3" model="1" name="NearPoolCoverSurface" type="0" value="1">
     1252       <property name="REFLECTIVITY" ref="REFLECTIVITY0x1e13840"/>
     1253       <property name="EFFICIENCY" ref="EFFICIENCY0x1e132a0"/>
     1254     </opticalsurface>


::

    epsilon:DayaBay_VGDX_20140414-1300 blyth$ grep -n name=\"EFFICIENCY\" -C 2 g4_00_CGeometry_export_v1.gdml 
    648-    <material name="/dd/Materials/Bialkali0x3e5d3e0" state="solid">
    649-      <property name="RINDEX" ref="RINDEX0x1d78d30"/>
    650:      <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>
    651-      <property name="GROUPVEL" ref="GROUPVEL0x1d7a220"/>
    652-      <property name="RAYLEIGH" ref="RAYLEIGH0x1d79be0"/>
    --
    --
    1251-    <opticalsurface finish="3" model="1" name="NearPoolCoverSurface" type="0" value="1">
    1252-      <property name="REFLECTIVITY" ref="REFLECTIVITY0x1e13840"/>
    1253:      <property name="EFFICIENCY" ref="EFFICIENCY0x1e132a0"/>
    1254-    </opticalsurface>
    1255-    <box lunit="mm" name="RPCStrip0xc04bcb00x3e737f0" x="2080" y="260" z="2"/>
    --
    --
    1629-    <opticalsurface finish="3" model="1" name="RSOilSurface" type="0" value="1">
    1630-      <property name="REFLECTIVITY" ref="REFLECTIVITY0x1f06190"/>
    1631:      <property name="EFFICIENCY" ref="EFFICIENCY0x1f05bf0"/>
    1632-    </opticalsurface>
    1633-    <tube aunit="deg" deltaphi="360" lunit="mm" name="TopESR0xbf9d3100x3e88c50" rmax="2223" rmin="144.5" startphi="0" z="0.1"/>
    --
    --
    1708-    <opticalsurface finish="0" model="1" name="ESRAirSurfaceTop" type="0" value="0">
    1709-      <property name="REFLECTIVITY" ref="REFLECTIVITY0x1f04910"/>
    1710:      <property name="EFFICIENCY" ref="EFFICIENCY0x1f04370"/>
    1711-    </opticalsurface>
    1712-    <tube aunit="deg" deltaphi="360" lunit="mm" name="TopReflector0xc3d71780x3e8b760" rmax="2250" rmin="127.5" startphi="0" z="20"/>
    --
    --


To add sensors need to find pair of pv names and add bordersurface between them

::
     5096     <volume name="/dd/Geometry/PMT/lvPmtHemi0xc1337400x3ee9b20">
     5097       <materialref ref="/dd/Materials/Pyrex0x3e60090"/>
     5098       <solidref ref="pmt-hemi0xc0fed900x3e85f00"/>
     5099       <physvol name="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0">             ##
     5100         <volumeref ref="/dd/Geometry/PMT/lvPmtHemiVacuum0xc2c7cc80x3ee9760"/>

         5081     <volume name="/dd/Geometry/PMT/lvPmtHemiVacuum0xc2c7cc80x3ee9760">
         5082       <materialref ref="/dd/Materials/Vacuum0x3e5e530"/>
         5083       <solidref ref="pmt-hemi-vac0xc21e2480x3e85290"/>
         5084       <physvol name="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720">   ##
         5085         <volumeref ref="/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400"/>

             5068     <volume name="/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400">
             5069       <materialref ref="/dd/Materials/Bialkali0x3e5d3e0"/>
             5070       <solidref ref="pmt-hemi-cathode0xc2f1ce80x3e842d0"/>
             5071       <auxiliary auxtype="SensDet" auxvalue="SD0"/>
             5072     </volume>

         5086       </physvol>
         5087       <physvol name="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom0xc21de780x3ee98d0">
         5088         <volumeref ref="/dd/Geometry/PMT/lvPmtHemiBottom0xc12ad600x3ee9530"/>
         5089         <position name="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom0xc21de780x3ee98d0_pos" unit="mm" x="0" y="0" z="69"/>

             5073     <volume name="/dd/Geometry/PMT/lvPmtHemiBottom0xc12ad600x3ee9530">
             5074       <materialref ref="/dd/Materials/OpaqueVacuum0x3e5d740"/>
             5075       <solidref ref="pmt-hemi-bot0xc22a9580x3e844c0"/>
             5076     </volume>

         5090       </physvol>
         5091       <physvol name="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode0xc04ad280x3ee99a0">
         5092         <volumeref ref="/dd/Geometry/PMT/lvPmtHemiDynode0xc02b2800x3ee9650"/>
         5093         <position name="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode0xc04ad280x3ee99a0_pos" unit="mm" x="0" y="0" z="-81.5"/>

             5077     <volume name="/dd/Geometry/PMT/lvPmtHemiDynode0xc02b2800x3ee9650">
             5078       <materialref ref="/dd/Materials/OpaqueVacuum0x3e5d740"/>
             5079       <solidref ref="pmt-hemi-dynode0xc346c500x3e84610"/>
             5080     </volume>

         5094       </physvol>
         5095     </volume>


     5101       </physvol>
     5102     </volume>


::

    epsilon:DayaBay_VGDX_20140414-1300 blyth$ grep -n lvPmtHemi0xc1337400x3ee9b20 g4_00_CGeometry_export_v0.gdml | head -4
    5096:    <volume name="/dd/Geometry/PMT/lvPmtHemi0xc1337400x3ee9b20">
    5332:        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc1337400x3ee9b20"/>
    5342:        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc1337400x3ee9b20"/>
    5352:        <volumeref ref="/dd/Geometry/PMT/lvPmtHemi0xc1337400x3ee9b20"/>
    epsilon:DayaBay_VGDX_20140414-1300 blyth$ grep -n lvPmtHemi0xc1337400x3ee9b20 g4_00_CGeometry_export_v0.gdml | wc -l 
         481





opticalsurface at tail of solids::

     3159   
     3160     <!-- SCB manual addition start : see notes/issues/sensor-gdml-review.rst -->
     3161     <!-- see bordersurface referencing at tail of structure -->
     3162     
     3163     <opticalsurface finish="0" model="0" name="SCB_photocathode_opsurf" type="0" value="1">
     3164          <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>   <!-- the non-zero efficiency-->
     3165     </opticalsurface>
     3166     <!-- SCB manual addition end : see notes/issues/sensor-gdml-review.rst -->
     3167     
     3168   </solids>


bordersurface at tail of structure::

    31964 
    31965     <!-- SCB manual addition start : see notes/issues/sensor-gdml-review.rst -->
    31966     <!-- see opticalsurface at tail of solids -->
    31967 
    31968     <bordersurface name="SCB_photocathode_logsurf1" surfaceproperty="SCB_photocathode_opsurf">
    31969        <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    31970        <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    31971     </bordersurface>
    31972 
    31973     <bordersurface name="SCB_photocathode_logsurf2" surfaceproperty="SCB_photocathode_opsurf">
    31974        <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    31975        <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    31976     </bordersurface>
    31977     <!-- SCB manual addition end : see notes/issues/sensor-gdml-review.rst -->
    31978   </structure>
    31979 


Add::

     610 geocache-dx1-notes(){ cat << EON
     611 
     612 Manually modified dx0 GDML to make dx1, with addition of bordersurface with non-zero associated efficiency.
     613 
     614 EON
     615 }
     616 geocache-dx1-(){  opticksaux- ; geocache-create- --gdmlpath $(opticksaux-dx1) --x4polyskip 211,232  --geocenter --noviz $* ; }
     617 geocache-dx1-comment(){ echo sensors-gdml-review.rst ; }     
     618 geocache-dx1(){   geocache-dx1- --runfolder $FUNCNAME --runcomment $(${FUNCNAME}-comment) $* ; }
     619 geocache-dx1-key(){ 
     620    case $(uname) in 
     621       Linux)  echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c ;;
     622       Darwin) echo OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3 ;;
     623    esac
     624 }
     625 geocache-dx1-keydir(){ OPTICKS_KEY=$(geocache-dx1-key) geocache-keydir ; }
     626 
     627 


geocache-dx1 looks like got the message::

    :set nowrap

    2020-10-14 16:12:30.713 INFO  [9085452] [GSurfaceLib::dumpSurfaces@752] X4PhysicalVolume::convertSurfaces num_surfaces 48
    ...
     index :  6 is_sensor : N type :        bordersurface name :                                NearOWSLinerSurface bpv1 /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS0xbf55b100x4128cf0 bpv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 .
     index :  7 is_sensor : N type :        bordersurface name :                               NearDeadLinerSurface bpv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 bpv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 .
     index :  8 is_sensor : Y type :        bordersurface name :                          SCB_photocathode_logsurf1 bpv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 bpv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 .
     index :  9 is_sensor : Y type :        bordersurface name :                          SCB_photocathode_logsurf2 bpv1 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 bpv2 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 .
     index : 10 is_sensor : N type :          skinsurface name :                               NearPoolCoverSurface sslv lvNearTopCover0xc1370600x3ebf2d0 .
     index : 11 is_sensor : N type :          skinsurface name :                                       RSOilSurface sslv lvRadialShieldUnit0xc3d7ec00x3eea9d0 .
     ...
     index : 43 is_sensor : N type :          skinsurface name :                                LegInDeadTubSurface sslv lvLegInDeadTub0xce5bea80x4129960 .
     index : 44 is_sensor : Y type :          testsurface name :                               perfectDetectSurface .
     index : 45 is_sensor : N type :          testsurface name :                               perfectAbsorbSurface .
     index : 46 is_sensor : N type :          testsurface name :                             perfectSpecularSurface .
     index : 47 is_sensor : N type :          testsurface name :                              perfectDiffuseSurface .
    2020-10-14 16:12:30.715 INFO  [9085452] [GPropertyLib::dumpSensorIndices@935] X4PhysicalVolume::convertSurfaces  NumSensorIndices 3 ( 8 9 44  ) 



GNodeLibTest show sensorIndex are in the volume identity::

    2020-10-14 16:20:32.629 INFO  [9095349] [test_getIdentity@90] 
     nidx 3199 nid[uvec4(3199, 83886080, 3080219, -1);rpo(5 0 0) 5000000]
     nidx 3200 nid[uvec4(3200, 83886081, 3014684, -1);rpo(5 0 1) 5000001]
     nidx 3201 nid[uvec4(3201, 83886082, 2818077, 0);rpo(5 0 2) 5000002]
     nidx 3202 nid[uvec4(3202, 83886083, 2883614, -1);rpo(5 0 3) 5000003]
     nidx 3203 nid[uvec4(3203, 83886084, 2949150, -1);rpo(5 0 4) 5000004]
     nidx 3204 nid[uvec4(3204, 1407, 3145759, -1);rpo(0 0 1407)     57f]
     nidx 3205 nid[uvec4(3205, 83886336, 3080219, -1);rpo(5 1 0) 5000100]
     nidx 3206 nid[uvec4(3206, 83886337, 3014684, -1);rpo(5 1 1) 5000101]
     nidx 3207 nid[uvec4(3207, 83886338, 2818077, 1);rpo(5 1 2) 5000102]
     nidx 3208 nid[uvec4(3208, 83886339, 2883614, -1);rpo(5 1 3) 5000103]
     nidx 3209 nid[uvec4(3209, 83886340, 2949150, -1);rpo(5 1 4) 5000104]
     nidx 3210 nid[uvec4(3210, 1408, 3145759, -1);rpo(0 0 1408)     580]
     nidx 3211 nid[uvec4(3211, 83886592, 3080219, -1);rpo(5 2 0) 5000200]
     nidx 3212 nid[uvec4(3212, 83886593, 3014684, -1);rpo(5 2 1) 5000201]
     nidx 3213 nid[uvec4(3213, 83886594, 2818077, 2);rpo(5 2 2) 5000202]
     nidx 3214 nid[uvec4(3214, 83886595, 2883614, -1);rpo(5 2 3) 5000203]
     nidx 3215 nid[uvec4(3215, 83886596, 2949150, -1);rpo(5 2 4) 5000204]
     nidx 3216 nid[uvec4(3216, 1409, 3145759, -1);rpo(0 0 1409)     581]
     nidx 3217 nid[uvec4(3217, 83886848, 3080219, -1);rpo(5 3 0) 5000300]
     nidx 3218 nid[uvec4(3218, 83886849, 3014684, -1);rpo(5 3 1) 5000301]
    epsilon:ggeo blyth$ 


::

    epsilon:GNodeLib blyth$ ggeo.py 3199 -i
    nidx:3199 triplet:5000000 sh:2f001b sidx:   -1   nrpo(  3199     5     0     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    nidx:3200 triplet:5000001 sh:2e001c sidx:   -1   nrpo(  3200     5     0     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    nidx:3201 triplet:5000002 sh:2b001d sidx:    0   nrpo(  3201     5     0     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0 Vacuum/SCB_photocathode_logsurf1/SCB_photocathode_logsurf2/Bialkali) 
    nidx:3202 triplet:5000003 sh:2c001e sidx:   -1   nrpo(  3202     5     0     3 )  shape(  44  30            pmt-hemi-bot0xc22a9580x3e844c0                    Vacuum///OpaqueVacuum) 
    nidx:3203 triplet:5000004 sh:2d001e sidx:   -1   nrpo(  3203     5     0     4 )  shape(  45  30         pmt-hemi-dynode0xc346c500x3e84610                    Vacuum///OpaqueVacuum) 
    nidx:3204 triplet:    57f sh:30001f sidx:   -1   nrpo(  3204     0     0  1407 )  shape(  48  31             AdPmtCollar0xc2c52600x3e86030          MineralOil///UnstStainlessSteel) 
    nidx:3205 triplet:5000100 sh:2f001b sidx:   -1   nrpo(  3205     5     1     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    nidx:3206 triplet:5000101 sh:2e001c sidx:   -1   nrpo(  3206     5     1     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    nidx:3207 triplet:5000102 sh:2b001d sidx:    1   nrpo(  3207     5     1     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0 Vacuum/SCB_photocathode_logsurf1/SCB_photocathode_logsurf2/Bialkali) 
    nidx:3208 triplet:5000103 sh:2c001e sidx:   -1   nrpo(  3208     5     1     3 )  shape(  44  30            pmt-hemi-bot0xc22a9580x3e844c0                    Vacuum///OpaqueVacuum) 
    nidx:3209 triplet:5000104 sh:2d001e sidx:   -1   nrpo(  3209     5     1     4 )  shape(  45  30         pmt-hemi-dynode0xc346c500x3e84610                    Vacuum///OpaqueVacuum) 
    nidx:3210 triplet:    580 sh:30001f sidx:   -1   nrpo(  3210     0     0  1408 )  shape(  48  31             AdPmtCollar0xc2c52600x3e86030          MineralOil///UnstStainlessSteel) 
    nidx:3211 triplet:5000200 sh:2f001b sidx:   -1   nrpo(  3211     5     2     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    nidx:3212 triplet:5000201 sh:2e001c sidx:   -1   nrpo(  3212     5     2     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    nidx:3213 triplet:5000202 sh:2b001d sidx:    2   nrpo(  3213     5     2     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0 Vacuum/SCB_photocathode_logsurf1/SCB_photocathode_logsurf2/Bialkali) 
    nidx:3214 triplet:5000203 sh:2c001e sidx:   -1   nrpo(  3214     5     2     3 )  shape(  44  30            pmt-hemi-bot0xc22a9580x3e844c0                    Vacuum///OpaqueVacuum) 
    nidx:3215 triplet:5000204 sh:2d001e sidx:   -1   nrpo(  3215     5     2     4 )  shape(  45  30         pmt-hemi-dynode0xc346c500x3e84610                    Vacuum///OpaqueVacuum) 
    nidx:3216 triplet:    581 sh:30001f sidx:   -1   nrpo(  3216     0     0  1409 )  shape(  48  31             AdPmtCollar0xc2c52600x3e86030          MineralOil///UnstStainlessSteel) 
    nidx:3217 triplet:5000300 sh:2f001b sidx:   -1   nrpo(  3217     5     3     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 




kcd/GNodeLib/GTreePresent.txt::


    3145      3144 [  5: 175/ 178]    0 ( 0)        /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearRightDiagSIRightY30xbf894500x3ee3fd0  near_diagonal_square_iron0xbf5f3f80x3e758d0
    3146      3145 [  5: 176/ 178]    0 ( 0)        /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearRightDiagSIRightY40xbf895400x3ee4100  near_diagonal_square_iron0xbf5f3f80x3e758d0
    3147      3146 [  5: 177/ 178]    0 ( 0)        /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearRightDiagSILeftY40xbf896300x3ee4230  near_diagonal_square_iron0xbf5f3f80x3e758d0
    3148      3147 [  2:   1/   2]   10 ( 0)     /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xcd2fa580x40f6eb0  near_hall_bot0xbf3d7180x3ebe9b0
    3149      3148 [  3:   0/  10]    9 ( 0)      /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090  near_pool_dead_box0xbf8a2800x3ebdb10
    3150      3149 [  4:   0/   9]    9 ( 0)       /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20  near_pool_liner_box0xc2dcc280x3ebcdc0
    3151      3150 [  5:   0/   9] 2938 ( 0)        /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS0xbf55b100x4128cf0  PLACEHOLDER_near_pool_ows_box0xbf8c8a80x3ebc050
    3152      3151 [  6:   0/2938]    9 ( 0)         /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain0xc5c5f200x3fa9070  near_pool_curtain_box0xc2cef480x3eb54a0
    3153      3152 [  7:   0/   9] 1619 ( 0)          /dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS0xc15a4980x3fa6c80  PLACEHOLDER_near_pool_iws_box0xc288ce80x3eb4760
    3154      3153 [  8:   0/1619]   11 ( 0)           /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf5280x3efb9c0  ade0xc2a74380x3eafdb0
    3155      3154 [  9:   0/  11]    4 ( 0)            /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100  sst0xbf4b0600x3e887c0
    3156      3155 [ 10:   0/   4]  520 ( 0)             /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0  oil0xbf5ed480x3e88410
    3157      3156 [ 11:   0/ 520]    3 ( 0)              /dd/Geometry/AD/lvOIL#pvOAV0xbf8f6380x3eeda30  oav0xc2ed7c80x3e83dc0
    3158      3157 [ 12:   0/   3]   35 ( 0)               /dd/Geometry/AD/lvOAV#pvLSO0xbf8e1200x3ee9070  lso0xc028a380x3e81200
    3159      3158 [ 13:   0/  35]    2 ( 0)                /dd/Geometry/AD/lvLSO#pvIAV0xc2d03480x3ee63b0  iav0xc346f900x3e7c860
    3160      3159 [ 14:   0/   2]    0 ( 0)                 /dd/Geometry/AD/lvIAV#pvGDS0xbf6ab000x3ee52e0  gds0xc28d3f00x3e7b440
    3161      3160 [ 14:   1/   2]    0 ( 0)                 /dd/Geometry/AD/lvIAV#pvOcrGdsInIAV0xbf6b0e00x3ee5400  OcrGdsInIav0xc405b100x3e7bbe0
    3162      3161 [ 13:   1/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvIavTopHub0xc34e6e80x3ee6450  IavTopHub0xc4059680x3e7ca80
    3163      3162 [ 13:   2/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvCtrGdsOflBotClp0xc2ce2a80x3ee6510  CtrGdsOflBotClp0xbf5dec00x3e7d440
    3164      3163 [ 13:   3/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvCtrGdsOflTfbInLso0xc2ca5380x3ee65e0  CtrGdsOflTfbInLso0xbfa2d300x3e7de40
    3165      3164 [ 13:   4/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvCtrGdsOflInLso0xbf742500x3ee66e0  CtrGdsOflInLso0xbfa11780x3e7df60
    3166      3165 [ 13:   5/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvOcrGdsPrt0xbf6d0d00x3ee6750  OcrGdsPrt0xc3525180x3e7ec30
    3167      3166 [ 13:   6/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvOcrGdsBotClp0xbfa16100x3ee6880  CtrGdsOflBotClp0xbf5dec00x3e7d440
    3168      3167 [ 13:   7/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvOcrGdsTfbInLso0xbfa18180x3ee68f0  OcrGdsTfbInLso0xbfa23700x3e7f040
    3169      3168 [ 13:   8/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvOcrGdsInLso0xbf6d2800x3ee69b0  OcrGdsInLso0xbfa21900x3e7f820
    3170      3169 [ 13:   9/  35]    0 ( 0)                /dd/Geometry/AD/lvLSO#pvOavBotRibs#OavBotRibs#OavBotRibRot0xbf5af900x3ee6b50  OavBotRib0xbfaafe00x3e7b3e0



::

    OKTest --domaintarget=3147 --gensteptarget=3154


Succeeds to get some hits, 7:SD terminated::

    2020-10-14 16:45:57.783 INFO  [9113079] [OpticksAttrSeq::dumpTable@422] OpticksIdx::makeHistoryItemIndex seqhis
        0      4420  0.488     8ccccd             TO BT BT BT BT SA 
        1       846  0.093 ccaccccccd TO BT BT BT BT BT BT SR BT BT 
        2       720  0.079    4cccccd          TO BT BT BT BT BT AB 
        3       489  0.054         4d                         TO AB 
        4       265  0.029  8cbcccccd    TO BT BT BT BT BT BR BT SA 
        5       257  0.028    8cccccd          TO BT BT BT BT BT SA 
        6       239  0.026 cccacccccd TO BT BT BT BT BT SR BT BT BT 
        7       235  0.026 cccc9ccccd TO BT BT BT BT DR BT BT BT BT 
        8       206  0.023   8ccccccd       TO BT BT BT BT BT BT SA 
        9       182  0.020    8cccc6d          TO SC BT BT BT BT SA 
       10       129  0.014     4ccccd             TO BT BT BT BT AB 
       11       123  0.014       4ccd                   TO BT BT AB 
       12       118  0.013 cccbcccccd TO BT BT BT BT BT BR BT BT BT 
       13       110  0.012 ccccbccccd TO BT BT BT BT BR BT BT BT BT 
       14        95  0.010 cccccccccd TO BT BT BT BT BT BT BT BT BT 
       15        85  0.009    8cccc5d          TO RE BT BT BT BT SA 
       16        56  0.006  4cccccccd    TO BT BT BT BT BT BT BT AB 
       17        55  0.006 cbcccccccd TO BT BT BT BT BT BT BT BR BT 
       18        50  0.006    7cccccd          TO BT BT BT BT BT SD 
       19        40  0.004    89ccccd          TO BT BT BT BT DR SA 
       20        37  0.004 cccccc6ccd TO BT BT SC BT BT BT BT BT BT 
       21        33  0.004   7ccccccd       TO BT BT BT BT BT BT SD 
       22        32  0.004 cccc6ccccd TO BT BT BT BT SC BT BT BT BT 
       23        31  0.003 bccbcccccd TO BT BT BT BT BT BR BT BT BR 
       24        30  0.003 abaccccccd TO BT BT BT BT BT BT SR BR SR 
       25        29  0.003    49ccccd          TO BT BT BT BT DR AB 
       26        27  0.003 c9cbcccccd TO BT BT BT BT BT BR BT DR BT 
       27        26  0.003 accccccccd TO BT BT BT BT BT BT BT BT SR 
       28        25  0.003    8bccccd          TO BT BT BT BT BR SA 
       29        24  0.003    86ccccd          TO BT BT BT BT SC SA 
       30        24  0.003    8cc6ccd          TO BT BT SC BT BT SA 
       31        23  0.003 ccbc9ccccd TO BT BT BT BT DR BT BR BT BT 
      TOT      9061






