FIXED : OKX4Test_missing_PMT_partIdx2_transform
===================================================

* Succeed to reproduce the problem in X4SolidTest:test_cathode

* the transform is getting thru to nnode model, but it is set on the difference operator
  (the sphere-with-inner is a primitive in G4 but a composite in OK) 

* think going to gtransforms too soon, should be setting level transforms
  and only calculating gtransforms as late as possible

* FIXED : the problem was that the parent links were not set for NCSG::FromNode
  so only the gtransform calc was not including transforms on operators



::

    2018-06-30 10:35:18.512 INFO  [1265168] [GParts::close@843] GParts::close DONE  verbosity 0
    /tmp/X4SolidTest/GParts
    prim (1, 4) part (7, 4, 4) tran (1, 3, 4, 4) 

    primIdx 0 prim array([0, 7, 0, 0], dtype=int32) partOffset 0 numParts 7 tranOffset 0 planOffset 0  
        Part  1  0            union     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  7  1          zsphere     tz:     0.000     r:    128.000 z1:    97.325 z2:   128.000   
        Part  7  1          zsphere     tz:     0.000     r:    127.950 z1:    97.287 z2:   127.950   
        Part  7  1          zsphere     tz:     0.000     r:     99.000 z1:    13.000 z2:    55.762   
        Part  7  1          zsphere     tz:     0.000     r:     98.950 z1:    12.993 z2:    55.734   
    2018-06-30 10:35:18.670 INFO  [1265168] [SSys::run@52] prim.py /tmp/X4SolidTest/GParts rc_raw : 0 rc : 0




primIdx 2 missing transform, when restrict to primitives

::

    epsilon:analytic blyth$ ab-prim
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5
    prim (5, 4) part (41, 4, 4) tran (12, 3, 4, 4) 

    primIdx 0 prim array([ 0, 15,  0,  0], dtype=int32) partOffset 0 numParts 15 tranOffset 0 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -84.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 1 prim array([15, 15,  4,  0], dtype=int32) partOffset 15 numParts 15 tranOffset 4 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -81.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 2 prim array([30,  7,  8,  0], dtype=int32) partOffset 30 numParts 7 tranOffset 8 planOffset 0  
        Part  1  0            union     0.0    
        Part  3  0       difference     0.0    
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  2          zsphere     43.0    
        Part  7  2          zsphere     43.0    

    primIdx 3 prim array([37,  3, 10,  0], dtype=int32) partOffset 37 numParts 3 tranOffset 10 planOffset 0  
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     69.0    
        Part  7  1          zsphere     69.0    

    primIdx 4 prim array([40,  1, 11,  0], dtype=int32) partOffset 40 numParts 1 tranOffset 11 planOffset 0  
        Part 12  1         cylinder     -81.5    


    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5
    prim (5, 4) part (41, 4, 4) tran (11, 3, 4, 4) 

    primIdx 0 prim array([ 0, 15,  0,  0], dtype=int32) partOffset 0 numParts 15 tranOffset 0 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -84.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 1 prim array([15, 15,  4,  0], dtype=int32) partOffset 15 numParts 15 tranOffset 4 planOffset 0  
        Part  1  0            union     0.0    
        Part  2  0     intersection     0.0    
        Part 12  4         cylinder     -81.5    
        Part  2  0     intersection     0.0    
        Part  5  3           sphere     69.0    
        Part  5  1           sphere     0.0    
        Part  5  2           sphere     43.0    

    primIdx 2 prim array([30,  7,  8,  0], dtype=int32) partOffset 30 numParts 7 tranOffset 8 planOffset 0  
        Part  1  0            union     0.0    
        Part  3  0       difference     0.0    
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0    
        Part  7  1          zsphere     0.0       <<<<< MISSING 
        Part  7  1          zsphere     0.0       <<<<  MISSING 

    primIdx 3 prim array([37,  3,  9,  0], dtype=int32) partOffset 37 numParts 3 tranOffset 9 planOffset 0  
        Part  3  0       difference     0.0    
        Part  7  1          zsphere     69.0    
        Part  7  1          zsphere     69.0    

    primIdx 4 prim array([40,  1, 10,  0], dtype=int32) partOffset 40 numParts 1 tranOffset 10 planOffset 0  
        Part 12  1         cylinder     -81.5    
    epsilon:analytic blyth$ 







::



  702     <sphere aunit="deg" deltaphi="360" deltatheta="82.45452026503" lunit="mm" name="pmt-hemi-bot0xc22a958" rmax="99" rmin="98" startphi="0" starttheta="97.54547973497"/>
  703     <tube aunit="deg" deltaphi="360" lunit="mm" name="pmt-hemi-dynode0xc346c50" rmax="27.5" rmin="0" startphi="0" z="166"/>






primIdx:0  pmt-hemi : union of cylinder with 3-sphere intersection
--------------------------------------------------------------------

::

  737     <union name="pmt-hemi0xc0fed90">
  738       <first ref="pmt-hemi-glass-bulb0xc0feb98"/>
  739       <second ref="pmt-hemi-base0xc0fecb0"/>
  740       <position name="pmt-hemi0xc0fed90_pos" unit="mm" x="0" y="0" z="-84.5"/>
  741     </union>

  736     <tube aunit="deg" deltaphi="360" lunit="mm" name="pmt-hemi-base0xc0fecb0" rmax="42.25" rmin="0" startphi="0" z="169"/>


  731     <intersection name="pmt-hemi-glass-bulb0xc0feb98">
  732       <first ref="pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0"/>

      725     <intersection name="pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0">
      726       <first ref="pmt-hemi-face-glass0xc0fde80"/>
           723     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-face-glass0xc0fde80" rmax="131" rmin="0" startphi="0" starttheta="0"/>

      727       <second ref="pmt-hemi-top-glass0xc0fdef0"/>
            724     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-top-glass0xc0fdef0" rmax="102" rmin="0" startphi="0" starttheta="0"/>

      728       <position name="pmt-hemi-face-glass*ChildForpmt-hemi-glass-bulb0xbf1f8d0_pos" unit="mm" x="0" y="0" z="43"/>
      729     </intersection>

  733       <second ref="pmt-hemi-bot-glass0xc0feac8"/>

        730     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-bot-glass0xc0feac8" rmax="102" rmin="0" startphi="0" starttheta="0"/>

  734       <position name="pmt-hemi-glass-bulb0xc0feb98_pos" unit="mm" x="0" y="0" z="69"/>
  735     </intersection>


::

    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5
    prim (5, 4) part (41, 4, 4) tran (11, 3, 4, 4) 

    primIdx 0 prim array([ 0, 15,  0,  0], dtype=int32) partOffset 0 numParts 15 tranOffset 0 planOffset 0  
        Part  1  0            union     tz:     0.000      
        Part  2  0     intersection     tz:     0.000      
        Part 12  4         cylinder     tz:   -84.500     r:     42.250 z1:   -84.500 z2:    84.500   
        Part  2  0     intersection     tz:     0.000      
        Part  5  3           sphere     tz:    69.000     r:    102.000   
        Part  5  1           sphere     tz:     0.000     r:    131.000   
        Part  5  2           sphere     tz:    43.000     r:    102.000   





primIdx 1 : pmt-hemi-vac same again slightly smaller
---------------------------------------------------------

::

  718     <union name="pmt-hemi-vac0xc21e248">
  719       <first ref="pmt-hemi-bulb-vac0xc21e200"/>
  720       <second ref="pmt-hemi-base-vac0xc133310"/>

      717     <tube aunit="deg" deltaphi="360" lunit="mm" name="pmt-hemi-base-vac0xc133310" rmax="39.25" rmin="0" startphi="0" z="166"/>

  721       <position name="pmt-hemi-vac0xc21e248_pos" unit="mm" x="0" y="0" z="-81.5"/>
  722     </union>


  712     <intersection name="pmt-hemi-bulb-vac0xc21e200">
  713       <first ref="pmt-hemi-face-vac*ChildForpmt-hemi-bulb-vac0xbf1f680"/>

      706     <intersection name="pmt-hemi-face-vac*ChildForpmt-hemi-bulb-vac0xbf1f680">
      707       <first ref="pmt-hemi-face-vac0xbf6d5e0"/>
               704     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-face-vac0xbf6d5e0" rmax="128" rmin="0" startphi="0" starttheta="0"/>

      708       <second ref="pmt-hemi-top-vac0xc2f4260"/>
               705     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-top-vac0xc2f4260" rmax="99" rmin="0" startphi="0" starttheta="0"/>

      709       <position name="pmt-hemi-face-vac*ChildForpmt-hemi-bulb-vac0xbf1f680_pos" unit="mm" x="0" y="0" z="43"/>
      710     </intersection>

  714       <second ref="pmt-hemi-bot-vac0xc2f4370"/>
        711     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="pmt-hemi-bot-vac0xc2f4370" rmax="99" rmin="0" startphi="0" starttheta="0"/>

  715       <position name="pmt-hemi-bulb-vac0xc21e200_pos" unit="mm" x="0" y="0" z="69"/>
  716     </intersection>


::

    primIdx 1 prim array([15, 15,  4,  0], dtype=int32) partOffset 15 numParts 15 tranOffset 4 planOffset 0  
        Part  1  0            union     tz:     0.000      
        Part  2  0     intersection     tz:     0.000      
        Part 12  4         cylinder     tz:   -81.500     r:     39.250 z1:   -83.000 z2:    83.000   
        Part  2  0     intersection     tz:     0.000      
        Part  5  3           sphere     tz:    69.000     r:     99.000   
        Part  5  1           sphere     tz:     0.000     r:    128.000   
        Part  5  2           sphere     tz:    43.000     r:     99.000   



primIdx 2 : pmt-hemi-cathode
-------------------------------------

* suspect a complication from G4 primitive -> OK composite due to inner (rmin)


::

  697     <union name="pmt-hemi-cathode0xc2f1ce8">
  698       <first ref="pmt-hemi-cathode-face0xc28c5f8"/>
        695     <sphere aunit="deg" deltaphi="360" deltatheta="40.5049977101673" lunit="mm" name="pmt-hemi-cathode-face0xc28c5f8" rmax="128" rmin="127.95" startphi="0" starttheta="0"/>

  699       <second ref="pmt-hemi-cathode-belly0xc28c668"/>
        696     <sphere aunit="deg" deltaphi="360" deltatheta="26.7358890588877" lunit="mm" name="pmt-hemi-cathode-belly0xc28c668" rmax="99" rmin="98.95" startphi="0" starttheta="55.7186312061423"/>
  700       <position name="pmt-hemi-cathode0xc2f1ce8_pos" unit="mm" x="0" y="0" z="43"/>
  701     </union>


old one without the bug::

    primIdx 2 prim array([30,  7,  8,  0], dtype=int32) partOffset 30 numParts 7 tranOffset 8 planOffset 0  
        Part  1  0            union     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  7  1          zsphere     tz:     0.000     r:    128.000 z1:    97.325 z2:   128.000   
        Part  7  1          zsphere     tz:     0.000     r:    127.950 z1:    97.287 z2:   127.950   
        Part  7  2          zsphere     tz:    43.000     r:     99.000 z1:    13.000 z2:    55.762   
        Part  7  2          zsphere     tz:    43.000     r:     98.950 z1:    12.993 z2:    55.734   

            55.762 + 43 = 98.762


buggered with missing tz 43::

    primIdx 2 prim array([30,  7,  8,  0], dtype=int32) partOffset 30 numParts 7 tranOffset 8 planOffset 0  
        Part  1  0            union     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  7  1          zsphere     tz:     0.000     r:    128.000 z1:    97.325 z2:   128.000   
        Part  7  1          zsphere     tz:     0.000     r:    127.950 z1:    97.287 z2:   127.950   
        Part  7  1          zsphere     tz:     0.000     r:     99.000 z1:    13.000 z2:    55.762   
        Part  7  1          zsphere     tz:     0.000     r:     98.950 z1:    12.993 z2:    55.734   


issue reproduced with X4SolidTest.test_cathode::

    primIdx 0 prim array([0, 7, 0, 0], dtype=int32) partOffset 0 numParts 7 tranOffset 0 planOffset 0  
        Part  1  0            union     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  3  0       difference     tz:     0.000      
        Part  7  1          zsphere     tz:     0.000     r:    128.000 z1:    97.325 z2:   128.000   
        Part  7  1          zsphere     tz:     0.000     r:    127.950 z1:    97.287 z2:   127.950   
        Part  7  1          zsphere     tz:     0.000     r:     99.000 z1:    13.000 z2:    55.762   
        Part  7  1          zsphere     tz:     0.000     r:     98.950 z1:    12.993 z2:    55.734   
    2018-06-30 10:35:18.670 INFO  [1265168] [SSys::run@52] prim.py /tmp/X4SolidTest/GParts rc_raw : 0 rc : 0
 



primIdx 3, 4
----------------

::

    primIdx 3 prim array([37,  3,  9,  0], dtype=int32) partOffset 37 numParts 3 tranOffset 9 planOffset 0  
        Part  3  0       difference     tz:     0.000      
        Part  7  1          zsphere     tz:    69.000     r:     99.000 z1:   -99.000 z2:   -13.000   
        Part  7  1          zsphere     tz:    69.000     r:     98.000 z1:   -98.000 z2:   -12.869   

    primIdx 4 prim array([40,  1, 10,  0], dtype=int32) partOffset 40 numParts 1 tranOffset 10 planOffset 0  
        Part 12  1         cylinder     tz:   -81.500     r:     27.500 z1:   -83.000 z2:    83.000   







