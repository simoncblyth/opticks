OKX4Test_partBuffer_mm0_global_differences
=============================================


Summary
---------

* seems that most of the tree height one level higher can be explained 
  by omission of polycone inner cylinder differencing in the old geometry  




Matched dimensions of the analytic geometry buffers : after duplicating omissions
--------------------------------------------------------------------------------------

* this means all the trees have same heights 

primBuffer matched::

    In [3]: pa
    Out[3]: 
    array([[    0,     1,     0,     0],
           [    1,     1,     1,     0],
           [    2,     1,     2,     0],
           ...,
           [11981,     1,  5341,   672],
           [11982,     1,  5342,   672],
           [11983,     1,  5343,   672]], dtype=int32)

    In [4]: pb
    Out[4]: 
    array([[    0,     1,     0,     0],
           [    1,     1,     1,     0],
           [    2,     1,     2,     0],
           ...,
           [11981,     1,  5341,   672],
           [11982,     1,  5342,   672],
           [11983,     1,  5343,   672]], dtype=int32)

    In [5]: np.all( pa == pb )
    Out[5]: True



::

    epsilon:~ blyth$ ab-diff
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/GParts.txt and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/GParts.txt differ
    Only in /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0: idxBuffer.npy
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/planBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/planBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy differ
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0
            ./GParts.txt : 11984 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 5eeee07e08a9a50278a2339dd0b47ac4
    MD5 (partBuffer.npy) = 8d837fba380dfc643968bd23f99d656f
    MD5 (planBuffer.npy) = 94e18d5e55d190c9ed73e04b45ebb404
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 77359e6d3d628e93cb7cf0a4a3824ab3
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0
            ./GParts.txt : 11984 
         ./idxBuffer.npy : (3116, 4) 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 53a257de5d4c5a071ddfd8bc41f2ad0a
    MD5 (idxBuffer.npy) = 237a00a2a01f1353d2b1110f320d0c6a
    MD5 (partBuffer.npy) = 03ba452b37fc15ccaa65953829517cbe
    MD5 (planBuffer.npy) = 43f2892dbf4b8e91231e5d830dee9e03
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 74a6d92ff0d830990e81e10434865714
    epsilon:0 blyth$ 


MATCHED by omission : lvIdx 131 : pill shape
------------------------------------------------

* Hmm : perhaps i avoided the phi-segment hassles with my zsphere. 

When making a pill shape, there is no need to use hemispheres 
(actually its problematic to do so : due to coincident surfaces)
just make a union of spheres and cylinder.

* hmm did i special case somehow ? nope the old py code happens to do the right thing for this case by accident
* it just ignores the phi-segmented sphere as deltaphi_slab_segment_enabled is False

gdml.py::

     393         has_deltaphi = self.deltaphi < 360
     394         if has_deltaphi and not only_inner and self.deltaphi_slab_segment_enabled:
     395              assert self.aunit == 'deg'
     396              phi0 = self.startphi
     397              phi1 = self.startphi + self.deltaphi
     398              rmax = self.rmax + 1
     399              ret_segment = self.deltaphi_slab_segment(ret, phi0, phi1, rmax)
     400              result = ret_segment
     401         else:
     402              result = ret
     403         pass


Hmm notice there is some rotation below, but its a sphere so it does nothing.

/tmp/blyth/opticks/gdml2gltf/extras/131/NNodeTest_131.cc::

    // generated by nnode_test_cpp.py : 20180702-1255 
    // opticks-;opticks-nnt 131 
    // opticks-;opticks-nnt-vi 131 
    
    ncylinder a = make_cylinder(0.000,0.000,0.000,10.035,-14.865,14.865,0.000,0.000) ; a.label = "a" ;   
    nsphere b = make_sphere(0.000,0.000,0.000,10.035) ; b.label = "b" ;   
    b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,0.000,-1.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,14.865,1.000) ;
    nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   
    
    nsphere c = make_sphere(0.000,0.000,0.000,10.035) ; c.label = "c" ;   
    c.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,-1.000,0.000,0.000,  0.000,0.000,-14.865,1.000) ;
    nunion abc = make_union(&ab, &c) ; abc.label = "abc" ; ab.parent = &abc ; c.parent = &abc ;  ;   

    abc.update_gtransforms();

::

     131 : AmCCo60AcrylicContainer0xc0b23b8 

     * this one is a different kettle of fish  


     In [2]: 29.73/2.
     Out[2]: 14.865


     1456     <union name="AmCCo60AcrylicContainer0xc0b23b8">
     1457       <first ref="AcrylicCylinder+ChildForAmCCo60AcrylicContainer0xc0b1f38"/>

         1449     <union name="AcrylicCylinder+ChildForAmCCo60AcrylicContainer0xc0b1f38">
         1450       <first ref="AcrylicCylinder0xc0b22c0"/>

             1447     <tube aunit="deg" deltaphi="360" lunit="mm" name="AcrylicCylinder0xc0b22c0" rmax="10.035" rmin="0" startphi="0" z="29.73"/>

         1451       <second ref="UpperAcrylicHemisphere0xc0b2ac0"/>

             1448     <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="UpperAcrylicHemisphere0xc0b2ac0" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>

                    * bizarre : hemisphere by phi segmenting so the hemisphere "points" to +y, then rotate 90 about x, so point it to +z
                    * easier to not phi segment and rotate, instead just use starttheta=0 deltatheta=90 ?

         1452       <position name="AcrylicCylinder+ChildForAmCCo60AcrylicContainer0xc0b1f38_pos" unit="mm" x="0" y="0" z="14.865"/>
         1453       <rotation name="AcrylicCylinder+ChildForAmCCo60AcrylicContainer0xc0b1f38_rot" unit="deg" x="90" y="0" z="0"/>
         1454     </union>

     1458       <second ref="LowerAcrylicHemisphere0xc0b2be8"/>

         1455     <sphere aunit="deg" deltaphi="180" deltatheta="180" lunit="mm" name="LowerAcrylicHemisphere0xc0b2be8" rmax="10.035" rmin="0" startphi="0" starttheta="0"/>

                  * again : hemisphere by phi segmenting to point to +y, then rotate -90 about x, so point it at -z
                  * easier to not phi segment and rotate, instead just use starttheta=90, deltatheta=90 

     1459       <position name="AmCCo60AcrylicContainer0xc0b23b8_pos" unit="mm" x="0" y="0" z="-14.865"/>
     1460       <rotation name="AmCCo60AcrylicContainer0xc0b23b8_rot" unit="deg" x="-90" y="0" z="0"/>
     1461     </union>


::

    2018-07-02 14:08:10.643 ERROR [171238] [*X4Solid::intersectWithPhiSegment@458]  special cased startPhi == 0.f && deltaPhi == 180.f 
    2018-07-02 14:08:10.644 ERROR [171238] [*X4Solid::intersectWithPhiSegment@458]  special cased startPhi == 0.f && deltaPhi == 180.f 
    2018-07-02 14:08:10.644 INFO  [171238] [*NTreeProcess<nnode>::Process@41] before
    NTreeAnalyse height 3 count 9
                          un            

          un                      in    

      cy          in          sp      bo

              sp      bo                


    2018-07-02 14:08:10.644 INFO  [171238] [*NTreeProcess<nnode>::Process@56] after
    NTreeAnalyse height 3 count 9
                          un            

          un                      in    

      cy          in          sp      bo

              sp      bo                


    2018-07-02 14:08:10.644 INFO  [171238] [*NTreeProcess<nnode>::Process@57]  soIdx 126 lvIdx 131 height0 3 height1 3 ### LISTED






FIXED : lvIdx 60 : new tree is 1-level more than old : the reason is ndisc handles rmin internally, i used it python side
---------------------------------------------------------------------------------------------------------------------------

* FIXED by implementing use of ndisc in X4Solid::convertTubs just like gdml.py 

* notice that 9 primitives is 1 too many : pushing this up to a height 4 tree

* I recall the reason, this shape is very thin : i implemeted ndisc to avoid the fuzzies in the hole
  which handles rmin within the primitive (ie without having one more di)  

::

    epsilon:extg4 blyth$ n=60;cat /tmp/blyth/opticks/gdml2gltf/extras/$n/NNodeTest_$n.cc 
    ...
        // generated by nnode_test_cpp.py : 20180702-1255 
        // opticks-;opticks-nnt 60 
        // opticks-;opticks-nnt-vi 60 
        
        ndisc a = make_disc(0.000,0.000,36.250,2223.000,-0.050,0.050,0.000,0.000) ; a.label = "a" ;   
        nbox b = make_box3(107.000,401.000,1.100,0.000) ; b.label = "b" ; b.complement = true ;   
        b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  2035.000,0.000,0.000,1.000) ;
        nintersection ab = make_intersection(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   
        
        nbox c = make_box3(107.000,401.000,1.100,0.000) ; c.label = "c" ; c.complement = true ;   
        c.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  -2035.000,0.000,0.000,1.000) ;
        nbox d = make_box3(401.000,107.000,1.100,0.000) ; d.label = "d" ; d.complement = true ;   
        d.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,2035.000,0.000,1.000) ;
        nintersection cd = make_intersection(&c, &d) ; cd.label = "cd" ; c.parent = &cd ; d.parent = &cd ;  ;   
        
        nintersection abcd = make_intersection(&ab, &cd) ; abcd.label = "abcd" ; ab.parent = &abcd ; cd.parent = &abcd ;  ;   
        
        nbox e = make_box3(401.000,107.000,1.100,0.000) ; e.label = "e" ; e.complement = true ;   
        e.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,-2035.000,0.000,1.000) ;
        ndisc f = make_disc(0.000,0.000,0.000,50.000,-0.550,0.550,0.000,0.000) ; f.label = "f" ; f.complement = true ;   
        f.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  141.784,-342.297,0.000,1.000) ;
        nintersection ef = make_intersection(&e, &f) ; ef.label = "ef" ; e.parent = &ef ; f.parent = &ef ;  ;   
        
        ndisc g = make_disc(0.000,0.000,0.000,50.000,-0.550,0.550,0.000,0.000) ; g.label = "g" ; g.complement = true ;   
        g.transform = nmat4triple::make_transform(0.383,0.924,0.000,0.000,  -0.924,0.383,0.000,0.000,  0.000,0.000,1.000,0.000,  909.276,-1541.906,0.000,1.000) ;
        ndisc h = make_disc(0.000,0.000,0.000,50.000,-0.550,0.550,0.000,0.000) ; h.label = "h" ; h.complement = true ;   
        h.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  -1252.327,-1747.620,0.000,1.000) ;
        nintersection gh = make_intersection(&g, &h) ; gh.label = "gh" ; g.parent = &gh ; h.parent = &gh ;  ;   
        
        nintersection efgh = make_intersection(&ef, &gh) ; efgh.label = "efgh" ; ef.parent = &efgh ; gh.parent = &efgh ;  ;   
        
        nintersection abcdefgh = make_intersection(&abcd, &efgh) ; abcdefgh.label = "abcdefgh" ; abcd.parent = &abcdefgh ; efgh.parent = &abcdefgh ;  ;   
        


        abcdefgh.update_gtransforms();





::

    2018-07-02 14:08:09.044 INFO  [171238] [*NTreeProcess<nnode>::Process@41] before
    NTreeAnalyse height 8 count 17
                                                                  di    

                                                          di          cy

                                                  di          cy        

                                          di          cy                

                                  di          bo                        

                          di          bo                                

                  di          bo                                        

          di          bo                                                

      cy      cy                                                        


    2018-07-02 14:08:09.045 INFO  [171238] [*NTreeProcess<nnode>::Process@56] after
    NTreeAnalyse height 4 count 17
                                                                  in    

                                  in                                 !cy

                  in                              in                    

          in              in              in              in            

      cy     !cy     !bo     !bo     !bo     !bo     !cy     !cy        


    2018-07-02 14:08:09.045 INFO  [171238] [*NTreeProcess<nnode>::Process@57]  soIdx 73 lvIdx 60 height0 8 height1 4 ### LISTED

::

     60 : BotESRCutHols0xbfa7368 

     * this one is a different kettle of fish  


      944     <subtraction name="BotESRCutHols0xbfa7368">
      945       <first ref="BotESR-ChildForBotESRCutHols0xbfa7168"/>

          937     <subtraction name="BotESR-ChildForBotESRCutHols0xbfa7168">
          938       <first ref="BotESR-ChildForBotESRCutHols0xbfa6f68"/>

              931     <subtraction name="BotESR-ChildForBotESRCutHols0xbfa6f68">
              932       <first ref="BotESR-ChildForBotESRCutHols0xbfa6d80"/>

                      925     <subtraction name="BotESR-ChildForBotESRCutHols0xbfa6d80">
                      926       <first ref="BotESR-ChildForBotESRCutHols0xbfa6c00"/>

                          919     <subtraction name="BotESR-ChildForBotESRCutHols0xbfa6c00">
                          920       <first ref="BotESR-ChildForBotESRCutHols0xbfa6a80"/>

                              913     <subtraction name="BotESR-ChildForBotESRCutHols0xbfa6a80">
                              914       <first ref="BotESR-ChildForBotESRCutHols0xbfa6340"/>

                                  907     <subtraction name="BotESR-ChildForBotESRCutHols0xbfa6340">
                                  908       <first ref="BotESR0xbfa5ca8"/>

                                      905     <tube aunit="deg" deltaphi="360" lunit="mm" name="BotESR0xbfa5ca8" rmax="2223" rmin="36.25" startphi="0" z="0.1"/>


                                  909       <second ref="BoxHolInBotESR10xbf96730"/>

                                      906     <box lunit="mm" name="BoxHolInBotESR10xbf96730" x="107" y="401" z="1.1"/>

                                  910       <position name="BotESR-ChildForBotESRCutHols0xbfa6340_pos" unit="mm" x="2035" y="0" z="0"/>
                                  911     </subtraction>

                              915       <second ref="BoxHolInBotESR20xbfa5de0"/>

                                 912     <box lunit="mm" name="BoxHolInBotESR20xbfa5de0" x="107" y="401" z="1.1"/>

                              916       <position name="BotESR-ChildForBotESRCutHols0xbfa6a80_pos" unit="mm" x="-2035" y="0" z="0"/>
                              917     </subtraction>

                          921       <second ref="BoxHolInBotESR30xbfa6bc0"/>

                              918     <box lunit="mm" name="BoxHolInBotESR30xbfa6bc0" x="401" y="107" z="1.1"/>

                          922       <position name="BotESR-ChildForBotESRCutHols0xbfa6c00_pos" unit="mm" x="0" y="2035" z="0"/>
                          923     </subtraction>

                      927       <second ref="BoxHolInBotESR40xbfa6d40"/>

                          924     <box lunit="mm" name="BoxHolInBotESR40xbfa6d40" x="401" y="107" z="1.1"/>

                      928       <position name="BotESR-ChildForBotESRCutHols0xbfa6d80_pos" unit="mm" x="0" y="-2035" z="0"/>
                      929     </subtraction>

              933       <second ref="HolFor2inGdsPMTInBotESR0xbfa6ec0"/>

                      930     <tube aunit="deg" deltaphi="360" lunit="mm" name="HolFor2inGdsPMTInBotESR0xbfa6ec0" rmax="50" rmin="0" startphi="0" z="1.1"/>

              934       <position name="BotESR-ChildForBotESRCutHols0xbfa6f68_pos" unit="mm" x="141.784211691266" y="-342.297366795432" z="0"/>
              935     </subtraction>

          939       <second ref="HolFor2inLsoPmtInBotESR0xbfa70c0"/>

                  936     <tube aunit="deg" deltaphi="360" lunit="mm" name="HolFor2inLsoPmtInBotESR0xbfa70c0" rmax="50" rmin="0" startphi="0" z="1.1"/>

          940       <position name="BotESR-ChildForBotESRCutHols0xbfa7168_pos" unit="mm" x="909.276266994944" y="-1541.90561328498" z="0"/>
          941       <rotation name="BotESR-ChildForBotESRCutHols0xbfa7168_rot" unit="deg" x="0" y="0" z="-67.5"/>
          942     </subtraction>

      946       <second ref="HolFor2inOilPmtInBotESR0xbfa72c0"/>

          943     <tube aunit="deg" deltaphi="360" lunit="mm" name="HolFor2inOilPmtInBotESR0xbfa72c0" rmax="50" rmin="0" startphi="0" z="1.1"/>

      947       <position name="BotESRCutHols0xbfa7368_pos" unit="mm" x="-1252.32704826577" y="-1747.62037187197" z="0"/>
      948     </subtraction>




After duplicate polycone inner omission : are down to 2 different lvIdx : 60,131
--------------------------------------------------------------------------------------

::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/i.py
    [2018-07-02 14:10:29,853] p29372 {opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-02 14:10:29,853] p29372 {opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    BotESRCutHols0xbfa7368
    AmCCo60AcrylicContainer0xc0b23b8

    In [1]: w
    Out[1]: array([ 317,  454,  542,  624, 1017, 1154, 1242, 1324])

    In [2]: np.hstack([pa[w],pb[w],xb[w]])
    Out[2]: 
    array([[1923,   15,  579,  160, 1923,   31,  579,  160,    0,   73,   60,    4],
           [2708,    7,  897,  336, 2724,   15,  897,  336,    0,  126,  131,    3],
           [3072,    7, 1060,  336, 3096,   15, 1060,  336,    0,  126,  131,    3],
           [3350,    7, 1198,  336, 3382,   15, 1198,  336,    0,  126,  131,    3],
           [5415,   15, 1865,  496, 5455,   31, 1865,  496,    0,   73,   60,    4],
           [6200,    7, 2183,  672, 6256,   15, 2183,  672,    0,  126,  131,    3],
           [6564,    7, 2346,  672, 6628,   15, 2346,  672,    0,  126,  131,    3],
           [6842,    7, 2484,  672, 6914,   15, 2484,  672,    0,  126,  131,    3]])


::

    epsilon:ggeo blyth$ ab-diff
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/GParts.txt and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/GParts.txt differ
    Only in /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0: idxBuffer.npy
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/planBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/planBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/primBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/primBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy differ
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0
            ./GParts.txt : 11984 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 5eeee07e08a9a50278a2339dd0b47ac4
    MD5 (partBuffer.npy) = 8d837fba380dfc643968bd23f99d656f
    MD5 (planBuffer.npy) = 94e18d5e55d190c9ed73e04b45ebb404
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 77359e6d3d628e93cb7cf0a4a3824ab3
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0
            ./GParts.txt : 12064 
         ./idxBuffer.npy : (3116, 4) 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (12064, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 2f288a875034810633798bd27d8149d8
    MD5 (idxBuffer.npy) = d9153caee56bf9968645be070023dcd7
    MD5 (partBuffer.npy) = 67699d42b671bb5c4de6c4add42bcbf4
    MD5 (planBuffer.npy) = 43f2892dbf4b8e91231e5d830dee9e03
    MD5 (primBuffer.npy) = 55c578bc96d6cd3d9b1f3f03cbec0aed
    MD5 (tranBuffer.npy) = 74a6d92ff0d830990e81e10434865714
    epsilon:0 blyth$ 


Summing the prim numParts gives the partBuffer sizes::

    In [4]: pa[:,1].sum()
    Out[4]: 11984

    In [5]: pb[:,1].sum()
    Out[5]: 12064





Issue : differences in the global parts for mm0 : 10/248 shapes differ
--------------------------------------------------------------------------

::

    epsilon:0 blyth$ ab-;AB_TAIL=0 ab-diff
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/GParts.txt and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/GParts.txt differ
    Only in /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0: idxBuffer.npy
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/partBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/partBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/planBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/planBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/primBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/primBuffer.npy differ
    Files /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0/tranBuffer.npy and /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0/tranBuffer.npy differ
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/0
            ./GParts.txt : 11984 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (11984, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = 5eeee07e08a9a50278a2339dd0b47ac4
    MD5 (partBuffer.npy) = 8d837fba380dfc643968bd23f99d656f
    MD5 (planBuffer.npy) = 94e18d5e55d190c9ed73e04b45ebb404
    MD5 (primBuffer.npy) = e21f1c240c4d5e9450aff3ddc0fb78d6
    MD5 (tranBuffer.npy) = 77359e6d3d628e93cb7cf0a4a3824ab3
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/0
            ./GParts.txt : 12208 
         ./idxBuffer.npy : (3116, 4) 
        ./planBuffer.npy : (672, 4) 
        ./partBuffer.npy : (12208, 4, 4) 
        ./tranBuffer.npy : (5344, 3, 4, 4) 
        ./primBuffer.npy : (3116, 4) 
    MD5 (GParts.txt) = dc48ac2d37176c9262e09444b0b32671
    MD5 (idxBuffer.npy) = 707efebd5158b7bf6795658bbfe477b0
    MD5 (partBuffer.npy) = a61393ad57ad26addbee90616e045afa
    MD5 (planBuffer.npy) = 43f2892dbf4b8e91231e5d830dee9e03
    MD5 (primBuffer.npy) = 1d3c0b9032dffb0384433b7539dbb37e
    MD5 (tranBuffer.npy) = 74a6d92ff0d830990e81e10434865714
    epsilon:0 blyth$ 


::

    In [10]: w = np.where( pa[:,1] != pb[:,1] )[0]  ## primBuffer indices pointing to prim of different heights

    In [11]: lv = np.unique(xb[w][:,2])   ## convert the primBuffer index (primIdx) into lvIdx using idxBuffer

    In [12]: lv
    Out[12]: array([ 25,  26,  29,  60,  68,  75,  77,  81,  85, 131], dtype=uint32)


Source IDPATH "ma" names make sense, dest IDPATH "mb" gives "random" selection::

    In [19]: print "\n".join(map(lambda _:"%3d : %s " % (_,ma.idx2name[_]), lv ))
     25 : IavTopHub0xc405968 
     26 : CtrGdsOflBotClp0xbf5dec0 
     29 : OcrGdsPrt0xc352518 

     60 : BotESRCutHols0xbfa7368 

     68 : SstTopHub0xc2643d8 
     75 : OavTopHub0xc2c9030 
     77 : CtrLsoOflTopClp0xc178498 
     81 : OcrGdsLsoPrt0xc104978 
     85 : OcrCalLsoPrt0xc1076b0 

    131 : AmCCo60AcrylicContainer0xc0b23b8 



gdml.py old omission of inner subtraction for Polycone::

     760     def as_ncsg(self):
     761         assert self.aunit == "deg" and self.lunit == "mm" and self.deltaphi == 360. and self.startphi == 0.
     762         try:
     763             prims = self.prims()
     764         except ValueError as e:
     765             log.fatal("Polycone.as_ncsg failed ValueError : %r " % e )
     766             return None
     767         pass
     768         cn = TreeBuilder.uniontree(prims, name=self.name + "_uniontree")
     769         inner = self.inner()
     770         #return CSG("difference", left=cn, right=inner ) if inner is not None else cn
     771         return cn


X4Solid.cc::

    748     
    749     bool duplicate_old_inner_omission = true ; 
    750     
    751     if( has_inner && duplicate_old_inner_omission )
    752     {
    753         LOG(error) << " duplicate_old_inner_omission " << duplicate_old_inner_omission ;
    754         inner = NULL ;
    755     }
    756     
    757     nnode* result = inner ? nnode::make_operator_ptr(CSG_DIFFERENCE, cn, inner )  : cn ;
    758     setRoot(result); 










Check GDML of the shapes causing problems : 8/10 involve polycone with inner
--------------------------------------------------------------------------------------

d_gdml **mark** root of the lvIdx, hmm i recall polycone abuse, z-flips::

     25 : IavTopHub0xc405968 

      555   **<polycone aunit="deg" deltaphi="360" lunit="mm" name="IavTopHub0xc405968" startphi="0">
      556       <zplane rmax="100" rmin="75" z="0"/>
      557       <zplane rmax="100" rmin="75" z="85.5603682281126"/>
      558       <zplane rmax="150" rmin="75" z="85.5603682281126"/>
      559       <zplane rmax="150" rmin="75" z="110.560368228113"/>
      560     </polycone>

     26 : CtrGdsOflBotClp0xbf5dec0 

      561    **<polycone aunit="deg" deltaphi="360" lunit="mm" name="CtrGdsOflBotClp0xbf5dec0" startphi="0">
      562       <zplane rmax="150" rmin="31.5" z="0"/>
      563       <zplane rmax="150" rmin="31.5" z="25"/>
      564       <zplane rmax="36.5" rmin="31.5" z="25"/>
      565       <zplane rmax="36.5" rmin="31.5" z="30"/>
      566     </polycone>

     29 : OcrGdsPrt0xc352518 

      569     <polycone aunit="deg" deltaphi="360" lunit="mm" name="OcrGdsPrtPln0xbfa1408" startphi="0">
      570       <zplane rmax="100" rmin="75" z="0"/>
      571       <zplane rmax="100" rmin="75" z="160"/>
      572       <zplane rmax="150" rmin="75" z="160"/>
      573       <zplane rmax="150" rmin="75" z="185"/>
      574     </polycone>
      575     <cone aunit="deg" deltaphi="360" lunit="mm" name="OcrGdsPrtCon0xc352418" rmax1="1520.39278882354" rmax2="100" rmin1="0" rmin2="0" startphi="0" z="74.4396317718873"/>
      576   **<subtraction name="OcrGdsPrt0xc352518">
      577       <first ref="OcrGdsPrtPln0xbfa1408"/>
      578       <second ref="OcrGdsPrtCon0xc352418"/>
      579       <position name="OcrGdsPrt0xc352518_pos" unit="mm" x="-516.622633692872" y="1247.23736889024" z="37.2198158859437"/>
      580     </subtraction>

     60 : BotESRCutHols0xbfa7368 

     * this one is a different kettle of fish  

     68 : SstTopHub0xc2643d8 

     * z-flips 

     1062     <polycone aunit="deg" deltaphi="360" lunit="mm" name="SstTopHubBot0xc2635b8" startphi="0">
     1063       <zplane rmax="220.5" rmin="150.5" z="-340"/>
     1064       <zplane rmax="220.5" rmin="150.5" z="-320"/>
     1065     </polycone>
     1066     <polycone aunit="deg" deltaphi="360" lunit="mm" name="SstTopHubMain0xc263d80" startphi="0">
     1067       <zplane rmax="170.5" rmin="150.5" z="-320"/>
     1068       <zplane rmax="170.5" rmin="150.5" z="0"/>
     1069     </polycone>
     1070   **<union name="SstTopHub0xc2643d8">
     1071       <first ref="SstTopHubBot0xc2635b8"/>
     1072       <second ref="SstTopHubMain0xc263d80"/>
     1073     </union>

     75 : OavTopHub0xc2c9030 

     1124     <polycone aunit="deg" deltaphi="360" lunit="mm" name="OavTopHub0xc2c9030" startphi="0">
     1125       <zplane rmax="125" rmin="50" z="0"/>
     1126       <zplane rmax="125" rmin="50" z="57"/>
     1127       <zplane rmax="68" rmin="50" z="57"/>
     1128       <zplane rmax="68" rmin="50" z="90"/>
     1129       <zplane rmax="98" rmin="50" z="90"/>
     1130       <zplane rmax="98" rmin="50" z="120"/>
     1131     </polycone>

     77 : CtrLsoOflTopClp0xc178498 

     1133     <polycone aunit="deg" deltaphi="360" lunit="mm" name="CtrLsoOflTopClp0xc178498" startphi="0">
     1134       <zplane rmax="102.5" rmin="50" z="0"/>
     1135       <zplane rmax="102.5" rmin="50" z="16"/>
     1136       <zplane rmax="100" rmin="50" z="16"/>
     1137       <zplane rmax="100" rmin="50" z="184"/>
     1138       <zplane rmax="112.5" rmin="50" z="184"/>
     1139       <zplane rmax="112.5" rmin="50" z="200"/>
     1140     </polycone>

     81 : OcrGdsLsoPrt0xc104978 

     1144     <polycone aunit="deg" deltaphi="360" lunit="mm" name="OcrGdsLsoPrtPln0xc104000" startphi="0">
     1145       <zplane rmax="68" rmin="50" z="0"/>
     1146       <zplane rmax="68" rmin="50" z="184.596041605889"/>
     1147       <zplane rmax="98" rmin="50" z="184.596041605889"/>
     1148       <zplane rmax="98" rmin="50" z="214.596041605889"/>
     1149     </polycone>
     1150     <cone aunit="deg" deltaphi="360" lunit="mm" name="OcrGdsLsoPrtCon0xc104870" rmax1="1930" rmax2="125" rmin1="0" rmin2="0" startphi="0" z="94.5960416058894"/>
     1151   ** <subtraction name="OcrGdsLsoPrt0xc104978">
     1152       <first ref="OcrGdsLsoPrtPln0xc104000"/>
     1153       <second ref="OcrGdsLsoPrtCon0xc104870"/>
     1154       <position name="OcrGdsLsoPrt0xc104978_pos" unit="mm" x="-516.622633692872" y="1247.23736889024" z="47.2980208029447"/>
     1155     </subtraction>

     85 : OcrCalLsoPrt0xc1076b0 

     1177     <polycone aunit="deg" deltaphi="360" lunit="mm" name="OcrCalLsoPrtPln0xc2fadd8" startphi="0">
     1178       <zplane rmax="68" rmin="50" z="0"/>
     1179       <zplane rmax="68" rmin="50" z="184.596041605889"/>
     1180       <zplane rmax="98" rmin="50" z="184.596041605889"/>
     1181       <zplane rmax="98" rmin="50" z="214.596041605889"/>
     1182     </polycone>
     1183     <cone aunit="deg" deltaphi="360" lunit="mm" name="OcrCalLsoPrtCon0xc1075a8" rmax1="1930" rmax2="125" rmin1="0" rmin2="0" startphi="0" z="94.5960416058894"/>
     1184    **<subtraction name="OcrCalLsoPrt0xc1076b0">
     1185       <first ref="OcrCalLsoPrtPln0xc2fadd8"/>
     1186       <second ref="OcrCalLsoPrtCon0xc1075a8"/>
     1187       <position name="OcrCalLsoPrt0xc1076b0_pos" unit="mm" x="678.306383867122" y="-1637.57647137626" z="47.2980208029447"/>
     1188     </subtraction>

     131 : AmCCo60AcrylicContainer0xc0b23b8 

     * this one is a different kettle of fish  





::

    ## discrepant b-numParts are all one level up from a-numParts 

    In [18]: np.hstack([pa[w],pb[w],xb[w]])
    Out[18]:        *A*                     *B*                 *soIdx* *lvIdx* height1
    array([[  38,    3,   14,    0,   38,    7,   14,    0,    0,   38,   25,    2],
           [  41,    3,   15,    0,   45,    7,   15,    0,    0,   39,   26,    2],
           [  48,    7,   18,    0,   56,   15,   18,    0,    0,   42,   29,    3],
           [  55,    3,   20,    0,   71,    7,   20,    0,    0,   39,   26,    2],
           [1923,   15,  579,  160, 1943,   31,  579,  160,    0,   73,   60,    4],
           [2229,    3,  692,  248, 2265,    7,  692,  248,    0,   79,   68,    2],
           [2448,    7,  781,  336, 2488,   15,  781,  336,    0,   86,   75,    3],
           [2458,    7,  783,  336, 2506,   15,  783,  336,    0,   88,   77,    3],
           [2468,    7,  787,  336, 2524,   15,  787,  336,    0,   92,   81,    3],
           [2478,    7,  790,  336, 2542,   15,  790,  336,    0,   88,   77,    3],
           [2494,    7,  797,  336, 2566,   15,  797,  336,    0,   96,   85,    3],
           [2504,    7,  800,  336, 2584,   15,  800,  336,    0,   88,   77,    3],
           [2708,    7,  897,  336, 2796,   15,  897,  336,    0,  126,  131,    3],
           [3072,    7, 1060,  336, 3168,   15, 1060,  336,    0,  126,  131,    3],
           [3350,    7, 1198,  336, 3454,   15, 1198,  336,    0,  126,  131,    3],
           [3530,    3, 1300,  336, 3642,    7, 1300,  336,    0,   38,   25,    2],
           [3533,    3, 1301,  336, 3649,    7, 1301,  336,    0,   39,   26,    2],
           [3540,    7, 1304,  336, 3660,   15, 1304,  336,    0,   42,   29,    3],
           [3547,    3, 1306,  336, 3675,    7, 1306,  336,    0,   39,   26,    2],
           [5415,   15, 1865,  496, 5547,   31, 1865,  496,    0,   73,   60,    4],
           [5721,    3, 1978,  584, 5869,    7, 1978,  584,    0,   79,   68,    2],
           [5940,    7, 2067,  672, 6092,   15, 2067,  672,    0,   86,   75,    3],
           [5950,    7, 2069,  672, 6110,   15, 2069,  672,    0,   88,   77,    3],
           [5960,    7, 2073,  672, 6128,   15, 2073,  672,    0,   92,   81,    3],
           [5970,    7, 2076,  672, 6146,   15, 2076,  672,    0,   88,   77,    3],
           [5986,    7, 2083,  672, 6170,   15, 2083,  672,    0,   96,   85,    3],
           [5996,    7, 2086,  672, 6188,   15, 2086,  672,    0,   88,   77,    3],
           [6200,    7, 2183,  672, 6400,   15, 2183,  672,    0,  126,  131,    3],
           [6564,    7, 2346,  672, 6772,   15, 2346,  672,    0,  126,  131,    3],
           [6842,    7, 2484,  672, 7058,   15, 2484,  672,    0,  126,  131,    3]])


::

    2018-07-02 11:12:46.440 INFO  [126742] [*NTreeProcess<nnode>::Process@41] before
    NTreeAnalyse height 2 count 5
                  di    

          un          cy

      cy      cy        


    2018-07-02 11:12:46.440 INFO  [126742] [*NTreeProcess<nnode>::Process@56] after
    NTreeAnalyse height 2 count 5
                  di    

          un          cy

      cy      cy        


    2018-07-02 11:12:46.440 INFO  [126742] [*NTreeProcess<nnode>::Process@57]  soIdx 38 lvIdx 25 height0 2 height1 2 ### LISTED




Rerun, writing to $TMP::

    op --gdml2gltf

    
Ensure writes to TMP via change to gdml2gltf.py prior to importing opticks_main::

     19 ## want to exercise python tree balancing without disturbing other things
     20 os.environ["OPTICKS_GLTFPATH"] = os.path.expandvars("$TMP/gdml2gltf/g4_00.gltf")
     22 from opticks.ana.base import opticks_main
     23 from opticks.analytic.sc import gdml2gltf_main
     26 if __name__ == '__main__':
     29     args = opticks_main()
     31     sc = gdml2gltf_main( args )



Hmm the old geometry misses inner difference "di cy", 
(from a cylinder with an rmin ?)
explaining the off-by-one.

cat /tmp/blyth/opticks/gdml2gltf/extras/26/NNodeTest_26.cc
cat /tmp/blyth/opticks/gdml2gltf/extras/25/NNodeTest_25.cc


::

    // generated by nnode_test_cpp.py : 20180702-1255 
    // opticks-;opticks-nnt 25 
    // opticks-;opticks-nnt-vi 25 

    // regenerate with gdml2gltf.py 

    #include <vector>
    #include "SSys.hh"
    #include "NGLMExt.hpp"
    #include "NCSG.hpp"
    #include "NSceneConfig.hpp"
    #include "NBBox.hpp"
    #include "NNode.hpp"
    #include "NPrimitives.hpp"
    #include "PLOG.hh"
    #include "NPY_LOG.hh"
    
    int main(int argc, char** argv)
    {
        PLOG_(argc, argv);
        NPY_LOG__ ; 

        // generated by nnode_test_cpp.py : 20180702-1255 
        // opticks-;opticks-nnt 25 
        // opticks-;opticks-nnt-vi 25 
        
        ncylinder a = make_cylinder(0.000,0.000,0.000,100.000,0.000,85.560,0.000,0.000) ; a.label = "a" ;   
        ncylinder b = make_cylinder(0.000,0.000,0.000,150.000,85.560,110.560,0.000,0.000) ; b.label = "b" ;   
        nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   
        


        ab.update_gtransforms();

        unsigned verbosity = SSys::getenvint("VERBOSITY", 1) ; 
        ab.verbosity = verbosity ; 
        //ab.dump() ; 

        const char* boundary = "Rock//perfectAbsorbSurface/Vacuum" ;




::

    2018-07-02 11:12:46.445 INFO  [126742] [*NTreeProcess<nnode>::Process@41] before
    NTreeAnalyse height 3 count 7
                          di    

                  di          co

          un          cy        

      cy      cy                


    2018-07-02 11:12:46.445 INFO  [126742] [*NTreeProcess<nnode>::Process@56] after
    NTreeAnalyse height 3 count 7
                          di    

                  di          co

          un          cy        

      cy      cy                


    2018-07-02 11:12:46.445 INFO  [126742] [*NTreeProcess<nnode>::Process@57]  soIdx 42 lvIdx 29 height0 3 height1 3 ### LISTED


::

    // generated by nnode_test_cpp.py : 20180702-1255 
    // opticks-;opticks-nnt 29 
    // opticks-;opticks-nnt-vi 29 
    
    ncylinder a = make_cylinder(0.000,0.000,0.000,100.000,0.000,160.000,0.000,0.000) ; a.label = "a" ;   
    ncylinder b = make_cylinder(0.000,0.000,0.000,150.000,160.000,185.000,0.000,0.000) ; b.label = "b" ;   
    nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   
    
    ncone c = make_cone(1520.393,-37.220,100.000,37.220) ; c.label = "c" ;   
    c.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  -516.623,1247.237,37.220,1.000) ;
    ndifference abc = make_difference(&ab, &c) ; abc.label = "abc" ; ab.parent = &abc ; c.parent = &abc ;  ;   
    


::


    2018-07-02 11:12:47.711 INFO  [126742] [*NTreeProcess<nnode>::Process@41] before
    NTreeAnalyse height 2 count 7
                  un            

          di              di    

      cy      cy      cy      cy


    2018-07-02 11:12:47.711 INFO  [126742] [*NTreeProcess<nnode>::Process@56] after
    NTreeAnalyse height 2 count 7
                  un            

          di              di    

      cy      cy      cy      cy


    2018-07-02 11:12:47.711 INFO  [126742] [*NTreeProcess<nnode>::Process@57]  soIdx 79 lvIdx 68 height0 2 height1 2 ### LISTED




    epsilon:extras blyth$ n=68;cat /tmp/blyth/opticks/gdml2gltf/extras/$n/NNodeTest_$n.cc 

    ...
        // generated by nnode_test_cpp.py : 20180702-1255 
        // opticks-;opticks-nnt 68 
        // opticks-;opticks-nnt-vi 68 
        
        ncylinder a = make_cylinder(0.000,0.000,0.000,220.500,-340.000,-320.000,0.000,0.000) ; a.label = "a" ;   
        ncylinder b = make_cylinder(0.000,0.000,0.000,170.500,-320.000,0.000,0.000,0.000) ; b.label = "b" ;   
        b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,0.000,0.000,1.000) ;
        nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   
        


        ab.update_gtransforms();






Update the listed in NTreeProcess with the discrepant heights::

     28 template <typename T>
     29 T* NTreeProcess<T>::Process( T* root_ , unsigned soIdx, unsigned lvIdx )  // static
     30 {
     31     //if( LVList == NULL )
     32     //     LVList = new std::vector<unsigned> {25,  26,  29,  60,  65,  68,  75,  77,  81,  85, 131, 140} ;
     33     if( LVList == NULL )
     34          LVList = new std::vector<unsigned> {25,  26,  29,  60,  68,  75,  77,  81,  85, 131};
     35     
     36     
     37     if( ProcBuffer == NULL ) ProcBuffer = NPY<unsigned>::make(0,4) ;
     38     
     39     bool listed = std::find(LVList->begin(), LVList->end(), lvIdx ) != LVList->end() ;
     40     
     41     if(listed) LOG(info) << "before\n" << NTreeAnalyse<T>::Desc(root_) ;
     42      // dump it here, prior to the inplace positivization 
     43      
     44     unsigned height0 = root_->maxdepth();
     45     
     46     NTreeProcess<T> proc(root_);
     47     
     48     assert( height0 == proc.balancer->height0 );
     49     
     50     T* result = proc.result ;
     51     
     52     unsigned height1 = result->maxdepth();
     53     
     54     if(listed)
     55     {
     56          LOG(info) << "after\n" << NTreeAnalyse<T>::Desc(result) ;
     57          LOG(info) 
     58          << " soIdx " << soIdx
     59          << " lvIdx " << lvIdx
     60          << " height0 " << height0
     61          << " height1 " << height1
     62          << " " << ( listed ? "### LISTED" : "" )
     63          ;
     64     }    
     65     
     66     if(ProcBuffer) ProcBuffer->add(soIdx, lvIdx, height0, height1);
     67 
     68     return result ;
     69 }


gdml.py::

     278     def as_cylinder(self, nudge_inner=0.01):
     279         hz = self.z/2.
     280         has_inner = self.rmin > 0.
     281 
     282         if has_inner:
     283             dz = hz*nudge_inner
     284             inner = self.make_cylinder(self.rmin, -(hz+dz), (hz+dz), self.name + "_inner")
     285         else:
     286             inner = None
     287         pass
     288         outer = self.make_cylinder(self.rmax, -hz, hz, self.name + "_outer" )
     289         tube = CSG("difference", left=outer, right=inner, name=self.name + "_difference" ) if has_inner else outer
     290 
     291         has_deltaphi = self.deltaphi < 360
     292         if has_deltaphi and self.deltaphi_segment_enabled:
     293 
     294              assert self.aunit == 'deg'
     295              phi0 = self.startphi
     296              phi1 = self.startphi + self.deltaphi
     297              sz  = self.z*1.01
     298              sr  = self.rmax*1.5
     299 
     300              ## TODO: calulate how much the segmenting prism needs to poke beyind the radius 
     301              ##       to avoid the outside plane from cutting the cylinder 
     302       
     303              segment = CSG.MakeSegment(phi0, phi1, sz, sr )
     304              log.info("as_cylinder doing phi0/phi1/sz/sr segmenting : name %s phi0 %s phi1 %s sz %s sr %s " % (self.name, phi0, phi1, sz, sr))
     305              tube_segment = CSG("intersection", left=tube, right=segment )
     306 
     307              #tube_segment = self.deltaphi_slab_segment(tube, phi0, phi1, dist)
     308              #result.balance_disabled = True 
     309 
     310              result = tube_segment
     311         else:
     312              result = tube
     313         pass
     314         return result


