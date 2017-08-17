j1707 node accounting
========================

Issue : more than quarter million nodes
-----------------------------------------

* is that really the case ?
* are they maximally instanced ? 
* get a feeling for tree structure, is sibling instance grouping possible ? 

  * somehow grouping 3inch together with 20inch would be big win ?


To tabulate
--------------

* node counts
* instance counts, node counts per instance
* node counts in non-instanced globals
* buffer sizes, opengl(triangulated) optix (analytic+triangulated)


Eyeballing source GDML
----------------------------------

::

    rg () 
    { 
        vim -R /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml
    }

    simon:issues blyth$ wc -l /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml
      277195 /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml


4 repeaters are apparent::

    lSteel0x14dde40         ~250
    lFasteners0x1506370     ~400
    lMaskVirtual0x1816910   ~18k 
    PMT_3inch_log0x1c9ef80  ~36k  

::

    000051   <materials>
       241   </materials>

       243   <solids>
       ...
       551   </solids> 

       553   <structure>


       737     <volume name="lInnerWater0x14dccf0">
       738       <materialref ref="Water0x14d1d00"/>
       739       <solidref ref="sInnerWater0x14dcb00"/>
       740       <physvol name="pAcylic0x14dda00">
       741         <volumeref ref="lAcrylic0x14dd290"/>
       742       </physvol>

       743       <physvol name="lSteel_phys0x14e01d0">     ~250 
       744         <volumeref ref="lSteel0x14dde40"/>
       745         <position name="lSteel_phys0x14e01d0_pos" unit="mm" x="3871.31568302668" y="0" z="18213.1083256635"/>
       746         <rotation name="lSteel_phys0x14e01d0_rot" unit="deg" x="180" y="12" z="180"/>
       747       </physvol>
       ...
      3138       <physvol name="lSteel_phys0x1504b20">
      3139         <volumeref ref="lSteel0x14dde40"/>
      3140         <position name="lSteel_phys0x1504b20_pos" unit="mm" x="3739.40379995337" y="-1001.97022837138" z="-18213.1083256635"/>
      3141         <rotation name="lSteel_phys0x1504b20_rot" unit="deg" x="3.1488779024914" y="11.5853397932875" z="15.3195239528622"/>
      3142       </physvol>

      3143       <physvol name="lFasteners_phys0x15072a0">   ~2000/5 ~400
      3144         <volumeref ref="lFasteners0x1506370"/>
      3145         <position name="lFasteners_phys0x15072a0_pos" unit="mm" x="3706.23380051738" y="0" z="17436.4591306808"/>
      3146         <rotation name="lFasteners_phys0x15072a0_rot" unit="deg" x="180" y="12" z="180"/>
      3147       </physvol>
      ....
      5538       <physvol name="lFasteners_phys0x152f3a0">
      5539         <volumeref ref="lFasteners0x1506370"/>
      5540         <position name="lFasteners_phys0x152f3a0_pos" unit="mm" x="3579.94694618522" y="-959.243893176594" z="-17436.4591306808"/>
      5541         <rotation name="lFasteners_phys0x152f3a0_rot" unit="deg" x="3.1488779024914" y="11.5853397932875" z="15.3195239528622"/>
      5542       </physvol>

      5543       <physvol name="lMaskVirtual_phys0x1868ad0">    ~90000/5 ~18k
      5544         <volumeref ref="lMaskVirtual0x1816910"/>
      5545         <position name="lMaskVirtual_phys0x1868ad0_pos" unit="mm" x="1065.41160578968" y="0" z="19470.8730700564"/>
      5546         <rotation name="lMaskVirtual_phys0x1868ad0_rot" unit="deg" x="180" y="3.132" z="180"/>
      5547       </physvol>
     .....
     94233       <physvol name="lMaskVirtual_phys0x1c9d5f0">
     94234         <volumeref ref="lMaskVirtual0x1816910"/>
     94235         <position name="lMaskVirtual_phys0x1c9d5f0_pos" unit="mm" x="19495.6188393558" y="-271.023178062762" z="-312.07772670818"/>
     94236         <rotation name="lMaskVirtual_phys0x1c9d5f0_rot" unit="deg" x="40.9726060827552" y="88.785428615014" z="40.9789798622846"/>
     94237       </physvol>

     94238       <physvol name="PMT_3inch_log_phys0x181f1b0">   ~ (277097-94238)/5 ~36k 
     94239         <volumeref ref="PMT_3inch_log0x1c9ef80"/>
     94240         <position name="PMT_3inch_log_phys0x181f1b0_pos" unit="mm" x="1402.8418375672" y="247.35886562974" z="19397.7665820157"/>
     94241         <rotation name="PMT_3inch_log_phys0x181f1b0_rot" unit="deg" x="-179.269408113041" y="4.13608063277865" z="-169.973618119703"/>
     94242       </physvol>
      ....
    277093       <physvol name="PMT_3inch_log_phys0x2547230">
    277094         <volumeref ref="PMT_3inch_log0x1c9ef80"/>
    277095         <position name="PMT_3inch_log_phys0x2547230_pos" unit="mm" x="529.088922853645" y="-305.469632034802" z="-19440.4025991135"/>
    277096         <rotation name="PMT_3inch_log_phys0x2547230_rot" unit="deg" x="0.900222120901556" y="1.55878160365665" z="30.0122466708415"/>
    277097       </physvol>



    277135     <volume name="lWorld0x14d9c00">
    277136       <materialref ref="Galactic0x1476410"/>
    277137       <solidref ref="sWorld0x14d9850"/>
    277138       <physvol name="pTopRock0x14da630">
    277139         <volumeref ref="lTopRock0x14da5a0"/>
    277140         <position name="pTopRock0x14da630_pos" unit="mm" x="0" y="0" z="32550"/>

    ::
        
        In [3]: a = np.load(os.path.expandvars("$TMP/NScene_triple.npy"))  

        In [4]: a.shape
        Out[4]: (290276, 3, 4, 4)

        In [10]: a[1]   ## so these are in traversal order from World
        Out[10]: 
        array([[[     1.,      0.,      0.,      0.],
                [     0.,      1.,      0.,      0.],
                [     0.,      0.,      1.,      0.],
                [     0.,      0.,  32550.,      1.]],

               [[     1.,      0.,      0.,      0.],
                [     0.,      1.,      0.,      0.],
                [     0.,      0.,      1.,      0.],
                [     0.,      0., -32550.,      1.]],

               [[     1.,      0.,      0.,      0.],
                [     0.,      1.,      0.,      0.],
                [     0.,      0.,      1., -32550.],
                [     0.,      0.,      0.,      1.]]], dtype=float32)


    277141       </physvol>
    277142       <physvol name="pBtmRock0x14db9f0">
    277143         <volumeref ref="lBtmRock0x14db220"/>
    277144       </physvol>
    277145     </volume>
    277146     <skinsurface name="Tube_surf" surfaceproperty="TubeSurface">
    ......
    277185     <bordersurface name="CDTyvekSurface" surfaceproperty="CDTyvekOpticalSurface">
    277186       <physvolref ref="pOuterWaterPool0x14dba40"/>
    277187       <physvolref ref="pCentralDetector0x14ddb50"/>
    277188     </bordersurface>
    277189   </structure>
    277190 
    277191   <setup name="Default" version="1.0">
    277192     <world ref="lWorld0x14d9c00"/>
    277193   </setup>
    277194 
    277195 </gdml>
     



