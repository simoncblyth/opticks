j1707 node accounting
========================

Issue : more than quarter million nodes
-----------------------------------------

* is that really the case ? YES
* are they maximally instanced ?  ALMOST, 480x fastener+strut COULD BE SIBLING INSTANCED
* get a feeling for tree structure


Sibling Instance Grouping ?
------------------------------

* grouping 3inch together with 20inch unfortunately impossible : they have complicated layouts
  that do not align to each other

* BUT the 480x fastener+strut looks like it could be instanced together 


To tabulate
--------------

* node counts
* instance counts, node counts per instance
* node counts in non-instanced globals
* buffer sizes, opengl(triangulated) optix (analytic+triangulated)


Analytic Buffer counts
------------------------

::

    2017-08-17 21:04:33.455 INFO  [348837] [OGeo::convert@203] OGeo::convert DONE  numMergedMesh: 5
    2017-08-17 21:04:33.455 INFO  [348837] [OGeo::dumpStats@572] OGeo::dumpStats num_stats 5
     mmIndex   0 numPrim    22 numPart   146 numTran(triples)    35 numPlan     0
     mmIndex   1 numPrim     5 numPart     7 numTran(triples)     5 numPlan     0
     mmIndex   2 numPrim     6 numPart   100 numTran(triples)    23 numPlan     0
     mmIndex   3 numPrim     1 numPart     7 numTran(triples)     2 numPlan     0
     mmIndex   4 numPrim     1 numPart     3 numTran(triples)     1 numPlan     0

* TODO: check these : 100 parts for mmIndex  2 ?



From geocache creation : note all 290276 volumes listed for mm0 : check implications on buffer sizes
-------------------------------------------------------------------------------------------------------

::

    op --j1707 -G

    210     2017-08-17 14:05:30.538 INFO  [213429] [GTreeCheck::labelTree@377] GTreeCheck::labelTree count of non-zero setRepeatIndex 290254
    211     2017-08-17 14:05:43.338 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 290276
    212     2017-08-17 14:05:43.596 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 36572 numSolids 5
    213     2017-08-17 14:05:43.809 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 17739 numSolids 6
    214     2017-08-17 14:05:44.019 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 480 numSolids 1
    215     2017-08-17 14:05:44.229 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 4 numPlacements 480 numSolids 1
    216

    36572*5 + 17739*6 + 480*1 + 480*1 = 290254

    290276 - 290254 = 22     ## 22 global volumes 

    35 - 5 - 6 - 1 - 1 = 22   ## subtract 4 instances solid counts from total number of distinct solids (35)   gives the remainder


NScene first/last mesh off-by-1 ? Where are the 22 ridx=0 global volumes ?
------------------------------------------------------------------------------

::

    2017-08-17 17:39:00.235 INFO  [282274] [NScene::labelTree@1391] NScene::labelTree label_count (non-zero ridx labelTree_r) 290254 num_repeat_candidates 4
    2017-08-17 17:39:00.235 INFO  [282274] [NScene::dumpRepeatCount@1429] NScene::dumpRepeatCount m_verbosity 1
     ridx   1 count 182860
     ridx   2 count 106434
     ridx   3 count   480
     ridx   4 count   480
    2017-08-17 17:39:00.236 INFO  [282274] [NScene::dumpRepeatCount@1446] NScene::dumpRepeatCount totCount 290254


::

    36572*5 = 182860
    17739*6 = 106434
    480*1   =    480
    480*1   =    480
             --------
              290254 




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


After fix to catch lSteel GTreeCheck agrees::

    2017-08-17 14:05:30.178 INFO  [213429] [GTreeCheck::dumpRepeatCandidates@305] GTreeCheck::dumpRepeatCandidates 
     pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 placements  36572 n PMT_3inch_log0x1c9ef80
     pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 placements  17739 n lMaskVirtual0x1816910
     pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 placements    480 n lFasteners0x1506370
     pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 placements    480 n lSteel0x14dde40


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
     



