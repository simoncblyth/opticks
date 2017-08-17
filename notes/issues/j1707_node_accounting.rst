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
     



Hmm, vertex_min dont make much sense in analytic context... 

Recall GTreeCheck is for triangulated

::

    2017-08-17 12:33:00.159 INFO  [183431] [*GScintillatorLib::createBuffer@102] GScintillatorLib::createBuffer  ni 1 nj 4096 nk 1
    2017-08-17 12:33:00.159 FATAL [183431] [*GScintillatorLib::constructInvertedReemissionCDF@170] GScintillatorLib::constructInvertedReemissionCDF  was expecting to trim 2 values  l_srrd 33 l_rrd 39
    2017-08-17 12:33:00.159 INFO  [183431] [GPropertyLib::close@384] GPropertyLib::close type GScintillatorLib buf 1,4096,1
    2017-08-17 12:33:10.310 INFO  [183431] [GTreeCheck::findRepeatCandidates@161] GTreeCheck::findRepeatCandidates nall 35 repeat_min 120 vertex_min 250 candidates marked with ** 
    2017-08-17 12:33:10.328 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   0 pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 nvert    802 n PMT_3inch_log0x1c9ef80
    2017-08-17 12:33:10.346 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   1 pdig 1f0e3dad99908345f7439f8ffabdffc4 ndig  36572 nprog      0 nvert     50 n PMT_3inch_cntr_log0x1c9f1f0
    2017-08-17 12:33:10.366 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   2 pdig 70efdf2ec9b086079795c442636b55fb ndig  36572 nprog      0 nvert    146 n PMT_3inch_inner2_log0x1c9f120
    2017-08-17 12:33:10.386 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   3 pdig c74d97b01eae257e44aa9d5bade97baf ndig  36572 nprog      0 nvert    122 n PMT_3inch_inner1_log0x1c9f050
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   4 pdig 873d395b9e0f186e0a9369ced7e84293 ndig  36572 nprog      2 nvert    486 n PMT_3inch_body_log0x1c9eef0
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   5 pdig d3d9446802a44259755d38e6d163e820 ndig  17739 nprog      0 nvert    484 n lMask0x18170e0
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   6 pdig c20ad4d76fe97759aa27a0c99bff6710 ndig  17739 nprog      0 nvert    482 n PMT_20inch_inner2_log0x1863310
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   7 pdig 6512bd43d9caa6e02c990b0a82652dca ndig  17739 nprog      0 nvert    194 n PMT_20inch_inner1_log0x1863280
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   8 pdig 511dd06bf687b1e989d4ac84e25bc0a3 ndig  17739 nprog      2 nvert   1254 n PMT_20inch_body_log0x1863160
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   9 pdig fd99bf5972e7592724cbd49bfb448953 ndig  17739 nprog      3 nvert   1832 n PMT_20inch_log0x18631f0
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i  10 pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 nvert   2366 n lMaskVirtual0x1816910
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i  11 pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 nvert    914 n lFasteners0x1506370
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i  12 pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 nvert     96 n lSteel0x14dde40
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i  13 pdig 1679091c5a880faf6fb5e6087eb1b2dc ndig      1 nprog      0 nvert    362 n lTarget0x14dd830
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i  14 pdig 3316c24dd38f0f2ca5d7250814b99d1a ndig      1 nprog      1 nvert    724 n lAcrylic0x14dd290
    2017-08-17 12:33:10.935 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig 873d395b9e0f186e0a9369ced7e84293 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.935 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig d3d9446802a44259755d38e6d163e820 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig c20ad4d76fe97759aa27a0c99bff6710 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig 511dd06bf687b1e989d4ac84e25bc0a3 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig fd99bf5972e7592724cbd49bfb448953 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::dumpRepeatCandidates@255] GTreeCheck::dumpRepeatCandidates 
     pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 placements  36572 n PMT_3inch_log0x1c9ef80
     pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 placements  17739 n lMaskVirtual0x1816910
     pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 placements    480 n lFasteners0x1506370
    2017-08-17 12:33:11.212 INFO  [183431] [GTreeCheck::labelTree@327] GTreeCheck::labelTree count of non-zero setRepeatIndex 289774
    2017-08-17 12:33:23.880 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 290276
    2017-08-17 12:33:24.124 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 36572 numSolids 5
    2017-08-17 12:33:24.340 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 17739 numSolids 6
    2017-08-17 12:33:24.549 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 480 numSolids 1
    2017-08-17 12:33:24.549 INFO  [183431] [TimesTable::dump@103] Timer::dump filter: NONE
              0.000      t_absolute        t_delta
              0.716           0.716          0.716 : deltacheck
             10.192          10.909         10.192 : traverse
              0.144          11.053          0.144 : labelTree
             13.337          24.390         13.337 : makeMergedMeshAndInstancedBuffers
    2017-08-17 12:33:24.549 INFO  [183431] [GColorizer::traverse@93] GColorizer::traverse START

