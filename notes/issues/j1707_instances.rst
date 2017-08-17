j1707 instances
===================






::

     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 

     **  ##  idx  11 pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 nvert    914 n lFasteners0x1506370
     **  ##  idx  12 pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 nvert     96 n lSteel0x14dde40



lFasteners0x1506370
---------------------

::

       610     <volume name="lFasteners0x1506370">
       611       <materialref ref="Copper0x14adbe0"/>
       612       <solidref ref="sFasteners0x1506180"/>
       613     </volume>


lSteel0x14dde40
-----------------

::

       606     <volume name="lSteel0x14dde40">
       607       <materialref ref="Steel0x14aa2a0"/>
       608       <solidref ref="sStrut0x14ddd50"/>
       609     </volume>



PMT_3inch_log0x1c9ef80
-------------------------


::

     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 

     **  ##  idx   0 pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 nvert    802 n PMT_3inch_log0x1c9ef80
     **      idx   1 pdig 1f0e3dad99908345f7439f8ffabdffc4 ndig  36572 nprog      0 nvert     50 n PMT_3inch_cntr_log0x1c9f1f0
     **      idx   2 pdig 70efdf2ec9b086079795c442636b55fb ndig  36572 nprog      0 nvert    146 n PMT_3inch_inner2_log0x1c9f120
     **      idx   3 pdig c74d97b01eae257e44aa9d5bade97baf ndig  36572 nprog      0 nvert    122 n PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 873d395b9e0f186e0a9369ced7e84293 ndig  36572 nprog      2 nvert    486 n PMT_3inch_body_log0x1c9eef0



::

       675     <volume name="PMT_3inch_log0x1c9ef80">
       676       <materialref ref="Water0x14d1d00"/>
       677       <solidref ref="PMT_3inch_pmt_solid0x1c9e270"/>
       678       <physvol name="PMT_3inch_body_phys0x1c9f2c0">
       679         <volumeref ref="PMT_3inch_body_log0x1c9eef0"/>
       680       </physvol>
       681       <physvol name="PMT_3inch_cntr_phys0x1c9f4d0">
       682         <volumeref ref="PMT_3inch_cntr_log0x1c9f1f0"/>
       683       </physvol>
       684     </volume>

           661     <volume name="PMT_3inch_body_log0x1c9eef0">
           662       <materialref ref="Pyrex0x14c4190"/>
           663       <solidref ref="PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0"/>
           664       <physvol name="PMT_3inch_inner1_phys0x1c9f370">
           665         <volumeref ref="PMT_3inch_inner1_log0x1c9f050"/>
           666       </physvol>
           667       <physvol name="PMT_3inch_inner2_phys0x1c9f420">
           668         <volumeref ref="PMT_3inch_inner2_log0x1c9f120"/>
           669       </physvol>
           670     </volume>

               653     <volume name="PMT_3inch_inner1_log0x1c9f050">
               654       <materialref ref="Vacuum0x14be220"/>
               655       <solidref ref="PMT_3inch_inner1_solid_ell_helper0x1c9e510"/>
               656     </volume>

               657     <volume name="PMT_3inch_inner2_log0x1c9f120">
               658       <materialref ref="Vacuum0x14be220"/>
               659       <solidref ref="PMT_3inch_inner2_solid_ell_helper0x1c9e5d0"/>
               660     </volume>


           671     <volume name="PMT_3inch_cntr_log0x1c9f1f0">
           672       <materialref ref="Steel0x14aa2a0"/>
           673       <solidref ref="PMT_3inch_cntr_solid0x1c9e640"/>
           674     </volume>





lMaskVirtual0x1816910
-----------------------


::

     **      idx   5 pdig d3d9446802a44259755d38e6d163e820 ndig  17739 nprog      0 nvert    484 n lMask0x18170e0
     **      idx   6 pdig c20ad4d76fe97759aa27a0c99bff6710 ndig  17739 nprog      0 nvert    482 n PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 6512bd43d9caa6e02c990b0a82652dca ndig  17739 nprog      0 nvert    194 n PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig 511dd06bf687b1e989d4ac84e25bc0a3 ndig  17739 nprog      2 nvert   1254 n PMT_20inch_body_log0x1863160
     **      idx   9 pdig fd99bf5972e7592724cbd49bfb448953 ndig  17739 nprog      3 nvert   1832 n PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 nvert   2366 n lMaskVirtual0x1816910


::

       643     <volume name="lMaskVirtual0x1816910">
       644       <materialref ref="Water0x14d1d00"/>
       645       <solidref ref="sMask_virtual0x18163c0"/>
       646       <physvol name="pMask0x18171e0">
       647         <volumeref ref="lMask0x18170e0"/>
       648       </physvol>
       649       <physvol name="PMT_20inch_log_phys0x1866410">
       650         <volumeref ref="PMT_20inch_log0x18631f0"/>
       651       </physvol>
       652     </volume>

           614     <volume name="lMask0x18170e0">
           615       <materialref ref="Acrylic0x14a82e0"/>
           616       <solidref ref="sMask0x1816f50"/>
           617     </volume>

           636     <volume name="PMT_20inch_log0x18631f0">
           637       <materialref ref="Pyrex0x14c4190"/>
           638       <solidref ref="PMT_20inch_pmt_solid0x1813600"/>
           639       <physvol name="PMT_20inch_body_phys0xe4d580">
           640         <volumeref ref="PMT_20inch_body_log0x1863160"/>
           641       </physvol>
           642     </volume>

               626     <volume name="PMT_20inch_body_log0x1863160">
               627       <materialref ref="Pyrex0x14c4190"/>
               628       <solidref ref="PMT_20inch_body_solid0x1813ec0"/>
               629       <physvol name="PMT_20inch_inner1_phys0x18012e0">
               630         <volumeref ref="PMT_20inch_inner1_log0x1863280"/>
               631       </physvol>
               632       <physvol name="PMT_20inch_inner2_phys0x1821730">
               633         <volumeref ref="PMT_20inch_inner2_log0x1863310"/>
               634       </physvol>
               635     </volume>

                   618     <volume name="PMT_20inch_inner1_log0x1863280">
                   619       <materialref ref="Vacuum0x14be220"/>
                   620       <solidref ref="PMT_20inch_inner1_solid0x1814a90"/>
                   621     </volume>

                   622     <volume name="PMT_20inch_inner2_log0x1863310">
                   623       <materialref ref="Vacuum0x14be220"/>
                   624       <solidref ref="PMT_20inch_inner2_solid0x1863010"/>
                   625     </volume>




