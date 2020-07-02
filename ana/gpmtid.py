#!/usr/bin/env python
"""
From perusing the gdml physvol/@copynumber

* 3inch inside lInnerWater      :   300000,..,325599
* 20inch inside lInnerWater     :    "0",1,2,..,17611   (covers both NNVT and Hamamatsu)
* 20inch inside lOuterWaterPool :   30000,...,32399


In [19]: g0.volume_summary()
25600 :                             PMT_3inch_log0x3a2e430 : lInnerWater0x30e63a0
12612 :                    NNVTMCPPMTlMaskVirtual0x32a56c0 : lInnerWater0x30e63a0
 5000 :               HamamatsuR12860lMaskVirtual0x3290b70 : lInnerWater0x30e63a0
 2400 :          mask_PMT_20inch_vetolMaskVirtual0x3297630 : lOuterWaterPool0x30e5550
  590 :                                    lUpper0x3176610 : lInnerWater0x30e63a0
  590 :                                lFasteners0x312e0c0 : lInnerWater0x30e63a0
  590 :                                    lSteel0x30e7840 : lInnerWater0x30e63a0
  590 :                                 lAddition0x31bd480 : lInnerWater0x30e63a0
   64 :                                  lCoating0x4507f00 : lPanelTape0x4507d70
   64 :                                lXJfixture0x320b990 : lTarget0x30e7080 lInnerWater0x30e63a0
   63 :                                  lWallff_0x45078b0 : lAirTT0x4507630
   56 :                                 lXJanchor0x32053d0 : lInnerWater0x30e63a0
   36 :                                lSJFixture0x3210660 : lTarget0x30e7080
    8 :                               lSJReceiver0x3215330 : lTarget0x30e7080
    4 :                                    lPanel0x4507be0 : lPlanef_0x4507ad0
    2 :                              lSJCLSanchor0x320f450 : lTarget0x30e7080
    2 :                                  lPlanef_0x4507ad0 : lWallff_0x45078b0
    1 :                             lUpperChimney0x4502a10 : lExpHall0x30dfd90
    1 :          NNVTMCPPMT_PMT_20inch_inner2_log0x32a40d0 : NNVTMCPPMT_PMT_20inch_body_log0x32a3db0
    1 :     HamamatsuR12860_PMT_20inch_inner2_log0x32a9750 : HamamatsuR12860_PMT_20inch_body_log0x32a9400
    1 :                                    lAirTT0x4507630 : lExpHall0x30dfd90



"""
import logging, textwrap
log = logging.getLogger(__name__)

from opticks.analytic.gdml import GDML 
specs_ = lambda s:filter(lambda s:s[0] != "#", filter(None,textwrap.dedent(s).split("\n")))


if __name__ == '__main__':


    lvx = specs_(r"""
    lInnerWater
    """
    )
    ilv = 0

    labels = specs_(r"""
    #tds_ngt
    tds_ngt_pcnk
    """
    )


    g0 = GDML.parse("$OPTICKS_PREFIX/%s.gdml" % labels[0])
    g0.smry()
    #g0.volume_summary()

    lv0 = g0.find_one_volume(lvx[ilv])
    print("lv0:%s [%s]" % (lv0.name, lvx[ilv]) )

    npmt = len(lv0.elem.xpath("./physvol[starts-with(@name,'pLPMT')]"))
    ncop = len(lv0.elem.xpath("./physvol[starts-with(@name,'pLPMT')]/@copynumber"))  # 1..17611
    print("npmt:%d ncop:%d" % (npmt, ncop)) 


    if len(labels)>1:
        g1 = GDML.parse("$OPTICKS_PREFIX/%s.gdml" % labels[1])
        g1.smry()
        #g1.volume_summary()

        lv1 = g1.find_one_volume(lvx[ilv])
        print("lv1:%s" % lv1 )
    pass




