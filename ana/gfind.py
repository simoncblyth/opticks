#!/usr/bin/env python

import logging, textwrap
log = logging.getLogger(__name__)

from opticks.analytic.gdml import GDML 
specs_ = lambda s:filter(lambda s:s[0] != "#", filter(None,textwrap.dedent(s).split("\n")))


if __name__ == '__main__':


    lvx = specs_(r"""
    PMT_3inch_log
    NNVTMCPPMTlMaskVirtual
    HamamatsuR12860lMaskVirtual
    mask_PMT_20inch_vetolMaskVirtual
    """
    )
    ilv = 3

    labels = specs_(r"""
    tds_ngt
    tds_ngt_pcnk
    """
    )


    g0 = GDML.parse("$OPTICKS_PREFIX/%s.gdml" % labels[0])
    g0.smry()
    #g0.volume_summary()

    lv0 = g0.find_one_volume(lvx[ilv])
    print("lv0:%s [%s]" % (lv0, lvx[ilv]) )


    if len(labels)>1:
        g1 = GDML.parse("$OPTICKS_PREFIX/%s.gdml" % labels[1])
        g1.smry()
        #g1.volume_summary()

        lv1 = g1.find_one_volume(lvx[ilv])
        print("lv1:%s" % lv1 )
    pass




