#!/usr/bin/env python
"""
pflags.py 
=============================================

Debugging inconsistent pflags in CFG4 evt

::

    In [3]: e.psel_dindex()
    Out[3]: '--dindex=3352,12902,22877,23065,41882,60653,68073,69957,93373,114425,116759,119820,121160,128896,140920,144796,149640,155511,172178,173508,181946,197721,206106,218798,226414,229472,245012,246679,247048,250061,256632,273737,277009,278330,283792,284688,302431,302522,310912,312485,322121,327125,328934,344304,348955,363391,385856,398678,405719,413374,427982,435697,440670,470050,474196,477693,479219,479671,482244,482334,483690,493571,499519,510053,512631,520014,528665,537688,572302,580525,582218,592832,603216,605660,609385,613092,616980,632731,643197,647969,648445,651609,652951,659879,661157,663245,666346,667822,668744,673617,685642,688649,699598,700202,710936,728978,733667,742167,745397,764234,764506,772722,776790,785381,798323,799789,800795,801821,816920,817527,821113,840075,863428,872134,878479,879868,898266,900382,900808,905903,909591,911618,917897,919938,925473,929891,929984,961725,967547,976708,978573,994454'


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt
from opticks.ana.hismask import HisMask
from opticks.ana.nbase import count_unique_sorted


if __name__ == '__main__':
    ok = opticks_main(det="concentric",src="torch",tag="1")
    hm = HisMask()

    e = Evt(tag="-%s"%ok.utag, src=ok.src, det=ok.det, args=ok, seqs=["PFLAGS_DEBUG"])
    e.history_table(slice(0,20))

    cu_pflags = count_unique_sorted(e.pflags[e.psel])
    cu_pflags2 = count_unique_sorted(e.pflags2[e.psel])

    print "\n".join(["","cu_pflags (masks from CRecorder)"]+map(lambda _:hm.label(_), cu_pflags[:,0]))
    print cu_pflags

    print "\n".join(["","cu_pflags2 (masks derived from seqhis)"]+map(lambda _:hm.label(_), cu_pflags2[:,0]))
    print cu_pflags2

    print e.psel_dindex()



