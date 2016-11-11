#!usr/bin/env python
"""

::

    In [20]: ab.his[:15]
    Out[20]: 
    .                seqhis_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       363.45/354 =  1.03  (pval:0.353 prob:0.647)  
       0               8ccccd        669843       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [6 ] TO BT BT BT BT SA
       1                   4d         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] TO AB
       2              8cccc6d         45490        45054             2.10        1.010 +- 0.005        0.990 +- 0.005  [7 ] TO SC BT BT BT BT SA
       3               4ccccd         28955        28649             1.63        1.011 +- 0.006        0.989 +- 0.006  [6 ] TO BT BT BT BT AB
       4                 4ccd         23187        23254             0.10        0.997 +- 0.007        1.003 +- 0.007  [4 ] TO BT BT AB
       5              8cccc5d         20239        19946             2.14        1.015 +- 0.007        0.986 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              86ccccd         10176        10396             2.35        0.979 +- 0.010        1.022 +- 0.010  [7 ] TO BT BT BT BT SC SA
       7              8cc6ccd         10214        10304             0.39        0.991 +- 0.010        1.009 +- 0.010  [7 ] TO BT BT SC BT BT SA
       8              89ccccd          7540         7694             1.56        0.980 +- 0.011        1.020 +- 0.012  [7 ] TO BT BT BT BT DR SA
       9             8cccc55d          5970         5814             2.07        1.027 +- 0.013        0.974 +- 0.013  [8 ] TO RE RE BT BT BT BT SA
      10                  45d          5780         5658             1.30        1.022 +- 0.013        0.979 +- 0.013  [3 ] TO RE AB
      11      8cccccccc9ccccd          5339         5367             0.07        0.995 +- 0.014        1.005 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      12              8cc5ccd          5113         4868             6.01        1.050 +- 0.015        0.952 +- 0.014  [7 ] TO BT BT RE BT BT SA
      13                  46d          4797         4815             0.03        0.996 +- 0.014        1.004 +- 0.014  [3 ] TO SC AB
      14          8cccc9ccccd          4494         4420             0.61        1.017 +- 0.015        0.984 +- 0.015  [11] TO BT BT BT BT DR BT BT BT BT SA
    .                               1000000      1000000       363.45/354 =  1.03  (pval:0.353 prob:0.647)  

    In [21]: ab.mat[:15]
    Out[21]: 
    .                seqmat_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       222.91/230 =  0.97  (pval:0.619 prob:0.381)  
       0               343231        669845       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [6 ] Gd Ac LS Ac MO Ac
       1                   11         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] Gd Gd
       2              3432311         65732        65001             4.09        1.011 +- 0.004        0.989 +- 0.004  [7 ] Gd Gd Ac LS Ac MO Ac
       3               443231         28955        28649             1.63        1.011 +- 0.006        0.989 +- 0.006  [6 ] Gd Ac LS Ac MO MO
       4                 2231         23188        23254             0.09        0.997 +- 0.007        1.003 +- 0.007  [4 ] Gd Ac LS LS
       5              3443231         17716        18090             3.91        0.979 +- 0.007        1.021 +- 0.008  [7 ] Gd Ac LS Ac MO MO Ac
       6              3432231         15327        15172             0.79        1.010 +- 0.008        0.990 +- 0.008  [7 ] Gd Ac LS LS Ac MO Ac
       7             34323111         10934        10826             0.54        1.010 +- 0.010        0.990 +- 0.010  [8 ] Gd Gd Gd Ac LS Ac MO Ac
       8                  111         10577        10474             0.50        1.010 +- 0.010        0.990 +- 0.010  [3 ] Gd Gd Gd
       9      343231323443231          6955         7001             0.15        0.993 +- 0.012        1.007 +- 0.012  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
      10          34323443231          6038         5954             0.59        1.014 +- 0.013        0.986 +- 0.013  [11] Gd Ac LS Ac MO MO Ac LS Ac MO Ac
      11          34323132231          4422         4532             1.35        0.976 +- 0.015        1.025 +- 0.015  [11] Gd Ac LS LS Ac Gd Ac LS Ac MO Ac
      12              4443231          3160         3272             1.95        0.966 +- 0.017        1.035 +- 0.018  [7 ] Gd Ac LS Ac MO MO MO
      13              4432311          3008         3002             0.01        1.002 +- 0.018        0.998 +- 0.018  [7 ] Gd Gd Ac LS Ac MO MO
      14            343231111          2859         2860             0.00        1.000 +- 0.019        1.000 +- 0.019  [9 ] Gd Gd Gd Gd Ac LS Ac MO Ac
    .                               1000000      1000000       222.91/230 =  0.97  (pval:0.619 prob:0.381)  


"""
import os, logging, numpy as np

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt

log = logging.getLogger(__name__)


class AB(object):
    def __init__(self, args):
        self.args = args
        self.load()
        self.compare()

    def load(self):
        """
        It takes aound 6s to load 1M full AB evt pair. So avoid needing to duplicate that.
        """
        log.info("AB.load START ")
        args = self.args
        try:
            a = Evt(tag="%s" % args.tag, src=args.src, det=args.det, args=args )
            b = Evt(tag="-%s" % args.tag, src=args.src, det=args.det, args=args )
        except IOError as err:
            log.fatal(err)
            sys.exit(args.mrc)
        pass
        self.a = a
        self.b = b 
        log.info("AB.load DONE ")

    def __repr__(self):
        return "AB(%s,%s,%s) " % (self.args.tag, self.args.src, self.args.det)
 
    def compare(self):
        log.info("AB.compare START ")
        self.his = self.cf("seqhis_ana")
        self.flg = self.cf("pflags_ana")
        #self.hfl = self.cf("hflags_ana")
        self.mat = self.cf("seqmat_ana")

        if self.args.prohis:self.prohis()
        if self.args.promat:self.promat()
        log.info("AB.compare DONE")

    def prohis(self, rng=range(1,8)):
        for imsk in rng:
            setattr(self, "his_%d" % imsk, self.cf("seqhis_ana_%d" % imsk)) 
        pass
    def promat(self, rng=range(1,8)):
        for imsk in rng:
            setattr(self, "mat_%d" % imsk, self.cf("seqmat_ana_%d" % imsk)) 
        pass

    def cf(self, ana="seqhis_ana"):
        a = self.a
        b = self.b
        print "AB a %s " % a.brief 
        print "AB b %s " % b.brief 
        c_tab = Evt.compare_ana( a, b, ana, lmx=self.args.lmx, cmx=self.args.cmx, c2max=None, cf=True)
        return c_tab

    # high level *sel* selection only, for lower level *psel* selections 
    # apply individually to evt a and b 

    def _get_sel(self):
        return self._sel

    def _set_sel(self, sel):
        self.a.sel = sel
        self.b.sel = sel
        self._sel = sel 

    sel = property(_get_sel, _set_sel)
 



if __name__ == '__main__':
    ok = opticks_main()
    ab = AB(ok)

