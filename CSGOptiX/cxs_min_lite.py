#!/usr/bin/env python

import os, logging, textwrap, numpy as np
import ast
log = logging.getLogger(__name__)
TEST = os.environ.get("TEST", "")

from opticks.ana.fold import Fold, EVAL

print("[from opticks.sysrap.sphoton import SPhoton")
from opticks.sysrap.sphoton import SPhoton
print("]from opticks.sysrap.sphoton import SPhoton")

print("[from opticks.sysrap.sphotonlite import SPhotonLite")
from opticks.sysrap.sphotonlite import SPhotonLite
print("]from opticks.sysrap.sphotonlite import SPhotonLite")


TEST = os.environ.get("TEST","")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    a = Fold.Load("$AFOLD", symbol="a")
    print(repr(a))

    _ph = a.photon
    _pl = a.photonlocal
    _hg = a.hit
    _hl = a.hitlocal
    _ht = a.hitlite

    ph = SPhoton.view(_ph)
    pl = SPhoton.view(_pl)
    hg = SPhoton.view(_hg)
    hl = SPhoton.view(_hl)
    ht = SPhotonLite.view(_ht)

    EVAL(r"""

    _ph.shape   # normal global photon array
    _pl.shape   # non-standard photonlocal array
    ph.shape    # ph recarray
    pl.shape    # pl recarray
    np.array_equal( pl.iindex, ph.iindex )

    w = np.flatnonzero(np.all(pl.pos == ph.pos, axis=1))  # where local pos are same as global
    w.shape

    np.all( pl.ident[w] == 0 )   # means photons end on global geometry



    _hg.shape   # normal global hit array (N,4,4)
    _hl.shape   # non-standard hitlocal array (N,4,4)
    _ht.shape   # non-standard hitlite array (N,4)

    hg.shape
    hl.shape
    ht.shape

    np.array_equal( hg.time, hl.time )
    np.array_equal( hg.time, ht.time )
    np.all( ht.hitcount == 1 )

    np.array_equal( ht.flagmask, hg.flagmask )

    np.allclose( hl.phi , ht.phi, atol=7e-4 )
    np.allclose( hl.cost, ht.lposcost_f, atol=1e-4 )

    """, ns=locals())


