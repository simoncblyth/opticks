#!/usr/bin/env python
"""

lob
lco
~/opticks/ana/re_match.sh


"""

import os, re, numpy as np
from opticks.ana.p import cf    # load CSGFoundry geometry info

#_ptn = 'Water'                      # only matches bnd starting with Water
#_ptn = 'Water/.*/Teflon$'           # OK
#_ptn = 'Water*Teflon'               # NOPE
#_ptn = 'Water.*/LatticedShellSteel' # OK
#_ptn = '.*Vacuum'                   # not just ending with Vacuum

#_ptn = '.*/Vacuum$'                 # OK imat:Vacuum
#_ptn = '.*/LatticedShellSteel$'     # OK
#_ptn = '.*/Steel$'                  # OK

_ptn = '(?P<Water_Virtuals>Water/.*/Water$)'            # OK : virtuals
#_ptn = '.*/Tyvek$'                  # OK

#_ptn = '.*'

PTN = os.environ.get("PTN", _ptn)

print("PTN:%s" % PTN)

ptn = re.compile(PTN)
_re_match = lambda s:bool(ptn.match(s))

def _re_key(s):
    keys = list(ptn.match(s).groupdict().keys())
    return keys[0] if len(keys) == 1 else PTN
pass


re_match = np.vectorize(_re_match)
re_key   = np.vectorize(_re_key)


_w = re_match(cf.bdn)
w = np.where(_w)[0]
#print(w)

bb = cf.bdn[w]
kk = re_key(bb)

print("\n".join(["bb"]+list(bb)))
print("\n".join(["kk"]+list(kk)))

