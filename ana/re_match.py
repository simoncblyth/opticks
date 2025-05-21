#!/usr/bin/env python
"""


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

_ptn = 'Water/.*/Water$'            # OK : virtuals 
_ptn = '.*/Tyvek$'                  # OK

#_ptn = '.*'  

PTN = os.environ.get("PTN", _ptn)

print("PTN:%s" % PTN)

ptn = re.compile(PTN)
_re_match_ = lambda elem:bool(ptn.match(elem))
re_match = np.vectorize(_re_match_)

_w = re_match(cf.bdn)
w = np.where(_w)
#print(w)

bb = cf.bdn[w]
print("\n".join(bb))

