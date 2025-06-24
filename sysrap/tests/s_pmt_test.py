#!/usr/bin/env python
"""
# s_pmt_test.py  CAUTION : THIS DOCSTRING IS EVALUATED
######################################################

f

f.lpmtidx.shape
f.lpmtidx.shape[0] == 17612 + 2400

f.lpmtidx_labels
f.lpmtidx[:17612]   # CD LPMT
np.all( f.lpmtidx[:17612,0] == f.lpmtidx[:17612,1] )
np.all( f.lpmtidx[:17612,0] == f.lpmtidx[:17612,2] )

f.lpmtidx[17612:]   # WP PMT


####################################################################

f.oldcontiguousidx.shape
f.oldcontiguousidx.shape[0] == 17612 + 25600 + 2400
f.oldcontiguousidx_labels

f.oldcontiguousidx[:17612]     # CD LPMT

np.all( f.oldcontiguousidx[:17612,0] == f.oldcontiguousidx[:17612,1] )
np.all( f.oldcontiguousidx[:17612,0] == f.oldcontiguousidx[:17612,2] )


f.oldcontiguousidx[17612:17612+25600]   # SPMT

np.all( f.oldcontiguousidx[17612:17612+25600,0] - 17612 == f.oldcontiguousidx[17612:17612+25600,1] - 20000 )
np.all( f.oldcontiguousidx[17612:17612+25600,2] == -1 )

f.oldcontiguousidx_labels

f.oldcontiguousidx[17612+25600:]   # WP PMT

np.all( f.oldcontiguousidx[17612+25600:,0] - 43212 == f.oldcontiguousidx[17612+25600:,1] - 50000 )
np.all( f.oldcontiguousidx[17612+25600:,1] - 50000 == f.oldcontiguousidx[17612+25600:,2] - 17612 )





#########################################

f.contiguousidx.shape
f.contiguousidx.shape[0] == 17612 + 2400 + 25600

f.contiguousidx_labels
f.contiguousidx[:17612]   # CD_LPMT

f.contiguousidx_labels
f.contiguousidx[17612:17612+2400]   # WP_PMT

f.contiguousidx_labels
f.contiguousidx[17612+2400:17612+2400+25600]   # S_PMT


"""

import textwrap, numpy as np
from np.fold import Fold


if __name__ == '__main__':
    f = Fold.Load(symbol="f")

    EXPR = list(map(str.strip,textwrap.dedent(__doc__).split("\n")))

    lines = []
    for expr in EXPR:
        lines.append(expr)
        if expr == "" or expr.startswith("#"): continue
        lines.append(repr(eval(expr)))
    pass
    print("\n".join(lines))
pass


