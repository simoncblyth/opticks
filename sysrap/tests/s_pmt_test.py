#!/usr/bin/env python
"""
# s_pmt_test.py  CAUTION : THIS DOCSTRING IS EVALUATED
######################################################

f

f.lpmtidx.shape
f.lpmtidx.shape[0] == 17612 + 2400 + 348 + 5         # without MPMT
f.lpmtidx.shape[0] == 17612 + 2400 + 348 + 600 + 5   # with MPMT

np.all( f.lpmtidx[:,0] == np.arange(len(f.lpmtidx)) )  # lpmtidx is contiguous from zero

f.lpmtidx_labels    #    ['lpmtidx', 'lpmtid', 'contiguousidx']
f.lpmtidx[:17612]   # CD LPMT
np.all( f.lpmtidx[:17612,0] == f.lpmtidx[:17612,1] )  # for CD LPMT  lpmtidx and lpmtid are same
np.all( f.lpmtidx[:17612,0] == f.lpmtidx[:17612,2] )  # for CD LPMT  lpmtidx and contiguousidx are same


f.lpmtidx[17612:17612+2400]

np.all( f.lpmtidx[17612:17612+2400,1] == np.arange(2400)+50000 )              ## WP_PMT with id from 50000

np.all( f.lpmtidx[17612+2400:17612+2400+348,1] == np.arange(348)+52400 )      ## WP_ATM_LPMT id from 52400

np.all( f.lpmtidx[17612+2400+348+600:17612+2400+348+600+5,1] == np.arange(5)+54000 )  ## WP_ARM_WAL id from 54000


####################################################################

f.oldcontiguousidx.shape
f.oldcontiguousidx.shape[0] == 17612 + 25600 + 2400 + 348 + 5
f.oldcontiguousidx.shape[0] == 17612 + 25600 + 2400 + 348 + 600 + 5
f.oldcontiguousidx_labels   # ['oldcontiguousidx', 'pmtid', 'lpmtidx']

f.oldcontiguousidx[:17612]     # CD LPMT

np.all( f.oldcontiguousidx[:17612,0] == np.arange(17612) + 0 )
np.all( f.oldcontiguousidx[:17612,1] == np.arange(17612) + 0 )
np.all( f.oldcontiguousidx[:17612,2] == np.arange(17612) + 0 )

np.cumsum([17612,25600,2400,348,5])

np.all( np.cumsum([17612,25600,2400,348,5]) == np.array([17612, 43212, 45612, 45960, 45965]))

f.oldcontiguousidx_labels    # ['oldcontiguousidx', 'pmtid', 'lpmtidx']

f.oldcontiguousidx[17612:17612+25600]   # SPMT

np.all( f.oldcontiguousidx[17612:17612+25600,0] == np.arange(25600) + 17612 )
np.all( f.oldcontiguousidx[17612:17612+25600,1] == np.arange(25600) + 20000 )
np.all( f.oldcontiguousidx[17612:17612+25600,2] == -1 )

f.oldcontiguousidx[17612+25600:17612+25600+2400]   # WP_PMT

np.all( f.oldcontiguousidx[17612+25600:17612+25600+2400,0] == np.arange(2400) + 43212 )
np.all( f.oldcontiguousidx[17612+25600:17612+25600+2400,1] == np.arange(2400) + 50000 )
np.all( f.oldcontiguousidx[17612+25600:17612+25600+2400,2] == np.arange(2400) + 17612 )

f.oldcontiguousidx[17612+25600+2400:17612+25600+2400+248]   # WP_ATM_LPMT

np.all( f.oldcontiguousidx[17612+25600+2400:17612+25600+2400+348,0] == np.arange(348) + 17612+25600+2400 )
np.all( f.oldcontiguousidx[17612+25600+2400:17612+25600+2400+348,1] == np.arange(348) + 52400 )
np.all( f.oldcontiguousidx[17612+25600+2400:17612+25600+2400+348,2] == np.arange(348) + 17612+0+2400 )    ## lpmtidx excludes SPMT


f.oldcontiguousidx[17612+25600+2400+348:17612+25600+2400+348+5]   # WP_WAL_PMT

np.all( f.oldcontiguousidx[17612+25600+2400+348:17612+25600+2400+348+5,0] == np.arange(5) + 17612+25600+2400+348 )
np.all( f.oldcontiguousidx[17612+25600+2400+348:17612+25600+2400+348+5,1] == np.arange(5) + 54000 )
np.all( f.oldcontiguousidx[17612+25600+2400+348:17612+25600+2400+348+5,2] == np.arange(5) + 17612+0+2400+348 )   ## lpmtidx excludes SPMT


#########################################

f.contiguousidx.shape
f.contiguousidx.shape[0] == 17612 + 2400 + 25600 + 348 + 5

f.contiguousidx_labels    # ['contiguousidx', 'pmtid', 'lpmtidx']
f.contiguousidx[:17612]   # CD_LPMT

f.contiguousidx_labels
f.contiguousidx[17612:17612+2400]   # WP_PMT

np.all( f.contiguousidx[17612:17612+2400,1] == np.arange(2400) + 50000 )

np.all( f.contiguousidx[17612+2400:17612+2400+348,1] == np.arange(348) + 52400 )

np.all( f.contiguousidx[17612+2400+348:17612+2400+348+5,1] == np.arange(5) + 54000 )


f.contiguousidx_labels
f.contiguousidx[17612+2400:17612+2400+25600]   # S_PMT

np.all( f.contiguousidx[17612+2400+348+5:17612+2400+348+5+25600,0] == np.arange(25600) + 17612+2400+348+5 )
np.all( f.contiguousidx[17612+2400+348+5:17612+2400+348+5+25600,1] == np.arange(25600) + 20000 )
np.all( f.contiguousidx[17612+2400+348+5:17612+2400+348+5+25600,2] == -1  )

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


