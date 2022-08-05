#!/usr/bin/env python
"""

    In [7]: wse = np.where(st.nds.sensor > -1 )[0]

    In [8]: wse.shape
    Out[8]: (91224,)


    In [9]: wse 
    Out[9]: array([ 70969,  70970,  70976,  70977,  70983, ..., 336639, 336644, 336645, 336650, 336651])

    In [10]: wse.min()
    Out[10]: 70969

    In [11]: wse.max()
    Out[11]: 336651

    In [12]: np.unique( wse )
    Out[12]: array([ 70969,  70970,  70976,  70977,  70983, ..., 336639, 336644, 336645, 336650, 336651])

    In [13]: np.unique( wse ).shape
    Out[13]: (91224,)

    In [15]: se = st.nds.sensor[wse]

    In [16]: se
    Out[16]: array([    0,     1,     2,     3,     4, ..., 91219, 91220, 91221, 91222, 91223], dtype=int32)




"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree
from opticks.CSG.CSGFoundry import CSGFoundry

if __name__ == '__main__':
    cf = CSGFoundry.Load()
    print(cf) 

    f = Fold.Load(symbol="f")
    st = stree(f)
    print(repr(st))

    se = np.where(st.nds.sensor > -1 )[0]  
pass
        

