#!/usr/bin/env python




template_head = r"""

tboolean-%(name)s(){ TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
tboolean-%(name)s-(){ $FUNCNAME- | python $* ; } 
tboolean-%(name)s--(){ cat << EOP

outdir = "$TMP/$FUNCNAME"
obj_ = "$(tboolean-testobject)"
con_ = "$(tboolean-container)"

import logging
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main()

CSG.boundary = obj_
CSG.kwa = dict(verbosity="1", poly="IM")

"""

template_tail = r"""

obj = %(root)s

con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=con_ , poly="HY", level="5" )
CSG.Serialize([con, obj], outdir )


EOP
}

"""

template_body = r"""

%(body)s

"""

class TBooleanBashFunction(dict):
     def __init__(self, *args, **kwa):
         dict.__init__(self, *args, **kwa)
     def __str__(self):
         return "\n".join([template_head % self, template_body % self, template_tail % self])  


if __name__ == '__main__':
     tbf = TBooleanBashFunction(name="esr", root="ab", body="# body goes here \n# and here\n# and here ")
     print tbf 



