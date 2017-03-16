#!/usr/bin/env python
"""
To regenerate the OpticksCSG.py::

    sysrap-;sysrap-cd 
    c_enums_to_python.py OpticksCSG.h  # check 
    c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 


"""
from opticks.sysrap.OpticksCSG import CSG_


if __name__ == '__main__':

     for k, v in CSG_.raw_enum():
         vv = getattr(CSG_, k)
         print k, v, vv



     for i in range(20):
         d = CSG_.desc(i)
         i2 = CSG_.fromdesc(d)

         print "%3d %15s %d " % (i, d, i2)



