#!/usr/bin/env python
"""

::

    pycuda._driver.LogicError: cuMemGetInfo failed: not initialized


"""
from collections import OrderedDict
from cuda_common import format_kv, timestamp

class DriverState(OrderedDict):
    """
    Usage::
    """
    def __init__(self, drv ):
        OrderedDict.__init__(self)
        free, total = drv.mem_get_info()
 
        self['timestamp'] = timestamp()
        self['memory total'] = total
        self['memory used'] = total-free
        self['memory free'] = free

    def __repr__(self):
        return "\n".join(["%-20s     %s" % (k,format_kv(k,v)) for k,v in self.items()])


if __name__ == '__main__':
    import pycuda.autoinit 
    from env.cuda.cuda_state import DriverState
    import pycuda.driver as drv
    ds = DriverState(drv)  
    print ds 



