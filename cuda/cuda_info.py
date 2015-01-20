#!/usr/bin/env python
"""

First thing after restart, using Terminal.app the only app running other than Finder::

    delta:~ blyth$ cuda_info.sh
    name                     GeForce GT 750M
    compute capability       (3, 0)
    device total             2.1G
    device used              470.5M
    device free              1.7G
    delta:~ blyth$ 

After several days of development::

    delta:~ blyth$ cuda_info.sh
    name                     GeForce GT 750M
    compute capability       (3, 0)
    device total             2.1G
    device used              1.6G
    device free              524.5M
    delta:~ blyth$ 

TODO: 

* check from ">console" login
* persist this state 

"""
import argparse
from collections import OrderedDict
from cuda_common import format_kv, timestamp

drv = None


class CUDAInfo(object):
    """
    Info not needing autoinit 
    """
    def __init__(self, dev ):
        self.count = drv.Device.count()
        print "%d device(s) found." % self.count



class DeviceInfo(object):
    def __init__(self, dev ):
        self.dev = dev

    name = property(lambda self:self.dev.name())
    compute_capability = property(lambda self:self.dev.compute_capability())
    total_memory = property(lambda self:self.dev.total_memory())

    def __repr__(self):
        state = self.current_state()
        return self.format_state(state)

    def format_state(self, state):
        return "\n".join(["%-20s     %s" % (k,format_kv(k,v)) for k,v in state.items()])

    def current_state(self, tag=None):
        if tag is None:
            tag = "default"
        free, total = drv.mem_get_info()
        qty = OrderedDict()
        qty['timestamp'] = timestamp()
        qty['tag'] = tag
        qty['name'] = self.name
        qty['compute capability'] = self.compute_capability
        qty['memory total'] = total
        qty['memory used'] = total-free
        qty['memory free'] = free
        return qty

    def properties(self):
        index = 0
        dev = self.dev
        print "Device #%d: %s" % (index, dev.name())
        print "  Compute Capability: %d.%d" % dev.compute_capability()
        print "  Total Memory: %s KB" % (dev.total_memory()//(1024))
        atts = [(str(att), value) for att, value in dev.get_attributes().iteritems()]
        atts.sort()
        for att, value in atts:
            print "  %s: %s" % (att, value)


def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    args = parser.parse_args()
    return args 

def main():
    global drv

    args = parse(__doc__)
    import pycuda.driver as drv
    #drv.init()              # this does not pick a device
    import pycuda.autoinit   # this picks device

    dev = DeviceInfo(pycuda.autoinit.device)
    print dev


if __name__ == '__main__':
    main()




