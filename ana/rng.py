#!/usr/bin/env python
"""
cuRANDWrapper_100000000_0_0.bin
cuRANDWrapper_10000000_0_0.bin
cuRANDWrapper_1000000_0_0.bin
cuRANDWrapper_10240_0_0.bin
cuRANDWrapper_200000000_0_0.bin
cuRANDWrapper_2000000_0_0.bin
cuRANDWrapper_3000000_0_0.bin
"""

import os, logging, re
log = logging.getLogger(__name__)
ptn = re.compile("^cuRANDWrapper_(?P<num>\d*)_(?P<a>\d*)_(?P<b>\d*).bin$")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    rngdir = os.path.expanduser("~/.opticks/rngcache/RNG")
    
    #lines = filter(None,__doc__.split("\n")) 
    lines = os.listdir(rngdir)  
    
    for line in lines:
        m = ptn.match(line)
        if not m:
            log.info("failed to match %s " % line)
        else:
            d = m.groupdict()
            num = d['num']
            millions = float(d['num'])/float(1e6)
            #print(d)
            print("%6.2fM : %10d : %s " % (millions, int(num), line ))
        pass
    pass



