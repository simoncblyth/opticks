#!/usr/bin/env python
"""
Parsing the OpticksPhoton.h enum 
"""
import os, re, logging
log = logging.getLogger(__name__)

class Enum(dict):
    lptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")
    vptn = re.compile("^0x1 <<\s*(\d+)$")
    def __init__(self, path):
        dict.__init__(self)
        path = os.path.expandvars(path)
        self.parse(path)

    def parse(self, path):
        lines = map(str.strip,file(path).readlines())
        for line in lines:
            lm = self.lptn.match(line)
            if lm:
                 lg = lm.groups()
                 assert len(lg) == 2 
                 label, val = lg

                 vm = self.vptn.match(val)
                 assert vm 
                 vg = vm.groups()
                 assert len(vg) == 1
                 n = int(vg[0])

                 emsk = eval(val)
                 msk = 0x1 << n 
                 assert emsk == msk

                 log.debug( "%-40s     ==> [%s]    [%s]  ==> [%d] ==> [%x]  " % (line, label, val, n, msk) )

                 self[label] = msk  


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)  

     d = Enum("$OPTICKS_HOME/optickscore/OpticksPhoton.h")
     print d
      

