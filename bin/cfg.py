#!/usr/bin/env python
"""
cfg.py
=======

NB configparser changed a lot betweeen py2.7 and py3
the below is for py2.7

"""
import StringIO, textwrap
from collections import OrderedDict as odict

try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser


class Cfg(object):
    def __init__(self, txt):
        sfp = StringIO.StringIO(txt)
        c = ConfigParser()
        c.readfp(sfp)
        self.c = c 
        self.d = self.full(c) 

    def full(self, c):
        d = odict()
        for s in c.sections(): 
            d[s] = dict(c.items(s))
        pass  
        return d

    def sections(self):
        return self.c.sections()

    def sect(self, s):
        return dict(self.c.items(s))

    def __repr__(self):
        return "\n".join(["%s\n%s" % (k,repr(v)) for k, v in self.d.items()])



if __name__ == '__main__':

     txt = textwrap.dedent("""
     [red]
     a=1
     b:2
     c=3
     [green]
     aa=2
     bb=3
     cc:4
     """)

     c = Cfg(txt)
     print(c.sections())

     for s in c.sections():
         print("%s " % s)
         d = c.sect(s)
         print(repr(d)) 
         for k,v in d.items():
             print("%10s : %s " % (k,v )) 
         pass
     pass
 
     print(c)
 

    



