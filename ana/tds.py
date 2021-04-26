#!/usr/bin/env python

import os, re, datetime, argparse

def dt_parse(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")  

class Line(object):
    ptn = re.compile("\((?P<a>\d);(?P<n>\d+),(?P<d>\d+)\)\s*launch time\s*(\S*)\s*$")

    def __init__(self, txt):
        self.t = dt_parse(txt[:23])
        self.c = txt[23:] 

        a,n,d,lt = self.ptn.findall(txt)[0]  

        self.a = int(a) 
        self.n = int(n) 
        self.fn = float(self.n)/1e6
        self.d = int(d) 
        self.lt = float(lt) 
        #self.s = float(self.t.strftime("%s"))   # just seconds
        self.s = self.t.timestamp()   # includes the sub-seconds, needs py3.3+
        self.prev = None

    dt = property(lambda self:self.s - self.prev.s if not self.prev is None else 0.)

    def __str__(self):
        return "%s : n:%8d fn:%6.2f lt:%10.4f  dt:%10.4f " % ( self.t.strftime("%c %f"), self.n, self.fn, self.lt, self.dt  )    


class TDSLog(object):
    """
    2021-04-26 23:49:33.732 INFO  [128777] [OPropagator::launch@287] 0 : (0;50214628,1)  launch time 41.9189
    """
    def __init__(self, path="/tmp/$USER/opticks/tds/python2.7.log"):
        path = os.path.expandvars(path)
        lines = open(path,"r").read().splitlines()
        lines = filter(lambda l:l.find("launch time") > -1, lines)

        lines = list(map(Line, lines)) 
        for i in range(len(lines)):
            lines[i].prev = lines[i-1] if i > 0 else None
        pass
        self.lines = lines
        self.path = path 

    def __str__(self):
        return "\n".join(list(map(str,self.lines)) + [self.path])          
    

if __name__ == '__main__':
    t = TDSLog()
    print(t)
    q = "OPropagator::launch@287"



