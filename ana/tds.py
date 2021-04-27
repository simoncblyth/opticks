#!/usr/bin/env python

import os, re, datetime, argparse
COLUMNS = os.environ.get("COLUMNS", 200)


def dt_parse(s):
    try:
        t = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")  
    except ValueError:
        t = None
    pass   
    return t 

class Line(object):
    ptn = re.compile("\((?P<a>\d);(?P<n>\d+),(?P<d>\d+)\)\s*launch time\s*(\S*)\s*$")

    @classmethod
    def ParseTime(cls, txt):
        t = dt_parse(txt[:23])
        if t is None:
            print("unexpected time format [%s]" % txt)
            s = None
        else:
            s = t.timestamp()       # includes the sub-seconds, needs py3.3+
            #s = t.strftime("%s"))   # just seconds
        pass
        return t, s


    def __init__(self, txt):

        t, s = self.ParseTime(txt)
        self.t = t 
        self.s = s 

        typ = self.Type(txt)
        self.typ = typ

        self.desc = ""
        if typ == "launch":
            self.parse_launch_time(txt)
        else:
            pass
        pass

        self.txt = txt 
        self.c = txt[23:] 
        self.prev = None

    def parse_launch_time(self, txt):
        a,n,d,lt = self.ptn.findall(txt)[0]  
        a = int(a) 
        n = int(n) 
        fn = float(n)/1e6
        d = int(d) 
        lt = float(lt) 
        self.desc = "%6.2f/%6.2f " % ( fn, lt ) 

    dt = property(lambda self:self.s - self.prev.s if not (self.prev is None or self.s is None or self.prev.s is None) else 0.)

    @classmethod
    def Type(cls, l):
        if l.find("launch time") > -1:
            typ = "launch"
        elif l.find("[[") > -1:
            typ = "open"
        elif l.find("]]") > -1:
            typ = "close"
        else:
            typ = None
        pass 
        return typ
 
    @classmethod
    def Select(cls, l):
        return not cls.Type(l) is None

    @classmethod
    def ShowTime(cls, t):
        #tfmt = "%c %f
        tfmt = "%H:%M:%S"
        return t.strftime(tfmt) if not t is None else "-"*8 

    def __str__(self):
        fline = "%s : %10.4f : %10s : %20ss : %s " % ( self.ShowTime(self.t), self.dt, self.typ, self.desc, self.txt )
        return fline[:COLUMNS]


class TDSLog(object):
    """
    2021-04-26 23:49:33.732 INFO  [128777] [OPropagator::launch@287] 0 : (0;50214628,1)  launch time 41.9189
    """
    def __init__(self, path="/tmp/$USER/opticks/tds/python2.7.log"):
        path = os.path.expandvars(path)
        lines = open(path,"r").read().splitlines()
        #lines = filter(Line.Select, lines)
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



