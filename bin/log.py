#!/bin/bash -l 

import os, sys
from datetime import datetime

class Line(object):
    FMT = '%Y-%m-%d %H:%M:%S.%f'
    LFMT = "%26s : %11s : %11s :%s"

    @classmethod
    def Parse(cls, stmp):
        try:
            d = datetime.strptime(stmp, cls.FMT)
        except ValueError:
            d = None
        pass
        return d 

    @classmethod
    def FVal(cls, dt ):
        return " %10s" % "" if dt < 0 else " %10.4f" % dt 

    @classmethod
    def Format(cls, t, dts, dfs, msg):
        stmp = "-" if t is None else datetime.strftime(t, cls.FMT)
        return cls.LFMT % ( stmp, cls.FVal(dts), cls.FVal(dfs), msg )
  
    @classmethod
    def Hdr(cls):
        return cls.LFMT % ( "timestamp", "DTS-prev", "DFS-frst", "pc:msg" )

    def __init__(self, line):
        t = self.Parse( line[:23] )
        msg = line if t is None else line[23:]
        self.t = t
        self.msg = msg
        self.prev = None 
        self.first = None
        self.total = 0.

    def __repr__(self):
        t = self.t
        total = self.total
        p = None if self.prev is None else self.prev.t  
        f = None if self.first is None else self.first.t  
        dts = -1 if (p is None or t is None) else (t - p).total_seconds() 
        dfs = -1 if (f is None or t is None) else (t - f).total_seconds() 

        percent = 100.*float(dts)/float(total) if dts > -1 and total > 0 else -1 
        if percent > -1:
            msg = "%2.0f:%s" % (percent, self.msg )  
        else:
            msg = "%2s:%s" % ( "", self.msg)
        pass 
        return self.Format( self.t, dts, dfs, msg )

class Log(object):
    def __init__(self, path):
        lines = []
        p = None
        first = None
        for line in open(path).read().splitlines():
            l = Line(line)
            if not l.t is None:
                if first is None:
                    first = l
                pass
                l.prev = p 
                l.first = first
                p = l
            pass
            lines.append(l)
        pass
        self.lines = lines

        for l in self.lines:
            l.total = self.total_seconds()
        pass 


    def time(self, reverse=False):
        lines = self.lines
        num_lines = len(lines) 
        t = None
        for i in range(num_lines):
            j = num_lines - 1 - i if reverse else i 
            if not lines[j].t is None:
                t = lines[j].t
                break 
            pass
        pass
        return t

    def start(self):
        return self.time(reverse=False)
    def end(self):
        return self.time(reverse=True)
    def total_seconds(self):
        t0 = self.start()
        t1 = self.end()
        dts = 0 if (t0 is None or t1 is None) else (t1 - t0).total_seconds() 
        return dts

    def smry(self):    
        return "\n".join([
                          Line.Format(None,         -1, -1,  path      ), 
                          Line.Format(self.start(), -1, -1,  "start"   ), 
                          Line.Format(self.end(),   -1, -1,  "end"     ), 
                          Line.Format(None,         -1, self.total_seconds(), "total seconds" )
                         ])

    def __repr__(self):    
        return "\n".join( [Line.Hdr(), str(self), "","",self.smry()] )     
    def __str__(self):    
        return "\n".join(map(repr,self.lines))


def test_time():
    s = "2022-08-23 02:35:33.112"
    now = datetime.now()
    print(datetime.strftime(now, Line.FMT))
    d = datetime.strptime(s, Line.FMT)
    print(d)

if __name__ == '__main__':
    path = os.environ["LOG"]
    log = Log(path)
    print(repr(log))

