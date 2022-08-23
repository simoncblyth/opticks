#!/bin/bash -l 
"""
log.py : logfile parsing and presenting with time filtering
=============================================================

::

    :set nowrap

    In [1]: log[1]  ## show log lines with delta times more than 1 percent of total time
    Out[1]: 
                     timestamp :        DTS-prev :        DFS-frst :msg
    2022-08-23 20:25:36.116000 :      0.3140[35] :      0.3330[37] : INFO  [22382] [QSim::UploadComponents@106] ] new QBase : latency here of about 0.3s from first device access, if latency of >1s need to start nvidia-persistenced 
    2022-08-23 20:25:36.294000 :      0.1780[20] :      0.5110[56] : INFO  [22382] [QSim::UploadComponents@110] QRng path /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin rngmax 1000000 qr 0x2b6cb70 d_qr 0x7f056dc00200
    2022-08-23 20:25:36.312000 :      0.0180[ 2] :      0.5290[58] : INFO  [22382] [QSim::UploadComponents@125] QBnd src NP  dtype <f4(45, 4, 2, 761, 4, ) size 1095840 uifc f ebyte 4 shape.size 5 data.size 4383360 meta.size 69 names.size 45 tex QTex width 761 height 360 texObj 1 meta 0x2d908f0 d_meta 0x7f056dc01000 tex 0x2d90880
    2022-08-23 20:25:36.425000 :      0.1030[11] :      0.6420[71] : INFO  [22382] [SBT::init@69] 
    2022-08-23 20:25:36.449000 :      0.0130[ 1] :      0.6660[73] : INFO  [22382] [IAS_Builder::CollectInstances@77]  i   25601 gasIdx   2 sbtOffset   3094 gasIdx_sbtOffset.size   3
    2022-08-23 20:25:36.488000 :      0.0130[ 1] :      0.7050[78] : INFO  [22382] [SBT::createHitgroup@849] gas_idx 1 so.numPrim 5 so.primOffset 3089
    2022-08-23 20:25:36.525000 :      0.0350[ 4] :      0.7420[82] : INFO  [22382] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 20:25:36.690000 :      0.1460[16] :      0.9070[100] : INFO  [22382] [G4CXOpticks::saveEvent@417] ]


                             - :                 :                 :G4CXSimtraceTest.log
    2022-08-23 20:25:35.783000 :                 :                 :start
    2022-08-23 20:25:36.690000 :                 :                 :end
                             - :                 :      0.9070[100] :total seconds
                             - :                 :      1.0000[100] :pc_cut





"""
import os, sys
from datetime import datetime

class Line(object):
    FMT = '%Y-%m-%d %H:%M:%S.%f'
    LFMT = "%26s : %16s : %16s :%s"

    @classmethod
    def Parse(cls, stmp):
        try:
            d = datetime.strptime(stmp, cls.FMT)
        except ValueError:
            d = None
        pass
        return d 

    @classmethod
    def FVal(cls, dt, pc_dt ):
        return " %15s" % "" if (dt < 0 or pc_dt < 0) else " %10.4f[%3.0f]" % (dt, pc_dt) 

    @classmethod
    def Format(cls, t, dts, pc_dts,  dfs, pc_dfs,  msg):
        stmp = "-" if t is None else datetime.strftime(t, cls.FMT)
        return cls.LFMT % ( stmp, cls.FVal(dts, pc_dts), cls.FVal(dfs, pc_dfs), msg )
  
    @classmethod
    def Hdr(cls, headline):
        return cls.LFMT % ( "timestamp", "DTS-prev", "DFS-frst", headline )

    def __init__(self, line):
        t = self.Parse( line[:23] )
        msg = line if t is None else line[23:]
        self.t = t
        self.msg = msg
        self.prev = None 
        self.first = None
        self.total = 0.
        self.line = line 
        self.is_first = False
        self.is_last = False

    def update(self):
        t = self.t
        total = self.total
        p = None if self.prev is None else self.prev.t  
        f = None if self.first is None else self.first.t  
        dts = -1 if (p is None or t is None) else (t - p).total_seconds() 
        dfs = -1 if (f is None or t is None) else (t - f).total_seconds() 
        pc_dts = 100.*float(dts)/float(total) if dts > -1 and total > 0 else -1 
        pc_dfs = 100.*float(dfs)/float(total) if dfs > -1 and total > 0 else -1 

        self.dts = dts
        self.dfs = dfs
        self.pc_dts = pc_dts
        self.pc_dfs = pc_dfs

    def __repr__(self):
        return self.Format( self.t, self.dts, self.pc_dts,  self.dfs, self.pc_dfs,  self.msg )

class Log(object):
    def __init__(self, path):
        raw = open(path, encoding='utf-8-sig').read().splitlines()  
        # avoid \ufeff BOM bill-of-materials at head with encoding='utf-8-sig'
        lines = []
        p = None
        first = None
        for line in raw:
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
        self.path = path 
        self.raw = raw 
        self.lines = lines
        self.pc_cut = 0 

        for l in self.lines:
            l.total = self.total_seconds()
            l.update()
        pass 

        num_lines = len(self.lines) 
        for i in range(num_lines):
            l = self.lines[i]
            if not l.t is None and l.is_first == False:
                l.is_first = True 
                break
            pass
        pass
        for i in range(num_lines):
            l = self.lines[num_lines - 1 - i]
            if not l.t is None and l.is_last == False:
                l.is_last = True 
                break
            pass
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

    def __getitem__(self, pc_cut):
        self.pc_cut = pc_cut
        return self 

    def __repr__(self):   
        headline = "path:%s pc_cut:%s " % ( self.path, self.pc_cut )
        return "\n".join( [Line.Hdr(headline), str(self) ] )     

    def __str__(self):    
        return "\n".join(map(repr,filter(self.select,self.lines)))

    def select(self, l):
        pc_cut = self.pc_cut
        if pc_cut == 0:
            ret = True
        else:
            ret = l.pc_dts is None or l.pc_dts > pc_cut or l.is_first or l.is_last
        pass
        return ret
        


def test_time():
    s = "2022-08-23 02:35:33.112"
    now = datetime.now()
    print(datetime.strftime(now, Line.FMT))
    d = datetime.strptime(s, Line.FMT)
    print(d)

if __name__ == '__main__':
    path = os.environ["LOG"]
    log = Log(path)
    print("repr(log[2])")
    print(repr(log[2]))

