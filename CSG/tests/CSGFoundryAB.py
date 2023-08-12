#!/usr/bin/env python 
"""
CSGFoundryAB.py
=================

"""

import numpy as np, os, textwrap
from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry, CSGPrim, CSGNode


class CSGFoundryAB(object):
    def __init__(self, a, b, _ip=0):
        self.a = a 
        self.b = b 
        self.ip = _ip

    def _set_ip(self, ip):
        self._ip = ip 

        a = self.a 
        b = self.b 

        ann = a.pr.numNode[ip]
        bnn = b.pr.numNode[ip]
        ano = a.pr.nodeOffset[ip]
        bno = b.pr.nodeOffset[ip]
        alv = a.pr.meshIdx[ip]
        blv = b.pr.meshIdx[ip]
        amn = a.meshname[alv]
        bmn = b.meshname[blv]
        anode = a.node[ano:ano+ann]  
        bnode = b.node[bno:bno+bnn]  

        atc = a.ni.typecode[ano:ano+ann]
        btc = b.ni.typecode[bno:bno+bnn]

        aco = a.ni.comptran[ano:ano+ann] >> 31
        bco = b.ni.comptran[bno:bno+bnn] >> 31

        self.alv = alv
        self.blv = blv
        self.amn = amn
        self.bmn = bmn
        self.anode = anode
        self.bnode = bnode
        self.atc = atc
        self.btc = btc
        self.aco = aco
        self.bco = bco

    def __repr__(self):
        return "AB %d alv %s:%s blv %s:%s " % ( self._ip, self.alv, self.amn, self.blv, self.bmn )

    def _get_ip(self):
        return self._ip

    ip = property(_get_ip, _set_ip) 


    def check_pr(self):
        a = self.a 
        b = self.b
        assert a.pr.dtype.names == b.pr.dtype.names

        lines = []
        lines.append("CSGFoundryAB.check_pr")
        expr_ = "len(np.where(a.pr.%(name)s != b.pr.%(name)s )[0])"
        for name in a.pr.dtype.names:
            expr = expr_ % locals()  
            lines.append("%80s : %s " % (expr, eval(expr)))
        pass
        return "\n".join(lines) 

    def check_solid(self):
        """
        [:,1] houses (numPrim, primOffset, type, padding)
        """
        a = self.a 
        b = self.b 
        lines = []
        lines.append("CSGFoundryAB.check_solid")
        EXPR = filter(None,textwrap.dedent(r"""
        np.all(a.solid[:,1]==b.solid[:,1]) 
        np.c_[a.solid[:,1,:2],b.solid[:,1,:2]]#(numNode,nodeOffset) 
        """).split("\n"))
        for expr in EXPR: 
            val = str(eval(expr)) if not expr.startswith("#") else "" 
            fmt = "%-80s \n%s\n" if len(val.split("\n")) > 1 else "%-80s : %s"
            lines.append(fmt % (expr, val))
        pass
        return "\n".join(lines) 

        



def checktran(a, b, ip, order="A" ):
    """
    ::
 
        at,bt,atran,btran,dtran = checktran(a,b,2000,"S")

        # order A:asis, S:sum_sort, U:unique
        #

    """
    ann = a.pr.numNode[ip]
    bnn = b.pr.numNode[ip]

    ano = a.pr.nodeOffset[ip]
    bno = b.pr.nodeOffset[ip]

    atr = ( a.ni.comptran[ano:ano+ann] & 0x7fffffff ) - 1  
    btr = ( b.ni.comptran[bno:bno+bnn] & 0x7fffffff ) - 1 

    at = atr[atr>-1]   # not -1 as thats done above 
    bt = btr[btr>-1]

    if len(at) == len(bt):
        if order == "U":
            atran = np.unique( a.tran[at], axis=0 )  
            btran = np.unique( b.tran[bt], axis=0 )  
        elif order == "S":
            # sort by the sum of the 16 elements 
            a_tran = a.tran[at]
            b_tran = b.tran[bt]

            ss_atran = np.argsort(np.sum(a_tran, axis=(1,2)))
            ss_btran = np.argsort(np.sum(b_tran, axis=(1,2)))

            atran = a_tran[ss_atran]  
            btran = b_tran[ss_btran]  
        elif order == "A":
            atran = a.tran[at]   
            btran = b.tran[bt] 
        pass
        dtran = np.sum( np.abs( atran - btran), axis=(1,2) ) 
    else:
        dtran = None
    pass
    return at,bt,atran,btran,dtran
 


def checkprim(a, b, ip, dump=False, order="A"):
    """
    :param a: cf
    :param b: cf
    :param ip: prim index
    :param dump:
    :param order: control tran order with one of "ASIS" "SS" "U"

    * A : default 
    * S : sorting by sum of 16 elements
    * U : apply np.unique 
 
    """
    fmt = "ip:%(ip)4d " 
    alv = a.pr.meshIdx[ip]
    blv = b.pr.meshIdx[ip]
    slv = "/" if alv == blv else "*"
    fmt += "lv:%(alv)3d%(slv)s%(blv)3d "

    ann = a.pr.numNode[ip]
    bnn = b.pr.numNode[ip]
    snn = "/" if ann == bnn  else "*"
    fmt += "nn:%(ann)3d%(snn)s%(bnn)3d "

    ano = a.pr.nodeOffset[ip]
    bno = b.pr.nodeOffset[ip]
    sno = "/" if ano == bno  else "%"   # not "*" as this is bound to get stuck after first deviation
    fmt += "no:%(ano)5d%(sno)s%(bno)5d " 

    anode = a.node[ano:ano+ann]  
    bnode = b.node[bno:bno+bnn]  

    amn = a.meshname[alv]
    bmn = b.meshname[blv]
    smn = "/" if amn == bmn  else "*"

    atc = a.ni.typecode[ano:ano+ann]
    btc = b.ni.typecode[bno:bno+bnn]

    aco = a.ni.comptran[ano:ano+ann] >> 31
    bco = b.ni.comptran[bno:bno+bnn] >> 31

    at,bt,atran,btran,dtran = checktran(a,b,ip,order=order)

    if not dtran is None:
        wtran = np.where( dtran > 1e-2 )[0]   # large epsilon to avoid any float/double diffs
        ltran = len(wtran)
    else:
        ltran = -1 
    pass  
    stran = ":" if ltran == 0 else "*"
    fmt += " %(order)s tr%(stran)s%(ltran)2d "

    fmt += "mn:%(amn)40s%(smn)s%(bmn)40s "


    if ip == 0: print(fmt)
    line = fmt % locals()
    if "*" in line or dump:
        print(line)
    pass

    if snn == "/":
        expr = "np.c_[atran, btran], np.c_[atc, aco, btc, bco], dtran"
        return expr, np.c_[atran, btran], np.c_[atc, aco, btc, bco], dtran
    else:
        return None
    pass


def checkprims(a, b, ip0=-1, ip1=-1, order="A"):
    """
    ::

        In [1]: checkprims(a,b)
         ip:2375  lv: 93/ 93 nn: 15*127 no:15209/15209 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2376  lv: 93/ 93 nn: 15*127 no:15224%15336 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2377  lv: 93/ 93 nn: 15*127 no:15239%15463 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2378  lv: 93/ 93 nn: 15*127 no:15254%15590 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2379  lv: 93/ 93 nn: 15*127 no:15269%15717 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2380  lv: 93/ 93 nn: 15*127 no:15284%15844 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2381  lv: 93/ 93 nn: 15*127 no:15299%15971 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:2382  lv: 93/ 93 nn: 15*127 no:15314%16098 mn:        solidSJReceiverFastern/        solidSJReceiverFastern 
         ip:3126  lv: 99/ 99 nn: 31*1023 no:23372%24268 mn:                          uni1/                          uni1 

    """
    if ip0 < 0 and len(a.pr.numNode) == len(b.pr.numNode):
        ip0 = 0 
        ip1 = len(a.pr.numNode)
    pass
    for i in range(ip0, ip1):
        checkprim(a,b, i, order=order)
    pass


if __name__ == '__main__':

    CSGPrim.Type()

    A = Fold.Load("$A_CFBASE/CSGFoundry", symbol="A")
    B = Fold.Load("$B_CFBASE/CSGFoundry", symbol="B")

    print(repr(A))
    print(repr(B))

    a = CSGFoundry.Load("$A_CFBASE", symbol="a")
    b = CSGFoundry.Load("$B_CFBASE", symbol="b")
    print(a.brief())
    print(b.brief())

    ab = CSGFoundryAB(a,b)
    print(ab.check_pr())
    print(ab.check_solid())

