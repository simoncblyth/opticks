#!/usr/bin/env python 
"""
CSGFoundryAB.py
=================

"""

import numpy as np, os, textwrap, builtins
from opticks.ana.fold import Fold, STR
from opticks.CSG.CSGFoundry import CSGFoundry, CSGPrim, CSGNode
from opticks.sysrap.stree import sn, snode, stree


class CSGFoundryAB(object):
    def __init__(self, a, b, _ip=0):
        self.a = a 
        self.b = b 
        self.ip = _ip

        self.check_4x4("node")
        self.check_4x4("prim")

        qwns = "npa nbb pbb".split()
        for qwn in qwns:
            self.compare_qwn(qwn)
        pass

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

    def brief(self):
        a = self.a
        b = self.b
        lines = []
        lines.append("CSGFoundryAB.brief")
        lines.append(a.brief())
        lines.append(b.brief())
        return "\n".join(lines)

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

    def check_prim_lv(self):
        a = self.a 
        b = self.b 
        a_lv = a.prim.view(np.int32)[:,1,1]  
        b_lv = b.prim.view(np.int32)[:,1,1]  
        assert np.all(a_lv==b_lv)    ## same G4VSolid => CSGPrim 
       
        assert np.all(np.unique(a_lv)==np.unique(b_lv)) 
        assert np.all(np.unique(a_lv,return_counts=True)[1]==np.unique(b_lv,return_counts=True)[1]) 

        a_lvu = np.unique(a_lv) 
        b_lvu = np.unique(b_lv) 
        assert np.all(np.arange(len(a_lvu))==a_lvu) 
        assert np.all(np.arange(len(b_lvu))==b_lvu) 
        assert np.all(a.meshname==b.meshname) 

    def check_node_tr(self):
        """
        """
        a = self.a 
        b = self.b 

        atr = a.node[:,3,3].view(np.int32) & 0x7fffffff
        btr = b.node[:,3,3].view(np.int32) & 0x7fffffff

        np.c_[np.unique(atr,return_counts=True)] 
        np.c_[np.unique(btr,return_counts=True)] 

    def check_node_index(self):
        """
        For A the node index increments until 15927, 
        thats probably the repeated unbalanced nodes 
        """
        a = self.a 
        b = self.b 

        aidx = a.node[:,1,3].view(np.int32)
        aidx_ = aidx[:15927]  
        assert np.all( aidx_ == np.arange(len(aidx_)) ) 

        bidx = b.node[:,1,3].view(np.int32)
        assert np.all( bidx == np.arange(len(bidx)) ) 


    def descLVDetail(self, lvid):
        a = self.a 
        b = self.b 
        lines = []
        lines.append("CSGFoundryAB.descLVDetail")
        lines.append(a.descLVDetail(lvid))
        lines.append(b.descLVDetail(lvid))
        return "\n".join(lines) 
 
    def check_prim(self):
        """
        np.all(a.prim.view(np.int32)[:,:2].reshape(-1,8)==b.prim.view(np.int32)[:,:2].reshape(-1,8)) 
        """
        a = self.a 
        b = self.b 
        ab = self
        setattr(self,"prim_numNode",np.where(a.prim[:,0,0].view(np.int32)!=b.prim[:,0,0].view(np.int32))[0])

        lines = []
        lines.append("CSGFoundryAB.check_prim")
        EXPR = filter(None,textwrap.dedent(r"""

        #(numNode,nodeOffset,tranOffset,planOffset)(sbtIndexOffset,meshIdx,repeatIdx,primIdx)
        a.prim.view(np.int32)[:,:2].reshape(-1,8)
        b.prim.view(np.int32)[:,:2].reshape(-1,8)

        #(numNode,nodeOffset,tranOffset,planOffset)
        a.prim[:,0].view(np.int32) 
        #(numNode,nodeOffset,tranOffset,planOffset)
        b.prim[:,0].view(np.int32) 
        len(np.where(a.prim[:,0]!=b.prim[:,0])[0])
        #numNode
        len(np.where(a.prim[:,0,0].view(np.int32)!=b.prim[:,0,0].view(np.int32))[0]) 
        len(ab.prim_numNode)
        np.c_[np.unique(a.primname[ab.prim_numNode],return_counts=True)]
        #nodeOffset
        len(np.where(a.prim[:,0,1].view(np.int32)!=b.prim[:,0,1].view(np.int32))[0]) 
        #tranOffset
        len(np.where(a.prim[:,0,2].view(np.int32)!=b.prim[:,0,2].view(np.int32))[0]) 
        #planOffset
        len(np.where(a.prim[:,0,3].view(np.int32)!=b.prim[:,0,3].view(np.int32))[0]) 
        #(sbtIndexOffset,meshIdx,repeatIdx,primIdx)
        a.prim[:,1].view(np.int32) 
        #(sbtIndexOffset,meshIdx,repeatIdx,primIdx)
        b.prim[:,1].view(np.int32) 
        len(np.where(a.prim[:,1]!=b.prim[:,1])[0])

        # A : (sbtIndexOffset,meshIdx,repeatIdx,primIdx) where prim_numNode discrepant 
        a.prim[ab.prim_numNode,1].view(np.int32)   
        # B : (sbtIndexOffset,meshIdx,repeatIdx,primIdx) where prim_numNode discrepant 
        b.prim[ab.prim_numNode,1].view(np.int32)   

        # A : (numNode,nodeOffset,tranOffset,planOffset) where prim_numNode discrepant
        a.prim[ab.prim_numNode,0].view(np.int32)  

        # B : (numNode,nodeOffset,tranOffset,planOffset) where prim_numNode discrepant
        b.prim[ab.prim_numNode,0].view(np.int32)  

        """).split("\n"))
        for expr in EXPR: 
            val = str(eval(expr)) if not expr.startswith("#") else "" 
            fmt = "%-80s \n%s\n" if len(val.split("\n")) > 1 else "%-80s : %s"
            lines.append(fmt % (expr, val))
        pass
        return "\n".join(lines) 


    def check_inst(self):
        a = self.a
        b = self.b
        ab = self
        setattr(self, "inst", np.max(np.abs(a.inst[:,:,:3]-b.inst[:,:,:3]).reshape(-1,12),axis=1))  

        lines = []
        lines.append("CSGFoundryAB.check_inst")
        EXPR = filter(None,textwrap.dedent(r"""
        np.all(a.inst[:,0,3].view(np.int32)==b.inst[:,0,3].view(np.int32)) 
        np.all(a.inst[:,1,3].view(np.int32)==b.inst[:,1,3].view(np.int32)) 
        np.all(a.inst[:,2,3].view(np.int32)==b.inst[:,2,3].view(np.int32)) 
        np.all(a.inst[:,3,3].view(np.int32)==b.inst[:,3,3].view(np.int32)) 
        np.all(a.inst[:,:,3].view(np.int32)==b.inst[:,:,3].view(np.int32)) 
        # deviation counts in the 12 floats of the inst transform
        len(np.where(ab.inst>1e-4)[0])
        len(np.where(ab.inst>1e-3)[0]) 
        len(np.where(ab.inst>1e-2)[0]) 
        """).split("\n"))

        for expr in EXPR: 
            val = str(eval(expr)) if not expr.startswith("#") else "" 
            fmt = "%-80s \n%s\n" if len(val.split("\n")) > 1 else "%-80s : %s"
            lines.append(fmt % (expr, val))
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
        np.c_[a.solid[:,1,:2],b.solid[:,1,:2]]#(numPrim,primOffset) 
        """).split("\n"))
        for expr in EXPR: 
            val = str(eval(expr)) if not expr.startswith("#") else "" 
            fmt = "%-80s \n%s\n" if len(val.split("\n")) > 1 else "%-80s : %s"
            lines.append(fmt % (expr, val))
        pass
        return "\n".join(lines) 

    def check_names(self):
        lines = []
        lines.append("CSGFoundryAB.check_names")
        EXPR = filter(None,textwrap.dedent(r"""
        np.all(a.mmlabel==b.mmlabel) 
        np.all(a.primname==b.primname) 
        np.all(a.meshname==b.meshname) 
        """).split("\n"))
        for expr in EXPR: 
            val = str(eval(expr)) if not expr.startswith("#") else "" 
            fmt = "%-80s \n%s\n" if len(val.split("\n")) > 1 else "%-80s : %s"
            lines.append(fmt % (expr, val))
        pass
        return "\n".join(lines) 



    def check_4x4(self, name):
        a = self.a 
        b = self.b        
        ab = self
        am = getattr(a, name)
        bm = getattr(b, name)

        lines = []
        lines.append("CSGFoundryAB.check_4x4 %s " % name )
        lines.append(" a.%s.shape : %s " % (name, str(am.shape)) )
        lines.append(" b.%s.shape : %s " % (name, str(bm.shape)) )
        if am.shape == bm.shape:
            abm = np.max(np.abs(am-bm).reshape(-1,16), axis=1 )
            setattr(self,name, abm )
            expr = "len(np.where(ab.%(name)s>0.01)[0])" % locals()
            lines.append(" %s : %s " % (expr, str(eval(expr))))
        else:
            setattr(self,name,None) 
        pass
        return STR("\n".join(lines))

    def compare_qwn(self, qwn="npa"):
        a = self.a 
        b = self.b        
        ab = self
        aq = getattr(a, qwn)
        bq = getattr(b, qwn)
        setattr(ab, qwn, np.max(np.abs(aq-bq), axis=1 ))
   

    def check_tran(self):
        lines = []
        lines.append(self.check_4x4("tran"))
        lines.append(self.check_4x4("itra"))
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


def eprint(_lines, _globals, _locals ):
    """
    Double hash "##" on the line suppresses printing the repr 
    of the evaluation. 

    Lines are first search for the position of the first "=" character.
    If present the position is used to extract the key and expression
    to be evaluated in the scope provided by the arguments. 
    The key with value is planted into the calling scope using builtins. 
    """
    lines = list(filter(None, textwrap.dedent(_lines).split("\n") ))

    print("-" * 100)
    print("\n".join(lines))
    print("-" * 100)

    for line in lines:
        eq = line.find("=") 
        eeq = line.find("==") 
        no_key = eq == -1 or ( eq > -1 and eq == eeq ) # no "=" or first "=" is from "==" 
        if no_key: 
            exp = line
            val = eval(exp, _globals, _locals )
            print(exp)
            print(repr(val))
        elif eq > -1:
            key, exp = line[:eq].strip(), line[1+eq:]  # before and after first "="
            val = eval(exp, _globals, _locals )
            #print("set [%s] " % key )
            setattr(builtins, key, val)
            print(line)
            if line.find("##") == -1:
                print(repr(val))
            pass
        else:
            print(line)
        pass
    pass       
    print("-" * 100)



if __name__ == '__main__':

    np.set_printoptions(edgeitems=10) 

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

    print(ab.check_names())
    print(ab.check_pr())
    print(ab.check_prim())
    print(ab.check_solid())
    print(ab.check_tran())
    print(ab.check_inst())

    print(ab.brief())

    a_meshname = np.char.partition(a.meshname.astype("U"),"0x")[:,0]  
    b_meshname = b.meshname.astype("U")
    assert np.all( a_meshname == b_meshname ) 

    a_primname = np.char.partition(a.primname.astype("U"),"0x")[:,0]  
    b_primname = b.primname.astype("U")
 
    an = A.SSim.stree._csg.sn
    bn = B.SSim.stree._csg.sn


    a_tc = a.node[:,3,2].view(np.int32)
    b_tc = b.node[:,3,2].view(np.int32)
    w_tc = np.where( a_tc != b_tc )[0]


    eprint(r"""
    w_npa3 = np.where( ab.npa > 1e-3 )[0]
    w_npa3.shape

    w_nbb3 = np.where( ab.nbb > 1e-3 )[0]  ##
    w_nbb3.shape

    w_nbb2 = np.where( ab.nbb > 1e-2 )[0]  ##
    w_nbb2.shape

    w_pbb3 = np.where( ab.pbb > 1e-3 )[0]  ##
    w_pbb3.shape

    w_pbb2 = np.where( ab.pbb > 1e-2 )[0]  ##
    w_pbb2.shape

    w_solid = np.where( a.solid != b.solid )[0]  ##
    w_solid.shape

    w_solid = np.where( a.solid != b.solid )[0] 

    w_nix = np.where(a.nix != b.nix)[0] ## 
    w_nix.shape

    np.all( a.ntc == b.ntc )  # node typecode
    np.all( a.ntr == b.ntr )  # node transform idx + 1 
    np.all( a.ncm == b.ncm )  # node complement 

    np.all( a.pix == b.pix )  # primIdx 

    """, globals(), locals() )


        

