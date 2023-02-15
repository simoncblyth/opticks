#!/usr/bin/env python 
"""


"""


from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry, CSGPrim, CSGNode


def checkprim(a, b, ip, dump=False):
    fmt = "ip:%(ip)3d " 
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
    fmt += "no:%(ano)3d%(sno)s%(bno)3d " 


    anode = a.node[ano:ano+ann]  
    bnode = b.node[bno:bno+bnn]  

    amn = a.meshname[alv]
    bmn = b.meshname[blv]
    smn = "/" if amn == bmn  else "*"


    atr = ( a.ni.comptran[ano:ano+ann] & 0x7fffffff ) - 1  
    btr = ( b.ni.comptran[bno:bno+bnn] & 0x7fffffff ) - 1 

    aco = a.ni.comptran[ano:ano+ann] >> 31
    bco = b.ni.comptran[bno:bno+bnn] >> 31

    atc = a.ni.typecode[ano:ano+ann]
    btc = b.ni.typecode[bno:bno+bnn]

    atran = a.tran[atr]   
    btran = b.tran[btr]   


    if snn == "/":
        dtran = np.sum( np.abs( atran - btran), axis=(1,2) ) 
        wtran = np.where( dtran > 1e-2 )[0]   # large epsilon to avoid any float/double diffs
        ltran = len(wtran)
    else:
        dtran = None
        ltran = -1 
    pass
    stran = ":" if ltran == 0 else "*"

    fmt += "tr%(stran)s%(ltran)2d "
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


def checkprims(a, b, ip0=-1, ip1=-1):
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
        checkprim(a,b, i)
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


