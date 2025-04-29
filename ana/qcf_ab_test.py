#!/usr/bin/env python
"""

https://github.com/numpy/numpy/pull/19388

"""

import numpy as np
import opticks.ana.qcf_ab as qcf_ab
from opticks.ana.nbase import chi2, chi2_pvalue

a = np.array([
       b'TO AB                                                                                           ',
       b'TO AB                                                                                           ',
       b'TO AB                                                                                           ',
       b'TO BT AB                                                                                        ',
       b'TO BT AB                                                                                        ',
       b'TO BT AB                                                                                        ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             '], dtype='|S96' )



b = np.array([
       b'TO AB                                                                                           ',
       b'TO AB                                                                                           ',
       b'TO BT AB                                                                                        ',
       b'TO BT AB                                                                                        ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             '], dtype='|S96' )

      
au, ax, an = np.unique(a, return_index=True, return_counts=True)
bu, bx, bn = np.unique(b, return_index=True, return_counts=True)

auf = np.asfortranarray(au)
buf = np.asfortranarray(bu)

print("au",au)
print("ax",ax)
print("an",an)

print("bu",bu)
print("bx",bx)
print("bn",bn)

qu = np.unique(np.concatenate([au,bu]))  ## unique histories of both A and B in uncontrolled order



ab = qcf_ab.foo(qu, au,ax,an,bu,bx,bn )

print("ab.shape %s " % str(ab.shape))
print("ab", ab)

# last dimension 0,1 corresponding to A,B 

abx = np.max(ab[:,2,:], axis=1 )   # max of aqn, bqn counts 
abxo = np.argsort(abx)[::-1]       # descending count order indices
abo = ab[abxo]                     # ab ordered  
quo = qu[abxo]                     # qu ordered 

c2,c2n,c2c = chi2( abo[:,2,0], abo[:,2,1], cut=0 )


iq = np.arange(len(qu))
siq = list(map(lambda _:"%2d" % _ , iq ))  # row index 
sc2 = list(map(lambda _:"%7.4f" % _, c2 ))


pstr_ = lambda _:_.strip().decode("utf-8")
_quo = list(map(pstr_, quo))
mxl = max(list(map(len, _quo)))
fmt = "%-" + str(mxl) + "s"
_quo = list(map(lambda _:fmt % _, _quo ))
_quo = np.array( _quo )

 

sabo2 = list(map(lambda _:"%6d %6d" % tuple(_), abo[:,2,:]))
sabo1 = list(map(lambda _:"%6d %6d" % tuple(_), abo[:,1,:]))
abexpr = "np.c_[siq,_quo,siq,sabo2,sc2,sabo1]"

lines = []
lines.append(str(eval(abexpr)))

print("\n".join(lines))





