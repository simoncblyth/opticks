#!/usr/bin/env python

"""
Labelling with the subdepth... 

* all *in* are at subdepth 1


args: deep.py
[2017-06-19 18:33:57,767] p82049 {/Users/blyth/opticks/analytic/csg.py:342} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/analytic/deep_py 
analytic=1_csgpath=/tmp/blyth/opticks/analytic/deep_py_name=deep_py_mode=PyCsgInBox
                                                                                            un 7            
                                                                            un 6                     in 1    
                                                            un 5                     in 1         cy 0     !cy 0
                                            un 4                     in 1         cy 0     !cy 0                
                            un 3                     in 1         cy 0     !cy 0                                
            un 2                     in 1         cy 0     !cy 0                                                
    in 1             in 1         cy 0     !cy 0                                                                
cy 0     !cy 0     cy 0     !cy 0                                                                                
delta:analytic blyth$ 



"""

import logging
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgpath="$TMP/analytic/deep_py" )

CSG.boundary = args.testobject
CSG.kwa = dict(verbosity="1")




a = CSG("cylinder", param = [0.000,0.000,0.000,660.000],param1 = [-5.000,5.000,0.000,0.000])
b = CSG("cylinder", param = [0.000,0.000,0.000,31.500],param1 = [-5.050,5.050,0.000,0.000],complement = True)
ab = CSG("intersection", left=a, right=b)

c = CSG("cylinder", param = [0.000,0.000,0.000,46.500],param1 = [-12.500,12.500,0.000,0.000])
d = CSG("cylinder", param = [0.000,0.000,0.000,31.500],param1 = [-12.625,12.625,0.000,0.000],complement = True)
cd = CSG("intersection", left=c, right=d)
cd.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-17.500,1.000]]

abcd = CSG("union", left=ab, right=cd)

e = CSG("cylinder", param = [0.000,0.000,0.000,660.000],param1 = [-67.500,67.500,0.000,0.000])
f = CSG("cylinder", param = [0.000,0.000,0.000,650.000],param1 = [-68.175,68.175,0.000,0.000],complement = True)
ef = CSG("intersection", left=e, right=f)
ef.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,72.500,1.000]]

abcdef = CSG("union", left=abcd, right=ef)

g = CSG("cylinder", param = [0.000,0.000,0.000,660.000],param1 = [-5.000,5.000,0.000,0.000])
h = CSG("cylinder", param = [0.000,0.000,0.000,122.000],param1 = [-5.050,5.050,0.000,0.000],complement = True)
gh = CSG("intersection", left=g, right=h)
gh.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,145.000,1.000]]

abcdefgh = CSG("union", left=abcdef, right=gh)

i = CSG("cylinder", param = [0.000,0.000,0.000,132.000],param1 = [-17.500,17.500,0.000,0.000])
j = CSG("cylinder", param = [0.000,0.000,0.000,122.000],param1 = [-17.675,17.675,0.000,0.000],complement = True)
ij = CSG("intersection", left=i, right=j)
ij.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,167.500,1.000]]

abcdefghij = CSG("union", left=abcdefgh, right=ij)

k = CSG("cylinder", param = [0.000,0.000,0.000,167.000],param1 = [-10.000,10.000,0.000,0.000])
l = CSG("cylinder", param = [0.000,0.000,0.000,122.000],param1 = [-10.100,10.100,0.000,0.000],complement = True)
kl = CSG("intersection", left=k, right=l)
kl.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,195.000,1.000]]

abcdefghijkl = CSG("union", left=abcdefghij, right=kl)

m = CSG("cylinder", param = [0.000,0.000,0.000,167.000],param1 = [-10.000,10.000,0.000,0.000])
n = CSG("cylinder", param = [0.000,0.000,0.000,41.500],param1 = [-10.100,10.100,0.000,0.000],complement = True)
mn = CSG("intersection", left=m, right=n)
mn.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,215.000,1.000]]

abcdefghijklmn = CSG("union", left=abcdefghijkl, right=mn)





obj = abcdefghijklmn

con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="HY", level="5" )
CSG.Serialize([con, obj], args.csgpath )


print obj.txt


subdepth = 1   # bileaf level, one step up from the primitives
subtrees = obj.subtrees_(subdepth=subdepth)   # collect the bileafs
ops = obj.operators_(minsubdepth=subdepth+1)  # look at operators above the bileafs 

print "ops:", map(CSG.desc, ops)

for i,sub in enumerate(subtrees):
    print "\n\nsub %s " % i
    sub.analyse()
    sub.dump(detailed=True)





