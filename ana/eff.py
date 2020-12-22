#!/usr/bin/env python
import os, subprocess, lxml.etree as ET, numpy as np
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
rc, gdmlpath = subprocess.getstatusoutput("bash -lc \"opticksaux-;opticksaux-dx1\"" )
gdml = parse_(gdmlpath)

#print(gdml)
effs = gdml.xpath("//matrix[starts-with(@name,'EFFICIENCY')]")  

nn = []
aa = []

for eff in effs:
    #print(eff)
    name = eff.attrib['name'] 
    vals = eff.attrib['values'] 
    a = np.fromstring(vals, sep=' ').reshape(-1,2)  
    #if a[:,1].max() > 0.:
    if True:
        nn.append(name)
        aa.append(a)
    pass
pass
print("matrix with max value greater than zero %d " % len(nn))

assert len(set(nn)) == len(nn), "looks like a non-unique efficiency name" 

n = nn[0]
a = aa[0]

print(n)
print(a)
print(" max %s " % a[:,1].max() )


