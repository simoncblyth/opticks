#!/usr/bin/env python
"""
"""

import os, json
j = 0 





def boundary(code):
    global j 

    bnd = j['lib']['boundary'][str(code)]
    imat = bnd['imat']
    omat = bnd['omat']
    isur = bnd['isur']
    osur = bnd['osur']
    return "(+1) %0.2d : %s %s %s %s " % (code + 1, imat['shortname'], omat['shortname'], isur['name'], osur['name'] )
   
def boundaries():
    global j 
    codes = sorted(map(int, j['lib']['boundary'].keys()))
    for code in codes:
        print boundary(code) 

if __name__ == '__main__':
    j = json.load(file(os.path.expandvars("$IDPATH/GBoundaryLibMetadata.json")))
    boundaries()
