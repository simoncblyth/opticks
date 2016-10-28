#!/usr/bin/env python

import os, numpy as np

from opticks.ana.proplib import PropLib

np.set_printoptions(precision=3, suppress=True)



if __name__ == '__main__':

    blib = PropLib("GBndLib")
    names = blib.names
    t = np.load(os.path.expandvars("$IDPATH/GBndLib/GBndLib.npy"))

    ext, nl = "interpol", 820-60+1

    o = np.load(os.path.expandvars("$TMP/InterpolationTest/OInterpolationTest_%s.npy" % ext)).reshape(-1,4,2,nl,4) 
    c = np.load(os.path.expandvars("$TMP/InterpolationTest/CInterpolationTest_%s.npy" % ext)).reshape(-1,4,2,nl,4) 
    

    assert len(t) == len(names)
    assert len(t) == len(o) 
    assert len(t) == len(c) 
    n = len(t)

    cf={}
    cfb={}

    for i in range(n):

        name = names[i]
        omat,osur,isur,imat = name.split("/")

        cf_omat = np.all( c[i,blib.B_OMAT,0] == o[i,blib.B_OMAT,0] )
        cf_imat = np.all( c[i,blib.B_IMAT,0] == o[i,blib.B_IMAT,0] )

        if omat in cf:
            assert cf[omat] == cf_omat
        else:
            cf[omat] = cf_omat
        pass

        if imat in cf:
            assert cf[imat] == cf_imat
        else:
            cf[imat] = cf_imat
        pass


        if len(osur)>0:
            cf_osur = np.all( c[i,blib.B_OSUR,0] == o[i,blib.B_OSUR,0] )
            if osur in cf:
                assert cf[osur] == cf_osur
            else:
                cf[osur] = cf_osur
            pass
            if not cf_osur:
               if not osur in cfb:
                   cfb[osur] = []
               cfb[osur].append( (i,blib.B_OSUR,0) )  

        else:
            cf_osur = None

        if len(isur)>0:
            cf_isur = np.all( c[i,blib.B_ISUR,0] == o[i,blib.B_ISUR,0] )
            if isur in cf:
                assert cf[isur] == cf_isur
            else:
                cf[isur] = cf_isur
            pass
            if not cf_isur:
               if not isur in cfb:
                   cfb[isur] = []
               cfb[isur].append( (i,blib.B_ISUR,0) )  
        else:
            cf_isur = None


        if not cf_omat:
           if not omat in cfb:
               cfb[omat] = []
           cfb[omat].append( (i,blib.B_OMAT,0) )  
 
 
        if not cf_imat:
           if not imat in cfb:
               cfb[imat] = []
           cfb[imat].append( (i,blib.B_IMAT,0) )  
 

        #print "%4d omat %25s imat %25s         cf_omat %7s cf_imat %7s  " % (  i, omat, imat, cf_omat, cf_imat )  

        if len(osur)>0 or len(isur)>0:
            print "%4d osur %35s isur %35s         cf_osur %7s cf_isur %7s  " % (  i, osur, isur, cf_osur, cf_isur )  

    pass


    print "cf"

    for b in [True,False]:
        for k,v in cf.items():
            if v == b:
                if v:
                    print "%30s  %s  " % (k, v) 
                else:
                    print "%30s  %s  %s " % (k, v, str(cfb[k])) 






