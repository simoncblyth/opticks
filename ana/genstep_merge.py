#!/usr/bin/env python
"""
genstep_merge.py : combine Cerenkov and scintillation gensteps in natural ones
================================================================================

NumPy level merging of gensteps with slicing of inputs.
Slicing is done as scintillation is already a bit heavy for mobile GPU.

Note the input gensteps are read from paths within the 
installed opticksdata (canonically within /usr/local/opticks/opticksdata/gensteps/)
whereas the output gensteps are in the "input" opticksdata clone 
in ~/opticksdata/gensteps.

Used for testing the "natural" genstep event type prior to 
implementing actual Geant4 level natural genstep collection. 

::

    ./genstep_merge.py
    [2016-08-23 17:57:05,323] p77451 {./genstep_merge.py:63} INFO - loaded a : (7836, 6, 4) /usr/local/opticks/opticksdata/gensteps/dayabay/cerenkov/1.npy 
    [2016-08-23 17:57:05,323] p77451 {./genstep_merge.py:64} INFO - loaded b : (13898, 6, 4) /usr/local/opticks/opticksdata/gensteps/dayabay/scintillation/1.npy 
    [2016-08-23 17:57:05,323] p77451 {./genstep_merge.py:70} INFO - sliced aa : (2612, 6, 4) 
    [2016-08-23 17:57:05,323] p77451 {./genstep_merge.py:71} INFO - sliced bb : (4633, 6, 4) 
    [2016-08-23 17:57:05,325] p77451 {./genstep_merge.py:82} INFO - merged (7245, 6, 4) /Users/blyth/opticksdata/gensteps/dayabay/natural/1.npy exists already skipping 

"""
import os, logging
import numpy as np

from opticks.ana.base import opticks_main
from opticks.ana.nload import A, gspath_

log = logging.getLogger(__name__)

if __name__ == '__main__':

    args = opticks_main(det="dayabay", src="cerenkov,scintillation", tag="1", sli="::3")

    srcs = args.src.split(",")
    assert len(srcs) == 2

    sli = slice(*map(lambda _:int(_) if len(_) > 0 else None,args.sli.split(":")))


    try:    
        a = A.load_("gensteps",srcs[0],args.tag,args.det)
        b = A.load_("gensteps",srcs[1],args.tag,args.det)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)


    log.info("loaded a : %s %s " % (repr(a.shape),a.path))
    log.info("loaded b : %s %s " % (repr(b.shape),b.path))

    aa = a[sli]
    bb = b[sli]
    assert aa.shape[1:] == (6,4) and bb.shape[1:] == (6,4)

    log.info("sliced aa : %s " % (repr(aa.shape)))
    log.info("sliced bb : %s " % (repr(bb.shape)))


    cc = np.empty((len(aa)+len(bb),6,4), dtype=np.float32)
    cc[0:len(aa)] = aa
    cc[len(aa):len(aa)+len(bb)] = bb

    gsp = gspath_("natural", args.tag, args.det, gsbase=os.path.expanduser("~/opticksdata/gensteps"))
    gsd = os.path.dirname(gsp) 

    if os.path.exists(gsp):
        log.info("merged %s %s exists already skipping " % (repr(cc.shape),gsp))
    else:
        if not os.path.isdir(gsd):
            log.info("creating directory gsd %s " % gsd )
            os.makedirs(gsd)
        pass
        log.info("saving merged gensteps to %s " % gsp )
        np.save(gsp, cc )




