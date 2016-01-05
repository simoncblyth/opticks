#!/usr/bin/env python

import os, re, logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from base import ini_, json_

class DateParser(object):
    ptn = re.compile("(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
    def __call__(self, txt):
        m = self.ptn.match(txt)
        if m is None:return None
        if len(m.groups()) != 6:return None
        dt = datetime(*map(int, m.groups()))
        return dt 

dateparser = DateParser() 

def finddir(base, dirfilter=lambda _:True):
    for root, dirs, files in os.walk(base):
        for name in dirs:
            path = os.path.join(root,name)
            d = dirfilter(path)
            if d is not None:
                yield path


class DatedFolder(object):
    def __call__(self, path):
        name = os.path.basename(path) 
        return dateparser(name)

class Metadata(object):

    COMPUTE = 0x1 << 1
    INTEROP = 0x1 << 2 
    CFG4    = 0x1 << 3 

    def __init__(self, path):
        self.path = path
        self.datefold = os.path.basename(path)
        tag = os.path.basename(os.path.dirname(path))        
        self.tag = tag 
        self.timestamp = dateparser(self.datefold) 
        self.parameters = json_(os.path.join(self.path, "parameters.json"))
        self.times = ini_(os.path.join(self.path, "t_delta.ini"))

    propagate = property(lambda self:float(self.times.get('propagate','-1')))

    mode = property(lambda self:self.parameters.get('mode',"no-mode") )
    photonData = property(lambda self:self.parameters.get('photonData',"no-photonData") )
    recordData = property(lambda self:self.parameters.get('recordData',"no-recordData") )
    sequenceData = property(lambda self:self.parameters.get('sequenceData',"no-sequenceData") )
    numPhotons = property(lambda self:self.parameters.get('NumPhotons',"no-NumPhotons") )

    def _flags(self):
        flgs = 0 
        if self.mode.lower() == "compute":
            flgs |= self.COMPUTE 
        elif self.mode.lower() == "interop":
            flgs |= self.INTEROP 
        elif self.mode.lower() == "cfg4":
            flgs |= self.CFG4 

        return flgs 
    flags = property(_flags)

    def __repr__(self):
        return "%s %32s %32s %s %s " % (self.path, self.photonData, self.recordData, self.propagate, self.mode )


class Catdir(object):
    def __init__(self, cat):
        path = os.path.expandvars("$LOCAL_BASE/env/opticks/%s" % cat )
        df = DatedFolder()

        metadata = {}
        for p in finddir(path, df):
            md = Metadata(p)
            tag = md.tag
            if tag not in metadata:
                metadata[tag] = []
            metadata[tag].append(md)

        self.metadata = metadata
        self.path = path


    def times(self, tag):
        """
        ::

            a[a.flgs & (0x1 << 1) != 0]

        """
        tags = [tag, "-%s" % tag]   # convention, negate to get equivalent cfg4 tag
        mds = []
        for tag in tags:
            mds.extend(self.metadata.get(tag, [])) 

        print "\n".join(map(str,mds))

        n = len(mds)
        a = np.recarray((n,), dtype=[("index", np.int32), ("time", "|O8"), ("propagate", np.float32), ("flgs", np.uint32 )])

        for i in range(n):
            md = mds[i]
            dat = (i, md.timestamp, md.propagate, md.flags )
            print dat 
            a[i] = dat
        pass
        return a 

    def __repr__(self):
         return "%s tags %s " % (self.path, repr(self.metadata.keys()))





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)    

    cat = "rainbow"
    tag = "6"
    title = "Propagate times (s) for 1M Photons with %s geometry, tag %s, [max/avg/min]" % (cat, tag)  

    catd = Catdir(cat)
    plt.close()
    plt.ion()

    a = catd.times(tag)


if 1:
    fig = plt.figure()
    fig.suptitle(title)

    compute = a.flgs & Metadata.COMPUTE != 0 
    interop = a.flgs & Metadata.INTEROP != 0 
    cfg4    = a.flgs & Metadata.CFG4 != 0 

    msks = [cfg4, interop, compute]
    ylims = [[0,60],[0,5],[0,1]]
    labels = ["CfGeant4", "Opticks Interop", "Opticks Compute"]

    n = len(msks)
    for i, msk in enumerate(msks):
        ax = fig.add_subplot(n,1,i+1)
        d = a[msk]

        t = d.propagate

        mn = t.min()
        mx = t.max()
        av = np.average(t)        

        label = "%s [%5.2f/%5.2f/%5.2f] " % (labels[i], mx,av,mn)
 
        loc = "lower right" if i == 0 else "upper right" 

        ax.plot( d.index, d.propagate, "o")
        ax.plot( d.index, d.propagate, drawstyle="steps", label=label)
        ax.set_ylim(ylims[i])
        ax.legend(loc=loc)
    pass


    ax.set_xlabel('All times from: MacBook Pro (2013), NVIDIA GeForce GT 750M 2048 MB (384 cores)')

    plt.show()




    



