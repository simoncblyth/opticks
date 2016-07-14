#!/usr/bin/env python
"""
Metadata
~~~~~~~~~~

Access the metadata json files written by Opticks runs, 
allowing evt digests and run times to be compared. 


"""

from datetime import datetime
import os, re, logging
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_environment
from opticks.ana.base import ini_, json_



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

    # parameter accessors (from the json)
    mode = property(lambda self:self.parameters.get('mode',"no-mode") )
    photonData = property(lambda self:self.parameters.get('photonData',"no-photonData") )
    recordData = property(lambda self:self.parameters.get('recordData',"no-recordData") )
    sequenceData = property(lambda self:self.parameters.get('sequenceData',"no-sequenceData") )
    numPhotons = property(lambda self:int(self.parameters.get('NumPhotons',"-1")) )

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
        return "%60s %32s %32s %7d %10.4f %s " % (self.path, self.photonData, self.recordData, self.numPhotons, self.propagate, self.mode )


class Catdir(object):
    """
    Reads in metadata from dated folders corresponding to runs of::

          ggv 
          ggv --compute
          ggv --cfg4

    ::

        INFO:__main__:Catdir searching for date stamped folders beneath : /usr/local/env/opticks/rainbow 
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160104_141234
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160104_144620
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160104_150114
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160105_155311
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160107_201949
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160224_120049
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-6/20160105_172431
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-6/20160105_172538
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/5/20160103_165049
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/5/20160103_165114
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/5/20160104_111515

    """
    def __init__(self, cat):
        path = os.path.expandvars("$LOCAL_BASE/env/opticks/%s" % cat )
        df = DatedFolder()

        log.info("Catdir searching for date stamped folders beneath : %s " % path)
        metadata = {}
        for p in finddir(path, df):
             
            log.debug("%s", p)
            md = Metadata(p)
            tag = md.tag
            if tag not in metadata:
                metadata[tag] = []
            metadata[tag].append(md)

        self.metadata = metadata
        self.path = path

        self.dump()


    def dump(self):
        log.info("Catdir %s tags: %s beneath %s " % ( len(self.metadata), repr(self.metadata.keys()), self.path))
        for tag, mds in self.metadata.items():
            print "%5s : %d " % (tag, len(mds)) 

    def times(self, tag):
        """
        :param tag: event tag, eg 1,2 (be negated to get cfg4 tags)
        :return: recarray with propagate times and corresponding flags indicating cfg4/interop/compute
        """
        tags = [tag, "-%s" % tag]   # convention, negate to get equivalent cfg4 tag
        mds = []
        for tag in tags:
            mds.extend(self.metadata.get(tag, [])) 

        log.info("times metadata for tag %s " % tag + "\n".join(map(str,mds)))

        n = len(mds)
        numpho0 = mds[0].numPhotons

        ## collect indices with consistent photon counts
        consistent = []
        for i in range(n):
            md = mds[i]
            if md.numPhotons == numpho0:
                consistent.append(i)
            
        nc = len(consistent)
        if nc != n:
            log.warning("skipped metadata with inconsistent photon count n %s nc %s " % (n, nc)) 


        a = np.recarray((nc,), dtype=[("index", np.int32), ("time", "|O8"), ("propagate", np.float32), ("flgs", np.uint32 ), ("numPhotons",np.uint32)])

        j = 0 
        for i in range(n):
            md = mds[i]
            dat = (i, md.timestamp, md.propagate, md.flags, md.numPhotons )
            if i in consistent:
                a[j] = dat
                j += 1 
                print dat 
            else:
                log.warning("skipped inconsistent %s " % repr(dat))
            pass
        pass

        return a 

    def __repr__(self):
         return "%s tags %s " % (self.path, repr(self.metadata.keys()))










if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)    

    #cat, tag= "rainbow", "6"
    cat, tag= "PmtInBox", "4"

    catd = Catdir(cat)

    a = catd.times(tag)



