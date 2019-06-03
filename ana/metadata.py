#!/usr/bin/env python
"""
Metadata
~~~~~~~~~~

Access the metadata json files written by Opticks runs, 
allowing evt digests and run times to be compared. 

See also meta.py a more generalized version of this, but 
not so fleshed out.

TODO: extract the good stuff from here as migrate from metadata.py to meta.py


"""

from datetime import datetime
import os, re, logging
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.base import ini_, json_, splitlines_
from opticks.ana.datedfolder import DatedFolder, dateparser
from opticks.ana.nload import tagdir_



class Metadata(object):
    """
    v2 layout::

        simon:ana blyth$ l $TMP/evt/PmtInBox/torch/10/
        total 55600
        drwxr-xr-x  6 blyth  wheel       204 Aug 19 15:32 20160819_153245
        -rw-r--r--  1 blyth  wheel       100 Aug 19 15:32 Boundary_IndexLocal.json
        -rw-r--r--  1 blyth  wheel       111 Aug 19 15:32 Boundary_IndexSource.json
        ...
        -rw-r--r--  1 blyth  wheel   6400080 Aug 19 15:32 ox.npy
        -rw-r--r--  1 blyth  wheel      1069 Aug 19 15:32 parameters.json
        -rw-r--r--  1 blyth  wheel   1600080 Aug 19 15:32 ph.npy
        -rw-r--r--  1 blyth  wheel    400080 Aug 19 15:32 ps.npy
        -rw-r--r--  1 blyth  wheel      2219 Aug 19 15:32 report.txt
        -rw-r--r--  1 blyth  wheel   4000096 Aug 19 15:32 rs.npy
        -rw-r--r--  1 blyth  wheel  16000096 Aug 19 15:32 rx.npy
        -rw-r--r--  1 blyth  wheel       763 Aug 19 15:32 t_absolute.ini
        -rw-r--r--  1 blyth  wheel       817 Aug 19 15:32 t_delta.ini
        drwxr-xr-x  6 blyth  wheel       204 Aug 18 20:53 20160818_205342

    timestamp folders contain just metadata for prior runs not full evt::

        simon:ana blyth$ l /tmp/blyth/opticks/evt/PmtInBox/torch/10/20160819_153245/
        total 32
        -rw-r--r--  1 blyth  wheel  1069 Aug 19 15:32 parameters.json
        -rw-r--r--  1 blyth  wheel  2219 Aug 19 15:32 report.txt
        -rw-r--r--  1 blyth  wheel   763 Aug 19 15:32 t_absolute.ini
        -rw-r--r--  1 blyth  wheel   817 Aug 19 15:32 t_delta.ini


    """

    COMPUTE = 0x1 << 1
    INTEROP = 0x1 << 2 
    CFG4    = 0x1 << 3 

    date_ptn = re.compile("\d{8}_\d{6}")  # eg 20160817_141731

    def __init__(self, path):
        """
        Path assumed to be a directory with one of two forms::

              $TMP/evt/PmtInBox/torch/10/                ## ending with tag
              $TMP/evt/PmtInBox/torch/10/20160817_141731 ## ending with datefold
            
        In both cases the directory must contain::

              parameters.json
              t_delta.ini 
 
        """
        self.path = path
        basename = os.path.basename(path)
        if self.date_ptn.match(basename):
             datefold = basename
             timestamp = dateparser(datefold)
             tag = os.path.basename(os.path.dirname(path))        
        else:
             datefold = None
             timestamp = None
             tag = basename 
        pass
        self.datefold = datefold
        self.tag = tag 
        self.timestamp = timestamp 

        self.parameters = json_(os.path.join(self.path, "parameters.json"))
        self.times = ini_(os.path.join(self.path, "t_delta.ini"))

    propagate = property(lambda self:float(self.times.get('propagate','-1')))

    # parameter accessors (from the json)
    mode = property(lambda self:self.parameters.get('mode',"no-mode") )
    photonData = property(lambda self:self.parameters.get('photonData',"no-photonData") )
    recordData = property(lambda self:self.parameters.get('recordData',"no-recordData") )
    sequenceData = property(lambda self:self.parameters.get('sequenceData',"no-sequenceData") )
    numPhotons = property(lambda self:int(self.parameters.get('NumPhotons',"-1")) )
    TestCSGPath = property(lambda self:self.parameters.get('TestCSGPath',None) )
    Note = property(lambda self:self.parameters.get('Note',"") )

    def _flags(self):
        flgs = 0 
        if self.mode.lower().startswith("compute"):
            flgs |= self.COMPUTE 
        elif self.mode.lower().startswith("interop"):
            flgs |= self.INTEROP 
        elif self.mode.lower().startswith("cfg4"):
            flgs |= self.CFG4 

        return flgs 
    flags = property(_flags)

    def __repr__(self):
        return "%60s %32s %32s %7d %10.4f %s " % (self.path, self.photonData, self.recordData, self.numPhotons, self.propagate, self.mode )




    def _get_csgbnd(self):
        csgtxt = os.path.join(self.TestCSGPath, "csg.txt")
        csgbnd = splitlines_(csgtxt) if os.path.exists(csgtxt) else []
        return csgbnd
    csgbnd = property(_get_csgbnd)
  

    def _get_csgmeta0(self):
        csgmeta0_ = os.path.join(self.TestCSGPath, "0", "meta.json")
        csgmeta0 = json_(csgmeta0_) if os.path.exists(csgmeta0_) else []
        return csgmeta0
    csgmeta0 = property(_get_csgmeta0)
       

    def dump(self):
        for k,v in self.parameters.items():
            print "%20s : %s " % (k, v)
        for k,v in self.times.items():
            print "%20s : %s " % (k, v)
     


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
    def __init__(self, path):
        """
        Path should be the folder above the tagdir in order
        to allow comparisons between equivalent (ie G4 negated) tags 
        """
        dirs, dfolds = DatedFolder.find(path)

        log.info("Catdir searching for date stamped folders beneath : %s " % path)
        metamap = {}
        for p in dirs:
             
            log.debug("%s", p)
            md = Metadata(p)
            tag = md.tag
            if tag not in metamap:
                metamap[tag] = []
            metamap[tag].append(md)

        self.metamap = metamap
        self.path = path

        self.dump()


    def dump(self):
        log.info("Catdir %s tags: %s beneath %s " % ( len(self.metamap), repr(self.metamap.keys()), self.path))
        for tag, mds in self.metamap.items():
            print "%5s : %d " % (tag, len(mds)) 

    def tags(self): 
        return self.metamap.keys()

    def times(self, tag):
        """
        :param tag: event tag, eg 1,2 (be negated to get cfg4 tags)
        :return: recarray with propagate times and corresponding flags indicating cfg4/interop/compute
        """
        tags = [tag, "-%s" % tag]   # convention, negate to get equivalent cfg4 tag
        mds = []
        for tag in tags:
            mds.extend(self.metamap.get(tag, [])) 

        log.info("times metadata for tag %s " % tag + "\n".join(map(str,mds)))

        n = len(mds)
        if n == 0:
            log.warning("no metadata found")
            return None  

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
         return "%s tags %s " % (self.path, repr(self.tags()))





def test_catdir():
    #cat, tag= "rainbow", "6"
    cat, tag= "PmtInBox", "4"
    catd = Catdir(cat)
    a = catd.times(tag)

def test_catdir_find():
    cat = Catdir(os.path.expandvars("/tmp/$USER/opticks/evt/PmtInBox/torch"))
    tags = cat.tags()
    for tag in tags:
        a = cat.times(tag)
        print "a", a



def test_metadata():
    from opticks.ana.nload import tagdir_
    td = tagdir_("PmtInBox", "torch", "10")
    md = Metadata(td)


def test_tagdir():
    td = os.path.expandvars("/tmp/$USER/opticks/evt/boolean/torch/1")
    md = Metadata(td)
    print md



if __name__ == '__main__':
    ok = opticks_main() 
    #print ok

    det = ok.det
    src = ok.src
    tag = ok.tag

    td = tagdir_(det, src, tag )
    print "td", td

    md = Metadata(td)
    print "md", md

    md.dump()

    csgpath = md.TestCSGPath
    print "csgpath", csgpath

    print "csgbnd", md.csgbnd
  
