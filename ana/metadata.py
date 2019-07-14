#!/usr/bin/env python
"""
metadata.py
=============

Access the metadata json files written by Opticks runs, 
allowing evt digests and run times to be compared. 

See also meta.py a more generalized version of this, but 
not so fleshed out.

TODO: extract the good stuff from here as migrate from metadata.py to meta.py

::

    [blyth@localhost tmp]$ OPTICKS_EVENT_BASE=tboolean-box metadata.py --det tboolean-box --src torch --pfx .

"""

from datetime import datetime
import os, re, logging
import numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main
from opticks.ana.base import ini_, json_, splitlines_
from opticks.ana.datedfolder import DatedFolder, dateparser


class DeltaTime(object):
    """
    hmm reading from delta is brittle, as will change
    on updates of profiling points.  Better to read two times
    and calculate the delta : as that does not depend in 
    interposing profile points.
    """
    NAME = "DeltaTime.ini"  
    PROPAGATE_G4 = "CG4::propagate_0"
    PROPAGATE_OK = "OPropagator::launch_0"

    def __init__(self, base, tag):
        path = os.path.join(base, self.NAME)
        ini = ini_(path) 
        itag = int(tag)        
        key = self.PROPAGATE_OK if itag > 0 else self.PROPAGATE_G4
        propagate = ini.get(key,-1)         
        self.propagate = propagate
        self.ini = ini
    pass
pass      

class LaunchTimes(object):
    NAME = "OpticksEvent_launch.ini"  
    PROPAGATE_OK = "launch001"
    def __init__(self, base, tag):
        path = os.path.join(base, self.NAME)
        if not os.path.exists(path):
            propagate = -99.
            log.info("path %s does not exist " % path )
        else:  
            ini = ini_(path) 
            itag = int(tag)        
            propagate = ini.get(self.PROPAGATE_OK,-1) if itag > 0 else -1 
        pass
        self.propagate = propagate
        self.ini = ini
    pass
pass      


class CommandLine(object):
    def __init__(self, cmdline):
        self.cmdline = cmdline
    def has(self, opt):
        return 1 if self.cmdline.find(opt) > -1 else 0 
    def __repr__(self):
        return "\n".join(self.cmdline.split())  


ratio_ = lambda num,den:float(num)/float(den) if den != 0 else -1 

class CompareMetadata(object):

    permit_mismatch = ["cmdline", "mode"]

    def __init__(self, am, bm):
        self.am = am 
        self.bm = bm 

        self.GEOCACHE = self.expected_common("GEOCACHE", parameter=False)
        self.numPhotons = self.expected_common("numPhotons", parameter=False)
        self.mode =  self.expected_common("mode", parameter=False)

        self.TestCSGPath =  self.expected_common("TestCSGPath", parameter=False)
        self.csgmeta0 = self.expected_common("csgmeta0", parameter=False)  # container metadata, usually an emitter 

        cmdline = self.expected_common( "cmdline", parameter=True) 
        self.cmdline = CommandLine(cmdline)
        self.Switches = self.expected_common( "Switches", parameter=True) 
        self.align = self.cmdline.has("--align ")
        self.reflectcheat = self.cmdline.has("--reflectcheat ")
        self.factor = ratio_(bm.propagate0, am.propagate0)

    

    def _get_crucial(self):
        """
        Amplify some of the crucial options for comparisons
        """
        return " ".join([ 
            "ALIGN" if self.align==1 else "non-align" , 
            "REFLECTCHEAT" if self.reflectcheat==1 else "non-reflectcheat" ,
            ]) 
    crucial = property(_get_crucial)

    def expected_common(self, key, parameter=True):
        if parameter:
            av = self.am.parameters.get(key,"") 
            bv = self.bm.parameters.get(key,"")
        else:
            av = getattr( self.am, key ) 
            bv = getattr( self.bm, key ) 
        pass     

        match = av == bv  

        if not match:
            if key in self.permit_mismatch:
                log.warning("note permitted expected_common mismatch for key %s " % key )
                log.info(av)
                log.info(bv)
            else: 
                log.fatal("expected_common mismatch for key %s " % key )
                log.fatal(av)
                log.fatal(bv)
                assert match, key           
            pass
        return av 

    def __repr__(self):
        return "\n".join([
              "ab.cfm",
              self.brief(),
              "ab.a.metadata:%s" % self.am,
              "ab.b.metadata:%s" % self.bm,
              self.Switches,
              str(self.csgmeta0),
              #repr(self.cmdline),
              "."
                        ])
 
    def brief(self):
        return "nph:%8d A:%10.4f B:%10.4f B/A:%10.1f %s %s " % (self.numPhotons, self.am.propagate0, self.bm.propagate0, self.factor, self.mode, self.crucial )  


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

    def __init__(self, path, base=None):
        """
        Path assumed to be a directory with one of two forms::

              $TMP/evt/PmtInBox/torch/10/                ## ending with tag
              $TMP/evt/PmtInBox/torch/10/20160817_141731 ## ending with datefold
            
        In both cases the directory must contain::

              parameters.json
              DeltaTime.ini
 
        """
        if base is not None:
            path = os.path.join(base, path)
        pass 
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

        self.delta_times = DeltaTime(self.path, tag)
        self.launch_times = LaunchTimes(self.path, tag)
        self.propagate0 = float(self.delta_times.propagate) 
        self.propagate = float(self.launch_times.propagate) 

        self._solids = None
 

    # parameter accessors (from the json)
    mode = property(lambda self:self.parameters.get('mode',"no-mode") )  # eg COMPUTE_MODE
    photonData = property(lambda self:self.parameters.get('photonData',"no-photonData") ) # digest
    recordData = property(lambda self:self.parameters.get('recordData',"no-recordData") )  # digest
    sequenceData = property(lambda self:self.parameters.get('sequenceData',"no-sequenceData") )  # digest
    numPhotons = property(lambda self:int(self.parameters.get('NumPhotons',"-1")) ) 
    TestCSGPath = property(lambda self:self.parameters.get('TestCSGPath',None) )  # eg tboolean-box
    GEOCACHE = property(lambda self:self.parameters.get('GEOCACHE',None) ) 
    Note = property(lambda self:self.parameters.get('Note',"") )

    def _flags(self):
        flgs = 0 
        if self.mode.lower().startswith("compute"):
            flgs |= self.COMPUTE 
        elif self.mode.lower().startswith("interop"):
            flgs |= self.INTEROP 
        elif self.mode.lower().startswith("cfg4"):
            flgs |= self.CFG4 
        pass
        return flgs 
    flags = property(_flags)

    def __repr__(self):
        return "%-60s ox:%32s rx:%32s np:%7d pr:%10.4f %s" % (self.path, self.photonData, self.recordData, self.numPhotons, self.propagate0, self.mode )


    def _get_csgbnd(self):
        """
        csg.txt contains a list of boundaries, one per line::

            [blyth@localhost tboolean-box]$ cat csg.txt 
            Rock//perfectAbsorbSurface/Vacuum
            Vacuum///GlassSchottF2

        * observed no newline at end
        """
        if self.TestCSGPath is None:
            return []
        pass 
        csgtxt = os.path.join(self.TestCSGPath, "csg.txt")    
        csgbnd = splitlines_(csgtxt) if os.path.exists(csgtxt) else []
        return csgbnd
    csgbnd = property(_get_csgbnd)
  

    def _get_csgmeta0(self):
        """
        Metadata from the outer CSG solid, typically the container::

            [blyth@localhost 0]$ js.py meta.json 
            {u'container': 1,
             u'containerscale': 3.0,
             u'ctrl': 0,
             u'emit': -1,
             u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55',
             u'poly': u'IM',
             u'resolution': u'20',
             u'verbosity': u'0'}

        """ 
        if self.TestCSGPath is None:
            return None
        pass 
        csgmeta0_ = os.path.join(self.TestCSGPath, "0", "meta.json")
        csgmeta0 = json_(csgmeta0_) if os.path.exists(csgmeta0_) else []
        return csgmeta0
    csgmeta0 = property(_get_csgmeta0)


    def _get_lv(self):
        path = self.TestCSGPath
        if path is None or len(path) == 0:
            lv = None
        else:  
            name = os.path.basename(path)
            lv = name.split("-")[-1]
        return lv
    lv = property(_get_lv)

    def _get_solid(self):
        lv = self.lv
        if lv is None: return None

        try:
            ilv = int(lv)
            solid = self.solids[ilv]
        except ValueError:  
            solid = lv
        pass
        return solid
    solid = property(_get_solid)

    def _get_solids(self):
        """
        List of solids (aka meshes) obtained from "GItemList/GMeshLib.txt" within the basis geocache 
        """
        if self._solids is None:  
            path = os.path.join(self.GEOCACHE, "GItemList/GMeshLib.txt")
            self._solids = splitlines_(path)
        return self._solids
    solids = property(_get_solids) 
       

    def dump(self):
        for k,v in self.parameters.items():
            print "%20s : %s " % (k, v)
        for k,v in self.delta_times.ini.items():
            print "%20s : %s " % (k, v)
        for k,v in self.launch_times.ini.items():
            print "%20s : %s " % (k, v)
      



def test_metadata():
    td = tagdir_("PmtInBox", "torch", "10")
    md = Metadata(td)


def test_tagdir():
    td = os.path.expandvars("/tmp/$USER/opticks/evt/boolean/torch/1")
    md = Metadata(td)
    print md



if __name__ == '__main__':
    ok = opticks_main() 
    print("ok.brief : %s " % ok.brief)
    print("ok.tagdir : %s " % ok.tagdir)

    md = Metadata(ok.tagdir)
    print("md : %s " % md)

if 0:
    md.dump()
    csgpath = md.TestCSGPath
    print("csgpath", csgpath)
    print("csgbnd", md.csgbnd)
  










  
