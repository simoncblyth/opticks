#!/usr/bin/env python
"""

::

    gs.numPhotons[gs.numPhotons > 1]

    In [13]: np.count_nonzero(gs.numPhotons == 1)   # reemission gensteps, need to be excluded 
    Out[13]: 12275

    In [14]: np.count_nonzero(gs.numPhotons > 1)
    Out[14]: 112


"""
import os, numpy as np, argparse
import logging
log = logging.getLogger(__name__)
from opticks.ana.OpticksGenstepEnum import OpticksGenstepEnum
from opticks.ana.PDGCodeEnum import PDGCodeEnum
from opticks.ana.nbase import count_unique_sorted
X,Y,Z,T = 0,1,2,3


def is_int(s):
    s = str(s)
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()    

class GS(object):
    etyp = OpticksGenstepEnum()
    epdg = PDGCodeEnum()

    @classmethod
    def parse_args(cls, doc, **kwa):
        np.set_printoptions(suppress=True, precision=3 )
        parser = argparse.ArgumentParser(doc)
        parser.add_argument(     "--pathtmpl",  help="Path template of genstep file", default=kwa.get("pathtmpl",None) )
        parser.add_argument(     "paths", nargs="*",  help="Paths of genstep files" )
        parser.add_argument(     "--level", default="info", help="logging level" ) 
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        return args  

    def __init__(self, path, pathtmpl):
        if is_int(path):
            path = pathtmpl % int(path)       
        pass
        f = np.load(os.path.expandvars(path))
        i = f.view(np.int32)
 
        self.path = path 
        self.f = f
        self.i = i

        log.info(" path %s shape %s " % (self.path, str(self.f.shape)))
        self.check_counts()
        self.check_pdgcode()
        self.check_ranges()

    
    ID = property(lambda self:self.i[:,0,0])  # gencode: the OpticksGenstep enum value
    PID = property(lambda self:self.i[:,0,1]) # parentID
    numPhotons = property(lambda self:self.i[:,0,3]) 

    xyzt = property(lambda self:self.f[:,1])

    deltaPositionLength = property(lambda self:self.f[:,2])

    pdgCode = property(lambda self:self.i[:,3,0])
    charge = property(lambda self:self.f[:,3,1])
    weight = property(lambda self:self.f[:,3,2])
    meanVelocity = property(lambda self:self.f[:,3,3])

    def check_counts(self):
        log.info("check_counts")
        i = self.i 

        num_gensteps = len(i)
        num_photons = self.numPhotons.sum()

        print("num_gensteps : %d " % num_gensteps )
        print("num_photons  : %d " % num_photons )

        cu = count_unique_sorted(self.ID)

        nph_tot = 0 
        ngs_tot = 0 
        fmt = " (%d)%-25s : ngs:%5d  npho:%5d "
        for typ,ngs in cu:
            sel = i[:,0,0] == typ
            nph = i[sel][:,0,3].sum()
            nph_tot += nph
            ngs_tot += ngs 
            print(fmt % (typ,self.etyp(typ),ngs,nph )) 
        pass
        print(fmt % (0, "TOTALS", ngs_tot, nph_tot))     

    def check_pdgcode(self):
        log.info("check_pdgcode")
        #cu = count_unique_sorted(self.pdgCode)  doesnt work with -ve coded
        #print(cu)
        pdgcodes, counts = np.unique(self.pdgCode, return_counts=True) # needs numpy > 1.9
        assert len(pdgcodes) == len(counts)

        fmt = " %7d : %10s : %d "  
        for i in range(len(pdgcodes)):
            pdgcode = pdgcodes[i]
            count = counts[i]
            print(fmt % (pdgcode,self.epdg(pdgcode),count )) 
        pass

    def check_ranges(self):
        log.info("check_ranges")
        f = self.f

        xyzt = f[:,1]
        x,y,z,t = xyzt[:,X], xyzt[:,Y], xyzt[:,Z], xyzt[:,T]

        tr = (t.min(), t.max())
        xr = (x.min(), x.max())
        yr = (y.min(), y.max())
        zr = (z.min(), z.max())

        print(" tr %10.4f %10.4f " % tr )
        print(" xr %10.4f %10.4f " % xr )
        print(" yr %10.4f %10.4f " % yr )
        print(" zr %10.4f %10.4f " % zr )



if __name__ == '__main__':
    pathtmpl = "$TMP/source/evt/g4live/natural/%d/gs.npy"
    args = GS.parse_args(__doc__, pathtmpl=pathtmpl)
    log.info(args)
    for path in args.paths:
        gs = GS(path, args.pathtmpl)
    pass



