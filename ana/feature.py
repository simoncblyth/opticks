#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

PV = not "NOPV" in os.environ 
if PV:
    try:
        from pyvista.plotting.colors import hexcolors  
    except ImportError:
        hexcolors = None
    pass
else:
    hexcolors = None
pass

def shorten_surfname(name=None, cut=20):
    if name is None: name="HamamatsuR12860_PMT_20inch_photocathode_logsurf2"
    name = name[:cut//2] + "..." + name[-cut//2:]
    return name 

def shorten_bndname(name=None, cut=20):
    if name is None: name = "Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum"
    elem = name.split("/")
    assert len(elem) == 4 
    if len(elem[1]) > cut: elem[1] = shorten_surfname(elem[1], cut) 
    if len(elem[2]) > cut: elem[2] = shorten_surfname(elem[2], cut) 
    return "/".join(elem)


def make_colors():
    """
    :return colors: large list of color names with easily recognisable ones first 
    """
    #colors = ["red","green","blue","cyan","magenta","yellow","pink","purple"]
    all_colors = list(hexcolors.keys()) if not hexcolors is None else []
    easy_colors = "red green blue cyan magenta yellow pink".split()
    skip_colors = "bisque beige white aliceblue antiquewhite aqua".split()    # skip colors that look too alike 

    colors = easy_colors 
    for c in all_colors:
        if c in skip_colors: 
            continue
        if not c in colors: 
            colors.append(c) 
        pass
    pass
    return colors

COLORS = make_colors()


class Feature(object):
    """
    Trying to generalize feature handling 
    """
    def __init__(self, name, val, vname={}):
        """
        :param name: string eg "bnd" or "primIdx"
        :param val: large array of integer feature values 
        :param namedict: dict relating feature integers to string names 

        The is an implicit assumption that the number of unique feature values is not enormous,
        for example boundary values or prim identity values.
        """


        uval, ucount = np.unique(val, return_counts=True)

        if len(vname) == 0:
            nn = ["%s%d" % (name,i) for i in uval]
            vname = dict(zip(uval,nn)) 
        pass
        pass 
        idxdesc = np.argsort(ucount)[::-1]  
        # indices of uval and ucount that reorder those arrays into descending count order

        ocount = [ucount[j]       for j in idxdesc]
        ouval  = [uval[j]         for j in idxdesc]

        # vname needs absolutes to get the names 
        onames = [vname[uval[j]]  for j in idxdesc]

        self.is_bnd = name == "bnd"
        self.name = name
        self.val = val
        self.vname = vname

        self.uval = uval
        self.unum = len(uval) 
        self.ucount = ucount
        self.idxdesc = idxdesc
        self.onames = onames
        self.ocount = ocount
        self.ouval = ouval

        ISEL = os.environ.get("ISEL","")  
        isel = self.parse_ISEL(ISEL, onames) 
        sisel = ",".join(map(str, isel))

        print( "Feature name %s ISEL: [%s] isel: [%s] sisel [%s] " % (name, ISEL, str(isel), sisel))

        self.isel = isel 
        self.sisel = sisel 

    @classmethod
    def parse_ISEL(cls, ISEL, onames):
        """ 
        :param ISEL: comma delimited list of strings or integers 
        :param onames: names ordered in descending frequency order
        :return isels: list of frequency order indices 

        Integers in the ISEL are interpreted as frequency order indices. 

        Strings are interpreted as fragments to look for in the ordered names,
        (which could be boundary names or prim names for example) 
        eg use Hama or NNVT to yield the list of frequency order indices 
        with corresponding names containing those strings. 
        """
        ISELS = list(filter(None,ISEL.split(",")))
        isels = []
        for i in ISELS:
            if i.isnumeric(): 
                isels.append(int(i))
            else:
                for idesc, nam in enumerate(onames):
                    if i in nam: 
                        isels.append(idesc)
                    pass
                pass
            pass
        pass    
        return isels 

    def __call__(self, idesc):
        """
        :param idesc: zero based index less than unum

        for frame photons, empty pixels give zero : so not including 0 in ISEL allows to skip
        if uval==0 and not 0 in isel: continue 

        """
        assert idesc > -1 and idesc < self.unum
        fname = self.onames[idesc]
        if self.is_bnd: fname = shorten_bndname(fname)

        uval = self.ouval[idesc] 
        count = self.ocount[idesc] 
        isel = self.isel  

        if fname[0] == "_":
            fname = fname[1:]
        pass
        color = COLORS[idesc % len(COLORS)]  # gives the more frequent boundary the easy_color names 
        msg = " %2d : %5d : %6d : %20s : %80s " % (idesc, uval, count, color, fname  )
        selector = self.val == uval

        if len(isel) == 0:
            skip = False
        else:
            skip = idesc not in isel
        pass 
        return uval, selector, fname, color, skip, msg 

    def __str__(self):
        lines = []
        lines.append(self.desc)  
        for idesc in range(self.unum):
            uval, selector, fname, color, skip, msg = self(idesc)
            lines.append(msg)
        pass
        return "\n".join(lines)

    desc = property(lambda self:"ph.%sfeat : %s " % (self.name, str(self.val.shape)))

    def __repr__(self):
        return "\n".join([
            "Feature name %s val %s" % (self.name, str(self.val.shape)),
            "uval %s " % str(self.uval),
            "ucount %s " % str(self.ucount),
            "idxdesc %s " % str(self.idxdesc),
            "onames %s " % " ".join(self.onames),
            "ocount %s " % str(self.ocount),
            "ouval %s " % " ".join(map(str,self.ouval)),
            ])


class SimtraceFeatures(object):
    """
    feat contriols how to select positions, eg  via boundary or identity 
    allow plotting of subsets with different colors
    """
    @classmethod
    def SubMock(cls, i, num):
        p = np.zeros([num, 4, 4], dtype=np.float32)  
        offset = i*100
        for j in range(10):
            for k in range(10): 
                idx = j*10+k
                if idx < num:
                    p[idx,0,0] = float(offset+j*10)
                    p[idx,0,1] = 0
                    p[idx,0,2] = float(offset+k*10)
                    p.view(np.int32)[idx,3,3] = i << 16
                pass
            pass
        pass
        return p

    @classmethod
    def Mock(cls):
        """
        Random number of items between 50 and 100 for each of 10 categories 
        """
        aa = []
        for i in range(10):
            aa.append(cls.SubMock(i, np.random.randint(0,100)))
        pass
        return np.concatenate(tuple(aa))

    def __init__(self, pos, cf=None, featname="pid", do_mok=False ):
        """
        :param pos: SimtracePositions instance

        Although at first sight it looks like could use photons array 
        argument rather than Positions instance, that is not the case when masks are applied. 
        The pos.p is changed by the mask as is needed such that feature 
        selectors can be used within the masked arrays. 
          
        HMM: identity access is only fully applicable to simtrace, not photons

        * ACTUALLY THE SAME INFO IS PRESENT IN PHOTON ARRAYS BUT IN DIFFERENT POSITIONS
        * TODO: accomodate the photon layout as well as the simtrace one by using 
          some common method names with different imps for SimtracePositions and PhotonPositions

          * OR: standardize the flag/identity layout between photons and simtrace ?
        

        bnd = p[:,2,3].view(np.int32)
        ids = p[:,3,3].view(np.int32)
        pid = ids >> 16          # prim_idx
        ins = ids & 0xffff       # instance_id  

             +  +  +  +
             +  +  +  +
             +  +  + bnd
             +  +  + ids

        qudarap/qevent.h::

            232 QEVENT_METHOD void qevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
            233 {
            234     float t = prd->distance() ;
            235     quad4 a ;
            236 
            237     a.q0.f  = prd->q0.f ;
            238 
            239     a.q1.f.x = p.q0.f.x + t*p.q1.f.x ;
            240     a.q1.f.y = p.q0.f.y + t*p.q1.f.y ;
            241     a.q1.f.z = p.q0.f.z + t*p.q1.f.z ;
            242     a.q1.i.w = 0.f ;
            243 
            244     a.q2.f.x = p.q0.f.x ;
            245     a.q2.f.y = p.q0.f.y ;
            246     a.q2.f.z = p.q0.f.z ;
            247     a.q2.u.w = prd->boundary() ; // used by ana/feature.py from CSGOpitXSimtraceTest,py 
            248 
            249     a.q3.f.x = p.q1.f.x ;
            250     a.q3.f.y = p.q1.f.y ;
            251     a.q3.f.z = p.q1.f.z ;
            252     a.q3.u.w = prd->identity() ;  // identity from __closesthit__ch (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) 
            253 
            254     simtrace[idx] = a ;
            255 }

        TODO: try to pass along the gas_idx in the prd ?  
        HMM: but maybe that would not distinguish between HighQE and ordinary probably ?
        suggests will need to do instance id lookup

        """
        p = pos.p
            
        log.info("[Photons p.ndim %d p.shape %s " % (int(p.ndim), str(p.shape)) )
        assert featname in ["pid", "bnd", "ins", "mok"]
        if p.ndim == 3:
            bnd = p[:,2,3].view(np.int32)
            ids = p[:,3,3].view(np.int32) 
        elif p.ndim == 4:
            bnd = p.view(np.int32)[:,:,2,3]
            ids = p.view(np.int32)[:,:,3,3] 
        else:
            log.info("unexpected p.shape %s " % str(p.shape))
        pass
        pid = ids >> 16
        ins = ids & 0xffff   # ridx?    

        log.debug("[ Photons.bndfeat ")
        bnd_namedict = {} if cf is None else cf.sim.bndnamedict 
        bndfeat = Feature("bnd", bnd, bnd_namedict)
        log.debug("] Photons.bndfeat ")

        log.debug("[ Photons.pidfeat ")
        pid_namedict = {} if cf is None else cf.primIdx_meshname_dict
        log.info(" pid_namedict: %d  " % len(pid_namedict))
        pidfeat = Feature("pid", pid, pid_namedict)
        log.debug("] Photons.pidfeat ")

        log.debug("[ Photons.insfeat ")
        ins_namedict = {} if cf is None else cf.insnamedict
        log.info(" ins_namedict: %d  " % len(ins_namedict))
        insfeat = Feature("ins", ins, ins_namedict)
        log.debug("] Photons.insfeat ")

        if do_mok:
            log.info("[ Photons.mokfeat ")
            mok_namedict = {} if cf is None else cf.moknamedict 
            mokfeat = Feature("mok", pid, mok_namedict)
            log.info("] Photons.mokfeat ")
        else: 
            mokfeat = None
        pass

        if featname=="pid":
            feat = pidfeat
        elif featname == "bnd":
            feat = bndfeat
        elif featname == "ins":
            feat = insfeat
        elif featname == "mok":
            feat = mokfeat
        else:
            feat = None
        pass

        self.cf = cf
        self.p = p 
        self.bnd = bnd
        self.ids = ids
        self.bndfeat = bndfeat
        self.pidfeat = pidfeat
        self.insfeat = insfeat
        self.mokfeat = mokfeat
        self.feat = feat
        log.info("]Photons")

        print(bndfeat)
        print(pidfeat)
        print(insfeat)


    def __repr__(self):
        return "\n".join([
               "p %s" % str(self.p.shape), 
               ])


def test_mok(cf):
    mock_photons = PhotonFeatures.Mock()
    ph = PhotonFeatures(mock_photons, cf, featname="mok", do_mok=True)
    print(ph.mokfeat)


if __name__ == '__main__':
    from opticks.CSG.CSGFoundry import CSGFoundry 
    cf = CSGFoundry.Load() 
    test_mok(cf)

