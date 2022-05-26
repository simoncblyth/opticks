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



def make_colors():
    """
    :return colors: large list of color names with easily recognisable ones first 
    """
    #colors = ["red","green","blue","cyan","magenta","yellow","pink","purple"]
    all_colors = list(hexcolors.keys()) if not hexcolors is None else []
    easy_colors = "red green blue cyan magenta yellow pink".split()
    skip_colors = "bisque beige white aliceblue antiquewhite".split()    # skip colors that look too alike 

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
        uval = self.ouval[idesc] 
        count = self.ocount[idesc] 
        isel = self.isel  

        if fname[0] == "_":
            fname = fname[1:]
        pass
        #label = "%s:%s" % (idesc, fname)
        label = "%s" % (fname)
        label = label.replace("solid","s")
        color = COLORS[idesc % len(COLORS)]  # gives the more frequent boundary the easy_color names 
        msg = " %2d : %4d : %6d : %20s : %40s : %s " % (idesc, uval, count, color, fname, label )
        selector = self.val == uval

        if len(isel) == 0:
            skip = False
        else:
            skip = idesc not in isel
        pass 
        return uval, selector, label, color, skip, msg 

    def __str__(self):
        lines = []
        lines.append(self.desc)  
        for idesc in range(self.unum):
            uval, selector, label, color, skip, msg = self(idesc)
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


class PhotonFeatures(object):
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
        :param pos: Positions instance

        Although at first sight it looks like could use photons array 
        argument rather than Positions instance, that is not the case when masks are applied. 
        The pos.p is changed by the mask as is needed such that feature 
        selectors can be used within the masked arrays. 
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

        log.info("[ Photons.bndfeat ")
        bnd_namedict = {} if cf is None else cf.sim.bndnamedict 
        bndfeat = Feature("bnd", bnd, bnd_namedict)
        log.info("] Photons.bndfeat ")

        log.info("[ Photons.pidfeat ")
        pid_namedict = {} if cf is None else cf.primIdx_meshname_dict()
        log.info(" pid_namedict: %d  " % len(pid_namedict))
        pidfeat = Feature("pid", pid, pid_namedict)
        log.info("] Photons.pidfeat ")

        log.info("[ Photons.insfeat ")
        ins_namedict = {} if cf is None else cf.insnamedict
        log.info(" ins_namedict: %d  " % len(ins_namedict))
        insfeat = Feature("ins", ins, ins_namedict)
        log.info("] Photons.insfeat ")

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

