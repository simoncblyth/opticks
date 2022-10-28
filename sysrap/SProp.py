#!/usr/bin/env python
"""
SProp.py 
==========


"""
import os, logging, numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

class SProp(object):
    BASE = os.environ.get("NP_PROP_BASE", "/tmp")

    @classmethod
    def Resolve_(cls, spec):
        relp = spec.replace(".","/")
        return os.path.join(cls.BASE, relp) 

    @classmethod
    def Resolve(cls, spec):
        return cls.Resolve_(spec) if spec.count(".") > 1 else spec  

    @classmethod
    def Names(cls, spec):
        rawnames = os.listdir(cls.Resolve(spec))
        names = []
        for name in rawnames: 
            if name.count(".") == 0:
                names.append(name)
            pass
        pass
        return names  

    @classmethod
    def Load(cls, spec):
        path = cls.Resolve(spec)
        log.debug(" spec %s path %s " % (spec, path))
        try:
            a = np.loadtxt(path, usecols=(0,2)) # skip "*eV" column 
        except (ValueError, IndexError):
            a = np.loadtxt(path, dtype=np.object)    
        pass
        if os.path.exists(path+".npy"):
            b = np.load(path+".npy")
            assert np.all( a == b )
            log.debug(" path %s matches %s " % (path, path+".npy"))
        pass 
        return a 

    def __init__(self, specbase="PMTProperty.R12860.", symbol="hama"):
        props = odict()
        specs = []
        paths = []
        for name in self.Names(specbase):
            spec = "%s%s" % ( specbase, name) 
            path = self.Resolve(spec)
            props[name] = self.Load(path)
            specs.append(spec)
            paths.append(path)
        pass
        self.props = props
        self.specs = specs
        self.paths = paths 
        self.specbase = specbase
        self.symbol = symbol

    def __repr__(self):
        lines = []
        lines.append("Prop %s %s " % (self.specbase, self.symbol) )
        for k, v in self.props.items():
            lines.append("%35s : %10s : %10s " % ("%s.%s" % (self.symbol, k), str(v.shape), str(v.dtype) ))
        pass
        return "\n".join(lines)

    def __getattr__(self, spec):
        if spec in self.props:
            return self.props[spec]
        else:
            raise AttributeError  
        pass
 
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    hama = SProp("PMTProperty.R12860.", symbol="hama")
    nnvt = SProp("PMTProperty.NNVTMCP.", symbol="nnvt")
    nnvtq = SProp("PMTProperty.NNVTMCP_HiQE.", symbol="nnvtq")
 
    print(repr(hama))
    print(repr(nnvt))
    print(repr(nnvtq))

    for symbol in ["hama", "nnvt", "nnvtq"]:
        expr = "np.c_[%(symbol)s.PHC_RINDEX, %(symbol)s.PHC_KINDEX, %(symbol)s.ARC_RINDEX ] " % locals()   
        print(expr)
        print(eval(expr))
    pass
    # energy domain consistent within but not between them 

