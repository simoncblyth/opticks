#!/bin/env python
import os, logging
log = logging.getLogger(__name__)
import numpy as np
from base import ini_

#DEFAULT_BASE = "$LOCAL_BASE/env/opticks"
DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"

DEFAULT_DIR_TEMPLATE = DEFAULT_BASE + "/$1/$2"  ## cf C++ NPYBase::

def path_(typ, tag, det="dayabay", name=None):
    tmpl = os.path.expandvars(DEFAULT_DIR_TEMPLATE.replace("$1", det).replace("$2",typ)) 
    if name is None:
        tmpl = os.path.join(tmpl, "%s.npy") 
    else:
        tmpl = os.path.join(tmpl, "%s", name) 
    pass
    return tmpl % str(tag)


def tpaths_(typ, tag, det="dayabay", name=None):
    assert name is not None 
    tmpl = os.path.expandvars(DEFAULT_DIR_TEMPLATE.replace("$1", det).replace("$2",typ)) 
    tagdir = os.path.join(tmpl, "%s") % str(tag) 
    names = os.listdir(tagdir)
    tdirs = filter( os.path.isdir, map(lambda name:os.path.join(tagdir, name), names))
    tdirs = sorted(tdirs, reverse=True)
    tnams = filter( os.path.exists, map(lambda tdir:os.path.join(tdir, name), tdirs))
    return tnams

 



class A(np.ndarray):
    @classmethod
    def load_(cls, typ, tag, det="dayabay", dbg=False):
        path = path_(typ,tag, det)
        a = None
        if os.path.exists(path):
            log.debug("loading %s " % path )
            if dbg: 
                os.system("ls -l %s " % path)
            arr = np.load(path)
            a = arr.view(cls)
            a.path = path 
            a.typ = typ
            a.tag = tag
            a.det = det 
        else:
            log.warning("cannot load %s " % path)
        pass
        return a

    def __repr__(self):
        if hasattr(self, 'typ'):
            return "A(%s,%s,%s)%s\n%s" % (self.typ, self.tag, self.det, getattr(self,'desc','-'),np.ndarray.__repr__(self))
        pass
        return "A()sliced\n%s" % (np.ndarray.__repr__(self))
 
    def derivative_path(self, postfix="track"):
        tag = "%s_%s" % (self.tag, postfix)
        return path_(self.typ, tag, self.det )

    def derivative_save(self, drv, postfix="track"): 
        path = self.derivative_path(postfix)
        if os.path.exists(path):
            log.warning("derivative of %s at path %s exists already, delete and rerun to update" % (repr(self),path) )
        else:
            log.info("saving derivative of %s to %s " % (repr(self), path ))
            np.save( path, drv )    



class I(dict):
    @classmethod
    def loadpath_(cls, path, dbg=False):
        if os.path.exists(path):
            log.debug("loading %s " % path )
            if dbg: 
                os.system("ls -l %s " % path)
            pass
            d = ini_(path)
        else:
            log.warning("cannot load %s " % path)
            d = None
        return d  

    @classmethod
    def load_(cls, typ, tag, det="dayabay", name="t_delta.ini", dbg=False):
        path = path_(typ, tag, det, name=name)
        i = cls(path, typ=typ, tag=tag, det=det, name=name)
        return i

    def __init__(self, path, typ=None, tag=None, det=None, name=None, dbg=False):
        d = self.loadpath_(path,dbg=dbg)
        dict.__init__(self, d)
        self.path = path
        self.fold = os.path.basename(os.path.dirname(path))
        self.typ = typ
        self.tag = tag
        self.det = det
        self.name = name

    def __repr__(self):
        return "I %5s %10s %10s %10s %10s " % (self.tag, self.typ, self.det, self.name, self.fold )


class II(list):
    """  
    List of I instances holding ini file dicts, providing a history of 
    event metadata
    """
    @classmethod
    def load_(cls, typ, tag, det="dayabay", name="t_delta.ini", dbg=False):
        tpaths = tpaths_(typ, tag, det, name=name)
        ii = map(lambda path:I(path, typ=typ, tag=tag, det=det, name=name, dbg=dbg), tpaths) 
        return cls(ii)

    def __init__(self, ii):
        list.__init__(self, ii)   

    def __getitem__(self, k):
        return map(lambda d:d.get(k, None), self) 

    def folds(self):
        return map(lambda i:i.fold, self) 

    def __repr__(self):
        return "\n".join(map(repr, self))



if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    a = A.load_("phtorch","5", "rainbow")
    i = I.load_("mdtorch","5", "rainbow", name="t_delta.ini")

    ii = II.load_("mdtorch","5", "rainbow", name="t_delta.ini")
    iprp = map(float, filter(None,ii['propagate']) )

    jj = II.load_("mdtorch","-5", "rainbow", name="t_delta.ini")
    jprp = map(float, filter(None,jj['propagate']) )


 
