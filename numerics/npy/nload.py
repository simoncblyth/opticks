#!/bin/env python
import os, logging
log = logging.getLogger(__name__)
import numpy as np


DEFAULT_PATH_TEMPLATE = "$LOCAL_BASE/env/opticks/$1/$2/%s.npy"  ## cf C++ NPYBase::

def path_(typ, tag, det="dayabay"):
    tmpl = os.path.expandvars(DEFAULT_PATH_TEMPLATE.replace("$1", det).replace("$2",typ)) 
    return tmpl % str(tag)

class A(np.ndarray):
    @classmethod
    def load_(cls, typ, tag, det="dayabay"):
        path = path_(typ,tag, det)
        a = None
        if os.path.exists(path):
            log.debug("loading %s " % path )
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


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    a = A.load_("phtorch","5", "rainbow")

 
