#!/usr/bin/env python
"""

NEXT:

* work out how to filter recarray to find large chi2 contributors
* summations for collective distrib chi2 ?


"""
import os, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.make_rst_table import recarray_as_rst

log = logging.getLogger(__name__)


class ABStat(object):
    """
    """
    STATPATH = "$TMP/stat.npy"

    @classmethod
    def path_(cls):
        return os.path.expandvars(cls.STATPATH)

    def __init__(self, st):
        self.st = st

    def save(self):
        np.save(self.path_(),self.st)  

    def __str__(self):
        return "\n".join([repr(self),recarray_as_rst(self.st),repr(self)])

    def __repr__(self):
        return "ABStat %s %s " % (len(self.st), ",".join(self.st.dtype.names) )

    @classmethod
    def load(cls):
        st = np.load(cls.path_()).view(np.recarray)
        return cls(st) 

    @classmethod
    def dump(cls, st=None): 
        if st is None:
            st = cls.load()
        print st


if __name__ == '__main__':
    ok = opticks_main()
    st = ABStat.load()
    print st 

          
