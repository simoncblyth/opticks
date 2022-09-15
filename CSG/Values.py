#!/usr/bin/env python
import os, numpy as np, logging
log = logging.getLogger(__name__)


class Values(object):
    @classmethod
    def FindDirUpTree(cls, origpath, name="Values"): 
        """
        :param origpath: to traverse upwards looking for sibling dirs with *name*
        :param name: sibling directory name to look for 
        :return full path to *name* directory 
        """
        elem = origpath.split("/")
        found = None
        for i in range(len(elem),0,-1):
            path = "/".join(elem[:i])
            cand = os.path.join(path, name)
            log.debug(cand) 
            if os.path.isdir(cand):
                found = cand
                break 
            pass  
        pass
        return found 

    @classmethod
    def Find(cls, origfold=None, symbol="v"):
        if origfold is None:
            origfold = os.environ.get("FOLD", "/tmp" )
        else:
            origfold = os.path.expandvars(origfold) 
        pass
        fold = cls.FindDirUpTree(origfold)
        return None if fold is None else cls(fold, symbol=symbol) 

    def __init__(self, fold_=None, symbol="v"): 
        fold = os.path.expandvars(fold_)
        a_path = os.path.join(fold, "values.npy")
        n_path = os.path.join(fold, "values_names.txt")
        a = np.load(a_path)
        n = np.loadtxt(n_path, dtype=np.object, delimiter="\n")     
        assert a.shape == n.shape
 
        self.fold = fold
        self.symbol = symbol
        self.a_path = a_path 
        self.n_path = n_path 
        self.a = a
        self.n = n
        self.contains = None

    def hdr(self):
        return "Values.%s : %s : contains:%s  " % (self.symbol,self.fold, self.contains ) 

    def __str__(self):
        return "\n".join([
               "Values", 
               "%s.fold: %s " % (self.symbol, self.fold), 
               "%s.a_path: %s " % (self.symbol, self.a_path), 
               "%s.n_path: %s " % (self.symbol, self.n_path), 
               "%s.a: %s " % (self.symbol, str(self.a.shape)),
               "%s.n: %s " % (self.symbol, str(self.n.shape))  
               ])


    def get_idx(self, q):
        n_idx = np.where( self.n == q )[0]   
        assert len(n_idx) in (0,1)
        return -1 if len(n_idx) == 0 else n_idx[0]

    def get(self, name):
        """
        :param name: eg SolidMask.SolidMaskVirtual.htop_out
        :return val: scalar

        sv.get("SolidMask.SolidMaskVirtual.htop_out")   194.0
        """
        idx = self.get_idx(name)
        return self.a[idx] if idx > -1 else None

    def __repr__(self):
        a = self.a
        n = self.n 
        contains = self.contains
        assert a.shape == n.shape
        lines = []
        lines.append(self.hdr())
        for i in range(len(a)):
            k = n[i]
            v = a[i]
            match = contains is None or (not contains is None and contains in k)
            if match:
                line = " %2d : %10.4f : %s " % (i, v, k )         
                lines.append(line)
            pass
        pass
        return "\n".join(lines)

    def __getitem__(self, contains):
        self.contains = contains
        return self



if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    v = Values.Find()
    print(repr(v))

 
