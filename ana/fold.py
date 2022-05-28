#!/usr/bin/env python

import os, sys, logging, numpy as np, datetime, builtins
from opticks.ana.npmeta import NPMeta

from opticks.sysrap.sframe import sframe



CMDLINE = " ".join(sys.argv)

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

class Fold(object):
    @classmethod
    def Load(cls, *args, **kwa):
        if len(args) == 0:
            args = [os.environ["FOLD"]] 
        pass
        relbase = os.path.join(*args[1:]) if len(args) > 1 else args[0]
        kwa["relbase"] = relbase   # relbase is the dir path excluding the first element 
        base = os.path.join(*args)
        base = os.path.expandvars(base) 
        quiet = kwa.get("quiet", False) == True 

        fold = cls(base, **kwa) if os.path.isdir(base) else None
        if fold is None and quiet == False:
            log.error("failed to load from base [%s]" % base )
        pass
        print(repr(fold))
        print(str(fold))
        return fold

    def __getattr__(self, name):
        """Only called when there is no *name* attr"""
        return None


    SFRAME = "sframe.npy"

    def __init__(self, base, **kwa):
        self.base = base
        self.kwa = kwa 
        self.symbol = kwa.get("symbol", "t")
        self.relbase = kwa.get("relbase")
        self.globals = kwa.get("globals", False) == True
        self.globals_prefix = kwa.get("globals_prefix", "") 
        print("Fold : setting globals %s globals_prefix %s " % (self.globals, self.globals_prefix)) 

        names = os.listdir(base)

        paths = []
        stems = []
        abbrev = []
        symbols = "abcdefghijklmnopqrsTuvwxyz"
        txts = {}

        for i, name in enumerate(filter(lambda n:n.endswith(".npy") or n.endswith(".txt"),names)):
            path = os.path.join(base, name)
            symbol = symbols[i]
            stem = name[:-4]

            paths.append(path)
            stems.append(stem)
            abbrev.append(symbol) 

            is_npy = name.endswith(".npy")

            if name == self.SFRAME:
                a = sframe.Load(path)
            elif is_npy:
                a = np.load(path)
            elif name.endswith("_meta.txt"):
                a = NPMeta.Load(path)
                txts[name] = a 
            elif name.endswith(".txt"):
                a = NPMeta.Load(path)
                txts[name] = a
            pass
            setattr(self, stem, a ) 

            if self.globals:
                gstem = self.globals_prefix + stem
                setattr( builtins, gstem, a )
                setattr( builtins, symbol, a )
                print("setting builtins symbol:%s gstem:%s" % (symbol, gstem) ) 
            pass
        pass
        self.paths = paths
        self.stems = stems
        self.abbrev = abbrev
        self.txts = txts

    def desc(self):
        now_stamp = datetime.datetime.now()
        l = []
        l.append(self.symbol)
        l.append("")
        l.append("CMDLINE:%s" % CMDLINE )
        l.append("%s.base:%s" % (self.symbol,self.base) )
        l.append("")
        stamps = []

        for i in range(len(self.paths)):
            path = self.paths[i]
            stem = self.stems[i]
            abbrev = self.abbrev[i]

            a = getattr(self, stem)

            kls = a.__class__.__name__
            ext = ".txt" if kls == 'NPMeta' else ".npy"
            name = "%s%s" % (stem,ext)
            aname = "%s.%s" % (self.symbol,stem)

            if os.path.exists(path):
                st = os.stat(path)
                stamp = datetime.datetime.fromtimestamp(st.st_ctime)
                age_stamp = now_stamp - stamp
                stamps.append(stamp)
                sh = str(len(a)) if ext == ".txt" else str(a.shape)
                abbrev_ = abbrev if self.globals else " " 
                line = "%1s : %-50s : %20s : %s " % ( abbrev_, aname, sh, age_stamp )
                l.append(line)
            else:
                msg = "ERROR non-existing path for : stem %s kls %s ext %s name %s aname %s path %s " % (stem, kls, ext, name, aname, path )
                l.append(msg)
            pass
        pass
        l.append("")

        if len(stamps) > 0:
            min_stamp = min(stamps)
            max_stamp = max(stamps)
            dif_stamp = max_stamp - min_stamp 
            age_stamp = now_stamp - max_stamp
            l.append(" min_stamp : %s " % str(min_stamp))
            l.append(" max_stamp : %s " % str(max_stamp))
            l.append(" dif_stamp : %s " % str(dif_stamp))
            l.append(" age_stamp : %s " % str(age_stamp))
            assert dif_stamp.microseconds < 1e6, "stamp divergence detected microseconds %d : so are seeing mixed up results from multiple runs " % dif_stamp.microseconds 
        else:
            l.append("WARNING THERE ARE NO TIME STAMPS")
        pass
        return l 

    def __repr__(self):
        return "\n".join(self.desc())    

    def __str__(self):
        l = []
        for k in self.txts:
            l.append("")
            l.append(k)
            l.append("")
            l.append(str(self.txts[k]))
        pass
        return "\n".join(l) 
  

if __name__  == '__main__':
    pass


