#!/usr/bin/env python

import os, sys, logging, numpy as np, datetime, builtins
from opticks.ana.npmeta import NPMeta

#import inspect
#SOURCE = inspect.getfile(inspect.currentframe())

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

    def __init__(self, base, **kwa):
        self.base = base
        self.kwa = kwa 
        self.relbase = kwa.get("relbase")
        self.globals = kwa.get("globals", False) == True
        self.globals_prefix = kwa.get("globals_prefix", "") 
        print("Fold : loading from base %s setting globals %s globals_prefix %s " % (base, self.globals, self.globals_prefix)) 

        names = os.listdir(base)
        stems = []
        abbrev = []
        symbols = "abcdefghijklmnopqrsTuvwxyz"
        txts = {}

        for i, name in enumerate(filter(lambda n:n.endswith(".npy") or n.endswith(".txt"),names)):
            path = os.path.join(base, name)
            symbol = symbols[i]

            is_npy = name.endswith(".npy")
            is_txt = name.endswith(".txt")

            stem = name[:-4]
            stems.append(stem)
            abbrev.append(symbol) 

            txt_dtype = "|S100" if stem.endswith("_meta") else np.object 

            if is_npy:
                a = np.load(path)
            else: 
                t = np.loadtxt(path, dtype=txt_dtype, delimiter="\t") 
                if t.shape == (): ## prevent one line file behaving different from multiline 
                    a = np.zeros(1, dtype=txt_dtype)
                    a[0] = NPMeta(str(t))   
                    # HMM: this messes up with double quoting
                    # workaround that by not writing a single metadata entries 
                else:
                    a = NPMeta(t)     
                pass
                txts[name] = a
            pass

            # use non-present delim so lines with spaces do not cause errors
            #list(map(str.strip,open(path).readlines())) 
            setattr(self, stem, a ) 
            ashape = str(a.shape) if is_npy else len(a)    
            if self.globals:
                gstem = self.globals_prefix + stem
                setattr( builtins, gstem, a )
                setattr( builtins, symbol, a )
                print("setting builtins symbol:%s gstem:%s" % (symbol, gstem) ) 
            pass
        pass
        self.stems = stems
        self.abbrev = abbrev
        self.txts = txts

    def desc(self):
        now_stamp = datetime.datetime.now()
        l = []
        l.append("t")
        l.append("")
        l.append("CMDLINE:%s" % CMDLINE )
        l.append("t.base:%s" % self.base )
        l.append("")
        stamps = []

        for i in range(len(self.stems)):
            stem = self.stems[i]
            abbrev = self.abbrev[i]

            a = getattr(self, stem)
            ext = ".txt" if a.__class__.__name__ == 'NPMeta' else ".npy"
            name = "%s%s" % (stem,ext)
            aname = "t.%s" % stem

            path = os.path.join(self.base, name)
            st = os.stat(path)
            stamp = datetime.datetime.fromtimestamp(st.st_ctime)
            age_stamp = now_stamp - stamp
            stamps.append(stamp)

            sh = str(len(a)) if ext == ".txt" else str(a.shape)
            abbrev_ = abbrev if self.globals else " " 
            line = "%1s : %-50s : %20s : %s " % ( abbrev_, aname, sh, age_stamp )
            l.append(line)
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


