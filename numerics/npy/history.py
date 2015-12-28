#!/bin/env python
import os, datetime, logging
log = logging.getLogger(__name__)
import numpy as np

from env.numerics.npy.base import ini_, json_, ihex_, ffs_
from env.numerics.npy.nbase import count_unique_sorted
from env.numerics.npy.nload import A

class Flags(object):
    def __init__(self, path="$IDPATH/GFlagsLocal.ini"):
        ini = ini_(path)
        ini = dict(zip(ini.keys(),map(int,ini.values())))  # convert values to int 

        names = map(str,ini.keys())
        codes = map(int,ini.values())

        self.names = names
        self.codes = codes
        self.name2code = dict(zip(names, codes)) 
        self.code2name = dict(zip(codes, names))


class Abbrev(object):
    def __init__(self, path="~/.opticks/GFlags/abbrev.json"):
        js = json_(path)

        names = map(str,js.keys())
        abbrs = map(str,js.values())

        self.names = names
        self.abbrs = abbrs
        self.name2abbr = dict(zip(names, abbrs))
        self.abbr2name = dict(zip(abbrs, names))


class AbbFlags(object):
    """
    ::

        In [17]: x=0x8cbbbcd

        In [18]: l = af.seqhis_label(x)

        In [19]: "%x" % af.seqhis_int(l)
        Out[19]: '8cbbbcd'

        In [20]: l
        Out[20]: 'TO BT BR BR BR BT SA'

    """
    def __init__(self):
        flags = Flags()
        abbrev = Abbrev()

        abbrs = map(lambda name:abbrev.name2abbr.get(name,None), flags.names )
        self.abbr2code = dict(zip(abbrs, flags.codes))
        self.code2abbr = dict(zip(flags.codes, abbrs))
        self.flags = flags 
        self.abbrev = abbrev 

    def seqhis_int(self, s):
        f = self.abbr2code
        return reduce(lambda a,b:a|b,map(lambda ib:ib[1] << 4*ib[0],enumerate(map(lambda n:f[n], s.split(" ")))))

    def seqhis_label(self, i):
        xs = ihex_(i)[::-1]  # top and tailed hex string in reverse order 
        seq = map(lambda _:int(_,16), xs ) 
        log.debug("seqhis xs %s seq %s " % (xs, repr(seq)) )
        d = self.code2abbr
        return " ".join(map(lambda _:d.get(_,'?%s?' % _ ), seq )) 

         

class HistoryTable(object):
    def __init__(self, cu): 
        assert len(cu.shape) == 2 and cu.shape[1] == 2 

        af = AbbFlags()
        seqs  = map(long, cu[:,0])
        counts = map(int, cu[:,1])
        labels = map(lambda i:af.seqhis_label(i), cu[:,0] )

        fmt = "%20s %10s : %40s "
        lines = map(lambda n:fmt % ( ihex_(seqs[n]), counts[n], labels[n]), range(len(cu)) ) 

        self.seqs = seqs
        self.counts = counts
        self.labels = labels
        self.lines = lines
        self.label2count = dict(zip(labels, counts))
        self.label2line = dict(zip(labels, lines))

        self.af = af
        self.cu = cu 
        self.tot = cu[:,1].astype(np.int32).sum()
        self.sli = slice(None)

    def __call__(self, labels):
        ll = sorted(list(labels), key=lambda _:self.label2count.get(_, None)) 
        return "\n".join(map(lambda _:self.label2line.get(_,None), ll )) 

    def __repr__(self):
        tail = [" tot: %s " % self.tot ]
        return "\n".join(self.lines[self.sli] + tail)



class History(object):
    @classmethod 
    def for_evt(cls, tag="1", src="torch", det="dayabay"):
        ph = A.load_("ph"+src,tag,det)
        seqhis = ph[:,0,0]
        return cls(seqhis)
    
    def __init__(self, seqhis):
        cu = count_unique_sorted(seqhis)
        self.table = HistoryTable(cu)
        self.seqhis = seqhis
        self.cu = cu

    def seqhis_or(self, seqs, not_=False):
        """
        :param seqs: sequence strings including source, eg "TO BR SA" "TO BR AB"
        :return: selection boolean array of photon length

        photon level selection based on history sequence 
        """
        af = self.table.af 

        s_seqhis = map(lambda _:self.seqhis == af.seqhis_int(_), seqs)

        psel = np.logical_or.reduce(s_seqhis)      
        if not_:
            psel = np.logical_not(psel)

        return psel 




if __name__ == '__main__':
     af = AbbFlags()

    
     src = "torch"
     tag = "5"
     det = "rainbow"  

     ph = A.load_("ph"+src,tag,det)
     seqhis = ph[:,0,0]

     cu = count_unique_sorted(seqhis)

     ht = HistoryTable(cu)




