#!/bin/env python
import os, datetime, logging
log = logging.getLogger(__name__)
import numpy as np

from env.numerics.npy.base import ini_, json_, ihex_, ffs_
from env.numerics.npy.nbase import count_unique_sorted, chi2
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

    def __call__(self, args):
        for a in args:
            return self.seqhis_int(a) 


    def seqhis_label(self, i):
        xs = ihex_(i)[::-1]  # top and tailed hex string in reverse order 
        seq = map(lambda _:int(_,16), xs ) 
        log.debug("seqhis xs %s seq %s " % (xs, repr(seq)) )
        d = self.code2abbr
        return " ".join(map(lambda _:d.get(_,'?%s?' % _ ), seq )) 

         

class HistoryTable(object):
    def __init__(self, cu, cnames=[]): 
        assert len(cu.shape) == 2 and cu.shape[1] >= 2 
        af = AbbFlags()
        ncol = cu.shape[1] - 1 

        self.cu = cu 
        self.ncol = ncol

        seqs = cu[:,0]
        tots = [cu[:,n].sum() for n in range(1,ncol+1)]

        if ncol == 2:
            a = cu[:,1].astype(np.float64)
            b = cu[:,2].astype(np.float64)
            c2, c2n = chi2(a, b, cut=30)
            c2p = c2.sum()/c2n
            cnames += ["c2"]
            tots += ["%10.2f" % c2p ]
        else:
            c2 = None
            c2p = None

        self.c2 = c2
        self.c2p = c2p

        self.seqs = seqs

        counts = cu[:,1]
        labels = map(lambda i:af.seqhis_label(i), cu[:,0] )
        nstep = map(lambda l:len(l.split(" ")),labels)

        self.label2nstep = dict(zip(labels, nstep))
        self.labels = labels

        lines = map(lambda n:self.line(n), range(len(cu)))

        self.counts = counts
        self.lines = lines

        self.label2count = dict(zip(labels, counts))
        self.label2line = dict(zip(labels, lines))
        self.label2code = dict(zip(labels, seqs))

        self.cnames = cnames
        self.tots = tots
        self.af = af
        self.sli = slice(None)

    def line(self, n):
        xs = "%20s " % ihex_(self.cu[n,0])        
        vals = map(lambda _:" %10s " % _, self.cu[n,1:] ) 
        label = self.labels[n]
        nstep = "[%-2d]" % self.label2nstep[label]

        if self.c2 is not None:
            sc2 = " %10.2f " % self.c2[n]
        else:
            sc2 = ""
        pass

        return " ".join([xs] + vals + ["   "]+ [sc2, nstep, label]) 

    def __call__(self, labels):
        ll = sorted(list(labels), key=lambda _:self.label2count.get(_, None)) 
        return "\n".join(map(lambda _:self.label2line.get(_,None), ll )) 

    def __repr__(self):
        space = "%20s " % ""
        body_ = lambda _:" %10s " % _
        head = space + " ".join(map(body_, self.cnames ))
        tail = space + " ".join(map(body_, self.tots ))
        return "\n".join([head] + self.lines[self.sli] + [tail])

    def compare(self, other):
        l = set(self.labels)
        o = set(other.labels)
        u = sorted(list(l | o), key=lambda _:max(self.label2count.get(_,0),other.label2count.get(_,0)), reverse=True)

        cf = np.zeros( (len(u),3), dtype=np.uint64 )

        cf[:,0] = map(lambda _:self.af.seqhis_int(_), u )
        cf[:,1] = map(lambda _:self.label2count.get(_,0), u )
        cf[:,2] = map(lambda _:other.label2count.get(_,0), u )

        cnames = self.cnames + other.cnames 

        return HistoryTable(cf, cnames=cnames)    




class History(object):
    @classmethod 
    def for_evt(cls, tag="1", src="torch", det="dayabay"):
        ph = A.load_("ph"+src,tag,det)
        seqhis = ph[:,0,0]
        return cls(seqhis, cnames=[tag])
    
    def __init__(self, seqhis, cnames=["noname"]):
        cu = count_unique_sorted(seqhis)
        self.table = HistoryTable(cu, cnames=cnames)
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



def test_HistoryTable(ht, seqhis):
     for seq in ht.labels:
         seqs = [seq]
         s_seqhis = map(lambda _:seqhis == af.seqhis_int(_), seqs )
         psel = np.logical_or.reduce(s_seqhis)      

         n = len(seqhis[psel])
         assert n == ht.label2count.get(seq)
         print "%10s %s " % (n, seq ) 



if __name__ == '__main__':
     af = AbbFlags()

    
     src = "torch"
     tag = "5"
     det = "rainbow"  

     ph = A.load_("ph"+src,tag,det)
     seqhis = ph[:,0,0]

     cu = count_unique_sorted(seqhis)

     ht = HistoryTable(cu)
     
     test_HistoryTable(ht, seqhis)






