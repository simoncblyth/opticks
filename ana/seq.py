#!/usr/bin/env python
import os, datetime, logging, re
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import ihex_
from opticks.ana.nbase import chi2, count_unique_sorted
from opticks.ana.nload import A


firstlast_ = lambda name:name[0] + name[-1]

class BaseType(object):
    hexstr = re.compile("^[0-9a-f]+$")
    def __init__(self, flags, abbrev, delim=" "):
        """
        When no abbreviation available, use first and last letter of name eg::

           MACHINERY -> MY
           FABRICATED -> FD
           G4GUN -> GN

        """
        abbrs = map(lambda name:abbrev.name2abbr.get(name,firstlast_(name)), flags.names )
        self.abbr2code = dict(zip(abbrs, flags.codes))
        self.code2abbr = dict(zip(flags.codes, abbrs))
        self.flags = flags
        self.abbrev = abbrev
        self.delim = delim

    def __call__(self, args):
        for a in args:
            return self.code(a) 

    def check(self, s):
        f = self.abbr2code
        bad = 0 
        for n in s.strip().split(self.delim):
            if f.get(n,0) == 0:
               log.warn("code bad abbr [%s] s [%s] " % (n, s) ) 
               bad += 1

        if bad>0:
           log.warn("code sees %s bad abbr in [%s] " % (bad, s )) 
        return bad


class MaskType(BaseType):
    def __init__(self, flags, abbrev):
         BaseType.__init__(self, flags, abbrev, delim="|")
         log.debug("abbr2code %s " % repr(self.abbr2code))
         log.debug("code2abbr %s " % repr(self.code2abbr))
         log.debug("flags.codes %s " % repr(self.flags.codes))

    def code(self, s):
        """
        :param s: abbreviation string eg "TO|BT|SD"  or hexstring 8ccccd  (without 0x prefix)
        :return: integer bitmask 
        """
        #log.info(" s [%s] " % s)
        if self.hexstr.match(s):
            c = int(s,16) 
            cs = "%x" % c 
            log.info("converted hexstr %s to hexint %x and back %s " % (s,c,cs)) 
            assert s == cs
        else:
            f = self.abbr2code
            bad = self.check(s) 
            c = reduce(lambda a,b:a|b,map(lambda n:f.get(n,0), s.split(self.delim)))
        pass
        return c 


    def label(self, i):
        """
        :param i: integer bitmask
        :return: abbreviation mask string 
        """
        log.debug(" i : %s %s " % (repr(i), type(i)))
        codes = filter(lambda c:int(i) & c, self.flags.codes)
        codes = sorted(codes,reverse=True)
        d = self.code2abbr
        return self.delim.join(map(lambda _:d.get(_,'?%s?' % _ ), codes )) 


class SeqType(BaseType):
    def __init__(self, flags, abbrev):
         BaseType.__init__(self, flags, abbrev, delim=" ")

    def code(self, s):
        """
        :param s: abbreviation sequence string eg "TO BT BR BR BR BT SA"
        :return: integer code eg 0x8cbbbcd
        """
        if self.hexstr.match(s):
            c = int(s,16) 
            cs = "%x" % c 
            log.info("converted hexstr %s to hexint %x and back %s " % (s,c,cs)) 
            assert s == cs
        else:
            f = self.abbr2code
            bad = self.check(s) 
            c = reduce(lambda a,b:a|b,map(lambda ib:ib[1] << 4*ib[0],enumerate(map(lambda n:f.get(n,0), s.split(self.delim)))))
        pass
        return c
   

    def label(self, i):
        """
        :param i: integer code
        :return: abbreviation sequence string 
        """
        xs = ihex_(i)[::-1]  # top and tailed hex string in reverse order 
        seq = map(lambda _:int(_,16), xs ) 
        log.debug("label xs %s seq %s " % (xs, repr(seq)) )
        d = self.code2abbr
        return self.delim.join(map(lambda _:d.get(_,'?%s?' % _ ), seq )) 




 
class SeqTable(object):
    def __init__(self, cu, af, cnames=[]): 
        """
        :param cu: count unique array, typically shaped (n, 2) 
        :param af: instance of SeqType subclass such as HisType
        :param cnames: column names 
        """
        log.debug("SeqTable.__init__")

        assert len(cu.shape) == 2 and cu.shape[1] >= 2 

        ncol = cu.shape[1] - 1 

        self.cu = cu 
        self.ncol = ncol

        seqs = cu[:,0]
        tots = [cu[:,n].sum() for n in range(1,ncol+1)]

        if ncol == 2:
            a = cu[:,1].astype(np.float64)
            b = cu[:,2].astype(np.float64)
            c2, c2n, c2c = chi2(a, b, cut=30)
            c2p = c2.sum()/c2n
            cnames += ["c2"]
            tots += ["%10.2f" % c2p ]
            cfcount = cu[:,1:]
        else:
            c2 = None
            c2p = None
            cfcount = None
        pass


        if len(tots) == 1:
            total = tots[0]           
            tots += ["%10.2f" % 1.0 ]
        else:
            total = None 
        pass

        self.total = total
        self.c2 = c2
        self.c2p = c2p

        self.seqs = seqs


        codes = cu[:,0]
        counts = cu[:,1]

        log.debug("codes  : %s " % repr(codes))
        log.debug("counts : %s " % repr(counts))

        labels = map(lambda i:af.label(i), codes )
        nstep = map(lambda l:len(l.split(af.delim)),labels)

        self.label2nstep = dict(zip(labels, nstep))
        self.labels = labels

        lines = map(lambda n:self.line(n), range(len(cu)))

        self.codes = codes  
        self.counts = counts
        self.lines = lines

        self.label2count = dict(zip(labels, counts))
        self.label2line = dict(zip(labels, lines))
        self.label2code = dict(zip(labels, seqs))

        if cfcount is not None:
            self.label2cfcount = dict(zip(labels, cfcount))

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

        if self.total is not None:
             frac = float(self.cu[n,1])/float(self.total)
             frac = " %10.3f   " % frac
        else:
             frac = ""
        pass

        return " ".join([xs] + [frac] + vals + ["   "]+ [sc2, nstep, label]) 

    def __call__(self, labels):
        ll = sorted(list(labels), key=lambda _:self.label2count.get(_, None)) 
        return "\n".join(map(lambda _:self.label2line.get(_,None), ll )) 

    def __repr__(self):

        spacer_ = lambda _:"%20s " % _
        space = spacer_("")
        title = spacer_(getattr(self,'title',""))

        body_ = lambda _:" %10s " % _
        head = title + " ".join(map(body_, self.cnames ))
        tail = space + " ".join(map(body_, self.tots ))
        return "\n".join([head] + self.lines[self.sli] + [tail])

    def compare(self, other):
        l = set(self.labels)
        o = set(other.labels)
        u = sorted(list(l | o), key=lambda _:max(self.label2count.get(_,0),other.label2count.get(_,0)), reverse=True)

        cf = np.zeros( (len(u),3), dtype=np.uint64 )

        cf[:,0] = map(lambda _:self.af.code(_), u )
        cf[:,1] = map(lambda _:self.label2count.get(_,0), u )
        cf[:,2] = map(lambda _:other.label2count.get(_,0), u )

        cnames = self.cnames + other.cnames 

        return SeqTable(cf, self.af, cnames=cnames)    


class SeqAna(object):
    """
    """
    @classmethod 
    def for_evt(cls, af, tag="1", src="torch", det="dayabay", offset=0):
        ph = A.load_("ph",src,tag,det)
        aseq = ph[:,0,offset]
        return cls(aseq, af, cnames=[tag])
    
    def __init__(self, aseq, af, cnames=["noname"]):
        """
        :param aseq: photon length sequence array 
        :param af: instance of SeqType subclass 
        """
        cu = count_unique_sorted(aseq)
        self.af = af
        self.table = SeqTable(cu, af, cnames=cnames)
        self.aseq = aseq
        self.cu = cu

    def seq_or(self, sseq, not_=False):
        """
        :param sseq: sequence strings including source, eg "TO BR SA" "TO BR AB"
        :return: selection boolean array of photon length

        photon level selection based on history sequence 
        """
        af = self.table.af 

        bseq = map(lambda _:self.aseq == af.code(_), sseq)

        psel = np.logical_or.reduce(bseq)      
        if not_:
            psel = np.logical_not(psel)

        return psel 



if __name__ == '__main__':
    pass 
    ## see histype.py and mattype.py for testing this
    



