#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

import os, datetime, logging, re, signal, sys
log = logging.getLogger(__name__)
import numpy as np
lfilter = lambda *args:list(filter(*args))

try:
    reduce
except NameError:
    from functools import reduce
pass


"""
About using IPython debugger
------------------------------

Plant an ipython debugger breakpoint inside some python module by 
duplicating the below code block into the head and entering
the below at the critical code point::

   ipdb.set_trace()   

1. previously thought this was not working in py3, but it seems OK now
2. attempting to import code doing the below doesnt work, have to include the code
3. breakpoint actually lands on the statement immediately after "ipdb.set_trace()"
4. when stopped at the breakpoint an >ipdb prompt should appear

Commands available from the ipdb prompt::

1. "bt" : show the stack backtrace 
2. "c"  : continue from the breakpoint

Observed that even when using python this can magically 
jump into ipython when a breakpoint is reached. However 
see notes/issues/ipython-ipdb-issue.rst which made
me add the check for an ipython invokation to prevent
such fragile magic. 
"""
#if sys.argv[0].find("ipython") > -1:
if True:
    try:
        from IPython.core.debugger import Pdb as MyPdb
    except ImportError:
        class MyPdb(object):
            def set_trace(self):
                log.error("IPython is required for ipdb.set_trace() " )
            pass  
        pass
    pass
    ipdb = MyPdb()
else:
    ipdb = None
pass


from opticks.ana.base import ihex_
from opticks.ana.nbase import chi2, chi2_pvalue, ratio, count_unique_sorted
from opticks.ana.nload import A

nibble_ = lambda n:0xf << (4*n)
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

        log.debug("flags.names %s " % repr(flags.names) )
        log.debug("abbrev.name2abbr %s " % abbrev.name2abbr )
        log.debug("abbrev %s " % repr(map(lambda _:abbrev.name2abbr.get(_,None),flags.names)))

        abbrs = list(map(lambda name:abbrev.name2abbr.get(name,firstlast_(name)), flags.names ))
        self.abbr2code = dict(zip(abbrs, flags.codes))
        self.code2abbr = dict(zip(flags.codes, abbrs))
        self.flags = flags
        self.abbrev = abbrev
        self.delim = delim

    def __call__(self, args):
        for a in args:
            return self.code(a)    # code from subtype

    def check(self, s):
        f = self.abbr2code
        bad = 0 
        for n in s.strip().split(self.delim):
            if f.get(n,0) == 0:
               #log.warn("code bad abbr [%s] s [%s] " % (n, s) ) 
               bad += 1

        #if bad>0:
        #   log.warn("code sees %s bad abbr in [%s] " % (bad, s )) 
        return bad


def seq2msk_procedural(isq):
    ifl = 0 
    for n in range(16):
        msk = 0xf << (4*n)
        nib = ( isq & msk ) >> (4*n)
        if nib == 0:continue   ## cannot vectorize with such procedural approach
        flg = 1 << (nib - 1) 
        ifl |= flg   
    pass
    return ifl

def seq2msk(isq):
    """
    Convert seqhis into mskhis

    OpticksPhoton.h uses a mask but seq use the index for bit-bevity::

          3 enum
          4 {
          5     CERENKOV          = 0x1 <<  0,
          6     SCINTILLATION     = 0x1 <<  1,
          7     MISS              = 0x1 <<  2,
          8     BULK_ABSORB       = 0x1 <<  3,
          9     BULK_REEMIT       = 0x1 <<  4,


    """
    ifl = np.zeros_like(isq)
    for n in range(16):
        msk = 0xf << (4*n)               ## nibble mask
        nib = ( isq & msk ) >> (4*n)     ## pick the nibble and shift to pole position
        flg = 1 << ( nib[nib>0] - 1 )    ## convert flag bit index into flag mask 
        ifl[nib>0] |= flg
    pass
    return ifl 


class MaskType(BaseType):
    def __init__(self, flags, abbrev):
         BaseType.__init__(self, flags, abbrev, delim="|")
         log.debug("abbr2code %s " % repr(self.abbr2code))
         log.debug("code2abbr %s " % repr(self.code2abbr))
         log.debug("flags.codes %s " % repr(self.flags.codes))

    def code(self, s):
        """
        :param s: abbreviation string eg "TO|BT|SD"  or hexstring 8ccccd (without 0x prefix)
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
            c = reduce(lambda a,b:a|b,list(map(lambda n:f.get(n,0), s.split(self.delim))))
        pass
        return c 


    def label(self, arg):
        """
        :param i: integer bitmask
        :return: abbreviation mask string 
        """

        if type(arg) is int:
            i = arg
        elif type(arg) in (np.uint64,np.int64,np.uint32,np.int32):
            i = arg
        elif type(arg) is str:
            if self.hexstr.match(arg):
                i = int(arg, 16)
            else:
                return arg
            pass
        else:
            log.fatal("unexpected argtype %s %s " % (arg, repr(type(arg))))        
            assert 0
        pass

        #log.debug(" i : %s %s " % (repr(i), type(i)))
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

            if bad>0:
               #assert 0
               log.warn("SeqType.code check [%s] bad %d " % (s, bad))

            c = reduce(lambda a,b:a|b,map(lambda ib:ib[1] << 4*ib[0],enumerate(map(lambda n:f.get(n,0), s.split(self.delim)))))
        pass
        return c
   

    def label(self, arg):
        """
        :param i: integer code
        :return: abbreviation sequence string 

        ::

            In [6]: from opticks.ana.histype import HisType
            In [7]: af = HisType()

            In [4]: af.label(0xccd)        # hexint 
            Out[4]: 'TO BT BT'

            In [5]: af.label("TO BT BT")   # already a label 
            Out[5]: 'TO BT BT'

            In [6]: af.label("ccd")        # hexstring  (NB without 0x)
            Out[6]: 'TO BT BT'

            In [7]: af.label(".ccd")       # hexstring with wildcard continuation char "."
            Out[7]: 'TO BT BT ..'

        """

        i = None
        wildcard = type(arg) == str and arg[0] == "." 
        if wildcard:
            arg = arg[1:]

        if type(arg) is int:
            i = arg
        elif type(arg) is np.uint64:
            i = arg
        elif type(arg) is str:
            if self.hexstr.match(arg):
                i = int(arg, 16)
            else:
                return arg
            pass
        else:
            log.fatal("unexpected argtype %s %s " % (arg, repr(type(arg))))        
            assert 0
        pass

        xs = ihex_(i)[::-1]  # top and tailed hex string in reverse order 
        seq = list(map(lambda _:int(_,16), xs )) 
        #log.debug("label xs %s seq %s " % (xs, repr(seq)) )
        d = self.code2abbr
        elem = list(map(lambda _:d.get(_,'?%s?' % _ ), seq )) 
        if wildcard:
            elem += [".."] 

        return self.delim.join(elem) 



class SeqList(object):
    def __init__(self, ls, af, sli ):
        """
        :param ls: seqhis or seqmat array of integers
        :param af: histype or mattype able to decode integers into labels 
        """
        self.ls = ls
        self.afl = af.label
        self.sli = sli

    def __repr__(self):
        return "\n".join(map(lambda _:self.afl(_), self.ls[self.sli]))

    def __getitem__(self, sli):
         self.sli = sli
         return self



 
class SeqTable(object):
    """
    Based on count_unique_sorted applied to a photon length array of sequence history codes

    """
    ptn_ = "^(?P<idx>\d{4})\s*(?P<code>[0-9a-f]+)\s*(?P<a>\d+)\s*(?P<b>\d+)\s*(?P<cf>\S*).*$" 
    ptn = re.compile(ptn_)

    @classmethod 
    def FromTxt(cls, txt, af, **kwa):
        """
        Hmm this assumes a comparison table with cu(count-unique) array of shape (n,3)
        """
        dd = []
        for line in txt.split("\n"):
            m = cls.ptn.match(line)
            if not m: continue
            dd.append(m.groupdict())
        pass
        cu = np.zeros( (len(dd),3), dtype=np.uint64 )  
        for i,d in enumerate(dd):
            cu[i] = ( int("0x%s"%d["code"],16), int(d["a"]), int(d["b"]) )
        pass
        return cls(cu, af, **kwa)


    def __init__(self, cu, af, cnames=[], dbgseq=0, dbgmsk=0, dbgzero=False, cmx=0, c2cut=30, smry=False, shortname="noshortname"): 
        """
        :param cu: count unique array, typically shaped (n, 2) or (n,3) for comparisons
        :param af: instance of SeqType subclass such as HisType
        :param cnames: column names 

        """

        assert len(cu.shape) == 2 and cu.shape[1] >= 2 
        ncol = cu.shape[1] - 1 

        log.debug("SeqTable.__init__ dbgseq %x" % dbgseq)
        log.debug("shortname %s cu.shape %s ncol: %s" % (shortname,repr(cu.shape), ncol))
        assert shortname != "noshortname"  

        if sys.version_info.major in (2,3):
            pass
            #ipdb.set_trace()  # plant an ipython debugger breakpoint
        pass


        self.smry = smry
        self.dirty = False
        self.cu = cu 

        self.ncol = ncol
        self.dbgseq = dbgseq
        self.dbgmsk = dbgmsk
        self.dbgzero = dbgzero
        self.cmx = cmx
        self.shortname = shortname

        seqs = cu[:,0]
        msks = seq2msk(seqs)

        tots = [cu[:,n].sum() for n in range(1,ncol+1)]

        if ncol == 2:
            a = cu[:,1].astype(np.float64)
            b = cu[:,2].astype(np.float64)

            ia = cu[:,1].astype(np.int64)
            ib = cu[:,2].astype(np.int64)
            idif = ia-ib    

            c2, c2n, c2c = chi2(a, b, cut=c2cut)

            #c2s = c2/c2n
            #c2s_tot = c2s.sum()  # same as c2p

            ndf = c2n - 1   ## totals are constrained to match, so one less degree of freedom ?

            c2sum = c2.sum()
            c2p = c2sum/max(1,ndf)

            c2_pval = chi2_pvalue( c2sum , ndf )


            log.debug(" c2sum %10.4f ndf %d c2p %10.4f c2_pval %10.4f " % (c2sum,ndf,c2p, c2_pval ))

            cnames += ["c2"]
            tots += ["%10.2f/%d = %5.2f  (pval:%0.3f prob:%0.3f) " % (c2sum,ndf,c2p,c2_pval,1-c2_pval) ]
            cfcount = cu[:,1:]

            ab, ba = ratio(a, b)
            cnames += ["ab"]
            cnames += ["ba"]

        else:
            c2 = None
            #c2s = None
            c2p = None
            cfcount = None
            ab = None
            ba = None
            idif = None
        pass
        self.idif = idif


        if len(tots) == 1:
            total = tots[0]           
            tots += ["%10.2f" % 1.0 ]
        else:
            total = None 
        pass

        self.total = total
        self.c2 = c2
        #self.c2s = c2s
        self.c2p = c2p
        self.ab = ab  
        self.ba = ba  

        self.seqs = seqs
        self.msks = msks

        codes = cu[:,0]
        counts = cu[:,1]

        #log.debug("codes  : %s " % repr(codes))
        #log.debug("counts : %s " % repr(counts))

        labels = list(map(lambda i:af.label(i), codes ))
        nstep = list(map(lambda l:len(l.split(af.delim)),labels))

        self.label2nstep = dict(zip(labels, nstep))
        self.labels = labels

        lines = list(filter(None, list(map(lambda n:self.line(n), range(len(cu))))))

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


    def cfo_line(self, n):
        if self.ncol == 2:
            cfo_key = getattr(self, 'cfordering_key', [] )
            if len(cfo_key) > 0:
                cfo_debug = " %10.5f " % self.cfordering_key[n]
            else:
                #log.info("no cfo_key ") 
                cfo_debug = " no cfo_key"
            pass
        else:
            cfo_debug = ""
        pass 
        return cfo_debug
 

    def line(self, n):
        iseq = int(self.cu[n,0]) 
        imsk = int(self.msks[n])

        if self.dbgseq > 0 and ( self.dbgseq & iseq ) != self.dbgseq:
           return None 
        pass
    
        if self.dbgmsk > 0:
           pick = (self.dbgmsk & imsk) == self.dbgmsk   # 
           if not pick: 
               return None 

        if self.smry == False:
            xs = "%0.4d %16s" % (n, ihex_(iseq))        
        else:
            xs = "%0.4d " % (n)        
        pass

        #cfo_debug = self.cfo_line(n)
        cfo_debug = ""

        vals = list(map(lambda _:" %7s " % _, self.cu[n,1:] ))

        idif = self.idif[n] if len(vals) == 2 else None
        idif = " %4d " % idif if idif is not None else " " 


        label = self.labels[n]
        if self.smry == False:
            nstep = "[%-2d]" % self.label2nstep[label]
        else:
            nstep = "" 
        pass

        # show only lines with chi2 contrib greater than cmx
        if self.c2 is not None:
            if self.cmx > 0 and self.c2[n] < self.cmx:
                return None
            pass
        pass

        # show only lines with zero counts
        if self.c2 is not None:
            if self.dbgzero and self.cu[n,1] > 0 and self.cu[n,2] > 0:
                return None
            pass
        pass

        if self.c2 is not None:
            sc2 = " %10.2f " % (self.c2[n])
        else:
            sc2 = ""
        pass

        if self.ab is not None and self.smry == False:
            sab = " %10.3f +- %4.3f " % ( self.ab[n,0], self.ab[n,1] )
        else:
            sab = ""
        pass

        if self.ba is not None and self.smry == False:
            sba = " %10.3f +- %4.3f " % ( self.ba[n,0], self.ba[n,1] )
        else:
            sba = ""
        pass


        if self.total is not None:
             frac = float(self.cu[n,1])/float(self.total)
             frac = " %10.3f   " % frac
        else:
             frac = ""
        pass
        cols = [cfo_debug] + [xs+" "] + [frac] + vals + [idif] + ["   "]+ [sc2, sab, sba, nstep, label]
        return " ".join(filter(lambda _:_ != "", cols)) 

    def __call__(self, labels):
        ll = sorted(list(labels), key=lambda _:self.label2count.get(_, None)) 
        return "\n".join(map(lambda _:self.label2line.get(_,None), ll )) 

    def __repr__(self):
        """
        title is externally set from evt.present_table
        """
        spacer_ = lambda _:"%1s%3s %22s " % (".","",_)
        space = spacer_("")
        title = spacer_(getattr(self,'title',"")+"  cfo:"+getattr(self,'cfordering',"-"))
        #print("title:[%s]" % title) 


        body_ = lambda _:" %7s " % _
        head = title + " ".join(map(body_, self.cnames ))
        tail = space + " ".join(map(body_, self.tots ))
        return "\n".join([self.shortname,head,tail]+ lfilter(None,self.lines[self.sli]) + [tail])

    def __getitem__(self, sli):
         self.sli = sli
         return self

    def compare(self, other, ordering="self", shortname="noshortname?"):
        """
        :param other: SeqTable instance
        :param ordering: string "self", "other", "max"  control descending count row ordering  
        """
        log.debug("SeqTable.compare START")
        l = set(self.labels)
        o = set(other.labels)
        lo = list(l | o)   
        # union of labels in self or other

        if ordering == "max":
            ordering_ = lambda _:max(self.label2count.get(_,0),other.label2count.get(_,0))
        elif ordering == "sum":
            ordering_ = lambda _:self.label2count.get(_,0)+other.label2count.get(_,0)
        elif ordering == "sum_code":
            ordering_ = lambda _:1e9*float(self.label2count.get(_,0)+other.label2count.get(_,0))+float(self.label2code.get(_,0)+other.label2code.get(_,0))
        elif ordering == "code":
            ordering_ = lambda _:self.label2code.get(_,0) + other.label2code.get(_,0)
        elif ordering == "self":
            ordering_ = lambda _:self.label2count.get(_,0)
        elif ordering == "other":
            ordering_ = lambda _:other.label2count.get(_,0)
        else:
            assert 0, "ordering_ must be one of max/sum/self/other "
        pass

        u = sorted( lo, key=ordering_, reverse=True)
        # order the labels union by descending maximum count in self or other

        cf = np.zeros( (len(u),3), dtype=np.uint64 )
        cf[:,0] = list(map(lambda _:self.af.code(_), u ))
        cf[:,1] = list(map(lambda _:self.label2count.get(_,0), u ))
        cf[:,2] = list(map(lambda _:other.label2count.get(_,0), u ))
        # form comparison table

        cnames = self.cnames + other.cnames 

        log.debug("compare dbgseq %x dbgmsk %x " % (self.dbgseq, self.dbgmsk))

        cftab = SeqTable(cf, self.af, cnames=cnames, dbgseq=self.dbgseq, dbgmsk=self.dbgmsk, dbgzero=self.dbgzero, cmx=self.cmx, smry=self.smry, shortname=shortname)    
        cftab.cfordering = ordering 

        cfordering_key = list(map(ordering_, u)) 
        log.info("cfordering_key for %s" % shortname)
        #log.info(cfordering_key)
  
        cftab.cfordering_key = cfordering_key 
        log.debug("SeqTable.compare DONE")
        return cftab


class SeqAna(object):
    """
    Canonical usage is from evt with::

        self.seqhis_ana = SeqAna(self.seqhis, self.histype) 
        self.seqmat_ana = SeqAna(self.seqmat, self.mattype)   

    In addition to holding the SeqTable instance SeqAna provides
    methods to make boolean array selections using the aseq and
    form labels. 

    SeqAna and its contained SeqTable exist within a particular selection, 
    ie changing selection entails recreation of SeqAna and its contained SeqTable

    Hmm: when searching for nibbles (eg RE) it would be convenient to view seqhis as an 
    np.int4/np.uint4 dtype, but there is no such thing.
    """
    @classmethod 
    def for_evt(cls, af, tag="1", src="torch", det="dayabay", pfx="source", offset=0):
        ph = A.load_("ph",src,tag,det, pfx=pfx)
        aseq = ph[:,0,offset]
        return cls(aseq, af, cnames=[tag])
    
    def __init__(self, aseq, af, cnames=["noname"], dbgseq=0, dbgmsk=0, dbgzero=False, cmx=0, smry=False, table_shortname="no_table_shortname"):
        """
        :param aseq: photon length sequence array, eg a.seqhis or a.seqmat 
        :param af: instance of SeqType subclass, which knows what the nibble codes mean 

        ::

            In [10]: sa.aseq
            A([  9227469,   9227469, 147639405, ...,   9227469,   9227469,     19661], dtype=uint64)

            In [11]: sa.aseq.shape
            Out[11]: (1000000,)

        """
        cu = count_unique_sorted(aseq)
        self.smry = smry 
        self.af = af
        self.dbgseq = dbgseq
        self.dbgmsk = dbgmsk
        self.dbgzero = dbgzero
        self.cmx = cmx

        self.table = SeqTable(cu, af, cnames=cnames, dbgseq=self.dbgseq, dbgmsk=self.dbgmsk, dbgzero=self.dbgzero, cmx=self.cmx, smry=self.smry, shortname=table_shortname)

        self.aseq = aseq
        self.cu = cu

    def labels(self, prefix=None):
        """
        :param prefix: string sequence label eg "TO BT BT SC"
        :return labels: list of string labels that start with the prefix   
        """
        codes = self.cu[:,0] 
        if not prefix is None:
            pfx = self.af.code(prefix)
            codes = codes[np.where( codes & pfx == pfx )]   
        pass
        labels =  map( lambda _:self.af.label(_), codes )
        return labels

    def seq_or(self, sseq):
        """
        :param sseq: list of sequence labels including source, eg "TO BR SA" "TO BR AB"
        :return psel: selection boolean array of photon length

        Selection of photons with any of the sequence arguments
        """
        #af = self.table.af 
        af = self.af 
        bseq = list(map(lambda _:self.aseq == af.code(_), sseq))  # full length boolean array
        psel = np.logical_or.reduce(bseq)      
        return psel 

    def seq_or_count(self, sseq):
        psel = self.seq_or(sseq)
        return np.count_nonzero(psel)

    def seq_startswith(self, prefix):
        """
        :param prefix: sequence string stub eg "TO BT BT SC"
        :return psel: selection boolean array of photon length

        Selection of all photons starting with prefix sequence
        """
        af = self.table.af 
        pfx = af.code(prefix)
        psel = self.aseq & pfx == pfx 
        return psel 

    def seq_startswith_count(self, prefix):
        psel = self.seq_startswith(prefix)
        return np.count_nonzero(psel)

    def seq_any_(self, co="RE"):
        code = self.af.code(co)
        aseq = self.aseq
        wk = np.zeros( (len(aseq), 16), dtype=np.bool )
        for n in range(16): wk[:, n] = ( aseq & ( 0xf << (n*4) ) == ( code << (n*4) ))
        return wk 

    def seq_any(self, co="RE" ):
        """
        :param co: string label for seqhis nibble
        :return psel: selection boolean array of photon length 
 
        Selects photons with the co nibble in any of the 16 slots 
        """
        wk = self.seq_any_(co)
        psel = np.any(wk, axis=1)
        return psel 

    def seq_any_count(self, co="RE" ):
        """
        :param co: string label for seqhis nibble
        :return count: count_nonzero of psel result of seq_any 

        The count is of photons with the co in any slot (this is not the count of nibbles)
        """
        psel = self.seq_any(co)
        return np.count_nonzero(psel)

    def seq_any_count_nibble(self, co="RE"):
        """
        Like seq_any_count but instead of counting photons count nibbles
        """
        wk = self.seq_any_(co)
        return np.count_nonzero(wk.ravel())


def test_simple_table():
    log.info("test_simple_table")
    from opticks.ana.histype import HisType
    af = HisType() 
    cu = np.zeros( (5,2), dtype=np.uint64 )  # mock up a count unique array 
    cu[0] = (af.code("TO BT AB"), 100)
    cu[1] = (af.code("TO BT AB SD"),200)
    cu[2] = (af.code("TO BT BR BT AB"),300)
    cu[3] = (af.code("TO BT BT BT AB SD"),400)
    cu[4] = (af.code("TO BT AB MI"),500)

    table = SeqTable(cu, af) 
    print(table)

def test_comparison_table():
    log.info("test_comparison_table")
    from opticks.ana.histype import HisType
    af = HisType() 
    cu = np.zeros( (8,3), dtype=np.uint64 )  # mock up a count unique array 

    a = 0 
    b = 0 

    cu[0] = (af.code("TO"), a,b)
    cu[1] = (af.code("TO BT"), a,b)
    cu[2] = (af.code("TO BT BT"),a,b)
    cu[3] = (af.code("TO BT BT BT"),a,b)
    cu[4] = (af.code("TO BT BT BT BT"),a,b)
    cu[5] = (af.code("TO BT BT BT BT BT"),a,b)
    cu[6] = (af.code("TO BT BT BT BT BT BT"),a,b)
    cu[7] = (af.code("TO BT BT BT BT BT BT BT"),a,b)

    table = SeqTable(cu, af) 
    print(table)


def test_comparison_table_2():
    txt = r"""
ab.ahis
.            all_seqhis_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba  
.                              10000     10000      2285.00/5 = 457.00  (pval:1.000 prob:0.000)  
0000             8ccd      8805      8807     -2             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
0001              3bd       580         0    580           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BR MI
0002            3cbcd       563         0    563           563.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BR BT MI
0003           8cbbcd        29        29      0             0.00        1.000 +- 0.186        1.000 +- 0.186  [6 ] TO BT BR BR BT SA
0004          3cbbbcd         6         0      6             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR BR BR BT MI
0005              36d         5         0      5             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SC MI
0006               4d         3         0      3             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO AB
0007            86ccd         2         2      0             0.00        1.000 +- 0.707        1.000 +- 0.707  [5 ] TO BT BT SC SA
0008            8c6cd         1         1      0             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO BT SC BT SA
0009            3b6bd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BR SC BR MI
0010           8b6ccd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT SC BR SA
0011            3cc6d         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SC BT BT MI
0012          3cc6ccd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT SC BT BT MI
0013             4ccd         1         1      0             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] TO BT BT AB
0014            3c6cd         1         0      1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT SC BT MI
0015              8bd         0       580   -580           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BR SA
0016           8cbc6d         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT BR BT SA
0017              86d         0         5     -5             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SC SA
0018        8cbbc6ccd         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT SC BT BR BR BT SA
0019        8cbbbb6cd         0         1     -1             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT SC BR BR BR BR BT SA
.                              10000     10000      2285.00/5 = 457.00  (pval:1.000 prob:0.000)  
    """
    log.info("test_comparison_table_2")
    from opticks.ana.histype import HisType
    af = HisType() 
    table = SeqTable.FromTxt(txt, af) 
    print(table)


if __name__ == '__main__':
    pass 
    ## see histype.py and mattype.py for other testing of this
    logging.basicConfig(level=logging.INFO) 
    #test_simple_table()
    #test_comparison_table()
    test_comparison_table_2()



