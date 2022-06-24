#!/usr/bin/env python 

import numpy as np
import os, re, logging
log = logging.getLogger(__name__)
from collections import OrderedDict as odict 






class stag_item(object):
    @classmethod
    def Placeholder(cls):
        return cls(-1,"placeholder","ERROR" )

    def __init__(self, code, name, note=""):
        self.code = code
        self.name = name
        self.note = note
    def __repr__(self):
        return "%2d : %10s : %s " % (self.code, self.name, self.note)


class stag(object):
    """
    # the below NSEQ, BITS, ... param need to correspond to stag.h static constexpr 
    """
    enum_ptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")
    note_ptn = re.compile("^\s*static constexpr const char\* (\w+)_note = \"(.*)\" ;\s*$")

    PATH = "$OPTICKS_PREFIX/include/sysrap/stag.h" 

    NSEQ = 4   ## must match stag.h:NSEQ 
    BITS = 5 
    MASK = ( 0x1 << BITS ) - 1 
    SLOTMAX = 64//BITS
    SLOTS = SLOTMAX*NSEQ


    @classmethod
    def NumStarts(cls, tg):
        ns = np.zeros( (len(tg)), dtype=np.uint8 ) 
        for i in range(len(tg)):
            starts = np.where( tg[i] == tg[0,0] )[0] 
            ns[i] = len(starts)
        pass
        return ns 
 

    @classmethod
    def StepSplit(cls, tg, fl=None,  step_slot=10):
        """
        :param tg: unpacked tag array of shape (n, SLOTS)
        :param fl: None or flat array of shape (n, SLOTS)
        :param step_slot: max random throws per step  
        :return tgs OR (tgs,fls): step split arrays of shape (n, max_starts, step_slot) 

        In [4]: at[0]
        Out[4]: array([ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0], dtype=uint8)

        In [8]: ats[0]
        Out[8]: 
        array([[ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
               [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
               [ 1,  2, 11, 12,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=uint8)

        """
        if not fl is None:
            assert fl.shape == tg.shape 
        pass

        max_starts = 0 
        for i in range(len(tg)):
            starts = np.where( tg[i] == tg[0,0] )[0] 
            if len(starts) > max_starts: max_starts = len(starts)
        pass

        tgs = np.zeros((len(tg), max_starts, step_slot), dtype=np.uint8)
        fls = np.zeros((len(tg), max_starts, step_slot), dtype=np.float32) if not fl is None else None

        for i in range(len(tg)):
            starts = np.where( tg[i] == tg[0,0] )[0] 
            ends = np.where( tg[i] == 0 )[0] 
            end = ends[0] if len(ends) > 0 else len(tg[i])  
            ## above handles when the tags do not get to zero due to collection truncation
            for j in range(len(starts)):
                st = starts[j]
                en = starts[j+1] if j+1 < len(starts) else end
                tgs[i, j,0:en-st] = tg[i,st:en] 
                if not fls is None:
                    fls[i, j,0:en-st] = fl[i,st:en] 
                pass
            pass
        pass
        return tgs if fls is None else tgs,fls         


    @classmethod
    def Unpack(cls, tag):
        """
        :param tag: (n, NSEQ) array of bitpacked tag enumerations
        :return tg: (n, SLOTS) array of unpacked tag enumerations

        Usage::

            # apply stag.Unpack to both as same stag.h bitpacking is used
            at = stag.Unpack(a.tag) if hasattr(a,"tag") else None
            bt = stag.Unpack(b.tag) if hasattr(b,"tag") else None

        """
        assert tag.shape == (len(tag), cls.NSEQ)

        st = np.zeros( (len(tag), cls.SLOTS), dtype=np.uint8 )   
        for i in range(cls.NSEQ):
            for j in range(cls.SLOTMAX):
                st[:,i*cls.SLOTMAX+j] = (tag[:,i] >> (cls.BITS*j)) & cls.MASK
            pass
        pass
        return st 

    def __init__(self, path=PATH):
        path = os.path.expandvars(path)
        lines = open(path, "r").read().splitlines()
        self.path = path 
        self.lines = lines 
        self.items = []
        self.d = self.parse()

    def find_item(self, name):
        for item in self.items:
            if item.name == name: return item
        pass
        return None 

    def parse(self):
        d=odict()
        for line in self.lines:
            enum_match = self.enum_ptn.match(line)
            note_match = self.note_ptn.match(line)
            if enum_match:
                name, val = enum_match.groups() 
                pfx = "stag_"
                assert name.startswith(pfx) 
                sname = name[len(pfx):]
                code = int(val)
                item = stag_item(code, sname, "") 
                self.items.append(item)
                d[code] = item
                log.debug("%40s : name:%20s  sname:%10s val:%10s code:%d " % (line,name,sname,val, code) )
            elif note_match:
                name, note = note_match.groups()
                item = self.find_item(name)
                assert not item is None
                item.note = note 
                log.debug(" note %10s : %s " % (name, note))
            pass
            pass
        pass
        return d

    def old_label(self, st):
        d = self.d
        label_ = lambda _:repr(d.get(_,stag_item.Placeholder()))
        ilabel_ = lambda _:"%2d : %s" % ( _, label_(st[_])) 
        return "\n".join(map(ilabel_, range(len(st))))


    def __call__(self, code):
        return self.d.get(code,stag_item.Placeholder())
 
    def label(self, st):
        lines = [] 
        for i in range(len(st)):
            item = self(st[i])
            if item.code == st[0] and i > 0:
                lines.append("")   
            pass
            label = "%2d : %s " % (i, repr(item)) 
            lines.append(label)
        pass
        return "\n".join(lines)



 
    def __str__(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return "\n".join(list(map(repr,self.items)))  




def test_label():
   tag = stag()
   #print(tag) 
   print(repr(tag))

   st = np.array([[ 1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [ 1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [ 1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=np.uint8)

   print(tag.label(st[0,:10]))


def test_StepSplit():
    from numpy import array, uint8
    at = array(
      [[ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0],
       [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0],
       [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0]], dtype=uint8)

    x_ats = array(
      [[[ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
        [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
        [ 1,  2, 11, 12,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],

       [[ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
        [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
        [ 1,  2, 11, 12,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],

       [[ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
        [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
        [ 1,  2, 11, 12,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]], dtype=uint8)

    ats = stag.StepSplit(at)
    assert np.all( ats == x_ats )




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
  
    #test_label()
    test_StepSplit()

    #at = np.array([[ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0],
    #               [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0],
    #               [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0]], dtype=np.uint8)
    #
    #ats = stag.StepSplit(at)



         
