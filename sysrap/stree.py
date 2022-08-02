#!/usr/bin/env python

import os, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold
from opticks.sysrap.sfreq import sfreq 

class snode(object):
    DTYPE = [('index', '<i4'), 
             ('depth', '<i4'), 
             ('sibdex', '<i4'), 
             ('parent', '<i4'), 
             ('num_child', '<i4'), 
             ('first_child', '<i4'), 
             ('next_sibling', '<i4'), 
             ('lvid', '<i4'), 
             ('copyno', '<i4')] 

    @classmethod
    def RecordsFromArrays(cls, a):
        """
        :param a: nodes ndarray
        :return nds: np.recarray
        """
        nds = np.core.records.fromarrays(a.T, dtype=cls.DTYPE )
        return nds 

    @classmethod
    def Desc(cls, rec):
        """
        :param rec:  single rec eg obtained by nds[0] 
        :return str: formatted description
        """
        return "snode ix:%7d dh:%2d sx:%5d pt:%7d nc:%5d fc:%7d ns:%7d lv:%3d cp:%7d." % tuple(rec)

    @classmethod
    def Brief(cls, rec):
        return "snode ix:%7d dh:%2s nc:%5d lv:%3d." % (rec.index, rec.depth, rec.num_child, rec.lvid)

class stree(object):
    @classmethod
    def MakeTxtArray(cls, lines):
        maxlen = max(list(map(len, lines)))  
        ta = np.array( lines, dtype="|S%d" % maxlen )
        return ta  

    def __init__(self, f):
        sff = Fold.Load(f.base,"subs_freq",  symbol="sf") 
        sf = sfreq(sff)
        nds = snode.RecordsFromArrays(f.nds)
        soname_ = self.MakeTxtArray(f.soname.lines)

        self.sf = sf
        self.f = f 
        self.nds = nds 
        self.raw_subs = None
        self.soname_ = soname_

    def find_lvid(self, q_soname, starting=True):
        """
        Pedestrian way to find the a string in a list 
        """
        lines = self.f.soname.lines
        lvid = -1  
        for i,soname in enumerate(lines):
            match = soname.startswith(q_soname) if starting else soname == q_soname
            if match:
                lvid = i
                break
            pass
        pass 
        return lvid 

    def find_lvid_(self, q_soname_, starting=True ):
        """
        find array indices starting or exactly matching q_soname
        unlike the above this returns an array with all matching indices
        """
        q_soname = q_soname_.encode() if type(q_soname_) is str else q_soname_ 
        assert type(q_soname) is bytes
        if starting:
            ii = np.where( np.core.defchararray.find(self.soname_, q_soname) == 0  )[0]  
        else:
            ii = np.where( self.soname_ == q_soname )[0]
        pass
        return ii 

    def find_lvid_nodes(self, arg):
        """
        :param arg: integer lvid or string soname used to obtain lvid 
        :return: array of indices of all nodes with the specified lvid   
        """
        if type(arg) in [int, np.int32, np.int64, np.uint32, np.uint64]:
            lvid = int(arg)
        elif type(arg) is str or type(arg) is bytes:
            ii = self.find_lvid_(arg)
            assert len(ii) > 0 
            lvid = int(ii[0])
            if len(ii) > 1:
               log.warning("multiple lvid %s correspond to arg:%s using first %s " % (str(ii), arg, lvid))
            pass
        else:
            print("arg %s unhandled %s " % (arg, type(arg)))
            assert(0)
        pass
        return np.where( self.nds.lvid == lvid )[0] 


    def get_children(self, nidx):
        """
        This is a direct translation of the C++ stree::get_children 
        which will inevitably be slow in python. 
        HMM: how to do this in a more numpy way ?
        """
        children = []
        nd = self.nds[nidx]
        assert nd.index == nidx
        ch = nd.first_child
        while ch > -1:
           child = self.nds[ch] 
           assert child.parent == nd.index
           children.append(child.index)
           ch = child.next_sibling
        pass
        return children

    def get_progeny_r(self, nidx, progeny):
        """
        :param nidx: node index
        :param progeny: in/out list of node indices

        Recursively collects node indices of progeny beneath nidx, 
        not including nidx itself.  
        """
        children = self.get_children(nidx)
        progeny.extend(children)
        for nix in children:
            self.get_progeny_r(nix, progeny)
        pass

    def get_progeny(self, nidx):
        progeny = []
        self.get_progeny_r(nidx, progeny)
        return np.array(progeny)  

    def get_soname(self, nidx):
        lvid = self.get_lvid(nidx)
        return self.f.soname[lvid] if lvid > -1 and lvid < len(self.f.soname) else None ; 
 
    def get_sub(self, nidx):
        return self.f.subs[nidx] if nidx > -1 else None ; 
 
    def get_depth(self, nidx):
        return self.nds[nidx].depth if nidx > -1 else  -1 ; 
   
    def get_parent(self, nidx): 
        return self.nds[nidx].parent if nidx > -1 else  -1 ; 

    def get_lvid(self, nidx): 
        return self.nds[nidx].lvid if nidx > -1 else  -1 ; 

    def get_transform(self, nidx):
        return self.f.trs[nidx] if nidx > -1 and nidx < len(self.f.trs) else None

    def get_ancestors(self, nidx):
        ancestors = [] 
        parent = self.get_parent(nidx)
        while parent > -1:
            ancestors.append(parent) 
            parent = self.get_parent(parent)
        pass
        ancestors.reverse()
        return ancestors 

    def __repr__(self):
        return repr(self.f)
    def __str__(self):
        return str(self.f)

    @classmethod
    def DepthSpacer(cls, dep, depmax=15):
        fmt = "|S%d" % depmax
        spc = np.zeros( (depmax,), dtype=np.int8 )
        spc[:] = ord(" ")
        spc[dep] = ord("+")
        return spc.view(fmt)[0].decode()   

    def desc_node(self, nidx, brief=False):
        return self.desc_node_(nidx, self.sf, brief=brief)

    def desc_nodes(self, nodes, brief=False):
        return "\n".join([self.desc_node(nix, brief=brief) for nix in nodes])

    def desc_node_(self, nidx, sf, brief=False):

        nd = self.nds[nidx]
        dep = self.get_depth(nidx)
        spc = self.DepthSpacer(dep)
        ndd = snode.Brief(nd) if brief else snode.Desc(nd) 
        sub = self.get_sub(nidx)
        sfd = "" if sf is None else sf.desc_key(sub)
        son = self.get_soname(nidx)
        return " ".join([spc,ndd,sfd,son])

    def desc_nodes_(self, nodes, sf):
        return "\n".join( [self.desc_node_(nix, sf) for nix in nodes])

    def make_freq(self, nodes):
        assert type(nodes) is np.ndarray 
        if self.raw_subs is None:
            path = os.path.join(self.f.base, "subs.txt")
            self.raw_subs = np.loadtxt( path, dtype="|S32")
        pass
        ssub = self.raw_subs[nodes] 
        ssf = sfreq.CreateFromArray(ssub) 
        return ssf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    f = Fold.Load(symbol="f")

    st = stree(f)
    print(repr(st))

    nidx = os.environ.get("NIDX", 13)

    ancestors = st.get_ancestors(nidx)
    print("NIDX %d ancestors: %s " % (nidx, repr(ancestors)) )
    print("st.desc_nodes(ancestors)")
    print(st.desc_nodes(ancestors))

    so = os.environ.get("SO", "sBar")  
    lvs = st.find_lvid_(so)
    print("SO %s lvs %s" % (so, str(lvs))) 
    
    for lv in lvs:
        bb = st.find_lvid_nodes(lv)
        b = int(bb[0])
        print("lv:%d bb=st.find_lvid_nodes(lv)  bb:%s b:%s " % (lv, str(bb),b)) 

        anc = st.get_ancestors(b)   
        print("b:%d anc=st.get_ancestors(b) anc:%s " % (b, str(anc))) 

        print("st.desc_nodes(anc, brief=True))")
        print(st.desc_nodes(anc, brief=True))
        print("st.desc_nodes([b], brief=True))")
        print(st.desc_nodes([b], brief=True))
    pass
        


if 0:
    progeny = st.get_progeny(nidx)
    print("NIDX %d progeny: %s " % (nidx, repr(progeny)) )
    #print("st.desc_nodes(progeny)")
    #print(st.desc_nodes(progeny))

    psf = st.make_freq(progeny) 
    print("st.desc_nodes_(progeny, psf)")
    print(st.desc_nodes_(progeny, psf))

    np.set_printoptions(edgeitems=600)

    print("st.f.trs[progeny].reshape(-1,16)")
    print(st.f.trs[progeny].reshape(-1,16))    
