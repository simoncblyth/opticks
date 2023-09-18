#!/usr/bin/env python
"""
stree.py
==========

TODO: improve stree.py repr 

"""

import os, numpy as np, logging, builtins
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold
from opticks.sysrap.sfreq import sfreq 
from opticks.sysrap.OpticksCSG import CSG_ 


class STR(str):
    """STR inherits from str and changes the repr to provide the str : useful for interactive ipython""" 
    def __repr__(self):
        return str(self)


class sobject(object):
    @classmethod
    def Label(cls, spc=5, pfx=10):
        prefix = " " * pfx 
        spacer = " " * spc 
        return prefix + spacer.join(cls.FIELD)

    @classmethod
    def Fields(cls, bi=False):
        kls = cls.__name__
        for i, field in enumerate(cls.FIELD):
            setattr(cls, field, i)
            if bi:setattr(builtins, field, i)
        pass

    @classmethod
    def Type(cls):
        cls.Fields()
        kls = cls.__name__
        print("%s.Type()" % kls )
        for i, field in enumerate(cls.FIELD):
            name = cls.DTYPE[i][0] 
            fieldname = "%s.%s" % (kls, field)
            print(" %2d : %20s : %s " % (i, fieldname, name))
        pass
        print("%s.Label() : " % cls.Label() )

    @classmethod
    def RecordsFromArrays(cls, a):
        """
        :param a: ndarray
        :return: np.recarray
        """
        ra = np.core.records.fromarrays(a.T, dtype=cls.DTYPE )
        return ra 

    @classmethod
    def Doc(cls):
        assert len(cls.FIELD) == len(cls.DTYPE)
        lines = []
        for i in range(len(cls.FIELD)):
            line = "%2d : %2s : %15s : %s " % (i, cls.FIELD[i], cls.DTYPE[i][0], cls.DTYPE[i][1] )
            lines.append(line)
        pass
        return STR("\n".join(lines))




class sn(sobject):
    """
    sn.h CSG constituent node 
    """
    DTYPE = [
             ('typecode', '<i4'), 
             ('complement', '<i4'), 
             ('lvid', '<i4'), 
             ('xform', '<i4'), 
             ('param', '<i4'), 
             ('aabb', '<i4'), 
             ('parent', '<i4'), 
             ('sibdex', '<i4'), 
             ('num_child', '<i4'),
             ('first_child', '<i4'),
             ('next_sibling', '<i4'),
             ('index', '<i4'),
             ('depth', '<i4'),
             ('note',  '<i4'),
             ('label0', '<i4'),
             ('label1', '<i4'),
             ('label2', '<i4'),
             ('label3', '<i4'),
             ] 

    FIELD = "tc cm lv xf pa bb pr sx nc fc ns ix dp nt l0 l1 l2 l3".split()







class snd(sobject):
    """
    snd.hh CSG constituent node 
    """
    DTYPE = [
             ('index', '<i4'), 
             ('depth', '<i4'), 
             ('sibdex', '<i4'), 
             ('parent', '<i4'), 
             ('num_child', '<i4'), 
             ('first_child', '<i4'), 
             ('next_sibling', '<i4'), 
             ('lvid', '<i4'), 
             ('typecode', '<i4'),
             ('param', '<i4'),
             ('aabb', '<i4'),
             ('xform', '<i4'),
             ] 

    FIELD = "ix dp sx pt nc fc sx lv tc pm bb xf".split()


class snode(sobject):
    """
    Volume structure node 
    """
    DTYPE = [
             ('index', '<i4'), 
             ('depth', '<i4'), 
             ('sibdex', '<i4'), 
             ('parent', '<i4'), 
             ('num_child', '<i4'), 
             ('first_child', '<i4'), 
             ('next_sibling', '<i4'), 
             ('lvid', '<i4'), 
             ('copyno', '<i4'),
             ('sensor_id', '<i4'),
             ('sensor_index', '<i4'),
             ('repeat_index', '<i4'),
             ('repeat_ordinal', '<i4'),
             ('boundary', '<i4'),
             ('sensor_name', '<i4')
             ] 

    FIELD = "ix dp sx pt nc fc sx lv cp se sx ri ro bd sn".split()

    @classmethod
    def Desc(cls, rec):
        """
        :param rec:  single rec eg obtained by nds[0] 
        :return str: formatted description
        """
        return "snode ix:%7d dh:%2d sx:%5d pt:%7d nc:%5d fc:%7d ns:%7d lv:%3d cp:%7d se:%7d." % tuple(rec)

    @classmethod
    def Brief(cls, rec):
        return "snode ix:%7d dh:%2s nc:%5d lv:%3d se:%7d." % (rec.index, rec.depth, rec.num_child, rec.lvid, rec.sensor_id)


class sfactor(sobject):
    """
    Just handles the first four fields, not the |S32 digest 
    """
    DTYPE = [
              ( 'index', '<i4'),
              ( 'freq',  '<i4'),
              ( 'sensors',  '<i4'),
              ( 'subtree',  '<i4')
            ]
    FIELD = "ix fr se su".split()   


class stree(object):
    @classmethod
    def MakeTxtArray(cls, lines):
        maxlen = max(list(map(len, lines)))  
        ta = np.array( lines, dtype="|S%d" % maxlen )
        return ta  

    def __init__(self, f, symbol="st"):

        #sff = Fold.Load(f.base,"subs_freq",  symbol="sf") 
        sff = f.subs_freq

        sf = None if sff is None else sfreq(sff)
        nds = None if f.nds is None else snode.RecordsFromArrays(f.nds)
        rem = None if f.rem is None else snode.RecordsFromArrays(f.rem)
        csg = None if f.csg is None else snd.RecordsFromArrays(f.csg.node[:,:12])
        factor = None if f.factor is None else sfactor.RecordsFromArrays(f.factor[:,:4])
        #soname_ = None if len(f.soname.lines) == 0  else self.MakeTxtArray(f.soname.lines)
        soname_ = f.soname_names

        self.sf = sf
        self.f = f 
        self.symbol = symbol
        self.nds = nds 
        self.rem = rem 
        self.csg = csg 
        self.factor = factor 
        self.raw_subs = None
        self.soname_ = soname_

    def find_lvid(self, q_soname, starting=True):
        """
        Pedestrian way to find string in a list 
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
        :return: array of indices of all nodes with the specified lvid index or solid name  
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

    def find_lvid_node( self, arg, ordinal=0 ):
        nn = self.find_lvid_nodes(arg)
        return nn[ordinal] 


    def get_children(self, nidx, prepend_arg=False):
        """
        This is a direct translation of the C++ stree::get_children 
        which will inevitably be slow in python. 
        HMM: how to do this in a more numpy way ?
        """
        children = []
        if prepend_arg:
            children.append(nidx)
        pass
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
        sub = "skip-sub" # self.get_sub(nidx)
        sfd = "skip-sfd" # "" if sf is None else sf.desc_key(sub)
        son = "skip-son" # self.get_soname(nidx)
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


    def desc_boundary_stats(self):
        """
        Counts of volumes with each boundary  
        
        desc_boundary_stats
        u_bd, n_bd = np.unique( st.nds.boundary, return_counts=True  )
           0 :    0 :      1 : Galactic///Galactic 
           1 :    1 :      3 : Galactic///Rock 
           2 :    2 :      1 : Rock///Galactic 
           3 :    3 :      1 : Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air 
           4 :    4 :      1 : Rock///Rock 
           5 :    5 :      1 : Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air 
           6 :    6 :      1 : Air/Implicit_RINDEX_NoRINDEX_pExpHall_pPoolCover//Steel 
         ...
         108 :  108 :   4997 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum 
         109 :  109 :   4997 : Vacuum/HamamatsuR12860_PMT_20inch_dynode_plate_opsurface//Steel 
         110 :  110 :   4997 : Vacuum/HamamatsuR12860_PMT_20inch_outer_edge_opsurface//Steel 
         111 :  111 :   4997 : Vacuum/HamamatsuR12860_PMT_20inch_inner_edge_opsurface//Steel 


        """ 
        lines = []
        lines.append("desc_boundary_stats")
        st = self
        lines.append("u_bd, n_bd = np.unique( st.nds.boundary, return_counts=True  )") 
        u_bd, n_bd = np.unique( st.nds.boundary, return_counts=True ) 

        for i in range(len(u_bd)):
            u = u_bd[i]
            n = n_bd[i]
            line = " %3d : %4d : %6d : %s " % (i, u, n, st.f.standard.bd_names[u] )
            lines.append(line)
        pass
        return STR("\n".join(lines))


    def desc_remainder(self):
        lines = []
        lines.append("desc_remainder")
        st = self
        u_lv, n_lv = np.unique( st.rem.lvid, return_counts=True  )
        lines.append("u_lv, n_lv = np.unique( st.rem.lvid, return_counts=True  )") 

        for i in range(len(u_lv)):
            u = u_lv[i]
            n = n_lv[i]
            line = " %3d : %4d : %6d : %s " % (i, u, n, st.f.soname[u] )
            lines.append(line)
        pass
        return "\n".join(lines)

   
    def desc_csg(self, lvid=105):
        """
        In [8]: snd.Label(3,8), n[n[:,snd.lv] == 105]
        Out[8]: 
        ('        ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf',
         array([[483,   4,   0, 484,   0,  -1,  -1, 105, 105, 294, 294,  -1],
                [484,   3,   0, 486,   1, 483, 485, 105,  11,  -1,  -1,  -1],
                [485,   3,   1, 486,   0,  -1,  -1, 105, 103, 295, 295, 183],
                [486,   2,   0, 488,   2, 484, 487, 105,   1,  -1,  -1,  -1],
                [487,   2,   1, 488,   0,  -1,  -1, 105, 105, 296, 296, 184],
                [488,   1,   0, 495,   2, 486, 494, 105,   1,  -1,  -1,  -1],
                [489,   4,   0, 490,   0,  -1,  -1, 105, 105, 297, 297,  -1],
                [490,   3,   0, 492,   1, 489, 491, 105,  11,  -1,  -1,  -1],
                [491,   3,   1, 492,   0,  -1,  -1, 105, 103, 298, 298, 186],
                [492,   2,   0, 494,   2, 490, 493, 105,   1,  -1,  -1,  -1],
                [493,   2,   1, 494,   0,  -1,  -1, 105, 105, 299, 299, 187],
                [494,   1,   1, 495,   2, 492,  -1, 105,   1,  -1,  -1, 188],
                [495,   0,  -1,  -1,   2, 488,  -1, 105,   3,  -1,  -1,  -1]], dtype=int32))

        In [9]: st.f.soname[105]
        Out[9]: 'HamamatsuR12860Tail0x61b5500'
        """
        csg = self.get_csg(lvid)
        lines = []
        lines.append("desc_csg lvid:%d st.f.soname[%d]:%s " % (lvid,lvid,self.get_lv_soname(lvid)))
        lines.append(snd.Label(3,8))
        lines.append("%s" % repr(csg))
        return "\n".join(lines)

    def get_csg(self, lvid):
        st = self 
        n = st.f.csg.node
        return n[n[:,snd.lv] == lvid]

    def get_csg_typecode(self, lvid):
        n = self.get_csg(lvid)
        return n[:,snd.tc] 


    def get_numSolid(self):
        return 1+len(self.factor)

    def get_numPrim(self, ridx):
        return len(self.rem) if ridx == 0 else self.factor[ridx-1].subtree

    def get_lvid(self, ridx):
        """
        :param ridx:
        :return lvid: array of meshIdx 

        In [3]: ridx = 1 ; st.nds.lvid[st.nds.repeat_index == ridx][:st.factor[ridx-1].subtree]
        Out[3]: array([133, 131, 129, 130, 132], dtype=int32)

        In [4]: ridx = 2 ; st.nds.lvid[st.nds.repeat_index == ridx][:st.factor[ridx-1].subtree]
        Out[4]: array([128, 118, 119, 127, 126, 120, 125, 121, 122, 123, 124], dtype=int32)

        In [9]: st.nds.lvid[st.nds.repeat_index == 0].shape
        Out[9]: (3089,)
        """
        st = self 
        return st.nds.lvid[st.nds.repeat_index == ridx] if ridx == 0 else st.nds.lvid[st.nds.repeat_index == ridx][:st.factor[ridx-1].subtree]

    def get_lv_soname(self, lv):
        return self.soname_[lv].decode("utf8")

    def descSolid(self, ridx, detail=False):
        """
        cf with CSGFoundry.descSolid
        """
        numPrim = self.get_numPrim(ridx)
        lvid = self.get_lvid(ridx) 
        u_lvid, n_lvid = np.unique(lvid, return_counts=True )
        n_lvid_one = np.all( n_lvid == 1 )
        p_lvid = lvid if n_lvid_one else u_lvid  # present in original order when not "unique" summarizing 
        pass
        lines = []
        lines.append("stree.descSolid ridx %3d numPrim %5d lvid %s n_lvid_one %d" % (ridx, numPrim, str(lvid), n_lvid_one)) 
        if detail:
            lines.append("")
            for pass_ in [0,1]:
                for i in range(len(p_lvid)):
                    ulv = p_lvid[i] 
                    nlv = n_lvid[i] 
                    lvn = self.get_lv_soname(ulv)
                    csg = self.get_csg(ulv)
                    tc  = self.get_csg_typecode(ulv)
                    assert len(csg) == len(tc)
                    tcn = " ".join(list(map(lambda _:"%d:%s"%(_,CSG_.desc(_)), tc))) 
                    if pass_ == 1:
                        lines.append("")
                    pass
                    lines.append(" lv:%3d nlv:%2d %50s csg %2d tcn %s " % (ulv, nlv, lvn, len(csg), tcn  ))  
                    if pass_ == 1:
                        lines.append(self.desc_csg(ulv))
                    pass  
                pass
            pass
        pass
        return "\n".join(lines)

    def descSolids(self, detail=False):
        lines = []
        numSolid = self.get_numSolid()
        lines.append("stree.descSolids numSolid:%d detail:%d " % (numSolid,detail) )

        q_ridx = int(os.environ.get("RIDX","-1")) 
        for ridx in range(numSolid):
            if q_ridx > -1 and ridx != q_ridx: continue
            lines.append(self.descSolid(ridx, detail=detail))
            if detail:
                lines.append("")
            pass
        pass
        return "\n".join(lines)




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
