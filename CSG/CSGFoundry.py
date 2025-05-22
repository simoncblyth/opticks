#!/usr/bin/env python
import os, sys, re,  numpy as np, logging, datetime
from configparser import ConfigParser
from collections import OrderedDict
log = logging.getLogger(__name__)

#from opticks.ana.key import keydir
from opticks.ana.fold import Fold, STR
from opticks.sysrap.OpticksCSG import CSG_


class BoundaryNameConfig(object):
    """
    Parses ini file of the below form::

        [key_boundary_regexp]
        red = (?P<Water_Virtuals>Water/.*/Water$)
        blue = Water///Acrylic$
        magenta = Water/Implicit_RINDEX_NoRINDEX_pOuterWaterInCD_pInnerReflector//Tyvek$
        yellow = DeadWater/Implicit_RINDEX_NoRINDEX_pDeadWater_pTyvekFilm//Tyvek$
        pink = Air/CDTyvekSurface//Tyvek$
        cyan = Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterInCD_pCentralDetector/Water$
        orange = Water/Steel_surface//Steel$
        grey =
        # empty value is special cased to mean all other boundary names

    Defaults path is $HOME/.opticks/GEOM/cxt_min.ini
    """

    @classmethod
    def Parse(cls, _path, _section ):
        cp = cls(_path, _section)
        return cp.bdict

    def __init__(self, _path, _section ):
        path = os.path.expandvars(_path)
        cfg = ConfigParser()
        cfg.read(path)
        sect = cfg[_section]
        bdict = OrderedDict(sect)

        self._path = _path
        self.path = path
        self._section = _section
        self.cfg = cfg
        self.bdict = bdict



class BoundaryTable(object):
    def __init__(self, bn, bdn, d_qbn={}, KEY=None):
        """
        :param bn: large array of boundary indices
        :param bdn: small array of boundary names
        :param d_qbn: dict keyed on colors with arrays of boundary indices as values
        """
        u, x, n = np.unique(bn, return_index=True, return_counts=True)

        # form array with the keys that each boundary index is grouped into
        akk = np.repeat("????????????????", len(u))
        for i in range(len(u)):
            kk = []
            for k, qbn in d_qbn.items():
                if len(np.where(np.isin(qbn, u[i]))[0]) == 1:
                    kk.append(k)
                pass
            pass
            akk[i] = ",".join(kk)
        pass
        o = np.argsort(n)[::-1]
        b = bdn[u]

        _tab = "np.c_[akk, u, n, x, b][o]"
        tab = eval(_tab)

        self.akk = akk[o]
        self.u = u[o]
        self.x = x[o]
        self.n = n[o]
        self.b = b[o]

        self.KEY = KEY

        self.d_qbn = d_qbn
        self._tab = _tab
        self.tab = tab

    def __repr__(self):
        """
        """
        vfmt = " %12s : %4d %8d %8d : %s "
        sfmt = " %12s : %4s %8s %8s : %s "
        tfmt = " %12s : %4s %8d %8s : %s "

        label = sfmt % ( ".akk", ".u", ".n", ".x", ".b" )
        lines = []
        lines.append("[cf.btab KEY:%s" % self.KEY )
        lines.append(label)
        ntot = 0

        if not self.KEY is None:
            KK = np.array(self.KEY.split(","))
            sel = np.where(np.isin(self.tab[:,0], KK))[0]
        else:
            sel = slice(None)
        pass

        for row in self.tab[sel]:
            akk = row[0]
            u = int(row[1])
            n = int(row[2])
            x = int(row[3])
            b = row[4]

            line = vfmt % ( akk, u, n, x, b )
            ntot += n
            lines.append(line)
        pass
        lines.append(label)
        lines.append(tfmt % ("", "",ntot, "",".ntot"))
        lines.append("]cf.btab")
        self.ntot = ntot
        return "\n".join(lines)






class CXT_Boundary(object):
    def __init__(self, bn, d_qbn, d_anno, cf_bdn):
        """
        :param bn: large array of boundary indices
        :param d_qbn: dict keyed on colors with boundary indices array values
        :param d_anno: dict keyed on colors with label values
        :param cf_bdn: small array of all boundary names
        """
        self.bn = bn
        self.d_qbn = d_qbn
        self.d_anno = d_anno
        self.cf_bdn = cf_bdn

        wdict = {}
        for k,qbn in self.d_qbn.items():
            _w = np.isin( bn, qbn )   # bool array indicating which elem of bn are in the qbn array
            w = np.where(_w)[0]       # indices of bn array with boundaries matching the regexp
            wdict[k] = w
        pass
        self.wdict = wdict

    def __repr__(self):
        lines = []
        lines.append("[CXT_Boundary___________________________________")

        for k,qbn in self.d_qbn.items():
            _w = np.isin( self.bn, qbn )   # bool array indicating which elem of bn are in the qbn array
            w = np.where(_w)[0]       # indices of bn array with boundaries matching the regexp
            label = self.d_anno[k]
            bb = self.cf_bdn[qbn]
            line = " %10s %100s nbb:%4d %s " % (k, label, len(bb), str(qbn[:10]))
            lines.append(line)
        pass
        lines.append("]CXT_Boundary___________________________________")
        return "\n".join(lines)






class CSGObject(object):
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



class CSGPrim(CSGObject):
    DTYPE = [
             ('numNode', '<i4'),
             ('nodeOffset', '<i4'),
             ('tranOffset', '<i4'),
             ('planOffset', '<i4'),
             ('sbtIndexOffset', '<i4'),
             ('meshIdx', '<i4'),
             ('repeatIdx', '<i4'),
             ('primIdx', '<i4'),
             ]

    EXTRA = [
             ('BBMin_x', '<f4'),
             ('BBMin_y', '<f4'),
             ('BBMin_z', '<f4'),
             ('BBMax_x', '<f4'),
             ('BBMax_y', '<f4'),
             ('BBMax_z', '<f4'),
             ('spare32', '<f4'),
             ('spare33', '<f4'),
             ]

    FIELD = "nn no to po sb lv ri pi".split()
    XFIELD = "nx ny nz mx my mz s2 s3".split()



class CSGNode(CSGObject):
    DTYPE = [
             ('p0', '<f4'),
             ('p1', '<f4'),
             ('p2', '<f4'),
             ('p3', '<f4'),
             ('p4', '<f4'),
             ('p5', '<f4'),
             ('boundary', '<i4'),
             ('index', '<i4'),
             ('BBMin_x', '<f4'),
             ('BBMin_y', '<f4'),
             ('BBMin_z', '<f4'),
             ('BBMax_x', '<f4'),
             ('BBMax_y', '<f4'),
             ('BBMax_z', '<f4'),
             ('typecode','<i4'),
             ('comptran','<i4'),
             ]
    FIELD = "p0 p1 p2 p3 p4 p5 bn ix nx ny nz mx my mz tc ct".split()




class MM(object):
    """
    The input file is now a standard resident of the CSGFoundry directory
    Note that the input file was formerly created using::

       ggeo.py --mmtrim > $TMP/mm.txt

    """
    PTN = re.compile("\\d+")
    def __init__(self, path ):
        mm = os.path.expandvars(path)
        mm = open(mm, "r").read().splitlines() if os.path.exists(mm) else None
        self.mm = mm
        if mm is None:
            log.fatal("missing %s, which is now a standard part of CSGFoundry " % path  )
            sys.exit(1)
        pass

    def imm(self, emm):
        return list(map(int, self.PTN.findall(emm)))

    def label(self, emm):
        imm = self.imm(emm)
        labs = [self.mm[i] for i in imm]
        lab = " ".join(labs)

        tilde = emm[0] == "t" or emm[0] == "~"
        pfx = (  "NOT: " if tilde else "     " )

        if emm == "~0" or emm == "t0":
            return "ALL"
        elif imm == [1,2,3,4]:
            return "ONLY PMT"
        elif "," in emm:
            return ( "EXCL: " if tilde else "ONLY: " ) +  lab
        else:
            return lab
        pass

    def __repr__(self):
        return STR("\n".join(self.mm))


class LV(object):
    PTN = re.compile("\\d+")
    def __init__(self, path):
        lv = os.path.expandvars(path)
        lv = open(lv, "r").read().splitlines() if os.path.exists(lv) else None
        self.lv = lv
        if lv is None:
            log.fatal("missing %s, which is now a standard part of CSGFoundry " % path  )
            sys.exit(1)
        pass

    def ilv(self, elv):
        return list(map(int, self.PTN.findall(elv)))

    def label(self, elv):
        ilv = self.ilv(elv)
        mns = [self.lv[i] for i in ilv]
        mn = " ".join(mns)
        tilde = elv[0] == "t"
        lab = ""
        if elv == "t":
            lab = "ALL"
        else:
            lab = ( "EXCL: " if tilde else "ONLY: " ) + mn
        pass
        return lab

    def __str__(self):
        return "\n".join(self.lv)

    def __repr__(self):
        return "\n".join(["%3d:%s " % (i, self.lv[i]) for i in range(len(self.lv))])



class Deprecated_NPFold(object):
    """
    HMM opticks.ana.fold.Fold looks more developed than this
    TODO: eliminate all use of this
    """
    INDEX = "NPFold_index.txt"

    @classmethod
    def IndexPath(cls, fold):
        return os.path.join(fold, cls.INDEX)

    @classmethod
    def HasIndex(cls, fold):
        return os.path.exists(cls.IndexPath(fold))

    @classmethod
    def Load(cls, *args, **kwa):
        return cls(*args, **kwa)

    def __init__(self, *args, **kwa):
        fold = os.path.join(*args)
        self.fold = fold
        self.has_index = self.HasIndex(fold)

        if self.has_index:
            self.load_idx(fold)
        else:
            self.load_fts(fold)
        pass

    def load_fts(self, fold):
        assert 0

    def load_idx(self, fold):
        keys = open(self.IndexPath(fold),"r").read().splitlines()
        aa = []
        kk = []
        ff = []
        subfold = []

        for k in keys:
            path = os.path.join(fold, k)
            if k.endswith(".npy"):
                assert os.path.exists(path)
                log.info("loading path %s k %s " % (path, k))
                a = np.load(path)
                aa.append(a)
                kk.append(k)
            else:
                log.info("skip non .npy path %s k %s " % (path, k))
            pass
        pass
        self.kk = kk
        self.aa = aa
        self.ff = ff
        self.subfold = subfold
        self.keys = keys


    def find(self, k):
        return self.kk.index(k) if k in self.kk else -1

    def has_key(self, k):
        return self.find(k) > -1

    def __repr__(self):
        lines = []
        lines.append("NPFold %s " % self.fold )
        for i in range(len(self.kk)):
            k = self.kk[i]
            a = self.aa[i]
            stem, ext = os.path.splitext(os.path.basename(k))
            path = os.path.join(self.fold, k)
            lines.append("%10s : %20s : %s " % (stem, str(a.shape), k ))
        pass
        return "\n".join(lines)


class Deprecated_SSim(object):
    """
    HMM: this adds little on top of ana.fold.Fold
    TODO: get rid of it
    """
    BND = "bnd.npy"


    @classmethod
    def Load(cls, simbase):
        ssdir = os.path.join(simbase, "SSim")
        log.info("SSim.Load simbase %s ssdir %s " % (simbase,ssdir))
        sim = Fold.Load(ssdir) if os.path.isdir(ssdir) else None
        return sim

    def __init__(self, fold):
        if self.has_key(self.BND):
            bnpath = os.path.join(fold, "bnd_names.txt")
            assert os.path.exists(bnpath)
            bnd_names = open(bnpath,"r").read().splitlines()
            self.bnd_names = bnd_names
        pass
        if hasattr(self, 'bnd_names'):  # names list from NP bnd.names metadata
             bndnamedict = SSim.namelist_to_namedict(self.bnd_names)
        else:
             bndnamedict = {}
        pass
        self.bndnamedict = bndnamedict
        pass



class CSGFoundry(object):
    FOLD = os.path.expandvars("$TMP/CSG_GGeo/CSGFoundry")
    FMT = "   %10s : %20s  : %s "


    @classmethod
    def namelist_to_namedict(cls, namelist):
        nd = {}
        if not namelist is None:
            nd = dict(zip(range(len(namelist)),list(map(str,namelist))))
        pass
        return nd


    @classmethod
    def CFBase(cls):
        """
        Precedence order for which envvar is used to derive the CFBase folder
        This allows scripts such as CSGOptiX/cxsim.sh to set envvars to control
        the geometry arrays loaded. For example this is used such that local
        analysis on laptop can use the geometry rsynced from remote.

        1. CFBASE_LOCAL
        2. CFBASE
        3. GEOM
        4. OPTICKS_KEY

        """
        if "CFBASE_LOCAL" in os.environ:
            cfbase= os.environ["CFBASE_LOCAL"]
            note = "via CFBASE_LOCAL"
        elif "CFBASE" in os.environ:
            cfbase= os.environ["CFBASE"]
            note = "via CFBASE"
        elif "GEOM" in os.environ:
            KEY = os.path.expandvars("${GEOM}_CFBaseFromGEOM")
            if KEY in os.environ:
                cfbase = os.environ[KEY]
                note = "via GEOM,%s" % KEY
            else:
                cfbase = os.path.expandvars("$HOME/.opticks/GEOM/$GEOM")
                note = "via GEOM"
            pass
        else:
            cfbase = None
            note = "via NO envvars"
        pass
        if cfbase is None:
            print("CSGFoundry.CFBase returning None, note:%s " % note )
        else:
            print("CSGFoundry.CFBase returning [%s], note:[%s] " % (cfbase,note) )
        pass
        return cfbase

    @classmethod
    def Load(cls, cfbase_=None, symbol=None):
        """
        :param cfbase_: typically None, but available when debugging eg comparing two geometries
        """
        if cfbase_ is None:
            cfbase = cls.CFBase()
        else:
            cfbase = os.path.expandvars(cfbase_)
        pass
        log.info("cfbase:%s " % cfbase)
        if cfbase is None or not os.path.exists(os.path.join(cfbase, "CSGFoundry")):
            print("ERROR CSGFoundry.CFBase returned None OR non-existing CSGFoundry dir so cannot CSGFoundry.Load" )
            return None
        pass
        assert not cfbase is None
        cf = cls(fold=os.path.join(cfbase, "CSGFoundry"))
        cf.symbol = symbol
        return cf

    @classmethod
    def FindDirUpTree(cls, origpath, name="CSGFoundry"):
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
    def FindDigest(cls, path):
        if path.find("*") > -1:  # eg with a glob pattern filename element
            base = os.path.dirname(path)
        else:
            base = path
        pass
        base = os.path.expandvars(base)
        return cls.FindDigest_(base)

    @classmethod
    def FindDigest_(cls, path):
        """
        :param path: Directory path in which to look for a 32 character digest path element
        :return 32 character digest or None:
        """
        hexchar = "0123456789abcdef"
        digest = None
        for elem in path.split("/"):
            if len(elem) == 32 and set(elem).issubset(hexchar):
                digest = elem
            pass
        pass
        return digest

    def __init__(self, fold):
        self.load(fold)
        self.meshnamedict = self.namelist_to_namedict(self.meshname)
        self.primIdx_meshname_dict = self.make_primIdx_meshname_dict()

        self.mokname = "zero one two three four five six seven eight nine".split()
        self.moknamedict = self.namelist_to_namedict(self.mokname)
        self.insnamedict = {}

        self.lv = LV(os.path.join(fold, "meshname.txt"))
        self.mm = MM(os.path.join(fold, "mmlabel.txt"))

        sim = Fold.Load(fold, "SSim")

        try:
            bdn = sim.stree.standard.bnd_names
        except AttributeError:
            bdn = None
        pass
        if bdn is None: log.fatal("CSGFoundry fail to access sim.stree.standard.bnd_names : geometry incomplete" )
        if type(bdn) is np.ndarray: sim.bndnamedict = self.namelist_to_namedict(bdn)
        pass
        self.bdn_config = BoundaryNameConfig.Parse("$HOME/.opticks/GEOM/cxt_min.ini", "key_boundary_regexp")
        self.bdn = bdn
        self.sim = sim

        self.npa = self.node.reshape(-1,16)[:,0:6]
        self.nbd = self.node.view(np.int32)[:,1,2]
        self.nix = self.node.view(np.int32)[:,1,3]
        self.nbb = self.node.reshape(-1,16)[:,8:14]
        self.ntc = self.node.view(np.int32)[:,3,2]
        self.ncm = self.node.view(np.uint32)[:,3,3] >> 31  # node complement
        self.ntr = self.node.view(np.uint32)[:,3,3] & 0x7fffffff  # node tranIdx+1

        self.pnn = self.prim.view(np.int32)[:,0,0] # prim num node
        self.pno = self.prim.view(np.int32)[:,0,1] # prim node offset



        self.pto = self.prim.view(np.int32)[:,0,2] # prim tran offset
        self.ppo = self.prim.view(np.int32)[:,0,3] # prim plan offset

        self.crn = self.pno[self.pnn>1]   # node indices of compound roots
        self.crn_subnum = self.node.view(np.int32)[self.crn,0,0]
        self.crn_suboff = self.node.view(np.int32)[self.crn,0,1]


        self.psb = self.prim.view(np.int32)[:,1,0] # prim sbtIndexOffset
        self.plv = self.prim.view(np.int32)[:,1,1] # prim lvid/meshIdx
        self.prx = self.prim.view(np.int32)[:,1,2] # prim ridx/repeatIdx
        self.pix = self.prim.view(np.int32)[:,1,3] # prim idx

        self.pbb = self.prim.reshape(-1,16)[:,8:14]


        self.snp = self.solid[:,1,0].view(np.int32) # solid numPrim
        self.spo = self.solid[:,1,1].view(np.int32) # solid primOffset
        self.sce = self.solid[:,2].view(np.float32)


    def boundary_table(self, bn, d_qbn={}, KEY=None):
        """
        :param bn: array of boundary indices, typically millions of them
        :param d_qbn: dict of color keys with values that are arrays of boundary indices
        :return bn_tab: string table showing unique boundary indices, names, counts and first indices into the bn array
        """
        btab = BoundaryTable(bn, self.bdn, d_qbn, KEY)
        self.btab = btab
        return btab


    def simtrace_boundary_analysis(self, bn, KEY=os.environ.get("KEY", None) ):
        """
        Group simtrace intersects according to their boundary indices
        using groups specified by regexp matching boundary names.

        Typically would have a few hundred boundary names
        in cf.bdn but potentially millions of simtrace intersects
        in the bn array.

        :param bn: large array of boundary indices
        :param KEY: color key OR None
        :return cxtb,btab:
        """
        d_qbn, d_anno = self.dict_find_boundary_indices_re_match(self.bdn_config)
        cxtb = CXT_Boundary(bn, d_qbn, d_anno, self.bdn)
        btab = self.boundary_table(bn, d_qbn, KEY)
        self.cxtb = cxtb
        self.btab = btab
        return cxtb, btab

    def dict_find_boundary_indices_re_match(self, _bndptn_dict):
        """
        :param _bndptn_dict: label keys, regexp pattern string values
        :return d: dict with same label keys and arrays of matching cf.bdn boundary indices

        Boundaries unmatched by the provided regexp values are
        grouped within the key with a blank value if provided.
        """
        d = {}
        anno = {}
        matched = np.array([], dtype=np.int64 )
        unmatched_k = None
        for k, v in _bndptn_dict.items():
            if v == '':
                unmatched_k = k
                continue
            pass
            qbn, bb, label = self.find_boundary_indices_re_match(v)
            matched = np.concatenate( (matched, qbn) )
            d[k] = qbn
            anno[k] = label
        pass

        c = dict(matched=matched, all=np.arange(len(self.bdn)))
        c['unmatched'] = np.where(np.isin(c['all'], c['matched'], invert=True ))[0]
        if not unmatched_k is None:
            d[unmatched_k] = c['unmatched']
            anno[unmatched_k] = "OTHER"
        pass
        return d, anno


    def find_boundary_indices_re_match(self, _ptn):
        """
        :param _ptn: boundary regexp string
        :return qbn,bb,label:

        qbn
           array of cf.bdn boundary indices that match the pattern
        bb
           array of boundary names matching the pattern
        label
           match key when the pattern string includes one eg (?<label>.*$)
           otherwise the pattern string itself


        Typically would have a few hundred boundary names for a geometry
        so the slowness of np.vectorize is not a problem.

        Example _ptn::

            .*
            .*/Vacuum$
            .*/LatticedShellSteel$
            .*/Steel$
            Water/.*/Water$
            (?P<Water_Virtuals>:Water/.*/Water$)
            .*/Tyvek$

        """
        ptn = re.compile(_ptn)
        _re_match = lambda s:bool(ptn.match(s))
        re_match = np.vectorize(_re_match)

        def _re_key(s):
            keys = list(ptn.match(s).groupdict().keys())
            return keys[0] if len(keys) == 1 else _ptn
        pass
        re_key = np.vectorize(_re_key)

        _qbn = re_match(self.bdn)   # boundary array bool match or not array
        qbn = np.where(_qbn)[0]     # indices of boundary array matching the regexp
        bb = self.bdn[qbn]

        if len(qbn) == 0:
            log.info("_ptn:[%s] did not match any boundaries" % _ptn )
            label = "NO_MATCH"
        else:
            _kk = re_key(bb)
            kk = np.unique(_kk)
            assert len(kk) == 1
            label = kk[0]
        pass
        return qbn, bb, label


    def find_primIdx_from_nodeIdx(self, nodeIdx_):
        """
        If nodeIdx is valid there should always be exactly
        one prim in which it appears.

        For example use to lookup primname that contain a selection
        of nodeIdx::

            In [4]: np.c_[b.primname[np.unique(a.find_primIdx_from_nodeIdx(w))]]
            Out[4]:
            array([['NNVTMCPPMTsMask_virtual'],
                   ['HamamatsuR12860sMask_virtual'],
                   ['HamamatsuR12860_PMT_20inch_pmt_solid_1_4'],
                   ['HamamatsuR12860_PMT_20inch_inner_solid_1_4'],
                   ['base_steel'],
                   ['uni_acrylic1']], dtype=object)


        """
        a = self

        numNode = a.prim[:,0,0].view(np.int32)
        nodeOffset = a.prim[:,0,1].view(np.int32)

        primIdx = np.zeros(len(nodeIdx_), dtype=np.int32)
        for i, nodeIdx in enumerate(nodeIdx_):
            primIdx[i]  = np.where(np.logical_and( nodeIdx >= nodeOffset, nodeIdx < nodeOffset+numNode ))[0]
        pass
        return primIdx


    def find_lvid_from_nodeIdx(self, nodeIdx_):
        """
        """
        primIdx = self.find_primIdx_from_nodeIdx(nodeIdx_)
        return self.prim[primIdx].view(np.int32)[:,1,1]

    def find_lvname_from_nodeIdx(self, nodeIdx_):
        lvid = self.find_lvid_from_nodeIdx(nodeIdx_)
        return self.meshname[lvid]



    def meshIdx(self, primIdx):
        """
        Lookup the midx of primIdx prim

        :param primIdx:
        :return midx:
        """
        assert primIdx < len(self.prim)
        midx = self.prim[primIdx].view(np.uint32)[1,1]
        return midx

    def make_primIdx_meshname_dict(self):
        """
        See notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst
        this method resolved an early naming bug

        CSG/CSGPrim.h::

             95     PRIM_METHOD unsigned  meshIdx() const {           return q1.u.y ; }  // aka lvIdx
             96     PRIM_METHOD void   setMeshIdx(unsigned midx){     q1.u.y = midx ; }

        """
        d = {}
        for primIdx in range(len(self.prim)):
            midx = self.meshIdx (primIdx)      # meshIdx method with contiguous primIdx argument
            assert midx < len(self.meshname)
            mnam = self.meshname[midx]
            d[primIdx] = mnam
            #print("CSGFoundry:primIdx_meshname_dict primIdx %5d midx %5d meshname %s " % (primIdx, midx, mnam))
        pass
        return d

    def loadtxt(self, path):
        """
        When the path contains only a single line using dtype np.object or |S100
        both result in an object with no length::

            array('solidXJfixture', dtype=object)

        """
        # both these do not yield an array when only a single line in the file
        #
        #a_txt = np.loadtxt(path, dtype=np.object)   # yields single object
        #a_txt = np.loadtxt(path, dtype="|S100")

        txt = open(path).read().splitlines()
        a_txt = np.array(txt, dtype=np.str_ )   # formerly np.object
        return a_txt


    def brief(self):
        symbol = getattr(self, "symbol", "no-symbol")
        return "CSGFoundry %s cfbase %s " % (symbol, self.cfbase)

    def load(self, fold):
        cfbase = os.path.dirname(fold)
        self.cfbase = cfbase
        self.base = cfbase
        log.info("load : fold %s cfbase: %s  " % (fold, cfbase) )

        if not os.path.isdir(fold):
            log.fatal("CSGFoundry folder %s does not exist " % fold)
            log.fatal("create foundry folder from OPTICKS_KEY geocache with CSG_GGeo/run.sh " )
            assert 0
        pass

        names = os.listdir(fold)
        stems = []
        stamps = []
        for name in filter(lambda name:name.endswith(".npy") or name.endswith(".txt"), names):
            path = os.path.join(fold, name)
            st = os.stat(path)
            stamp = datetime.datetime.fromtimestamp(st.st_ctime)
            stamps.append(stamp)
            stem = name[:-4]
            a = np.load(path) if name.endswith(".npy") else self.loadtxt(path)
            setattr(self, stem, a)
            stems.append(stem)
        pass

        min_stamp = min(stamps)
        max_stamp = max(stamps)
        now_stamp = datetime.datetime.now()
        dif_stamp = max_stamp - min_stamp
        age_stamp = now_stamp - max_stamp

        #print("min_stamp:%s" % min_stamp)
        #print("max_stamp:%s" % max_stamp)
        #print("dif_stamp:%s" % dif_stamp)
        #print("age_stamp:%s" % age_stamp)

        assert dif_stamp.microseconds < 1e6, "stamp divergence detected microseconds %d : so are seeing mixed up results from multiple runs " % dif_stamp.microseconds

        self.stems = stems
        self.min_stamp = min_stamp
        self.max_stamp = max_stamp
        self.age_stamp = age_stamp
        self.stamps = stamps
        self.fold = fold
        self.pr = CSGPrim.RecordsFromArrays(self.prim[:,:2].reshape(-1,8).view(np.int32))
        self.nf = CSGNode.RecordsFromArrays(self.node.reshape(-1,16).view(np.float32))
        self.ni = CSGNode.RecordsFromArrays(self.node.reshape(-1,16).view(np.int32))


    def desc(self, stem):
        a = getattr(self, stem)
        ext = ".txt" if a.dtype == 'O' else ".npy"
        pstem = "bnd" if stem == "bndname" else stem
        path = os.path.join(self.fold, "%s%s" % (pstem, ext))
        return self.FMT % (stem, str(a.shape), path)

    def head(self):
        return "\n".join(map(str, [self.fold, "min_stamp:%s" % self.min_stamp, "max_stamp:%s" % self.max_stamp, "age_stamp:%s" % self.age_stamp]))

    def body(self):
        return "\n".join(map(lambda stem:self.desc(stem),self.stems))

    def __repr__(self):
        return "\n".join([self.head(),self.body()])

    def dump_node_boundary(self):
        logging.info("dump_node_boundary")
        node = self.node

        node_boundary = node.view(np.uint32)[:,1,2]
        ubs, ubs_count = np.unique(node_boundary, return_counts=True)

        for ub, ub_count in zip(ubs, ubs_count):
            bn = self.bdn[ub] if not self.bdn is None else "-"
            print(" %4d : %6d : %s " % (ub, ub_count, bn))
        pass

    def descSolid(self, ridx, detail=True):
        """
        After CSGFoundry::dumpSolid
        """
        label = self.solid[ridx,0,:4].copy().view("|S16")[0].decode("utf8")
        numPrim = self.solid[ridx,1,0]
        primOffset = self.solid[ridx,1,1]
        prs = self.prim[primOffset:primOffset+numPrim]
        iprs = prs.view(np.int32)

        lvid = iprs[:,1,1]
        u_lvid, x_lvid, n_lvid = np.unique( lvid, return_index=True, return_counts = True )
        lv_one = np.all( n_lvid == 1 )

        lines = []
        lines.append("CSGFoundry.descSolid ridx %2d label %16s numPrim %6d primOffset %6d lv_one %d " % (ridx,label, numPrim, primOffset, lv_one))

        if detail:
            if lv_one:
                for pidx in range(primOffset, primOffset+numPrim):
                    lines.append(self.descPrim(pidx))
                pass
            else:
                x_order = np.argsort(x_lvid)  # order by prim index of first occurence of each lv
                for i in x_order:
                    ulv = u_lvid[i]
                    nlv = n_lvid[i]
                    xlv = x_lvid[i]
                    pidx = primOffset + xlv   # index of the 1st prim with each lv
                    lines.append(" i %3d ulv %3d xlv %4d nlv %3d : %s " % (i, ulv, xlv, nlv, self.descPrim(pidx) ))
                pass
            pass
        pass
        return "\n".join(lines)

    def descLV(self, lvid, detail=False):
        lv = self.prim.view(np.int32)[:,1,1]
        pidxs = np.where(lv==lvid)[0]

        lines = []
        lines.append("descLV lvid:%d meshname:%s pidxs:%s" % (lvid, self.meshname[lvid],str(pidxs)))
        for pidx in pidxs:
            lines.append(self.descPrim(pidx, detail=detail))
        pass
        return STR("\n".join(lines))

    def descLVDetail(self, lvid):
        return self.descLV(lvid, detail=True)

    def descPrim(self, pidx, detail=False):

        pr = self.prim[pidx]
        ipr = pr.view(np.int32)

        numNode    = ipr[0,0]
        nodeOffset = ipr[0,1]
        repeatIdx  = ipr[1,2]
        primIdxLocal = ipr[1,3]

        lvid = ipr[1,1]
        lvn = self.meshname[lvid]
        tcn = self.descNodeTC(nodeOffset, numNode)

        bnd_ = self.node[nodeOffset:nodeOffset+numNode,1,2].view(np.int32)
        ubnd_ = np.unique(bnd_)
        assert len(ubnd_) == 1, "all nodes of prim should have same boundary "
        ubnd = ubnd_[0]
        line = "pidx %4d lv %3d pxl %4d : %50s : %s : bnd %s : %s " % (pidx, lvid, primIdxLocal, lvn, tcn, ubnd, self.bdn[ubnd] )

        lines = []
        lines.append(line)
        if detail:
            lines.append(self.descNodeParam(nodeOffset,numNode))
            lines.append(self.descNodeBB(nodeOffset,numNode))
            lines.append(self.descNodeBoundaryIndex(nodeOffset,numNode))
            lines.append(self.descNodeTCTran(nodeOffset,numNode, mask_signbit=False))
            lines.append(self.descNodeTCTran(nodeOffset,numNode, mask_signbit=True))
        pass
        return STR("\n".join(lines))

    def descPrimDetail(self, pidx):
        return self.descPrim(pidx,detail=True)

    def descNodeFloat(self, nodeOffset, numNode, sli="[:,:6]", label=""):
        symbol = self.symbol
        locals()[symbol] = self
        expr = "%(symbol)s.node[%(nodeOffset)d:%(nodeOffset)d+%(numNode)d].reshape(-1,16)"
        expr += sli
        expr = expr % locals()
        lines = []
        lines.append("%s # %s " % (expr,label))
        lines.append(str(eval(expr)))
        return STR("\n".join(lines))

    def descNodeParam(self, nodeOffset, numNode):
        return self.descNodeFloat(nodeOffset,numNode,"[:,:6]", "descNodeParam" )
    def descNodeBB(self, nodeOffset, numNode):
        return self.descNodeFloat(nodeOffset,numNode,"[:,8:14]", "descNodeBB" )

    def descNodeInt(self, nodeOffset, numNode, sli="[:,6:8]", label="", mask_signbit=False):
        symbol = self.symbol
        locals()[symbol] = self
        expr = "%(symbol)s.node[%(nodeOffset)d:%(nodeOffset)d+%(numNode)d].reshape(-1,16).view(np.int32)"
        expr += sli
        if mask_signbit:
            expr += " & 0x7ffffff "
        pass
        expr = expr % locals()
        lines = []
        lines.append("%s # %s " % (expr,label))
        lines.append(str(eval(expr)))
        return STR("\n".join(lines))

    def descNodeBoundaryIndex(self, nodeOffset, numNode):
        return self.descNodeInt(nodeOffset,numNode,"[:,6:8]", "descNodeBoundaryIndex" )
    def descNodeTCTran(self, nodeOffset, numNode, mask_signbit=False):
        return self.descNodeInt(nodeOffset,numNode,"[:,14:16]", "descNodeTCTran", mask_signbit=mask_signbit )




    def descNodeTC(self, nodeOffset, numNode, sumcut=7):
        """
        :param nodeOffset: absolute index within self.node array
        :param numNode: number of contiguous nodes
        :param sumcut: number of nodes above which a summary output is returned
        :return str: representing CSG typecodes

        (nodeOffset,numNode) will normally correspond to the node range of a single prim pidx

        Output examples::

            tcn 1:union 1:union 108:cone 105:cylinder 105:cylinder 0:zero 0:zero
            tcn 2:intersection 1:union 2:intersection 103:zsphere 105:cylinder 103:!zsphere 105:!cylinder
            tcn 3(2:intersection) 2(1:union) 4(105:cylinder) 2(103:zsphere) 4(0:zero)

        """
        nds = self.node[nodeOffset:nodeOffset+numNode].view(np.uint32)
        neg = ( nds[:,3,3] & 0x80000000 ) >> 31    # complemented
        tcs  = nds[:,3,2]

        if len(tcs) > sumcut:
            u_tc, x_tc, n_tc = np.unique(tcs, return_index=True, return_counts=True)
            x_order = np.argsort(x_tc)  # order by index of first occurence of each typecode
            tce = []
            for i in x_order:
                utc = u_tc[i]
                ntc = n_tc[i]
                xtc = x_tc[i]
                xng = neg[xtc]
                tce.append( "%d(%d:%s%s)" % (ntc,utc,"!" if xng else "", CSG_.desc(utc) ))
            pass
            tcn = " ".join(tce)
        else:
            tcn = " ".join(list(map(lambda _:"%d:%s%s" % (tcs[_],"!" if neg[_] else "",CSG_.desc(tcs[_])),range(len(tcs)))))
        pass

        return "no %5d nn %4d tcn %s tcs %s" % (nodeOffset, numNode, tcn, str(tcs) )



    def descSolids(self, detail=True):
        num_solid = len(self.solid)
        lines = []
        q_ridx = int(os.environ.get("RIDX", "-1"))
        for ridx in range(num_solid):
            if q_ridx > -1 and ridx != q_ridx: continue
            if detail:
                lines.append("")
            pass
            lines.append(self.descSolid(ridx, detail=detail))
        pass
        return STR("\n".join(lines))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cf = CSGFoundry.Load()
    print(cf)

    #cf.dump_node_boundary()
    #d = cf.primIdx_meshname_dict()
    #print(d)

    #cf.dumpSolid(1)


