#!/bin/env python
import os, datetime, logging, json
log = logging.getLogger(__name__)
import numpy as np
import ctypes
libcpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')

ffs_ = lambda _:libcpp.ffs(_)

lsb2_ = lambda _:(_ & 0xFF) 
msb2_ = lambda _:(_ & 0xFF00) >> 8 
hex_ = lambda _:"0x%x" % _

def ihex_(i):
    xs = hex(i)[2:]
    xs = xs[:-1] if xs[-1] == 'L' else xs 
    # trim the 0x and L
    return xs 


def count_unique(vals):
    """  
    http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    """
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    return np.vstack((uniq, np.bincount(bins))).T


idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )

pro_ = lambda _:load_("prop",_)
ppp_ = lambda _:load_("photon",_)
hhh_ = lambda _:load_("hit",_)
ttt_ = lambda _:load_("test",_)
stc_ = lambda _:load_("cerenkov",_)
sts_ = lambda _:load_("scintillation",_)
chc_ = lambda _:load_("opcerenkov",_)
chs_ = lambda _:load_("opscintillation",_)
oxc_ = lambda _:load_("oxcerenkov",_)
auc_ = lambda _:load_("aucerenkov",_)
oxs_ = lambda _:load_("oxscintillation",_)
aus_ = lambda _:load_("auscintillation",_)
rxc_ = lambda _:load_("rxcerenkov",_)
rxs_ = lambda _:load_("rxscintillation",_)
dom_ = lambda _:load_("domain",_)
idom_ = lambda _:load_("idomain",_)
seqc_ = lambda _:load_("seqcerenkov",_)
seqs_ = lambda _:load_("seqscintillation",_)
phc_ =  lambda _:load_("phcerenkov",_)
phs_ =  lambda _:load_("phscintillation",_)

recsel_cerenkov_ = lambda _:load_("recsel_cerenkov", _)
phosel_cerenkov_ = lambda _:load_("phosel_cerenkov", _)
recsel_scintillation_ = lambda _:load_("recsel_scintillation", _)
phosel_scintillation_ = lambda _:load_("phosel_scintillation", _)



g4c_ = lambda _:load_("gopcerenkov",_)
g4s_ = lambda _:load_("gopscintillation",_)
pmt_ = lambda _:load_("pmthit",_)




_json = {}
def json_(path):
    global _json 
    if _json.get(path,None):
        log.debug("return cached json for key %s" % path)
        return _json[path] 
    try: 
        log.info("parsing json for key %s" % path)
        _json[path] = json.load(file(os.path.expandvars(os.path.expanduser(path))))
    except IOError:
        log.warning("failed to load json from %s" % path)
        _json[path] = {}
    pass
    return _json[path] 

def mat_(path="~/.opticks/GMaterialIndexLocal.json"):
    """
    Customized names to codes arranged to place 
    more important materials at indices less than 0xF::

        simon:~ blyth$ cat ~/.opticks/GMaterialIndexLocal.json
        {
            "ADTableStainlessSteel": "19",
            "Acrylic": "3",
            "Air": "15",
            "Aluminium": "18",
            "Bialkali": "5",
            "DeadWater": "8",
            "ESR": "10",
            "Foam": "20",
            "GdDopedLS": "1",
            "IwsWater": "6",
            "LiquidScintillator": "2",
            "MineralOil": "4",

    """
    js = json_(path)
    return dict(zip(map(str,js.keys()),map(int,js.values())))



def imat_():
    """
    Inverse mat providing names for the custom integer codes::

        In [90]: im[0xF]
        Out[90]: 'Air'

        im = imat_()
        im_ = lambda _:im.get(_,'?')

    """
    mat = mat_()  
    return dict(zip(map(int,mat.values()),map(str,mat.keys())))

def cmm_(path="$IDPATH/../ChromaMaterialMap.json"):
    """
    Longname to chroma material code:: 

        simon:~ blyth$ cat $IDPATH/../ChromaMaterialMap.json
        {"/dd/Materials/OpaqueVacuum": 18, "/dd/Materials/Pyrex": 21, "/dd/Materials/PVC": 20, "/dd/Materials/NitrogenGas": 16,

    """ 
    js = json_(path)
    return dict(zip(map(lambda _:str(_)[len("/dd/Materials/"):],js.keys()),map(int,js.values())))

def icmm_():
    cmm = cmm_()
    return dict(zip(cmm.values(),cmm.keys()))
 

def lmm_(path="$IDPATH/GBoundaryLibMetadataMaterialMap.json"):
    """
    Shortname to wavelength line number::

        simon:~ blyth$ cat $IDPATH/GBoundaryLibMetadataMaterialMap.json
        {
            "ADTableStainlessSteel": "330",
            "Acrylic": "84",
            "Air": "12",
            "Aluminium": "24",
            "Bialkali": "126",
            "DeadWater": "42",
             ...

    """
    js = json_(path)
    return dict(zip(map(str,js.keys()),map(int,js.values())))

  

def bnd_(path="$IDPATH/GBoundaryLibMetadata.json"):
    js = json_(path) 

    boundary = js['lib']['boundary']

    bnd = {}
    bnd[0] = "MISS" 

    def shorten_sur_(sur):
        return sur if len(sur) < 2 else sur.split("__")[-1]

    for i in sorted(map(int,boundary.keys())):
        b = boundary[str(i)] 
        j = i + 1
        imat = b['imat']['shortname']
        omat = b['omat']['shortname']
        isur = shorten_sur_(b['isur']['name'])
        osur = shorten_sur_(b['osur']['name'])
        d = "%s/%s/%s/%s" % ( imat, omat, isur, osur )

        bnd[j]   = "(+%0.2d) %s " % (j,d) 
        bnd[-j] =  "(-%0.2d) %s " % (j,d) 
    pass
    return bnd
 

def c2g_():
    """
    From chroma material indice to ggeo customized 
    """
    cmm = cmm_() 
    gmm = mat_() 
    return dict(zip(map(lambda _:cmm.get(_,-1),gmm.keys()),gmm.values()))

def c2l_():
    """
    Equivalent to G4StepNPY lookup 
    providing mapping from chroma material index that is present in the gensteps
    to GGeo wavelength texture line number
    """
    cmm = cmm_()  # chroma material map
    lmm = lmm_()  # ggeo line numbers into wavelength texture
    return dict(zip(map(lambda _:cmm.get(_,-1),lmm.keys()),lmm.values()))

def check_c2l():
    """
    Rock is discrepant as not present in cmm
    """
    c2l = c2l_()    # int to int mapping

    icmm = icmm_()  # chroma int to name 
    mat = mat_()    # ggeo name to int   (customized(
    line2mat = line2mat_()   # wavelength buffer line  

    cnam = map(lambda _:icmm.get(_,None),c2l.keys())
    lnam = map(lambda _:line2mat.get(_,None),c2l.values())
    assert lnam[:-1] == cnam[:-1]
    print set(lnam) - set(cnam)



def check_gs():
    """ 
    In [71]: map(lambda _:icmm.get(_,None),np.unique(gsmat))
    Out[71]: 
    ['Acrylic',
     'DeadWater',
     'GdDopedLS',
     'IwsWater',
     'LiquidScintillator',
     'MineralOil',
     'OwsWater']
    """

    gs = stc_(1) 
    icmm = icmm_()
    gsmat = gs.view(np.int32)[:,0,2] 
    gnam = map(lambda _:icmm.get(_,None),np.unique(gsmat)) 



_line2g = {}

def line2g_():
    """
    ::

        In [93]: l2g = line2g_() 

        In [96]: l2g
        Out[96]: 
        {0: 13,
         1: 13,
         6: 16,
         7: 13,
         12: 15,
         13: 16,
         18: 17,

 
    """
    global _line2g
    if _line2g:
        return _line2g
    o = np.load(os.path.expandvars("$IDPATH/optical.npy")).reshape(-1,6,4)
    _line2g = {}
    for i,b in enumerate(o):
        _line2g[i*6+0] = b[0,0]
        _line2g[i*6+1] = b[1,0]    
    return _line2g

def line2mat_():
    im = imat_()
    line2g = line2g_()
    return dict(zip(line2g.keys(),map(lambda _:im.get(_,None),line2g.values())))

_ini = {}
def ini_(path):
    global _ini 
    if _ini.get(path,None):
        log.debug("return cached ini for key %s" % path)
        return _ini[path] 
    try: 
        log.debug("parsing ini for key %s" % path)
        _ini[path] = iniparse(file(os.path.expandvars(os.path.expanduser(path))).read())
    except IOError:
        log.fatal("failed to load ini from %s" % path)
        _ini[path] = {}
    pass
    return _ini[path] 

       
def iniparse(ini):
    return dict(map(lambda _:_.split("="), filter(None,ini.split("\n")) ))

def flags_(path=None):
    if path is None:
        #path = "$IDPATH/GFlagIndexLocal.ini"
        path = "$IDPATH/GFlagsLocal.ini"
    pass
    ini = ini_(path) 
    return dict(zip(ini.keys(),map(int,ini.values())))

def iflags_():
    flg = flags_()  
    return dict(zip(map(int,flg.values()),map(str,flg.keys())))



_abbrev_flags = {
     "CERENKOV":"CK",
     "SCINTILLATION":"SC",
     "MISS":"MI",
     "BULK_ABSORB":"AB",
     "BULK_REEMIT":"RE",
     "BULK_SCATTER":"BS",
     "SURFACE_DETECT":"SD",
     "SURFACE_ABSORB":"SA",
     "SURFACE_DREFLECT":"DR",
     "SURFACE_SREFLECT":"SR",
     "BOUNDARY_REFLECT":"BR",   
     "BOUNDARY_TRANSMIT":"BT",
     "NAN_ABORT":"NA" }

def abbrev_flags_(flg):
    return _abbrev_flags.get(flg,flg)

def abflags_():
    flg = flags_()
    return dict(zip(map(abbrev_flags_,flg.keys()),flg.values()))    

def iabflags_():
    flg = abflags_()  
    return dict(zip(map(int,flg.values()),map(str,flg.keys())))

def maskflags_():
    iaf = iabflags_()
    return dict(zip(map(lambda b:0x1 << (b-1),iaf.keys()),iaf.values()))


class Index(object):
    def table(self, cu, hex_=False):
        assert len(cu.shape) == 2 and cu.shape[1] == 2 
        for msk,cnt in sorted(cu, key=lambda _:_[1], reverse=True):
            label = ihex_(msk) if hex_ else int(msk)
            log.debug("Index msk %d cnt %d label %s " % (msk, cnt, label))
            print "%20s %10d : %40s " % ( label, int(cnt), self(msk) ) 
        pass

class GFlags(Index):
    def __init__(self):
        self.mf = maskflags_()
        self.skip = ["TORCH"]
    def __call__(self, i):
        return "|".join(map(lambda kv:kv[1], filter(lambda kv:kv[1] not in self.skip,filter(lambda kv:i & kv[0], self.mf.items()))))

def gflags_table(cu):
    gf = GFlags()
    gf.table(cu)
 

class SeqHis(Index):
    def __init__(self, abbrev=True): 
        self.f  = abflags_() if abbrev else flags_()
        self.fi = iabflags_() if abbrev else iflags_()
    def __call__(self, i):
        x = ihex_(i) 
        log.debug("seqhis %s " % x )
        return " ".join(map(lambda _:self.fi.get(int(_,16),'?%s?' % int(_,16) ), x[::-1] )) 
    def seqhis_int(self, s):
        f = self.f
        return reduce(lambda a,b:a|b,map(lambda ib:ib[1] << 4*ib[0],enumerate(map(lambda n:f[n], s.split(" ")))))

def seqhis_int(s, abbrev=True):
    f  = abflags_() if abbrev else flags_()
    return reduce(lambda a,b:a|b,map(lambda ib:ib[1] << 4*ib[0],enumerate(map(lambda n:f[n], s.split(" ")))))

def seqhis_table(cu):
    sh = SeqHis()
    sh.table(cu, hex_=True)

def seqhis_table_(): 
    seq = load_("phtorch","1","dayabay") 
    seqhis = seq[:,0,0]
    cu = count_unique(seqhis)
    sh = SeqHis()
    sh.table(cu, hex_=True)





def maskflags_string(i, skip=["TORCH"]):
    mf = maskflags_()
    return "|".join(map(lambda kv:kv[1], filter(lambda kv:kv[1] not in skip,filter(lambda kv:i & kv[0], mf.items()))))

def maskflags_int(s):
    mf = maskflags_()
    fm = dict(zip(mf.values(),mf.keys()))
    return reduce(lambda a,b:a|b, map(lambda n:fm[n], s.split("|")))




_abbrev_mat = {
    "NitrogenGas":"NG", 
    "ADTableStainlessSteel":"SS", 
    "LiquidScintillator":"LS", 
    "MineralOil":"MO" } 

def abbrev_mat_(mat):
    return _abbrev_mat.get(mat,mat[0:2])


def abmat_():
    mat = mat_()
    return dict(zip(map(abbrev_mat_,mat.keys()),mat.values()))    

def iabmat_():
    mat =  abmat_()  
    d = dict(zip(map(int,mat.values()),map(str,mat.keys())))
    d[0] = '??'
    return d 

def seqhis_(i, abbrev=True): 
    fi = iabflags_() if abbrev else iflags_()
    x = ihex_(i) 
    return " ".join(map(lambda _:fi[int(_,16)], x[::-1] )) 

def seqmat_(i, abbrev=True): 
    mi = iabmat_() if abbrev else imat_()
    x = ihex_(i) 
    return " ".join(map(lambda _:mi[int(_,16)], x[::-1] ))




DEFAULT_PATH_TEMPLATE = "$LOCAL_BASE/env/opticks/$1/$2/%s.npy"  ## cf C++ NPYBase::

def path_(typ, tag, det="dayabay"):
    tmpl = os.path.expandvars(DEFAULT_PATH_TEMPLATE.replace("$1", det).replace("$2",typ)) 
    return tmpl % str(tag)

def load_(typ, tag, det="dayabay"):
    path = path_(typ,tag, det)

    if os.path.exists(path):
        log.debug("loading %s " % path )
        os.system("ls -l %s " % path)
        return np.load(path)
    pass
    log.warning("load_ no such path %s " % path )  
    return None


gjspath_ = lambda _:os.path.expandvars("$IDPATH/%s" % _)
geopath_ = lambda _:os.path.expandvars("$IDPATH/%s.npy" % _)
geoload_ = lambda _:np.load(geopath_(_)) 




global typs
typs = "photon hit test cerenkov scintillation opcerenkov opscintillation gopcerenkov gopscintillation".split()

global typmap
typmap = {}

class NPY(np.ndarray):
    shape2type = {
            (4,4):"photon",
            (6,4):"g4step",
                 }

    @classmethod
    def from_array(cls, arr ):
        return arr.view(cls)

    @classmethod
    def empty(cls):
        a = np.array((), dtype=np.float32)
        return a.view(cls)

    @classmethod
    def detect_type(cls, arr ):
        """
        distinguish opcerenkov from opscintillation ?
        """
        assert len(arr.shape) == 3 
        itemshape = arr.shape[1:]
        typ = cls.shape2type.get(itemshape, None) 
        if typ == "g4step":
            zzz = arr[0,0,0].view(np.int32)
            if zzz < 0:
                typ = "cerenkov"
            else:
                typ = "scintillation"
            pass
        pass
        return typ 

    @classmethod
    def get(cls, tag):
        """
        # viewing an ndarray as a subclass allows adding customizations 
          on top of the ndarray while using the same storage
        """
        a = load_(cls.typ, tag).view(cls)
        a.tag = tag
        return a

    label = property(lambda self:"%s.get(%s)" % (self.__class__.__name__, self.tag))

    @classmethod
    def summary(cls, tag):
        for typ in typs:
            path = path_(typ,tag)
            if os.path.exists(path):
                mt = os.path.getmtime(path)
                dt = datetime.datetime.fromtimestamp(mt)
                msg = dt.strftime("%c")
            else:
                msg = "-"
            pass
            print "%20s : %60s : %s " % (typ, path, msg)

    @classmethod
    def mget(cls, tag, *typs):
        """
        Load multiple typed instances::

            chc, g4c, tst = NPY.mget(1,"opcerenkov","gopcerenkov","test")

        """
        if len(typs) == 1:
            typs = typs[0].split()

        klss = map(lambda _:typmap[_], typs)
        arys = map(lambda kls:kls.get(tag), klss)
        return arys


class Record(NPY):
    """
    OpenGL normalized shorts, a form of float -1.f:1.f compression  
    """
    posx         = property(lambda self:self[:,0,0])
    posy         = property(lambda self:self[:,0,1])
    posz         = property(lambda self:self[:,0,2])
    time         = property(lambda self:self[:,0,3])


class Photon(NPY):
    posx         = property(lambda self:self[:,0,0])
    posy         = property(lambda self:self[:,0,1])
    posz         = property(lambda self:self[:,0,2])
    time         = property(lambda self:self[:,0,3])
    position     = property(lambda self:self[:,0,:3]) 

    dirx         = property(lambda self:self[:,1,0])
    diry         = property(lambda self:self[:,1,1])
    dirz         = property(lambda self:self[:,1,2])
    wavelength   = property(lambda self:self[:,1,3])
    direction    = property(lambda self:self[:,1,:3]) 

    polx         = property(lambda self:self[:,2,0])
    poly         = property(lambda self:self[:,2,1])
    polz         = property(lambda self:self[:,2,2])
    weight       = property(lambda self:self[:,2,3])
    polarization = property(lambda self:self[:,2,:3]) 

    aux0         = property(lambda self:self[:,3,0].view(np.int32))
    aux1         = property(lambda self:self[:,3,1].view(np.int32))
    aux2         = property(lambda self:self[:,3,2].view(np.int32))
    aux3         = property(lambda self:self[:,3,3].view(np.int32))

    photonid     = property(lambda self:self[:,3,0].view(np.int32)) 
    spare        = property(lambda self:self[:,3,1].view(np.int32)) 
    flgs         = property(lambda self:self[:,3,2].view(np.uint32))
    pmt          = property(lambda self:self[:,3,3].view(np.int32))

    history      = property(lambda self:self[:,3,2].view(np.uint32))   # cannot name "flags" as that shadows a necessary ndarray property
    pmtid        = property(lambda self:self[:,3,3].view(np.int32)) 

    hits         = property(lambda self:self[self.pmtid > 0]) 
    aborts       = property(lambda self:self[np.where(self.history & 1<<31)])


    last_hit_triangle = property(lambda self:self[:,3,0].view(np.int32)) 

    def history_zero(self):
        log.info("filling history with zeros %s " % repr(self.history.shape))
        self.history.fill(0)

    #def _get_last_hit_triangles(self):
    #    if self._last_hit_triangles is None:
    #        self._last_hit_triangles = np.empty(len(self), dtype=np.int32)
    #        self._last_hit_triangles.fill(-1)
    #    return self._last_hit_triangles
    #last_hit_triangles = property(_get_last_hit_triangles)
    #
    #def _get_history(self):
    #    if self._last_hit_triangles is None:
    #        self._last_hit_triangles = np.empty(len(self), dtype=np.int32)
    #        self._last_hit_triangles.fill(-1)
    #    return self._last_hit_triangles
    #last_hit_triangles = property(_get_last_hit_triangles)




    def dump(self, index):
        log.info("dump index %d " % index)
        print self[index]
        print "photonid: ", self.photonid[index]
        print "history: ",  self.history[index]
        print "pmtid: ",    self.pmtid[index]    # is this still last_hit_triangle index when not a hit ?


class G4CerenkovPhoton(Photon):
    """DsChromaG4Cerenkov.cc"""
    typ = "gopcerenkov"
    cmat = property(lambda self:self[:,3,0].view(np.int32)) # chroma material index
    sid = property(lambda self:self[:,3,1].view(np.int32)) 
    pass
typmap[G4CerenkovPhoton.typ] = G4CerenkovPhoton

class G4ScintillationPhoton(Photon):
    """DsChromaG4Scintillation.cc"""
    typ = "gopscintillation"
    cmat = property(lambda self:self[:,3,0].view(np.int32)) # chroma material index
    sid = property(lambda self:self[:,3,1].view(np.int32)) 
    pdg = property(lambda self:self[:,3,2].view(np.int32)) 
    scnt = property(lambda self:self[:,3,3].view(np.int32)) 
    pass
typmap[G4ScintillationPhoton.typ] = G4ScintillationPhoton

class ChCerenkovPhoton(Photon):
    typ = "opcerenkov"
    pass
typmap[ChCerenkovPhoton.typ] = ChCerenkovPhoton

class OxCerenkovPhoton(Photon):
    typ = "oxcerenkov"
    pass
typmap[OxCerenkovPhoton.typ] = OxCerenkovPhoton





class ChCerenkovPhotonGen(Photon):
    typ = "opcerenkovgen"
    pass
typmap[ChCerenkovPhotonGen.typ] = ChCerenkovPhotonGen


class ChScintillationPhoton(Photon):
    typ = "opscintillation"
    pass
typmap[ChScintillationPhoton.typ] = ChScintillationPhoton

class ChScintillationPhotonGen(Photon):
    typ = "opscintillationgen"
    pass
typmap[ChScintillationPhotonGen.typ] = ChScintillationPhotonGen

class OxScintillationPhoton(Photon):
    typ = "oxscintillation"
    pass
typmap[OxScintillationPhoton.typ] = OxScintillationPhoton



class RxScintillationRecord(Record):
    typ = "rxscintillation"
    pass
typmap[RxScintillationRecord.typ] = RxScintillationRecord

class RxCerenkovRecord(Record):
    typ = "rxcerenkov"
    pass
typmap[RxCerenkovRecord.typ] = RxCerenkovRecord




class TestPhoton(Photon):
    typ = "test"
    pass
typmap[TestPhoton.typ] = TestPhoton


class ChromaPhoton(Photon):
    typ = "chromaphoton"

    GENERATE_SCINTILLATION = 0x1 << 16
    GENERATE_CERENKOV      = 0x1 << 17

    cerenkov      = property(lambda self:self[np.where( self.history & self.GENERATE_CERENKOV )[0]])
    scintillation = property(lambda self:self[np.where( self.history & self.GENERATE_SCINTILLATION )[0]])

    @classmethod
    def from_arrays(cls, pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights, hit=0): 
        """
        #. NB a kludge setting of pmtid into lht using the map argument of propagate_hit.cu 
        """
        nall = len(pos)
        a = np.zeros( (nall,4,4), dtype=np.float32 )       
        pmtid = np.zeros( nall, dtype=np.int32 )
 
        a[:,0,:3] = pos
        a[:,0, 3] = t 

        a[:,1,:3] = dir
        a[:,1, 3] = wavelengths

        a[:,2,:3] = pol
        a[:,2, 3] = weights 

        assert len(last_hit_triangles) == len(flags)

        SURFACE_DETECT = 0x1 << 2
        detected = np.where( flags & SURFACE_DETECT  )
        pmtid[detected] = last_hit_triangles[detected]             # sparsely populate, leaving zeros for undetected

        a[:,3, 0] = np.arange(nall, dtype=np.int32).view(a.dtype)  # photon_id
        a[:,3, 1] = 0                                              # used in comparison againt vbo prop
        a[:,3, 2] = flags.view(a.dtype)                            # history flags 
        a[:,3, 3] = pmtid.view(a.dtype)                            # channel_id ie PmtId

        if hit:
            pp = a[pmtid > 0].view(cls)
        else:
            pp = a.view(cls)  
        pass
        return pp
    pass
typmap[ChromaPhoton.typ] = ChromaPhoton









class Prop(NPY):
    """
    See test_ScintillationIntegral from gdct-
    """
    typ = "prop"
    flat = property(lambda self:self[:,0])  # unform draw from 0 to max ScintillationIntegral 
    wavelength = property(lambda self:1./self[:,1])# take reciprocal to give wavelength
    pass
typmap[Prop.typ] = Prop

class G4Step(NPY):
    typ = "g4step"
    sid = property(lambda self:self[:,0,0].view(np.int32))    # 0
    parentId = property(lambda self:self[:,0,1].view(np.int32))
    materialIndex = property(lambda self:self[:,0,2].view(np.int32))
    numPhotons = property(lambda self:self[:,0,3].view(np.int32))  

    position      = property(lambda self:self[:,1,:3]) 
    time          = property(lambda self:self[:,1,3])

    deltaPosition = property(lambda self:self[:,2,:3]) 
    stepLength    = property(lambda self:self[:,2,3])


    code = property(lambda self:self[:,3,0].view(np.int32))   # 3 

    totPhotons = property(lambda self:int(self.numPhotons.sum()))
    materialIndices = property(lambda self:np.unique(self.materialIndex))

    def materials(self, _cg):
        """
        :param _cg: chroma geometry instance
        :return: list of chroma material instances relevant to this evt 
        """
        return [_cg.unique_materials[materialIndex] for materialIndex in self.materialIndices]
    pass
typmap[G4Step.typ] = G4Step


class ScintillationStep(G4Step):
    """
    see DsChromaG4Scintillation.cc
    """
    typ = "scintillation"
    pass
typmap[ScintillationStep.typ] = ScintillationStep

 
class CerenkovStep(G4Step):
    """
    see DsChromaG4Cerenkov.cc
    """
    typ = "cerenkov"
    BetaInverse = property(lambda self:self[:,4,0])
    maxSin2 = property(lambda self:self[:,5,0])
    bialkaliIndex = property(lambda self:self[:,5,3].view(np.int32))  
    pass
typmap[CerenkovStep.typ] = CerenkovStep


class PmtHit(Photon):
    typ = "pmthit"
typmap[PmtHit.typ] = PmtHit

class G4PmtHit(Photon):
    typ = "g4pmthit"
typmap[G4PmtHit.typ] = G4PmtHit



class VBOMixin(object):
    numquad = 6 
    force_attribute_zero = "position_weight"

    @classmethod
    def vbo_from_array(cls, arr, max_slots=None):
        if arr is None:return None 
        assert max_slots
        a = arr.view(cls)
        a.initialize(max_slots)
        return a 

    def initialize(self, max_slots): 
        self.max_slots = max_slots
        self._vbodata = None
        self._ccolor = None
        self._indices = None

    def _get_ccolor(self):
        """
        #. initialize to red, reset by CUDA kernel
        """
        if self._ccolor is None:
            self._ccolor = np.tile( [1.,0.,0,1.], (len(self),1)).astype(np.float32)    
        return self._ccolor
    ccolor = property(_get_ccolor, doc=_get_ccolor.__doc__)

    def ccolor_from_code(self):
        ccolor = np.tile( [1.,1.,1.,1.], (len(self),1)).astype(np.float32)    
        ccolor[np.where(self.code == 13),:] = [1,0,0,1]  #     mu:red  
        ccolor[np.where(self.code == 11),:] = [0,1,0,1]  #      e:green 
        ccolor[np.where(self.code == 22),:] = [0,0,1,1]  #  gamma:blue 
        return ccolor

    def _get_indices(self):
        """
        List of indices
        """ 
        if self._indices is None:
            self._indices = np.arange( len(self), dtype=np.uint32)  
        return self._indices 
    indices = property(_get_indices, doc=_get_indices.__doc__)

    def _get_vbodata(self):
        if self._vbodata is None:
            self._vbodata = self.create_vbodata(self.max_slots)
        return self._vbodata
    vbodata = property(_get_vbodata)         




class VBOStep(G4Step, VBOMixin):
    def create_vbodata(self, max_slots):
        """
        """
        if len(self) == 0:return None
        dtype = np.dtype([ 
            ('position_time'   ,          np.float32, 4 ), 
            ('direction_wavelength',      np.float32, 4 ), 
            ('polarization_weight',       np.float32, 4 ), 
            ('ccolor',                    np.float32, 4 ), 
            ('flags',                     np.uint32,  4 ), 
            ('last_hit_triangle',         np.int32,   4 ), 
          ])
        assert len(dtype) == self.numquad

        data = np.zeros(len(self)*max_slots, dtype )
        log.info( "create_data items %d max_slots %d nvert %d (with slot scaleups) " % (len(self),max_slots,len(data)) )


        def pack31_( name, a, b ):
            data[name][::max_slots,:3] = a
            data[name][::max_slots,3] = b
        def pack1_( name, a):
            data[name][::max_slots,0] = a
        def pack4_( name, a):
            data[name][::max_slots] = a

        pack31_( 'position_time',        self.position ,    self.time )
        pack31_( 'direction_wavelength', self.deltaPosition, self.time )

        # not setting, leaving at zero initially 
        #pack31_( 'polarization_weight',  self.polarization, self.weight  )
        #pack1_(  'flags',                self.history )            # flags is used already by numpy 
        #pack1_(  'last_hit_triangle',    self.last_hit_triangle )

        # attempting to get gensteps sensitive to time selection
        data['flags'][::max_slots, 1] = self.time   
        data['flags'][::max_slots, 2] = self.time   

        pack4_(  'ccolor',               self.ccolor_from_code()) 

        return data



class VBOPhoton(Photon, VBOMixin):
    @classmethod
    def from_vbo_propagated(cls, vbo ):
        """
        Pulling the correct slot (-2) ?
        Hmm seems to be pulling all slots ?
        """
        r = np.zeros( (len(vbo),4,4), dtype=np.float32 )  
        r[:,0,:4] = vbo['position_time'] 
        r[:,1,:4] = vbo['direction_wavelength'] 
        r[:,2,:4] = vbo['polarization_weight'] 
        r[:,3,:4] = vbo['last_hit_triangle'].view(r.dtype) # must view as target type to avoid coercion of int32 data into float32
        return r.view(cls) 

    def create_vbodata(self, max_slots):
        """
        :return: numpy named constituent array with numquad*quads structure 

        Trying to replace DAEPhotonsData

        The start data is splayed out into the slots, leaving loadsa free slots
        this very sparse data structure with loadsa empty space limits 
        the number of photons that can me managed, but its for visualization  
        anyhow so do not need more than 100k or so. 

        Caution sensitivity to data structure naming:

        #. using 'position' would use traditional glVertexPointer furnishing gl_Vertex to shader
        #. using smth else eg 'position_weight' uses generic attribute , 
           which requires force_attribute_zero for anythinh to appear

        """
        if len(self) == 0:return None
        dtype = np.dtype([ 
            ('position_time'   ,          np.float32, 4 ), 
            ('direction_wavelength',      np.float32, 4 ), 
            ('polarization_weight',       np.float32, 4 ), 
            ('ccolor',                    np.float32, 4 ), 
            ('flags',                     np.uint32,  4 ), 
            ('last_hit_triangle',         np.int32,   4 ), 
          ])
        assert len(dtype) == self.numquad

        data = np.zeros(len(self)*max_slots, dtype )
        log.info( "create_data items %d max_slots %d nvert %d (with slot scaleups) " % (len(self),max_slots,len(data)) )

        def pack31_( name, a, b ):
            data[name][::max_slots,:3] = a
            data[name][::max_slots,3] = b
        def pack1_( name, a):
            data[name][::max_slots,0] = a
        def pack4_( name, a):
            data[name][::max_slots] = a


        pack31_( 'position_time',        self.position ,    self.time )
        pack31_( 'direction_wavelength', self.direction,    self.wavelength )
        pack31_( 'polarization_weight',  self.polarization, self.weight  )

        # not setting leaving at zero 
        #pack1_(  'flags',                self.history )            # flags is used already by numpy 
        #pack1_(  'last_hit_triangle',    self.last_hit_triangle )

        pack4_(  'ccolor',               self.ccolor) 

        return data





if __name__ == '__main__':
    pass



