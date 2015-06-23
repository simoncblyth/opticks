#!/bin/env python
import os, datetime, logging
log = logging.getLogger(__name__)
import numpy as np
import ctypes
libcpp = ctypes.cdll.LoadLibrary('libc++.1.dylib')

ffs_ = lambda _:libcpp.ffs(_)

lsb2_ = lambda _:(_ & 0xFF) 
msb2_ = lambda _:(_ & 0xFF00) >> 8 
hex_ = lambda _:"0x%x" % _

pro_ = lambda _:load_("prop",_)
ppp_ = lambda _:load_("photon",_)
hhh_ = lambda _:load_("hit",_)
ttt_ = lambda _:load_("test",_)
stc_ = lambda _:load_("cerenkov",_)
sts_ = lambda _:load_("scintillation",_)
chc_ = lambda _:load_("opcerenkov",_)
chs_ = lambda _:load_("opscintillation",_)
oxc_ = lambda _:load_("oxcerenkov",_)
oxs_ = lambda _:load_("oxscintillation",_)
rxc_ = lambda _:load_("rxcerenkov",_)
rxs_ = lambda _:load_("rxscintillation",_)
dom_ = lambda _:load_("domain",_)
idom_ = lambda _:load_("idomain",_)
seqidx_ = lambda _:load_("seqidx",_)

g4c_ = lambda _:load_("gopcerenkov",_)
g4s_ = lambda _:load_("gopscintillation",_)
pmt_ = lambda _:load_("pmthit",_)

path_ = lambda typ,tag:os.environ["DAE_%s_PATH_TEMPLATE" % typ.upper()] % str(tag)
#load_ = lambda typ,tag:np.load(path_(typ,tag))     

def load_(typ, tag):
    path = path_(typ,tag)
    log.info("loading %s " % path )
    os.system("ls -l %s " % path)
    return np.load(path)


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



