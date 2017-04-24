#!/usr/bin/env python
"""
gdml.py : parsing GDML
=========================

"""
import os, re, logging, math
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG 
from opticks.dev.csg.glm import make_trs

import numpy as np
import lxml.etree as ET
import lxml.html as HT

tostring_ = lambda _:ET.tostring(_)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])


def construct_transform(obj):
    pos = obj.position
    rot = obj.rotation
    sca = obj.scale
    return make_trs( 
              pos.xyz if pos is not None else None,  
              rot.xyz if rot is not None else None, 
              sca.xyz if sca is not None else None, three_axis_rotate=True)


class G(object):
    typ = property(lambda self:self.__class__.__name__)
    name  = property(lambda self:self.elem.attrib.get('name',None))
    xml = property(lambda self:tostring_(self.elem))

    def att(self, name, default=None, typ=None):
        assert typ is not None
        v = self.elem.attrib.get(name, default)
        return typ(v)

    def __init__(self, elem, g=None):
        self.elem = elem 
        self.g = g 

    def findall_(self, expr):
        """
        lxml findall result elements are wrapped in the class appropriate to their tags,
        note the global g gets passed into all Elem

        g.kls is a dict associating tag names to classes, Elem 
        is fallback, all the classes have signature of  elem-instance, g 

        """
        wrap_ = lambda e:self.g.kls.get(e.tag,G)(e,self.g)
        fa = map(wrap_, self.elem.findall(expr) )
        kln = self.__class__.__name__
        name = self.name 
        log.debug("findall_ from %s:%s expr:%s returned %s " % (kln, name, expr, len(fa)))
        return fa 

    def findone_(self, expr):
        all_ = self.findall_(expr)
        assert len(all_) == 1
        return all_[0]

    def find1_(self, expr):
        all_ = self.findall_(expr)
        assert len(all_) in [0,1]
        return all_[0] if len(all_) == 1 else None

    def find_(self, expr):
        e = self.elem.find(expr) 
        wrap_ = lambda e:self.g.kls.get(e.tag,G)(e,self.g)
        return wrap_(e) if e is not None else None

    def __repr__(self):
        return "%15s : %s " % ( self.elem.tag, repr(self.elem.attrib) )



class Material(G):
    state = property(lambda self:self.att('state', '', typ=str ))
    def __repr__(self):
        return "%s %s %s" % (self.typ, self.name, self.state )
 

class Transform(G):
    unit = property(lambda self:self.att('unit', "", typ=str ))
    x = property(lambda self:self.att('x', 0, typ=float))
    y = property(lambda self:self.att('y', 0, typ=float))
    z = property(lambda self:self.att('z', 0, typ=float))
    xyz = property(lambda self:[self.x, self.y, self.z] )

    def __repr__(self):
        return "%s %s %s %s %s " % (self.typ, self.unit, self.x, self.y, self.z )

class Position(Transform):
    pass
class Rotation(Transform):
    pass
class Scale(Transform):
    pass





class Geometry(G):
    def as_ncsg(self):
        assert 0, "Geometry.as_ncsg needs to be overridden in the subclass: %s " % self.__class__ 

class Boolean(Geometry):
    firstref = property(lambda self:self.elem.find("first").attrib["ref"])
    secondref = property(lambda self:self.elem.find("second").attrib["ref"])

    position = property(lambda self:self.find1_("position"))
    rotation = property(lambda self:self.find1_("rotation"))
    scale = None
    secondtransform = property(lambda self:construct_transform(self))
   
    first = property(lambda self:self.g.solids[self.firstref])
    second = property(lambda self:self.g.solids[self.secondref])

    def __repr__(self):
        line = "%s %s  " % (self.typ, self.name )
        lrep_ = lambda label,obj:"     %s:%r"%(label,obj)
        return "\n".join([line, lrep_("l",self.first), lrep_("r",self.second)])

    def as_ncsg(self):
        left = self.first.as_ncsg()
        right = self.second.as_ncsg()
        right.transform = self.secondtransform

        cn = CSG(self.operation, name=self.name)
        cn.left = left
        cn.right = right 
        return cn 

class Intersection(Boolean):
    operation = "intersection"

class Subtraction(Boolean):
    operation = "difference"

class Union(Boolean):
    operation = "union"


class Primitive(Geometry):
    lunit = property(lambda self:self.att('lunit', 'mm', typ=str))
    aunit = property(lambda self:self.att('aunit', 'deg', typ=str))
    startphi = property(lambda self:self.att('startphi', 0, typ=float))
    deltaphi = property(lambda self:self.att('deltaphi',  360, typ=float))
    starttheta = property(lambda self:self.att('starttheta', 0, typ=float))
    deltatheta = property(lambda self:self.att('deltatheta', 180, typ=float))
    rmin = property(lambda self:self.att('rmin', 0, typ=float))
    rmax = property(lambda self:self.att('rmax', 0, typ=float))

    x = property(lambda self:self.att('x', 0, typ=float))
    y = property(lambda self:self.att('y', 0, typ=float))
    z = property(lambda self:self.att('z', 0, typ=float))

    def __repr__(self):
        return "%s %s %s rmin %s rmax %s  x %s y %s z %s  " % (self.typ, self.name, self.lunit, self.rmin, self.rmax, self.x, self.y, self.z)

class Tube(Primitive):
    def as_ncsg(self):
        cn = CSG("cylinder", name=self.name)
        cn.param[0] = 0
        cn.param[1] = 0
        cn.param[2] = 0    
        cn.param[3] = self.rmax
        cn.param1[0] = self.z    # assuming the z actually means dimension "sizeZ"
        assert self.rmin == 0.

        PCAP = 0x1 << 0 
        QCAP = 0x1 << 1 
        flags = PCAP | QCAP 
        cn.param1.view(np.uint32)[1] = flags  

        return cn


class Sphere(Primitive):
    def as_ncsg(self, only_inner=False):
        pass
        assert self.aunit == "deg" and self.lunit == "mm"

        has_inner = not only_inner and self.rmin > 0.
        if has_inner:
            inner = self.as_ncsg(only_inner=True)  # recursive call to make inner 
        pass

        radius = self.rmin if only_inner else self.rmax 
        assert radius is not None

        startThetaAngle = self.starttheta
        deltaThetaAngle = self.deltatheta

        x = 0
        y = 0
        z = 0

        # z to the right, theta   0 -> z=r, theta 180 -> z=-r
        rTheta = startThetaAngle
        lTheta = startThetaAngle + deltaThetaAngle

        assert rTheta >= 0. and rTheta <= 180.
        assert lTheta >= 0. and lTheta <= 180.

        log.info("Sphere.as_ncsg radius:%s only_inner:%s  has_inner:%s " % (radius, only_inner, has_inner)) 

        zslice = startThetaAngle > 0. or deltaThetaAngle < 180.

        if zslice:
            zmin = radius*math.cos(lTheta*math.pi/180.)
            zmax = radius*math.cos(rTheta*math.pi/180.)
            assert zmax > zmin, (startThetaAngle, deltaThetaAngle, rTheta, lTheta, zmin, zmax )

            log.info("Sphere.as_ncsg rTheta:%5.2f lTheta:%5.2f zmin:%5.2f zmax:%5.2f azmin:%5.2f azmax:%5.2f " % (rTheta, lTheta, zmin, zmax, z+zmin, z+zmax ))

            cn = CSG("zsphere", name=self.name, param=[x,y,z,radius], param1=[zmin,zmax,0,0], param2=[0,0,0,0]  )

            ZSPHERE_QCAP = 0x1 << 1   # zmax
            ZSPHERE_PCAP = 0x1 << 0   # zmin
            flags = ZSPHERE_QCAP | ZSPHERE_PCAP 

            cn.param2.view(np.uint32)[0] = flags 
            pass
        else:
            cn = CSG("sphere", name=self.name)
            cn.param[0] = x
            cn.param[1] = y
            cn.param[2] = z
            cn.param[3] = radius
        pass 
        if has_inner:
            ret = CSG("difference", left=cn, right=inner )
        else: 
            ret = cn 
        pass
        return ret



class Box(Primitive):
    """
    Need to overhaul nbox, currently using x,y,z for placement
    and just w for extent...  instead need to generalize to 
    use x,y,z for dimensions and leave placement to transforms. 
    """
    def as_ncsg(self):
        assert self.lunit == 'mm' 
        cn = CSG("box", name=self.name)
        cn.param[0] = self.x
        cn.param[1] = self.y
        cn.param[2] = self.z
        cn.param[3] = 0
        return cn




class Volume(G):
    materialref = property(lambda self:self.elem.find("materialref").attrib["ref"])
    solidref = property(lambda self:self.elem.find("solidref").attrib["ref"])
    solid = property(lambda self:self.g.solids[self.solidref])
    material = property(lambda self:self.g.materials[self.materialref])

    physvol = property(lambda self:self.findall_("physvol"))
    children = property(lambda self:self.physvol)

    def filterpv(self, pfx):
        return filter(lambda pv:pv.name.startswith(pfx), self.physvol) 

    def rdump(self, depth=0):
        print self
        for pv in self.physvol:
            lv = pv.volume
            lv.rdump(depth=depth+1)


    def as_ncsg(self):
        """
        Hmm pv level transforms need to be applied to the
        csg top nodes of the solids, as the pv transforms
        are not being propagated GPU side, and even when they
        are will want to be able to handle a csg instance
        comprising a few solids only (eg the PMT) 
        as is with its own transforms transforms.

        Essentially are collapsing a subtree for the 
        handful of solids into the self contained instance
        of a list of csgnodes.
        """
        pass
 

    def __repr__(self):
        repr_ = lambda _:"   %r" % _ 
        pvs = map(repr_, self.physvol) 
        line = "%s %s %s %s" % (self.typ, self.name, self.materialref, self.solidref)
        return "\n".join([line, repr_(self.solid), repr_(self.material)] + pvs )

class PhysVol(G):
    volumeref = property(lambda self:self.elem.find("volumeref").attrib["ref"])
    volume = property(lambda self:self.g.volumes[self.volumeref])
    children = property(lambda self:[self.volume])

    position = property(lambda self:self.find1_("position"))
    rotation = property(lambda self:self.find1_("rotation"))
    scale = property(lambda self:self.find1_("scale"))
    transform = property(lambda self:construct_transform(self))

    def __repr__(self):
        return "\n".join(["%s %s" % (self.typ, self.name)," %r %r " % ( self.position, self.rotation)])
     

class GDML(G):
    kls = {
        "material":Material,

        "tube":Tube,
        "sphere":Sphere,
        "box":Box,

        "intersection":Intersection,
        "subtraction":Subtraction,
        "union":Union,

        "position":Position,
        "rotation":Rotation,
        "scale":Scale,

        "volume":Volume,
        "physvol":PhysVol,
    }

    @classmethod
    def parse(cls, path):
        log.info("parsing gdmlpath %s " % path )
        gg = cls(parse_(path))
        gg.g = gg
        gg.path = path 
        gg.init()
        return gg 

    def find_by_prefix(self, d, prefix):
        return filter(lambda v:v.name.startswith(prefix), d.values())

    def find_volumes(self, prefix="/dd/Geometry/PMT/lvPmtHemi"):
        return self.find_by_prefix(self.volumes, prefix)

    def find_solids(self, prefix="pmt-hemi"):
        return self.find_by_prefix(self.solids, prefix)

    def find_materials(self, prefix="/dd/Materials/Acrylic"):
        return self.find_by_prefix(self.materials, prefix)

    world = property(lambda self:self.volumes[self.worldvol])

    def init(self):

        self.materials = {}
        self.solids = {}
        self.volumes = {}

        for e in self.findall_("materials/material"):
            self.materials[e.name] = e 

        for e in self.findall_("solids/*"):
            self.solids[e.name] = e 
        pass
        for e in self.findall_("structure/*"):
            self.volumes[e.name] = e
        pass
        self.worldvol = self.elem.find("setup/world").attrib["ref"]




if __name__ == '__main__':

    args = opticks_main()
    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # envvar set within opticks_main 

    gdml = GDML.parse(gdmlpath)

    #print gdml.world
    #gdml.world.rdump()

    #gdml.volumes["/dd/Geometry/PMT/lvPmtHemi0xc133740"].rdump()

    from treebase import Tree
    t = Tree(gdml.world, postype="position") 

    lvn = "/dd/Geometry/PMT/lvPmtHemi0x"
 
    l = t.findlv(lvn)  # all nodes 
    assert len(l) == 672

    #a = np.array(map(lambda n:n.index, l))  # 3 groups of index pitches apparent in  a[1:] - a[:-1]

    n = l[0] 
    sibs = n.siblings 
    assert len(sibs) == 192
  
   
    cn = n.lv.solid.as_ncsg()
    cn.dump()

    v = n.children[0]
    for c in v.children:
        print c.pv.transform





