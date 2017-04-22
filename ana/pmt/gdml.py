#!/usr/bin/env python
"""
gdml.py : parsing GDML
=========================

"""
import os, re, logging, math
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main

import numpy as np
import lxml.etree as ET
import lxml.html as HT

tostring_ = lambda _:ET.tostring(_)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])


class G(object):
    typ = property(lambda self:self.__class__.__name__)
    name  = property(lambda self:self.elem.attrib.get('name',None))

    def att(self, name, default=None):
        return self.elem.attrib.get(name, default)

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
    state = property(lambda self:self.att('state', None))
    def __repr__(self):
        return "%s %s %s" % (self.typ, self.name, self.state )
 

class Transform(G):
    unit = property(lambda self:self.att('unit', None))
    x = property(lambda self:self.att('x', None))
    y = property(lambda self:self.att('y', None))
    z = property(lambda self:self.att('z', None))

    def __repr__(self):
        return "%s %s %s %s %s " % (self.typ, self.unit, self.x, self.y, self.z )

class Position(Transform):
    pass
class Rotation(Transform):
    pass


class Boolean(G):
    firstref = property(lambda self:self.elem.find("first").attrib["ref"])
    secondref = property(lambda self:self.elem.find("second").attrib["ref"])
    position = property(lambda self:self.find1_("position"))
    rotation = property(lambda self:self.find1_("rotation"))
   
    first = property(lambda self:self.g.solids[self.firstref])
    second = property(lambda self:self.g.solids[self.secondref])

    def __repr__(self):
        line = "%s %s  " % (self.typ, self.name )
        lrep_ = lambda label,obj:"     %s:%r"%(label,obj)
        return "\n".join([line, lrep_("l",self.first), lrep_("r",self.second)])


class Intersection(Boolean):
    pass
class Subtraction(Boolean):
    pass
class Union(Boolean):
    pass


class Primitive(G):
    lunit = property(lambda self:self.att('lunit', None))
    aunit = property(lambda self:self.att('aunit', None))
    startphi = property(lambda self:self.att('startphi', None))
    deltaphi = property(lambda self:self.att('deltaphi', None))
    starttheta = property(lambda self:self.att('starttheta', None))
    deltatheta = property(lambda self:self.att('deltatheta', None))
    rmin = property(lambda self:self.att('rmin', None))
    rmax = property(lambda self:self.att('rmax', None))

    x = property(lambda self:self.att('x', None))
    y = property(lambda self:self.att('y', None))
    z = property(lambda self:self.att('z', None))

    def __repr__(self):
        return "%s %s %s rmin %s rmax %s  x %s y %s z %s  " % (self.typ, self.name, self.lunit, self.rmin, self.rmax, self.x, self.y, self.z)

class Tube(Primitive):
    pass
class Sphere(Primitive):
    pass
class Box(Primitive):
    pass


class Volume(G):
    materialref = property(lambda self:self.elem.find("materialref").attrib["ref"])
    solidref = property(lambda self:self.elem.find("solidref").attrib["ref"])
    solid = property(lambda self:self.g.solids[self.solidref])
    material = property(lambda self:self.g.materials[self.materialref])

    physvol = property(lambda self:self.findall_("physvol"))

    def rdump(self, depth=0):
        print self
        for pv in self.physvol:
            lv = pv.volume
            lv.rdump(depth=depth+1)
            

    def __repr__(self):
        repr_ = lambda _:"   %r" % _ 
        pvs = map(repr_, self.physvol) 
        line = "%s %s %s %s" % (self.typ, self.name, self.materialref, self.solidref)
        return "\n".join([line, repr_(self.solid), repr_(self.material)] + pvs )

class PhysVol(G):
    volumeref = property(lambda self:self.elem.find("volumeref").attrib["ref"])
    position = property(lambda self:self.find1_("position"))
    rotation = property(lambda self:self.find1_("rotation"))

    volume = property(lambda self:self.g.volumes[self.volumeref])

    def __repr__(self):
        return "%s %s  %r %r " % (self.typ, self.volumeref, self.position, self.rotation)
     

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

        "volume":Volume,
        "physvol":PhysVol,
    }

    @classmethod
    def parse(cls, path):
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
    gg = GDML.parse("/tmp/g4_00.gdml")
    #print gg.world
    #gg.world.rdump()

    gg.volumes["/dd/Geometry/PMT/lvPmtHemi0xc133740"].rdump()

