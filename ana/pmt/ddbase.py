#!/usr/bin/env python

import os, re, logging, math
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main

import numpy as np
import lxml.etree as ET
import lxml.html as HT
from math import acos   # needed for detdesc string evaluation during context building

tostring_ = lambda _:ET.tostring(_)
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])

X,Y,Z = 0,1,2


class Att(object):
    def __init__(self, expr, g, evaluate=True):
        self.expr = expr
        self.g = g  
        self.evaluate = evaluate

    value = property(lambda self:self.g.ctx.evaluate(self.expr) if self.evaluate else self.expr)

    def __repr__(self):
        return "%s : %s " % (self.expr, self.value)

class Elem(object):
    posXYZ = None
    is_rev = False
    name  = property(lambda self:self.elem.attrib.get('name',None))

    # structure avoids having to forward declare classes
    is_primitive = property(lambda self:type(self) in self.g.primitive)
    is_composite = property(lambda self:type(self) in self.g.composite)
    is_intersection = property(lambda self:type(self) in self.g.intersection)
    is_tubs = property(lambda self:type(self) in self.g.tubs)
    is_sphere = property(lambda self:type(self) in self.g.sphere)
    is_union = property(lambda self:type(self) in self.g.union)
    is_posXYZ = property(lambda self:type(self) in self.g.posXYZ)
    is_geometry  = property(lambda self:type(self) in self.g.geometry)
    is_logvol  = property(lambda self:type(self) in self.g.logvol)
    typ = property(lambda self:self.__class__.__name__)

    @classmethod
    def link_posXYZ(cls, ls, posXYZ):
        """
        Attach *posXYZ* attribute to all primitives in the list 
        """
        for i in range(len(ls)):
            if ls[i].is_primitive:
                ls[i].posXYZ = posXYZ 
                log.debug("linking %s to %s " % (posXYZ, ls[i]))

    @classmethod
    def combine_posXYZ(cls, prim, plus):
        assert 0, "not yet implemented"
 
    @classmethod
    def link_prior_posXYZ(cls, ls, base=None):
        """
        Attach any *posXYZ* instances in the list to preceeding primitives
        """
        for i in range(len(ls)):
            if base is not None and ls[i-1].is_primitive:
                ls[i-1].posXYZ = base 
            pass
            if ls[i].is_posXYZ and ls[i-1].is_primitive:
                if ls[i-1].posXYZ is not None:
                    cls.combine_posXYZ(ls[i-1], ls[i])
                else:
                    ls[i-1].posXYZ = ls[i] 
                log.debug("linking %s to %s " % (ls[i], ls[i-1]))

    def _get_desc(self):
        return "%10s %15s %s " % (type(self).__name__, self.xyz, self.name )
    desc = property(_get_desc)

    def _get_xyz(self):
       x = y = z = 0
       if self.posXYZ is not None:
           x = self.posXYZ.x.value 
           y = self.posXYZ.y.value 
           z = self.posXYZ.z.value 
       pass
       return [x,y,z] 
    xyz = property(_get_xyz)

    def _get_z(self):
       """
       z value from any linked *posXYZ* or 0 
       """
       z = 0
       if self.posXYZ is not None:
           z = self.posXYZ.z.value 
       return z
    z = property(_get_z)

    def __init__(self, elem, g=None):
        self.elem = elem 
        self.g = g 

    def att(self, k, dflt=None):
        v = self.elem.attrib.get(k, None)
        return Att(v, self.g) if v is not None else Att(dflt, self.g, evaluate=False) 

    def findall_(self, expr):
        """
        lxml findall result elements are wrapped in the class appropriate to their tags,
        note the global g gets passed into all Elem

        g.kls is a dict associating tag names to classes, Elem 
        is fallback, all the classes have signature of  elem-instance, g 

        """
        wrap_ = lambda e:self.g.kls.get(e.tag,Elem)(e,self.g)
        fa = map(wrap_, self.elem.findall(expr) )
        kln = self.__class__.__name__
        name = self.name 
        log.debug("findall_ from %s:%s expr:%s returned %s " % (kln, name, expr, len(fa)))
        return fa 

    def findone_(self, expr):
        all_ = self.findall_(expr)
        assert len(all_) == 1
        return all_[0]

    def find_(self, expr):
        e = self.elem.find(expr) 
        wrap_ = lambda e:self.g.kls.get(e.tag,Elem)(e,self.g)
        return wrap_(e) if e is not None else None

    def __repr__(self):
        return "%15s : %s " % ( self.elem.tag, repr(self.elem.attrib) )

    def children(self):
        """
        Defines the nature of the tree. 

        * for Physvol returns single item list containing the referenced Logvol
        * for Logvol returns list of all contained Physvol
        * otherwise returns empty list 

        NB bits of geometry of a Logvol are not regarded as children, 
        but rather are constitutent to it.
        """
        if type(self) is Physvol:
            posXYZ = self.find_("./posXYZ")
            lvn = self.logvolref.split("/")[-1]
            lv = self.g.logvol_(lvn)
            lv.posXYZ = posXYZ
            if posXYZ is not None:
                log.debug("%s positioning %s  " % (self.name, repr(lv))) 
            return [lv]
        elif type(self) is Logvol:
            pvs = self.findall_("./physvol")
            return pvs
        else:
            return []  


class Logvol(Elem):
    material = property(lambda self:self.elem.attrib.get('material', None))
    sensdet = property(lambda self:self.elem.attrib.get('sensdet', None))
    def __repr__(self):
        return "LV %-20s %20s %s : %s " % (self.name, self.material, self.sensdet, repr(self.posXYZ))

class Physvol(Elem):
    logvolref = property(lambda self:self.elem.attrib.get('logvol', None))
    def __repr__(self):
        return "PV %-20s %s " % (self.name, self.logvolref)

class Union(Elem):
    def __repr__(self):
        return "Union %20s  " % (self.name)

class Intersection(Elem):
    def __repr__(self):
        return "Intersection %20s  " % (self.name)

class Parameter(Elem):
    expr = property(lambda self:self.elem.attrib['value'])

    def hasprefix(self, prefix):
        return self.name.startswith(prefix)

    def __repr__(self):
        return "%30s : %s " % ( self.name, self.expr )


class PosXYZ(Elem):
    x = property(lambda self:self.att('x',0))
    y = property(lambda self:self.att('y',0))
    z = property(lambda self:self.att('z',0))
    def __repr__(self):
        return "PosXYZ  %s  " % (repr(self.z))


class Primitive(Elem):
    is_rev = True
    outerRadius = property(lambda self:self.att('outerRadius'))
    innerRadius = property(lambda self:self.att('innerRadius'))

    def bbox(self, zl, zr, yn, yp ):
        assert yn < 0 and yp > 0 and zr > zl
        return BBox([yn,yn,zl], [yp,yp,zr])
 

class Sphere(Primitive):
    startThetaAngle = property(lambda self:self.att('startThetaAngle'))
    deltaThetaAngle = property(lambda self:self.att('deltaThetaAngle'))

    def has_inner(self):
        return self.innerRadius.value is not None 

    def __repr__(self):
        return "sphere %20s : %s :  %s " % (self.name, self.outerRadius, self.posXYZ)

class Tubs(Primitive):
    sizeZ = property(lambda self:self.att('sizeZ'))
    def __repr__(self):
        return "Tubs %20s : outerRadius %s  sizeZ %s  :  %s " % (self.name, self.outerRadius, self.sizeZ, self.posXYZ)




class Dddb(Elem):
    kls = {
        "parameter":Parameter,
        "sphere":Sphere,
        "tubs":Tubs,
        "logvol":Logvol,
        "physvol":Physvol,
        "posXYZ":PosXYZ,
        "intersection":Intersection,
        "union":Union,
    }

    primitive = [Sphere, Tubs]
    intersection = [Intersection]
    union = [Union]
    tubs = [Tubs]
    sphere = [Sphere]
    composite = [Union, Intersection, Physvol]
    geometry = [Sphere, Tubs, Union, Intersection]
    posXYZ= [PosXYZ]
    logvol = [Logvol]

    expand = {
        "(PmtHemiFaceROCvac^2-PmtHemiBellyROCvac^2-(PmtHemiFaceOff-PmtHemiBellyOff)^2)/(2*(PmtHemiFaceOff-PmtHemiBellyOff))":
         "(PmtHemiFaceROCvac*PmtHemiFaceROCvac-PmtHemiBellyROCvac*PmtHemiBellyROCvac-(PmtHemiFaceOff-PmtHemiBellyOff)*(PmtHemiFaceOff-PmtHemiBellyOff))/(2*(PmtHemiFaceOff-PmtHemiBellyOff))" 

    }

    @classmethod
    def parse(cls, path, analytic_version=0):
        log.info("Dddb parsing %s " % path )
        g = Dddb(parse_(path))
        g.init(analytic_version=analytic_version)
        return g

    def __call__(self, expr):
        return self.ctx.evaluate(expr)

    def init(self, analytic_version):
        self.g = self
        self.analytic_version = analytic_version

        pctx = {}
        pctx["mm"] = 1.0 
        pctx["cm"] = 10.0 
        pctx["m"] = 1000.0 
        pctx["degree"] = 1.0
        pctx["radian"] = 180./math.pi

        self.ctx = Context(pctx, self.expand)
        self.ctx.build_context(self.params_())

    def logvol_(self, name):
        if name[0] == '/':
            name = name.split("/")[-1]
        return self.find_(".//logvol[@name='%s']"%name)

    def logvols_(self):
        return self.findall_(".//logvol")

    def params_(self, prefix=None):
        pp = self.findall_(".//parameter")
        if prefix is not None:
            pp = filter(lambda p:p.hasprefix(prefix), pp)
        return pp  

    def context_(self, prefix=None):
        dd = {}
        for k,v in self.ctx.d.items():
            if k.startswith(prefix) or prefix is None:
                dd[k] = v
        return dd   

    def dump_context(self, prefix=None):
        print pp_(self.context_(prefix))





class Context(object):
    def __init__(self, d, expand):
        """
        :param d: context dict pre-populated with units
        :param expand: manual expansion dict  
        """
        self.d = d
        self.expand = expand

    def build_context(self, params):
        """
        :param params: XML parameter elements  
        """
        name_error = params
        type_error = []
        for wave in range(3):
            name_error, type_error = self._build_context(name_error, wave)
            log.debug("after wave %s remaining name_error %s type_error %s " % (wave, len(name_error), len(type_error)))
        pass
        assert len(name_error) == 0
        assert len(type_error) == 0

    def evaluate(self, expr):
        txt = "float(%s)" % expr
        try:
            val = eval(txt, globals(), self.d)
        except NameError, ex:
            log.fatal("%s :failed to evaluate expr %s " % (repr(ex), expr))
            val = None 
        pass 
        return val    

    def _build_context(self, params, wave):
        """
        :param params: to be evaluated into the context
        """
        name_error = []
        type_error = []
        for p in params:
            if p.expr in self.expand:
                expr = self.expand[p.expr]
                log.debug("using manual expansion of %s to %s " % (p.expr, expr))
            else:
                expr = p.expr

            txt = "float(%s)" % expr
            try:
                val = eval(txt, globals(), self.d)
                pass
                self.d[p.name] = float(val)  
            except NameError:
                name_error.append(p)
                log.debug("NameError %s %s " % (p.name, txt ))
            except TypeError:
                type_error.append(p)
                log.debug("TypeError %s %s " % (p.name, txt ))
            pass
        return name_error, type_error
          
    def dump_context(self, prefix):
        log.info("dump_context %s* " % prefix ) 
        return "\n".join(["%25s : %s " % (k,v) for k,v in filter(lambda kv:kv[0].startswith(prefix),self.d.items())])
 
    def __repr__(self):
        return "\n".join(["%25s : %s " % (k,v) for k,v in self.d.items()])




if __name__ == '__main__':
    args = opticks_main(apmtidx=2)

    xmlpath = args.apmtddpath 
    log.info("parsing %s -> %s " % (xmlpath, os.path.expandvars(xmlpath)))

    g = Dddb.parse(xmlpath)

    lv = g.logvol_("lvPmtHemi")


