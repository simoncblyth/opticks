#!/usr/bin/env python
"""
ddbase.py : Detdesc parsing into wrapped Elem tree
====================================================


Questions 
-----------


Fly in ointment is that there is detdesc xml generation
which means cannot understand whats going on from sources
alone.


Detdesc Cross File Referencing
-------------------------------------


Catalog declares things that can be referenced from other files and
defines the references for them eg /dd/Geometr


DDDB/geometry.xml::

     04 <DDDB>
      5 
      6   <catalog name="Geometry">
      7 
     ..
     30 
     31     <catalogref href="PMT/geometry.xml#PMT" />
     ..
     41   </catalog>
     42 
     43 </DDDB>


DDDB/PMT/geometry.xml::

     06 <DDDB>
      7 
      8   <catalog name="PMT">
      9 
     10     <logvolref href="hemi-pmt.xml#lvPmtHemiFrame"/>
     11     <logvolref href="hemi-pmt.xml#lvPmtHemi"/>
     12     <logvolref href="hemi-pmt.xml#lvPmtHemiwPmtHolder"/>
     13     <logvolref href="hemi-pmt.xml#lvAdPmtCollar"/>




::

    simon:DDDB blyth$ pmt-dfind /dd/Geometry/PMT/lvPmtHemi\"
    ./PMT/hemi-pmt.xml:     <physvol name="pvPmtHemi" logvol="/dd/Geometry/PMT/lvPmtHemi" />
    ./PMT/hemi-pmt.xml:     <physvol name="pvAdPmt" logvol="/dd/Geometry/PMT/lvPmtHemi" />
    ./PMT/pmt.xml:       volsecond="/dd/Geometry/PMT/lvPmtHemi">
    ./PmtBox/geometry.xml:    <physvol name="pvPmtBoxPmt" logvol="/dd/Geometry/PMT/lvPmtHemi">
    ./PmtPanel/properties.xml:     volfirst="/dd/Geometry/PMT/lvPmtHemi"
    ./PmtPanel/properties.xml:     volfirst="/dd/Geometry/PMT/lvPmtHemi"
    simon:DDDB blyth$ 

    simon:DDDB blyth$ pmt-dfind /dd/Geometry/PMT/lvPmtHemiFrame\"
    ./PmtPanel/vetopmt.xml:    <physvol name="pvVetoPmtUnit" logvol="/dd/Geometry/PMT/lvPmtHemiFrame">
    simon:DDDB blyth$ 

    simon:DDDB blyth$ pmt-dfind /dd/Geometry/PMT/lvPmtHemiwPmtHolder\"
    ./AdPmts/geometry.xml:  <physvol name="pvAdPmtUnit" logvol="/dd/Geometry/PMT/lvPmtHemiwPmtHolder">
    simon:DDDB blyth$ 




* Relies on all paramters being defined within the file
  (XML entity inclusion is regarded as being withn the same file)


Classes
--------

Att(object) 
    holds string expression "expr" and top level global "g" 
    allowing the expression to be evaluated within the global context  
    via the "value" property 
 
    For simple properties this is not needed,  only need for expression 
    properties.


Elem(object)
    ctor simply holds raw lxml elem and global "g", 

    Primary functionality invoked via findall_ which takes the 
    lxml elements returned by lxml findall and wraps them into 
    Elem subclasses appropriate to the lxml tag
     
    Also provides introspection. 


Dddb(Elem)
    top level global g that holds the context within which txt expr 
    are evaluated via Att "value" properties 

Parameter(Elem)
    prop: expr

PosXYZ(Elem)
    att: x,y,z 

    att rather than simple props are needed as they are expressions 
    that must be evaluated within the global context 

Primitive(Elem)
    att: outerRadius, innerRadius

Sphere(Primitive)
    att: startThetaAngle, deltaThetaAngle

Tubs(Primitive)
    att: sizeZ

Logvol(Elem)
    prop: material, sensdet 

Physvol(Elem)
    prop: logvolref

Union(Elem)
Intersection(Elem)
Subtraction(Elem)
    naming types




"""
import os, re, logging, math
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main

import numpy as np
import lxml.etree as ET
import lxml.html as HT
from math import acos   # needed for detdesc string evaluation during context building

tostring_ = lambda _:ET.tostring(_)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
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



class E(object):
    lvtype = 'Logvol'
    pvtype = 'Physvol'
    postype = 'posXYZ'

    typ = property(lambda self:self.__class__.__name__)
    name  = property(lambda self:self.elem.attrib.get('name',None))
    shortname = property(lambda self:self.name)   # for correspondence with GDML branch 

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
        wrap_ = lambda e:self.g.kls.get(e.tag,E)(e,self.g)
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
        wrap_ = lambda e:self.g.kls.get(e.tag,Elem)(e,self.g)
        return wrap_(e) if e is not None else None

    def __repr__(self):
        return "%15s : %s " % ( self.elem.tag, repr(self.elem.attrib) )





class Elem(E):
    posXYZ = None
    is_rev = False

    # structure avoids having to forward declare classes
    is_primitive = property(lambda self:type(self) in self.g.primitive)
    is_composite = property(lambda self:type(self) in self.g.composite)
    is_intersection = property(lambda self:type(self) in self.g.intersection)
    is_difference = property(lambda self:type(self) in self.g.difference)
    is_operator = property(lambda self:type(self) in self.g.operator)
    is_tubs = property(lambda self:type(self) in self.g.tubs)
    is_sphere = property(lambda self:type(self) in self.g.sphere)
    is_union = property(lambda self:type(self) in self.g.union)
    is_posXYZ = property(lambda self:type(self) in self.g.posXYZ)
    is_geometry  = property(lambda self:type(self) in self.g.geometry)
    is_logvol  = property(lambda self:type(self) in self.g.logvol)

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
        for i in range(1,len(ls)):   # start from 1 to avoid list wraparound ?
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

    def _get_children(self):
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
    
            ## Hmm: Associating a posXYZ to an lv should probably be done 
            ##      on a clone of the lv ?  As there may be multiple such placements
            ##      and dont want prior placements to get stomped on.
            ##
            ## OR use a different class to hold the original lv   
            ##    together with its posXYZ ?
            ##

            if posXYZ is not None:
                log.info("children... %s passing pv posXYZ to lv %s  " % (self.name, repr(lv))) 
            return [lv]

        elif type(self) is Logvol:
            pvs = self.findall_("./physvol")
            return pvs
        else:
            return []  
        pass

    children = property(_get_children)


    def comps(self):
        """
        :return comps: immediate constituents of an Elem, not recursive
        """
        comps = self.findall_("./*")
        self.link_prior_posXYZ(comps)
        return comps

    def geometry(self):
        return filter(lambda c:c.is_geometry, self.comps())





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

class Subtraction(Elem):
    def __repr__(self):
        return "Subtraction %20s  " % (self.name)



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

    def has_innerRadius(self):
        return self.innerRadius.value is not None 

    def has_startThetaAngle(self):
        return self.startThetaAngle.value is not None 

    def has_deltaThetaAngle(self):
        return self.deltaThetaAngle.value is not None 

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
        "subtraction":Subtraction,    # not seen in wild
    }


    primitive = [Sphere, Tubs]
    tubs = [Tubs]
    sphere = [Sphere]

    intersection = [Intersection]
    union = [Union]
    subtraction = [Subtraction]
    operator = [Union, Intersection, Subtraction]

    composite = [Union, Intersection, Subtraction, Physvol]  # who uses this ? funny combo

    geometry = [Sphere, Tubs, Union, Intersection, Subtraction]

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
        """
        Hmm this referencing is single file only, 
        need to build registry of logvol from all files
        """
        log.info("logvol_ %s " % name)

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
        :param expand: manual expansion dict, workaround for detdesc 
                       expressions that are not valid python 
        """
        self.d = d
        self.expand = expand

    def build_context(self, params):
        """
        :param params: XML parameter elements  

        Three wave context building avoids having to work out dependencies, 
        just repeat and rely on simpler expressions getting set into 
        context on prior waves.
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





    



class Ref(E):
    href = property(lambda self:self.elem.attrib['href'])

    xmlname = property(lambda self:self.href.split("#")[0])
    anchor = property(lambda self:self.href.split("#")[1])

    xmlpath = property(lambda self:os.path.join(self.parent.xmldir, self.xmlname))
    ddpath = property(lambda self:os.path.join(self.parent.ddpath, self.anchor))

    def __repr__(self):
        return "%20s %20s %s %s " % (self.typ, self.anchor, self.xmlpath, self.ddpath )

class CatalogRef(Ref):
    pass

class LogvolRef(Ref):
    pass


class Catalog(E):
    """
    Role of catalog is to provide easy access to 
    logvol via simple path access... so this 
    needs to maintain the connection between 
    
    * filesystem paths
    * dd paths
    * in memory elements 

    """ 
    name = property(lambda self:self.elem.attrib['name'])
    parent = None

    def _get_ddpath(self):
        lev = [""]
        if self.parent is not None:
            lev.append( self.parent.ddpath )
        pass
        lev.append(self.name) 
        return "/".join(lev)

    ddpath = property(_get_ddpath)


    def __repr__(self):
        return "%20s %20s %s " % (self.typ, self.name, self.ddpath)

    def refs(self):
        """
        Catalogs often contain catalogref, logvolref but they
        can also directly contain logvol.
    
        Hmm splitting into separate files is an 
        implementation detail, should not impact the 
        tree being built.
        """
        refs = self.findall_("./catalogref")
        for ref in refs:
            ref.parent = self
            if not exists_(ref.xmlpath):
                log.warning("ref.xmlpath doesnt exist %s " % ref.xmlpath)
        pass
        return refs 






class DD(E):

    ddr = {}

    kls = {
        "catalog":Catalog,
        "catalogref":CatalogRef,
        "logvolref":LogvolRef,
    }

    xmldir = property(lambda self:os.path.dirname(self.xmlpath))

    @classmethod
    def parse(cls, path):
        log.info("DD parsing %s " % path )
        g = cls 
        dd = DD(parse_(path), g)
        dd.xmlpath = path
        dd.init()
        return dd

    def init(self):
        cat = self.find1_("./catalog")
        if cat is not None:
            print cat  
            cat.xmldir = self.xmldir
            for ref in cat.refs():
                print ref, ref.ddpath
                self.g.ddr[ref.ddpath] = DD.parse(ref.xmlpath) 

    def __repr__(self):
        return "%20s %s %s " % (self.typ, self.xmlpath, self.xmldir)









if __name__ == '__main__':
    args = opticks_main(apmtidx=2)



    xmlpath = args.apmtddpath 
    log.info("parsing %s -> %s " % (xmlpath, os.path.expandvars(xmlpath)))

    g = Dddb.parse(xmlpath)

    lv = g.logvol_("lvPmtHemi")


    dd = DD.parse(args.addpath)


    print dd



