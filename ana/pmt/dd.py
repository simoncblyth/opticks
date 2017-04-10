#!/usr/bin/env python
"""
dd.py
=======

Approach:

* kicking off from Dddb.logvol_ wraps results of 
  lxml find into do-everything Elem subclass appropriate to the tags of the elements

* primary purpose is the Elem.parts method, trace back from there to understand



"""
import os, re, logging, math
import numpy as np
import lxml.etree as ET
import lxml.html as HT
from math import acos   # needed for detdesc string evaluation during context building

from geom import Part, BBox, ZPlane, Rev
from csg import CSG

log = logging.getLogger(__name__)

tostring_ = lambda _:ET.tostring(_)
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])

X,Y,Z = 0,1,2


class Parts(list):
   def __repr__(self):
       return "Parts " + list.__repr__(self)

   def __init__(self, *args, **kwa):
       list.__init__(self, *args, **kwa)


class Uncoincide(object):
    def __init__(self, top="CONTAINING_MATERIAL", sensor="SENSOR_SURFACE"):
         self.top = top
         self.sensor = sensor
         pass
    def face_bottom_tubs(self, name):
        """
        Only face and bottom have coincident surfaces in literal
        translation that need fixing.

        ::

               tubs                      bottom                    face

               OM///Pyrex                OM///Pyrex               OM///Pyrex

               Pyrex///Vacuum            Pyrex///OpaqueVacuum     Pyrex/SENSOR//Bialkali

               Vacuum///OpaqueVacuum     OpaqueVacuum///Vacuum    Bialkali///Vacuum


        """
        log.warning("UNCOINCIDE boundary setting for %s " % name)
        face = bottom = tubs = None
        if name == "pmt-hemi":
            pass
            boundary = "%s///Pyrex" % self.top   
            face = boundary 
            bottom = boundary   
            tubs = boundary    
            pass
        elif name == "pmt-hemi-vac": 
            pass
            face = "Pyrex/%s//Bialkali" % self.sensor
            bottom = "Pyrex///OpaqueVacuum"
            tubs = "Pyrex///Vacuum"
            pass
        elif name == "pmt-hemi-cathode": 
            pass
            face = "Bialkali///Vacuum"
            bottom = None
            tubs = None
            pass
        elif name == "pmt-hemi-bot": 
            pass
            face = None
            bottom = "OpaqueVacuum///Vacuum"
            tubs = None
            pass
        elif name == "pmt-hemi-dynode":
            pass
            face = None
            bottom = None
            tubs = "Vacuum///OpaqueVacuum"
            pass
        else:
            assert 0
        pass
        return face, bottom, tubs

UNCOINCIDE = Uncoincide()


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
        lxml findall result elements are wrapped in the class appropriate to their tags 
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

    def partition_intersection_3spheres(self, spheres, material=None):
        """
        :param spheres:  list of three *Sphere* in ascending center z order, which are assumed to intersect
        :return parts:  list of three sphere *Part* instances 

        Consider splitting the lens shape made from the boolean CSG intersection 
        of two spheres along the plane of the intersection. 
        The left part of the lens comes from the right Sphere 
        and the right part comes left Sphere.

        Extending from two sphere intersection to three spheres
        numbered s1,s2,s3 from left to right, with two ZPlane intersections z23, z12.

        left
            from s3, bounded by z23 on right  

        middle
            from s2, bounded by z23 on left, z12 on right

        right 
            from s1, bounded by z12 on left

        """
        s1, s2, s3 = spheres

        assert s1.z < s2.z < s3.z

        z12 = Sphere.intersect("z12",s1,s2)   # ZPlane of s1 s2 intersection
        z23 = Sphere.intersect("z23",s2,s3)   # ZPlane of s2 s3 intersection

        assert z23.z < z12.z 

        p1 = s3.part_zleft(z23)       
        p2 = s2.part_zmiddle(z23, z12)       
        p3 = s1.part_zright(z12)       

        assert p1.bbox.z < p2.bbox.z < p3.bbox.z

        pts = Parts([p3,p2,p1])
        return pts 
 
    def partition_intersection_2spheres(self, spheres, material=None, unoverlap=True):
        """
        :param spheres: list of two *Sphere* in ascending center z order
        :return parts: list of two sphere *Part* instances

        Used for the very thin photocathode

        For pmt-hemi-cathode-face_part pmt-hemi-cathode-belly_part the 
        bbox of the Sphere parts from just the theta angle ranges are overlapping
        each other very slightly with the plane of sphere intersection in between.
        Because of the thin photocathode this creates an dangerous sharp protuding edge.

        Suspect overlapping bbox may be problematic, so unoverlap by adjusting z 
        edges of bbox, and adjust in xy. 
        """ 
        s1, s2 = spheres
        assert s1.z < s2.z 

        # parts from just the theta ranges
        p1, p2 = Part.ascending_bbox_zmin([s1.as_part(),s2.as_part()])
        assert p1.bbox.zmin < p2.bbox.zmin
        p12 = Sphere.intersect("p12",s1,s2) 

        if unoverlap:
            p1.bbox.zmax = p12.z
            p2.bbox.zmin  = p12.z 
            p2.bbox.xymax = p12.y
            p2.bbox.xymin = -p12.y

        log.warning("material %s name %s " % (material, self.name))
        log.warning("p1 %s " % p1) 
        log.warning("p2 %s " % p2) 

        # pluck inner radii, for the insides 
        i1, i2 = Part.ascending_bbox_zmin([s1.as_part(inner=True),s2.as_part(inner=True)])
        assert i1.bbox.zmin < i2.bbox.zmin
        i12 = Sphere.intersect("p12",s1,s2, inner=True)

        if unoverlap:
            i1.bbox.zmax = i12.z
            i2.bbox.zmin  = i12.z 
            i2.bbox.xymax = i12.y
            i2.bbox.xymin = -i12.y
 
        log.warning("i1 %s" % i1) 
        log.warning("i2 %s" % i2) 
        if UNCOINCIDE and self.name == "pmt-hemi-cathode": 
            face, bottom, tubs = UNCOINCIDE.face_bottom_tubs(self.name)
            i2.boundary = face 
            i1.boundary = face  
            ret = [i2,i1]    # skipping the coincidents 
        else:
            #i1.parent = p1
            #i2.parent = p2
            ret = [p2,i2,p1,i1]
        pass
        return Parts(ret)

    def partition_intersection(self, material=None):
        #log.info(self)

        spheres = []
        comps = self.findall_("./*")
        self.link_prior_posXYZ(comps)

        other = []
        for c in comps:
            if type(c) is Sphere:
                spheres.append(c)
            elif type(c) is PosXYZ:
                pass
            else:
                other.append(c)
            pass

        assert len(other) == 0, "only 2/3-sphere intersections handled"    

        for i,s in enumerate(spheres):
            log.debug("s%d: %s %s " % (i, s.desc, s.outerRadius.value))
 
        if len(spheres) == 3:
            pts = self.partition_intersection_3spheres(spheres, material=material) 
        elif len(spheres) == 2:
            pts = self.partition_intersection_2spheres(spheres, material=material) 
        else:
            assert 0 


        pts.csg = CSG(self, spheres) 

        return pts


    def partition_union_intersection_tubs(self, comps, material=None, verbose=False):
        """ 
        """
        log.info("material %s name %s " % (material, self.name))

        ipts = comps[0].partition_intersection()
        sparts = Part.ascending_bbox_zmin(ipts)
        assert len(sparts) == 3 

        tpart = comps[1].as_part()
        ts = Part.intersect_tubs_sphere("ts", tpart, sparts[0], -1)   # -ve root for leftmost


        if verbose:
            log.info(self)
            log.info("ts %s " % repr(ts))
            for i, s in enumerate(sparts):
                log.info("sp(%s) %s " % (i, repr(s)))
            log.info("tp(0) %s " % repr(tpart))
        pass

        # CRUCIAL setting of same zmax for ztubs and zmin for zsphere
        # done via the bbox...
        # TODO: store z-range in parameters (2*float), not the costly bbox (6*float)

        tpart.bbox.zmax = ts.z
        sparts[0].bbox.zmin = ts.z   

        tpart.enable_endcap("P")  # smaller Z endcap 


        if UNCOINCIDE: 
            if self.name == "pmt-hemi" or self.name == "pmt-hemi-vac":
                face, bottom, tubs = UNCOINCIDE.face_bottom_tubs(self.name)
            else:
                assert 0  
            pass
            tpart.boundary = tubs
            sparts[0].boundary = bottom
            sparts[1].boundary = face
            sparts[2].boundary = face
        pass 

        rparts = Parts()
        rparts.extend(sparts)
        rparts.extend([tpart])

        # had CSG list/tree confusion here, from thinking ahead to serialization
        # dont do that, use the best representation for the immediate problem at hand 
        # of grabbing the CSG tree into an in-memory representation

        rparts.csg = CSG(self, [ipts.csg, comps[1]] )

        return rparts 


    def partition_union(self, material=None, verbose=False):
        """
        union of a 3-sphere lens shape and a tubs requires:

        * adjust bbox of the abutting part Sphere to the intersection z of tubs and Sphere
        * avoid a surface at the interface of tubs endcap and part Sphere

        """
        comps = self.findall_("./*")
        self.link_prior_posXYZ(comps)

        rparts = Parts()
        xret = None

        if len(comps) == 3 and comps[0].is_intersection and comps[1].is_tubs and comps[2].is_posXYZ:
            # pmt-hemi-glass-bulb, pmt-hemi-bulb-vac        
            xret = self.partition_union_intersection_tubs(comps[0:3], material=material)
        elif len(comps) == 3 and comps[0].is_sphere and comps[1].is_sphere and comps[2].is_posXYZ:
            xret = self.partition_intersection_2spheres(comps[0:2], material=material)
            if not hasattr(xret, 'csg'):
                xret.csg = CSG(self, comps[0:2])
        else:
            xret = self.parts()   

        if xret is not None:
            rparts.extend(xret)
            if hasattr(xret, 'csg'):
                rparts.csg = xret.csg
        pass
        return rparts  ; 

    def parts_sphere_with_inner(self, c):
        log.info("name %s " % c.name)

        p = c.as_part()
        i = c.as_part(inner=True)
        log.info("    part   %s " % p ) 
        log.info("    inner  %s " % i ) 

        if UNCOINCIDE and c.name == "pmt-hemi-bot":
            face, bottom, tubs = UNCOINCIDE.face_bottom_tubs(c.name)
            i.boundary = bottom   
            ret = [i]
        else:
            #i.parent = p
            ret = [p,i]
        pass
        pts = Parts(ret) 
        pts.csg = CSG(c)
        return pts


    def parts_primitive(self, c):
        log.info("name %s " % c.name)
        p = c.as_part()
        if UNCOINCIDE and c.name == "pmt-hemi-dynode":
            face, bottom, tubs = UNCOINCIDE.face_bottom_tubs(c.name)
            p.boundary = tubs
        pass
        pts = Parts([p])
        pts.csg = CSG(c)
        return pts 


    def parts(self):
        """
        :return: list of Part instances

        Provides parts from a single LV only, ie not
        following pv refs. Recursion is needed 
        in order to do link posXYZ transforms with geometry
        and skip them from the parts returned.
        """
        if type(self) is Physvol:
            return [] 

        if type(self) is Logvol and not self.is_logvol: 
            log.warning("inconsistent LV %s " % repr(self)) 

        if self.is_logvol:
            base = self.posXYZ 
            material = self.material
        else:
            base = None
            material = None
        pass

        comps = self.findall_("./*")  # one lev only
        self.link_prior_posXYZ(comps, base)

        rparts = Parts()

        csg_ = []

        for c in comps:
            xret = []
            if c.is_sphere and c.has_inner():
                log.info("-> sphere with inner  %s " % c.name)
                xret = self.parts_sphere_with_inner(c)  
            elif c.is_primitive:
                log.info("-> primitive %s " % c.name)
                xret = self.parts_primitive(c)
            elif c.is_intersection:
                log.info("-> intersection %s " % c.name)
                xret = c.partition_intersection(material=material) 
            elif c.is_union:
                log.info("-> union %s " % c.name)
                xret = c.partition_union(material=material) 
            elif c.is_composite:
                log.info("-> composite %s " % c.name )
                xret = c.parts() 
            elif c.is_posXYZ:
                pass
            else:
                log.warning("skipped component %s " % repr(c))
            pass

            if len(xret) > 0:
                rparts.extend(xret)
                if hasattr(xret, 'csg'):
                    csg_.append(xret.csg)
                pass
            pass
        pass

        log.info("%s : %s comps yield %s rparts %s csg " % (repr(self), len(comps), len(rparts), len(csg_)))

        if self.is_logvol:
            for p in rparts:
                if p.material is None:
                    p.material = self.material
                pass
            pass
            for cn in csg_:
                # associate lv to the first csg node
                if cn.lv is None:
                    cn.lv = self
                    #log.info("cn lv : %s " % repr(cn.lv))


        rparts.csg = csg_

        return rparts

    def geometry(self):
        return filter(lambda c:c.is_geometry, self.components())

   
 
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

    @classmethod
    def intersect(cls, name, a_, b_, inner=False):
        """
        Find Z intersect of two Z offset spheres 

        * http://mathworld.wolfram.com/Circle-CircleIntersection.html

        :param name: identifer passed to ZPlane
        :param a_: *Sphere* instance
        :param b_: *Sphere* instance

        :return zpl: *ZPlane* instance with z and y attributes where:
                     z is intersection coordinate 
                     y is radius of the intersection circle 
        """

        if inner:
            R = a_.innerRadius.value
            r = b_.innerRadius.value
        else:
            R = a_.outerRadius.value
            r = b_.outerRadius.value
        pass
        assert R is not None and r is not None

        a = a_.xyz
        b = b_.xyz

        log.debug(" R %s a %s " % ( R, repr(a)) )  
        log.debug(" r %s b %s " % ( r, repr(b)) )  

        dx = b[X] - a[X]
        dy = b[Y] - a[Y]
        dz = b[Z] - a[Z]

        assert dx == 0
        assert dy == 0
        assert dz != 0 

        d = dz             # use Sphere a_ frame
        dd_m_rr_p_RR = d*d - r*r + R*R 
        z = dd_m_rr_p_RR/(2.*d)
        yy = (4.*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.*d*d)
        y = math.sqrt(yy)

        # add a[Z] to return to original frame
        return ZPlane(name, z+a[Z], y ) 


    def has_inner(self):
        return self.innerRadius.value is not None 

    def as_csg(self):
        R = self.outerRadius.value
        if self.has_inner():
            r = self.innerRadius.value  
        else: 
            r = -1.0
        pass

        sta = self.startThetaAngle.value
        dta = self.deltaThetaAngle.value
        if sta is None:
            sta = 0. 
        if dta is None:
            dta = 180. 

        csg = []
        csg.append( [self.xyz[0], self.xyz[1], self.xyz[2], R] )
        csg.append( [sta,dta,0,r] )
        csg.append( [0,0,0,0] )
        csg.append( [0,0,0,0] )
        return csg

    def as_part(self, inner=False):
        if inner:
            radius = self.innerRadius.value 
        else:
            radius = self.outerRadius.value 
        pass
        if radius is None:
            return None

        sta = self.startThetaAngle.value
        dta = self.deltaThetaAngle.value

        assert self.xyz[2] == self.z
        z = self.z

        p = Part('Sphere', self.name + "_part", self.xyz, radius )

        if sta is None and dta is None:
            thetacut = False
            bb = self.bbox(z-radius, z+radius, -radius, radius)
        else: 
            # above case is equivalent to sta=0 dta=180
            if sta is None:
                sta = 0.
            rta = sta
            lta = sta + dta
            log.debug("Sphere.as_part %s  leftThetaAngle %s rightThetaAngle %s " % (self.name, lta, rta))
            assert rta >= 0. and rta <= 180.
            assert lta >= 0. and lta <= 180.
            zl = radius*math.cos(lta*math.pi/180.)
            yl = radius*math.sin(lta*math.pi/180.)
            zr = radius*math.cos(rta*math.pi/180.)
            yr = radius*math.sin(rta*math.pi/180.)
            ym = max(abs(yl),abs(yr))
            bb = self.bbox(z+zl, z+zr, -ym, ym)
        pass
        p.bbox = bb 

        return p 

    def part_zleft(self, zpl):
        radius = self.outerRadius.value 
        z = self.xyz[2]
        ymax = zpl.y 
        p = Part('Sphere', self.name + "_part_zleft", self.xyz, radius )
        p.bbox = self.bbox(z-radius, zpl.z, -ymax, ymax)
        return p 

    def part_zright(self, zpr):
        radius = self.outerRadius.value 
        z = self.xyz[2]
        ymax = zpr.y 
        p = Part('Sphere', self.name + "_part_zright", self.xyz, radius )
        p.bbox = self.bbox(zpr.z,z+radius, -ymax, ymax)
        return p 

    def part_zmiddle(self, zpl, zpr):
        p = Part('Sphere', self.name + "_part_zmiddle", self.xyz, self.outerRadius.value )
        ymax = max(zpl.y,zpr.y)
        p.bbox = self.bbox(zpl.z,zpr.z,-ymax,ymax )
        return p 

    def __repr__(self):
        return "sphere %20s : %s :  %s " % (self.name, self.outerRadius, self.posXYZ)

    def asrev(self):
        xyz = self.xyz 
        ro = self.outerRadius.value
        ri = self.innerRadius.value
        st = self.startThetaAngle.value
        dt = self.deltaThetaAngle.value
        sz = None
        wi = None
        if ri is not None and ri > 0:
            wi = ro - ri 

        return [Rev('Sphere', self.name,xyz, ro, sz, st, dt, wi)]

class Tubs(Primitive):

    sizeZ = property(lambda self:self.att('sizeZ'))

    def __repr__(self):
        return "Tubs %20s : outerRadius %s  sizeZ %s  :  %s " % (self.name, self.outerRadius, self.sizeZ, self.posXYZ)

    def as_part(self):
        sizeZ = self.sizeZ.value
        radius = self.outerRadius.value 
        z = self.xyz[2]
        p = Part('Tubs', self.name + "_part", self.xyz, radius, sizeZ )
        p.bbox = self.bbox(z-sizeZ/2, z+sizeZ/2, -radius, radius)
        return p 

    def as_csg(self):
        sizeZ = self.sizeZ.value
        outer = self.outerRadius.value 
        inner = self.innerRadius.value  
        csg = []
        csg.append( [self.xyz[0], self.xyz[1], self.xyz[2], outer] )
        csg.append( [0,0,sizeZ, inner] )
        csg.append( [0,0,0,0] )
        csg.append( [0,0,0,0] )
        return csg

    def asrev(self):
        sz = self.sizeZ.value
        r = self.outerRadius.value
        xyz = self.xyz 
        return [Rev('Tubs', self.name,xyz, r, sz )]


class PosXYZ(Elem):
    x = property(lambda self:self.att('x',0))
    y = property(lambda self:self.att('y',0))
    z = property(lambda self:self.att('z',0))
    def __repr__(self):
        return "PosXYZ  %s  " % (repr(self.z))



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
    def parse(cls, path):
        g = Dddb(parse_(path))
        g.init()
        return g

    def __call__(self, expr):
        return self.ctx.evaluate(expr)

    def init(self):
        self.g = self

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


if __name__ == '__main__':
    format_ = "[%(filename)s +%(lineno)3s %(funcName)20s() ] %(message)s" 
    logging.basicConfig(level=logging.DEBUG, format=format_)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")

    g.dump_context('PmtHemi')

    lv = g.logvol_("lvPmtHemi")

    pts = lv.parts()

    for pt in pts:
        log.info(pt)



