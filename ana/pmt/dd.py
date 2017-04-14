#!/usr/bin/env python
"""
dd.py
=======

Approach:

* kicking off from Dddb.logvol_ wraps results of 
  lxml find into do-everything Elem subclass appropriate to the tags of the elements

* primary purpose is the Elem.parts method, trace back from there to understand




Need to play some mixin tricks 

* http://stackoverflow.com/questions/8544983/dynamically-mixin-a-base-class-to-an-instance-in-python


"""
import os, re, logging, math
from opticks.ana.base import opticks_main


from ddbase import Dddb, Elem, Sphere, Tubs

from geom import Part, BBox, ZPlane, Rev
from gcsg import GCSG

log = logging.getLogger(__name__)


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


# intended to be a mixin to Elem
class ElemPartioner(object):
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


        pts.gcsg = GCSG(self, spheres) # so pts keep hold of reference to the basis Elem and shapes list they came from

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

        rparts.gcsg = GCSG(self, [ipts.gcsg, comps[1]] )

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
            if not hasattr(xret, 'gcsg'):
                xret.gcsg = GCSG(self, comps[0:2])
        else:
            xret = self.parts()   

        if xret is not None:
            rparts.extend(xret)
            if hasattr(xret, 'gcsg'):
                rparts.gcsg = xret.gcsg
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
        pts.gcsg = GCSG(c)
        return pts


    def parts_primitive(self, c):
        log.info("name %s " % c.name)
        p = c.as_part()
        if UNCOINCIDE and c.name == "pmt-hemi-dynode":
            face, bottom, tubs = UNCOINCIDE.face_bottom_tubs(c.name)
            p.boundary = tubs
        pass
        pts = Parts([p])
        pts.gcsg = GCSG(c)
        return pts 


    def parts(self, analytic_version=0):
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

        gcsg_ = []

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
                if hasattr(xret, 'gcsg'):
                    gcsg_.append(xret.gcsg)
                pass
            pass
        pass

        log.info("%s : %s comps yield %s rparts %s gcsg " % (repr(self), len(comps), len(rparts), len(gcsg_)))

        if self.is_logvol:
            for p in rparts:
                if p.material is None:
                    p.material = self.material
                pass
            pass
            for cn in gcsg_:
                # associate lv to the first gcsg node
                if cn.lv is None:
                    cn.lv = self
                    #log.info("cn lv : %s " % repr(cn.lv))


        rparts.gcsg = gcsg_

        return rparts

    def geometry(self):
        return filter(lambda c:c.is_geometry, self.components())




  

class SpherePartitioner(Sphere):

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


    def as_gcsg(self):
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

        gcsg = []
        gcsg.append( [self.xyz[0], self.xyz[1], self.xyz[2], R] )
        gcsg.append( [sta,dta,0,r] )
        gcsg.append( [0,0,0,0] )
        gcsg.append( [0,0,0,0] )
        return gcsg

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



class TubsPartitioner(Tubs):

    def as_part(self):
        sizeZ = self.sizeZ.value
        radius = self.outerRadius.value 
        z = self.xyz[2]
        p = Part('Tubs', self.name + "_part", self.xyz, radius, sizeZ )
        p.bbox = self.bbox(z-sizeZ/2, z+sizeZ/2, -radius, radius)
        return p 

    def as_gcsg(self):
        sizeZ = self.sizeZ.value
        outer = self.outerRadius.value 
        inner = self.innerRadius.value  
        gcsg = []
        gcsg.append( [self.xyz[0], self.xyz[1], self.xyz[2], outer] )
        gcsg.append( [0,0,sizeZ, inner] )
        gcsg.append( [0,0,0,0] )
        gcsg.append( [0,0,0,0] )
        return gcsg

    def asrev(self):
        sz = self.sizeZ.value
        r = self.outerRadius.value
        xyz = self.xyz 
        return [Rev('Tubs', self.name,xyz, r, sz )]






if __name__ == '__main__':
    args = opticks_main(apmtidx=2)

    g = Dddb.parse(args.apmtddpath)


    # override the element tag to wrapper class mapping  
    Dddb.kls["sphere"] = SpherePartitioner  
    Dddb.kls["tubs"] = TubsPartitioner  


    g.dump_context('PmtHemi')

    lv = g.logvol_("lvPmtHemi")

    pts = lv.parts()

    for pt in pts:
        log.info(pt)



