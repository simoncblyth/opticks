#!/usr/bin/env python
import os, logging, math
import numpy as np
X,Y,Z = 0,1,2

log = logging.getLogger(__name__)

class Rev(object):
    def __init__(self, typ, name, xyz, radius, sizeZ=None, startTheta=None, deltaTheta=None, width=None):
        self.typ = typ
        self.name = name
        self.xyz = xyz 
        self.radius = radius
        self.sizeZ = sizeZ
        self.startTheta = startTheta
        self.deltaTheta = deltaTheta
        self.width = width

    def __repr__(self):
        return "Rev('%s','%s' xyz:%s, r:%s, sz:%s, st:%s, dt:%s wi:%s)" % \
            (self.typ, self.name, self.xyz, self.radius, self.sizeZ, self.startTheta, self.deltaTheta, self.width)


class ZPlane(object):
    def __init__(self, name, z, y):
        """
        xy symmetry assumed
        """
        self.name = name
        self.z = z
        self.y = y   
    def __repr__(self):
        return "ZPlane %s z:%s y:%s " % (self.name, self.z, self.y )

class Part(object):

    @classmethod 
    def ascending_bbox_zmin(cls, parts):
        return sorted(parts, key=lambda p:p.bbox.zmin)

    @classmethod 
    def intersect_tubs_sphere(cls, name, tubs, sphere, sign ):
        """
        :param name: identifier of ZPlane created
        :param tubs: tubs Part instance  
        :param sphere: sphere Part instance  
        :param sign: 1 or -1 sign of the sqrt  

        Sphere at zp on Z axis

            xx + yy + (z-zp)(z-zp) = RR    

        Cylinder along Z axis from -sizeZ/2 to sizeZ/2
           
            xx + yy = rr

        Intersection is a circle in Z plane  

            (z-zp) = sqrt(RR - rr) 

        """ 
        R = sphere.radius 
        r = tubs.radius

        RR_m_rr = R*R - r*r
        assert RR_m_rr > 0

        iz = math.sqrt(RR_m_rr)  
        assert iz < tubs.sizeZ 

        return ZPlane(name, sphere.xyz[Z] + sign*iz, r) 

        
    def enable_endcap(self, tag):
        ENDCAP_P = 0x1 <<  0
        ENDCAP_Q = 0x1 <<  1 
        pass
        if tag == "P":
            self.flags |= ENDCAP_P 
        elif tag == "Q":
            self.flags |= ENDCAP_Q 
        else:
            log.warning("tag is not P or Q, for the low Z endcap (P) and higher Z endcap (Q)")


    def __init__(self, typ, name, xyz, radius, sizeZ=0.):
        """
        see cu/hemi-pmt.cu for where these are used 
        """
        self.typ = typ
        self.name = name
        self.xyz = xyz
        self.radius = radius
        self.sizeZ = sizeZ   # used for Tubs
        self.bbox = None
        self.parent = None
        self.node = None
        self.material = None
        self.boundary = None

        self.flags = 0
        # Tubs endcap control

        if typ == 'Sphere':
            self.typecode = 1
        elif typ == 'Tubs':
            self.typecode = 2
        elif typ == 'Box':
            self.typecode = 3
        else:
            assert 0


    @classmethod
    def make_container(cls, parts, factor=3. ):
        """
        create container box for all the parts 
        optionally enlarged by a multiple of the bbox extent
        """
        bb = BBox([0,0,0],[0,0,0])
        for pt in parts:
            #print pt
            bb.include(pt.bbox)
        pass
        bb.enlarge(factor)

        p = Part('Box', "make_container_box", bb.xyz, 0., 0. )
        p.bbox = bb
        log.info(p)
        return p 


    def __repr__(self):
        return "Part %6s %12s %32s %15s r:%6s sz:%5s %40s %s" % (self.typ, self.material, self.name, repr(self.xyz), self.radius, self.sizeZ, repr(self.bbox), self.boundary) 

    def as_quads(self):
        quads = []
        quads.append( [self.xyz[0], self.xyz[1], self.xyz[2], self.radius] )
        quads.append( [self.sizeZ, 0, 0, 0] )
        for q in self.bbox.as_quads():
            quads.append(q)
        return quads
           

class BBox(object):
    def __init__(self, min_, max_):
        self.min_ = np.array(min_)
        self.max_ = np.array(max_)

    def include(self, other):
        """
        Expand this bounding box to encompass another
        """
        self.min_ = np.minimum(other.min_, self.min_)
        self.max_ = np.maximum(other.max_, self.max_)

    def enlarge(self, factor):
        """
        Intended to duplicate ggeo-/GVector.hh/gbbox::enlarge
        """
        dim = self.max_ - self.min_
        ext = dim.max()/2.0
        vec = np.repeat(ext*factor, 3)
        self.min_ = self.min_ - vec 
        self.max_ = self.max_ + vec 

    def _get_zmin(self):
        return self.min_[Z]
    def _set_zmin(self, val):
        self.min_[Z] = val 
    zmin = property(_get_zmin, _set_zmin)

    def _get_zmax(self):
        return self.max_[Z]
    def _set_zmax(self, val):
        self.max_[Z] = val 
    zmax = property(_get_zmax, _set_zmax)

    def _get_xymin(self):
        """
        xy symmetry assumed
        """ 
        assert self.min_[X] == self.min_[Y]
        return self.min_[X]
    def _set_xymin(self, val):
        self.min_[X] = val 
        self.min_[Y] = val 
    xymin = property(_get_xymin, _set_xymin)

    def _get_xymax(self):
        assert self.max_[X] == self.max_[Y]
        return self.max_[X]
    def _set_xymax(self, val):
        self.max_[X] = val 
        self.max_[Y] = val 
    xymax = property(_get_xymax, _set_xymax)

    x = property(lambda self:self.xyz[X])
    y = property(lambda self:self.xyz[Y])
    z = property(lambda self:self.xyz[Z])
    xyz = property(lambda self:(self.min_+self.max_)/2.)

    def as_quads(self,scale=1):
        qmin = np.zeros(4)
        qmin[:3] = self.min_*scale
        qmax = np.zeros(4)
        qmax[:3] = self.max_*scale
        return qmin, qmax 

    def __repr__(self):
        xyz = self.xyz 

        assert xyz[X] == xyz[Y] == 0. 
        return "BB %30s %30s z %6.2f" % (str(self.min_), str(self.max_), xyz[Z] )



if __name__ == '__main__':

    bb = BBox([-100,-100,-100],[100,100,100])
    print bb 

