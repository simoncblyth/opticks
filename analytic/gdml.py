#!/usr/bin/env python
"""
gdml.py : parsing GDML
=========================
       

"""
import os, re, logging, math, collections

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import find_ranges
from opticks.analytic.csg import CSG 
from opticks.analytic.treebuilder import TreeBuilder
from opticks.analytic.glm import make_trs, make_transform
from opticks.analytic.prism import make_trapezoid

import numpy as np
import lxml.etree as ET
import lxml.html as HT


tostring_ = lambda _:ET.tostring(_)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])


def construct_transform(obj):
    tla = obj.position.xyz if obj.position is not None else None
    rot = obj.rotation.xyz if obj.rotation is not None else None
    sca = obj.scale.xyz if obj.scale is not None else None
    order = "trs"

    #elem = filter(None, [tla,rot,sca])
    #if len(elem) > 1:
    #    log.warning("construct_transform multi %s " % repr(obj))
    #pass

    return make_transform( order, tla, rot, sca , three_axis_rotate=True, transpose_rotation=True, suppress_identity=False, dtype=np.float32 )


class G(object):
    """
    Base wrapper type for lxml parsed GDML elements 
    """
    pvtype = 'PhysVol'
    lvtype = 'Volume'
    postype = 'position'

    typ = property(lambda self:self.__class__.__name__)
    name  = property(lambda self:self.elem.attrib.get('name',None))
    xml = property(lambda self:tostring_(self.elem))
    gidx = property(lambda self:"[%d]" % self.idx if hasattr(self, 'idx') else '[]' )  # materials, volumes and top level solids have idx attribute 

    is_primitive = property(lambda self:issubclass(self.__class__ , Primitive )) 
    is_boolean   = property(lambda self:issubclass(self.__class__ , Boolean )) 


    def _get_shortname(self):
        """/dd/Geometry/PMT/lvPmtHemi0xc133740 -> lvPmtHemi"""
        base = self.name.split("/")[-1]
        return base[:-9] if base[-9:-7] == '0x' else base
    shortname = property(_get_shortname)

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

    def __str__(self):
        return self.xml
   


class Material(G):
    state = property(lambda self:self.att('state', '', typ=str ))
    def __repr__(self):
        return "%s %s %s %s" % (self.gidx, self.typ, self.name, self.state )
 

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



    def _get_subsolids(self):
        ss = []
        def subsolids_r(solid, top=False):
            if not top:
                ss.append(solid.idx)
            pass
            if solid.is_boolean:
                subsolids_r(solid.first)
                subsolids_r(solid.second)
            pass
        pass
        subsolids_r(self, top=True)
        return ss
    subsolids = property(_get_subsolids)
    subsolidranges = property(lambda self:list(find_ranges(sorted(self.subsolids))))
    subsolidcount = property(lambda self:len(self.subsolids))



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
        line = "%s %s %s  " % (self.gidx, self.typ, self.name )
        lrep_ = lambda label,obj:"     %s:%r"%(label,obj)
        return "\n".join([line, lrep_("l",self.first), lrep_("r",self.second)])

    def as_ncsg(self):

        if not hasattr(self.first, 'as_ncsg'):
            print self.first 
        pass
        left = self.first.as_ncsg()
        right = self.second.as_ncsg()

        assert left, " left fail as_ncsg for first : %r self: %r " % (self.first, self)
        assert right, "right fail as_ncsg for second : %r self: %r " % (self.second, self)

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

    @classmethod
    def fromstring(cls, st ):
        gg = GDML.fromstring("<gdml><solids>" + st + "</solids></gdml>")
        so = gg.solids(-1)   # pick last to allow composite booleans
        return so

    @classmethod
    def deltaphi_segment_via_slab_DEPRECATED(cls, obj, phi0, phi1, dist):
        xyzw_ = lambda phi:(np.cos(phi*np.pi/180.), np.sin(phi*np.pi/180.),0,0)

        slab_a = 0
        slab_b = dist 

        slab0 = CSG("slab", param=xyzw_(phi0+90),param1=[slab_a,slab_b,0,0] ) 
        slab1 = CSG("slab", param=xyzw_(phi1-90),param1=[slab_a,slab_b,0,0] ) 

        # flipped signs to get segment from intended quadrant
        
        obj_slab0 = CSG("intersection", left=obj, right=slab0 )
        obj_slab0_slab1 = CSG("intersection", left=obj_slab0, right=slab1 )        

        return obj_slab0_slab1

    def __repr__(self):
        return "%s %s %s %s rmin %s rmax %s  x %s y %s z %s  " % (self.gidx, self.typ, self.name, self.lunit, self.rmin, self.rmax, self.x, self.y, self.z)

class Tube(Primitive):
    """
    G4Tubs is GDML serialized as tube 
    """
    deltaphi_segment_enabled = True

    @classmethod 
    def make_cylinder(cls, radius, z1, z2, name):
        cn = CSG("cylinder", name=name)
        cn.param[0] = 0
        cn.param[1] = 0
        cn.param[2] = 0    
        cn.param[3] = radius
        cn.param1[0] = z1
        cn.param1[1] = z2
        return cn

    @classmethod 
    def make_disc(cls, x, y, inner, radius, z1, z2, name):
        cn = CSG("disc", name=name)
        cn.param[0] = x
        cn.param[1] = y
        cn.param[2] = inner    
        cn.param[3] = radius
        cn.param1[0] = z1
        cn.param1[1] = z2
        return cn

    def as_cylinder(self, nudge_inner=0.01):
        hz = self.z/2.
        has_inner = self.rmin > 0.

        if has_inner:
            dz = hz*nudge_inner 
            inner = self.make_cylinder(self.rmin, -(hz+dz), (hz+dz), self.name + "_inner") 
        else:
            inner = None
        pass
        outer = self.make_cylinder(self.rmax, -hz, hz, self.name + "_outer" )
        tube = CSG("difference", left=outer, right=inner, name=self.name + "_difference" ) if has_inner else outer

        has_deltaphi = self.deltaphi < 360
        if has_deltaphi and self.deltaphi_segment_enabled:

             assert self.aunit == 'deg'
             phi0 = self.startphi
             phi1 = self.startphi + self.deltaphi
             sz  = self.z*1.01
             sr  = self.rmax*1.5

             ## TODO: calulate how much the segmenting prism needs to poke beyind the radius 
             ##       to avoid the outside plane from cutting the cylinder 
             
             segment = CSG.MakeSegment(phi0, phi1, sz, sr )
             log.info("as_cylinder doing phi0/phi1/sz/sr segmenting : name %s phi0 %s phi1 %s sz %s sr %s " % (self.name, phi0, phi1, sz, sr))
             tube_segment = CSG("intersection", left=tube, right=segment )

             #tube_segment = self.deltaphi_slab_segment(tube, phi0, phi1, dist)
             #result.balance_disabled = True 

             result = tube_segment
        else:
             result = tube
        pass
        return result

    def as_disc(self):
        hz = self.z/2.
        return self.make_disc( self.x, self.y, self.rmin, self.rmax, -hz, hz, self.name + "_disc" ) 

    def as_ncsg(self, hz_disc_cylinder_cut=1.):
        hz = self.z/2.
        rmin = self.rmin
        rmax = self.rmax
        pick_disc = hz < hz_disc_cylinder_cut 
        if pick_disc:
            log.debug("Tube.as_ncsg.CSG_DISC %s as hz < cut,  hz:%s cut:%s rmin:%s rmax:%s " % (self.name, hz, hz_disc_cylinder_cut, rmin, rmax))
        pass
        return self.as_disc() if pick_disc else self.as_cylinder()



class Sphere(Primitive):
    deltaphi_slab_segment_enabled = False

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

        log.debug("Sphere.as_ncsg radius:%s only_inner:%s  has_inner:%s " % (radius, only_inner, has_inner)) 

        zslice = startThetaAngle > 0. or deltaThetaAngle < 180.

        if zslice:
            zmin = radius*math.cos(lTheta*math.pi/180.)
            zmax = radius*math.cos(rTheta*math.pi/180.)
            assert zmax > zmin, (startThetaAngle, deltaThetaAngle, rTheta, lTheta, zmin, zmax )

            log.debug("Sphere.as_ncsg rTheta:%5.2f lTheta:%5.2f zmin:%5.2f zmax:%5.2f azmin:%5.2f azmax:%5.2f " % (rTheta, lTheta, zmin, zmax, z+zmin, z+zmax ))

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
            ret = CSG("difference", left=cn, right=inner, name=self.name + "_difference"  )
        else: 
            ret = cn 
        pass

        has_deltaphi = self.deltaphi < 360
        if has_deltaphi and not only_inner and self.deltaphi_slab_segment_enabled:
             assert self.aunit == 'deg'
             phi0 = self.startphi
             phi1 = self.startphi + self.deltaphi
             rmax = self.rmax + 1
             ret_segment = self.deltaphi_slab_segment(ret, phi0, phi1, rmax)
             result = ret_segment
        else:
             result = ret
        pass

        return result





class Ellipsoid(Primitive):
    ax = property(lambda self:self.att('ax', 0, typ=float))
    by = property(lambda self:self.att('by', 0, typ=float))
    cz = property(lambda self:self.att('cz', 0, typ=float))
    zcut1 = property(lambda self:self.att('zcut1', 0, typ=float))
    zcut2 = property(lambda self:self.att('zcut2', 0, typ=float))


    def _get_semi_axes(self):
        ax = self.ax
        by = self.by
        cz = self.cz
        a = np.array([ax,by,cz], dtype=np.float32)
        return a
    semi_axes = property(_get_semi_axes)    


    def as_ncsg(self):
        ax = self.semi_axes
        cn = CSG.MakeEllipsoid(axes=ax, name=self.name, zcut1=self.zcut1, zcut2=self.zcut2)
        return cn


class Torus(Primitive):
    rtor = property(lambda self:self.att('rtor', 0, typ=float))
    def as_ncsg(self):
        cn = CSG.MakeTorus(R=self.rtor, r=self.rmax, name=self.name)
        return cn


class Box(Primitive):
    def as_ncsg(self):
        assert self.lunit == 'mm' 
        cn = CSG("box3", name=self.name)
        cn.param[0] = self.x
        cn.param[1] = self.y
        cn.param[2] = self.z
        cn.param[3] = 0
        return cn

class Cone(Primitive):
    """
    GDML inner cone translated into CSG difference 

    rmin1: inner radius at base of cone 
    rmax1: outer radius at base of cone 
    rmin2: inner radius at top of cone 
    rmax2: outer radius at top of cone
    z height of cone segment

    """
    rmin1 = property(lambda self:self.att('rmin1', 0, typ=float))
    rmin2 = property(lambda self:self.att('rmin2', 0, typ=float))

    rmax1 = property(lambda self:self.att('rmax1', 0, typ=float))
    rmax2 = property(lambda self:self.att('rmax2', 0, typ=float))
    z = property(lambda self:self.att('z', 0, typ=float))
    pass

    @classmethod
    def make_cone(cls, r1,z1,r2,z2, name):
        cn = CSG("cone", name=name)
        cn.param[0] = r1
        cn.param[1] = z1
        cn.param[2] = r2
        cn.param[3] = z2
        return cn 

    def as_ncsg(self, only_inner=False):
        pass
        assert self.aunit == "deg" and self.lunit == "mm" and self.deltaphi == 360. and self.startphi == 0. 
        has_inner = not only_inner and (self.rmin1 > 0. or self.rmin2 > 0. )
        if has_inner:
            inner = self.as_ncsg(only_inner=True)  # recursive call to make inner 
        pass

        r1 = self.rmin1 if only_inner else self.rmax1 
        z1 = -self.z/2

        r2 = self.rmin2 if only_inner else self.rmax2 
        z2 = self.z/2
   
        cn = self.make_cone( r1,z1,r2,z2, self.name )

        return CSG("difference", left=cn, right=inner ) if has_inner else cn

    def __repr__(self):
        return "%s %s z:%s rmin1:%s rmin2:%s rmax1:%s rmax2:%s  " % (self.gidx, self.typ, self.z, self.rmin1, self.rmin2, self.rmax1, self.rmax2 )

    def plot(self, ax):
        zp0 = FakeZPlane(z=0., rmin=self.rmin1,rmax=self.rmax1) 
        zp1 = FakeZPlane(z=self.z, rmin=self.rmin2,rmax=self.rmax2) 
        zp = [zp0, zp1]
        PolyCone.Plot(ax, zp)



class FakeZPlane(object):
    def __init__(self, z, rmin, rmax):
        self.z = z
        self.rmin = rmin
        self.rmax = rmax

    def __repr__(self):
        return "%s %s %s %s " % (self.__class__.__name__, self.z, self.rmin, self.rmax)


class ZPlane(Primitive):
    pass
    

class Trapezoid(Primitive):
    """
    The GDML Trapezoid is formed using 5 dimensions:

    x1: x length at -z
    x2: x length at +z
    y1: y length at -z
    y2: y length at +z
    z:  z length

    The general form is ConvexPolyhedron, modelled by a list of planes 
    defining half spaces which come together to define the polyhedron.
    
    * 4:tetrahedron
    * 5:triangular prism
    * 6:trapezoid, cube, box

    Q&A: 

    * where to transition from specific Trapezoid to generic ConvexPolyhedron ? Did this immediately with python input. 
    * where to extract the bbox, thats easier before conversion to generic ? Again immediately in python.
    
    * :google:`convex polyhedron bounding box`

    Enumerating the vertices of a convex polyhedron defined by its planes
    looks complicated, hence need to extract and store the bbox prior
    to generalization of the shape. 

    * http://cgm.cs.mcgill.ca/~avis/doc/avis/AF92b.pdf

    ::

        In [1]: run gdml.py 

        In [2]: trs = g.findall_("solids//trd")

        In [3]: len(trs)
        Out[3]: 2

        In [13]: trs = g.findall_("solids//trd")

        In [14]: trs
        Out[14]: 
        [Trapezoid name:SstTopRadiusRibBase0xc271078 z:2228.5 x1:160.0 y1:20.0 x2:691.02 y2:20.0  ,
         Trapezoid name:SstInnVerRibCut0xbf31118 z:50.02 x1:100.0 y1:27.0 x2:237.2 y2:27.0  ]

        In [4]: print trs[0]
        <trd xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" lunit="mm" name="SstTopRadiusRibBase0xc271078" x1="160" x2="691.02" y1="20" y2="20" z="2228.5"/>
            

        In [5]: print trs[1]
        <trd xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" lunit="mm" name="SstInnVerRibCut0xbf31118" x1="100" x2="237.2" y1="27" y2="27" z="50.02"/>


    Tips for finding LV with a solid... use filternodes_so to get some nodes
    then can use parent links to get idea of local tree around the solid.

    ::

        In [17]: sivr = t.filternodes_so("SstInnVerRib")

        In [18]: sivr[0]
        Out[18]: 
        Node 4473 : dig a60e pig 8be9 depth 11 nchild 0  
        pv:PhysVol /dd/Geometry/AD/lvOIL#pvSstInnVerRibs#SstInnVerRibs#SstInnVerRibRot0xbf1abc0
         Position mm 2428.0 0.0 -40.0  None 
        lv:Volume /dd/Geometry/AdDetails/lvSstInnVerRibBase0xbf31748 /dd/Materials/StainlessSteel0xc2adc00 SstInnVerRibBase0xbf30b50
           Subtraction SstInnVerRibBase0xbf30b50  
             l:Box SstInnVerRibBox0xbf310d8 mm rmin 0.0 rmax 0.0  x 120.0 y 25.0 z 4875.0  
             r:Trapezoid name:SstInnVerRibCut0xbf31118 z:50.02 x1:100.0 y1:27.0 x2:237.2 y2:27.0  
           Material /dd/Materials/StainlessSteel0xc2adc00 solid : Position mm 2428.0 0.0 -40.0  

        In [26]: print list(map(lambda n:n.index, sivr))
        [4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 6133, 6134, 6135, 6136, 6137, 6138, 6139, 6140]


        In [28]: sivr[0].parent
        Out[28]: 
        Node 3155 : dig 8be9 pig a856 depth 10 nchild 520  
        pv:PhysVol /dd/Geometry/AD/lvSST#pvOIL0xc241510
         Position mm 0.0 0.0 7.5  Rotation deg 0.0 0.0 -180.0  
        lv:Volume /dd/Geometry/AD/lvOIL0xbf5e0b8 /dd/Materials/MineralOil0xbf5c830 oil0xbf5ed48
           Tube oil0xbf5ed48 mm rmin 0.0 rmax 2488.0  x 0.0 y 0.0 z 4955.0  
           Material /dd/Materials/MineralOil0xbf5c830 solid
           PhysVol /dd/Geometry/AD/lvOIL#pvOAV0xbf8f638



    """
    x1 = property(lambda self:self.att('x1', 0, typ=float))
    x2 = property(lambda self:self.att('x2', 0, typ=float))
    y1 = property(lambda self:self.att('y1', 0, typ=float))
    y2 = property(lambda self:self.att('y2', 0, typ=float))
    z = property(lambda self:self.att('z', 0, typ=float))

    def __repr__(self):
        return "%s name:%s z:%s x1:%s y1:%s x2:%s y2:%s  " % (self.typ, self.name, self.z, self.x1, self.y1, self.x2, self.y2 )

    def as_ncsg(self):
        assert self.lunit == 'mm' 
        cn = CSG("trapezoid", name=self.name)
        planes, verts, bbox, src = make_trapezoid(z=self.z, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2 )
        cn.planes = planes
        cn.param2[:3] = bbox[0]
        cn.param3[:3] = bbox[1]

        cn.meta.update(src.srcmeta)

        return cn


class PolyCone(Primitive):
    """
    ::

        In [1]: run gdml.py 

        In [2]: pcs = gdml.findall_("solids//polycone")

        In [3]: len(pcs)
        Out[3]: 13

        In [6]: print pcs[0]
        <polycone xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" aunit="deg" deltaphi="360" lunit="mm" name="gds_polycone0xc404f40" startphi="0">
          <zplane rmax="1520" rmin="0" z="3070"/>              # really flat cone 
          <zplane rmax="75" rmin="0" z="3145.72924106399"/>    # little cylinder 
          <zplane rmax="75" rmin="0" z="3159.43963177189"/>    
        </polycone>

       In [7]: print pcs[1]
        <polycone xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" aunit="deg" deltaphi="360" lunit="mm" name="iav_polycone0xc346448" startphi="0">
          <zplane rmax="1565" rmin="0" z="3085"/>                 # flat cylinder (repeated rmax)
          <zplane rmax="1565" rmin="0" z="3100"/>                 # repeated z, changed r   <--- polycone abuse, zero height truncated cone
          <zplane rmax="1520.39278882354" rmin="0" z="3100"/>     #    big taper cone
          <zplane rmax="100" rmin="0" z="3174.43963177189"/>
        </polycone>


    Summarizing with zp, see that

    * all using same rmin, thats polycone abuse : better to subtract a cylinder
    * common pattern repeated z with different rmax to model "lips"

    ::

        pcs[ 0].zp :          gds_polycone0xc404f40  3 z:      [3145.72924106399, 3159.43963177189, 3070.0] rmax:                     [1520.0, 75.0] rmin:               [0.0]  
        pcs[ 7].zp :          SstTopHubBot0xc2635b8  2 z:                                  [-320.0, -340.0] rmax:                            [220.5] rmin:             [150.5]  
        pcs[ 8].zp :         SstTopHubMain0xc263d80  2 z:                                     [-320.0, 0.0] rmax:                            [170.5] rmin:             [150.5]  


        pcs[ 1].zp :          iav_polycone0xc346448  4 z:                [3100.0, 3085.0, 3174.43963177189] rmax:  [100.0, 1520.39278882354, 1565.0] rmin:               [0.0]  
        pcs[ 2].zp :             IavTopHub0xc405968  4 z:         [0.0, 85.5603682281126, 110.560368228113] rmax:                     [100.0, 150.0] rmin:              [75.0]  
        pcs[ 3].zp :       CtrGdsOflBotClp0xbf5dec0  4 z:                                 [0.0, 25.0, 30.0] rmax:                      [36.5, 150.0] rmin:              [31.5]  
        pcs[ 4].zp :          OcrGdsPrtPln0xbfa1408  4 z:                               [0.0, 160.0, 185.0] rmax:                     [100.0, 150.0] rmin:              [75.0]  
        pcs[ 5].zp :          lso_polycone0xc02a418  4 z:      [4076.62074383385, 4058.59604160589, 3964.0] rmax:              [1930.0, 50.0, 125.0] rmin:               [0.0]  
        pcs[ 6].zp :          oav_polycone0xbf1c840  4 z:      [4094.62074383385, 3937.0, 4000.02470222796] rmax:            [2040.0, 1930.0, 125.0] rmin:               [0.0]  

        pcs[ 9].zp :             OavTopHub0xc2c9030  6 z:                          [0.0, 57.0, 90.0, 120.0] rmax:                [98.0, 68.0, 125.0] rmin:              [50.0]  
        pcs[10].zp :       CtrLsoOflTopClp0xc178498  6 z:                         [0.0, 16.0, 184.0, 200.0] rmax:              [102.5, 112.5, 100.0] rmin:              [50.0]  
        pcs[11].zp :       OcrGdsLsoPrtPln0xc104000  4 z:         [0.0, 214.596041605889, 184.596041605889] rmax:                       [98.0, 68.0] rmin:              [50.0]  
        pcs[12].zp :       OcrCalLsoPrtPln0xc2fadd8  4 z:         [0.0, 214.596041605889, 184.596041605889] rmax:                       [98.0, 68.0] rmin:              [50.0]  


    """
    pass
    zplane = property(lambda self:self.findall_("zplane"))

    zp_num = property(lambda self:len(self.zplane))
    zp_rmin = property(lambda self:list(set(map(lambda _:_.rmin,self.zplane))))
    zp_rmax = property(lambda self:list(set(map(lambda _:_.rmax,self.zplane))))
    zp_z = property(lambda self:list(set(map(lambda _:_.z,self.zplane))))
    zp = property(lambda self:"%s %30s %2d z:%50r rmax:%35r rmin:%20r " % (self.gidx, self.name, self.zp_num, self.zp_z, self.zp_rmax, self.zp_rmin)) 

    def __repr__(self):
        return self.zp


    def prims(self):
        """
        Auto correct simple case of wrong z-order::

               421     <polycone aunit="deg" deltaphi="360" lunit="mm" name="PMT_3inch_pmt_solid_cyl0x1c9da50" startphi="0">
               422       <zplane rmax="30.001" rmin="0" z="-15.8745078663875"/>
               423       <zplane rmax="30.001" rmin="0" z="-75.8755078663876"/>
               424     </polycone>

        """
        zp = self.zplane 
        zz = map(lambda _:_.z, zp)
 
        if len(zp) == 2 and zz[0] > zz[1]:
           log.warning("Polycone swap misordered pair of zplanes for %s " % self.name)
           zp = list(reversed(zp))
        pass   

        prims = []
        for i in range(1,len(zp)):
            zp1 = zp[i-1]
            zp2 = zp[i]

            r1 = zp1.rmax
            r2 = zp2.rmax
            z1 = zp1.z
            z2 = zp2.z

            if z2 == z1:
                log.debug("skipping z2 == z1 zp" )
            else:
                #assert z2 > z1, (z2,z1)
                if not z2 > z1:
                    raise ValueError("Polycone bad z-order expect z2>z1 : but z1 %s z2 %s " % (z1,z2 ))
                pass 
                name = self.name + "_zp_%d" % i 
                pr = Tube.make_cylinder( r1, z1, z2, name ) if r1 == r2 else Cone.make_cone( r1, z1, r2, z2, name )
                prims.append(pr)
            pass
        pass
        return prims

    def inner(self):
        rmin = self.zp_rmin
        assert len(rmin) == 1 
        rmin = rmin[0]

        zp = self.zplane 
        z = map(lambda _:_.z, zp)
        zmax = max(z) 
        zmin = min(z)

        has_inner = rmin > 0.
        if has_inner:
            inner = Tube.make_cylinder( rmin, zmin, zmax, self.name + "_inner_cylinder" )
        else:
            inner = None
        pass
        return inner


    def as_ncsg(self):
        assert self.aunit == "deg" and self.lunit == "mm" and self.deltaphi == 360. and self.startphi == 0. 
        try:
            prims = self.prims()
        except ValueError as e:
            log.fatal("Polycone.as_ncsg failed ValueError : %r " % e )  
            return None 
        pass
        cn = TreeBuilder.uniontree(prims, name=self.name + "_uniontree")
        inner = self.inner()
        #return CSG("difference", left=cn, right=inner ) if inner is not None else cn
        return cn


    def plot(self, ax):
        self.Plot(ax, self.zplane)

    @classmethod
    def Plot(cls, ax, zp):

        rmin = map(lambda _:_.rmin, zp)
        rmax = map(lambda _:_.rmax, zp)

        z = map(lambda _:_.z, zp)
        zmax = max(z) 
        zmin = min(z)
        zsiz = zmax - zmin

        for i in range(1,len(zp)):
            zp0 = zp[i-1]
            zp1 = zp[i]
            ax.plot( [zp0.rmax,zp1.rmax], [zp0.z,zp1.z])        
            ax.plot( [-zp0.rmax,-zp1.rmax], [zp0.z,zp1.z])        

            if i == 1:
                ax.plot( [-zp0.rmax, zp0.rmax], [zp0.z, zp0.z] )
            pass
            if i == len(zp) - 1:
                ax.plot( [-zp1.rmax, zp1.rmax], [zp1.z, zp1.z] )
            pass
        pass

        for i in range(1,len(zp)):
            zp0 = zp[i-1]
            zp1 = zp[i]
            ax.plot( [zp0.rmin,zp1.rmin], [zp0.z,zp1.z])        
            ax.plot( [-zp0.rmin,-zp1.rmin], [zp0.z,zp1.z])        

            if i == 1:
                ax.plot( [-zp0.rmin, zp0.rmin], [zp0.z, zp0.z] )
            pass
            if i == len(zp) - 1:
                ax.plot( [-zp1.rmin, zp1.rmin], [zp1.z, zp1.z] )
            pass
        pass

        xmax = max(rmax)*1.2
        ax.set_xlim(-xmax, xmax )
        ax.set_ylim( zmin - 0.1*zsiz, zmax + 0.1*zsiz )






class Volume(G):
    """
    ::

        In [15]: for v in gdml.volumes.values():print v.material.shortname
        PPE
        MixGas
        Air
        Bakelite
        Air
        Bakelite
        Foam
        Aluminium
        Air
        ...

    """
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
        as is with its own transforms.

        Essentially are collapsing a subtree for the 
        handful of solids into the self contained instance
        of a list of csgnodes.
        """
        pass
 

    def __repr__(self):
        repr_ = lambda _:"   %r" % _ 
        pvs = map(repr_, self.physvol) 
        line = "%s %s %s %s %s" % (self.gidx, self.typ, self.name, self.materialref, self.solidref)
        return "\n".join([line, repr_(self.solid), repr_(self.material)] + pvs )

class PhysVol(G):
    """

    volume
         opticks.ana.pmt.gdml.Volume  

    transform
         4x4 np.ndarray homogeneous TRS matrix ie Translate-Rotate-Scale 
         combining the position, rotation and scale attributes

    position
    rotation
    scale
         dont use these, use transform that combines them


    """
    volumeref = property(lambda self:self.elem.find("volumeref").attrib["ref"])
    volume = property(lambda self:self.g.volumes[self.volumeref])
    children = property(lambda self:[self.volume])

    position = property(lambda self:self.find1_("position"))
    rotation = property(lambda self:self.find1_("rotation"))
    scale = property(lambda self:self.find1_("scale"))
    transform = property(lambda self:construct_transform(self))

    def __repr__(self):
        return "\n".join(["%s %s" % (self.typ, self.name)," %r %r " % ( self.position, self.rotation)])
     


class odict(collections.OrderedDict):
    """
    Used for GDML collections of materials, solids and volumes which are 
    always keyed by name.  Call method for convenient indexed access. 
    """
    def __call__(self, iarg):
        return self.items()[iarg][1]


class GDML(G):
    kls = {
        "material":Material,

        "tube":Tube,
        "torus":Torus,
        "sphere":Sphere,
        "ellipsoid":Ellipsoid,
        "box":Box,
        "cone":Cone,
        "polycone":PolyCone,
        "zplane":ZPlane,
        "trd":Trapezoid,

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
    def parse(cls, path=None):
        if path is None:
            path = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main
            log.info("gdmlpath defaulting to OPTICKS_GDMLPATH %s which is derived by opticks.ana.base from the IDPATH input envvar " % path )
        pass
        log.info("parsing gdmlpath %s " % path )
        gg = parse_(path)
        wgg = cls.wrap(gg, path=path)
        return wgg 

    @classmethod
    def fromstring(cls, st ):
        log.info("parsing string %s " % st )
        gg = ET.fromstring(st) 
        wgg = cls.wrap(gg)
        return wgg 

    @classmethod
    def wrap(cls, gdml, path=None):
        log.info("wrapping gdml element  ")
        gg = cls(gdml)
        gg.g = gg
        gg.path = path
        gg.string = tostring_(gdml) 
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
        """
        Heart of the GDML parsing.
        """
        self.materials = odict()
        self.solids = odict()
        self.volumes = odict()

        for i, e in enumerate(self.findall_("materials/material")):
            e.idx = i
            self.materials[e.name] = e 
        pass
        for i, e in enumerate(self.findall_("solids/*")):
            e.idx = i
            self.solids[e.name] = e 
        pass
        for i, e in enumerate(self.findall_("structure/*")):
            e.idx = i
            self.volumes[e.name] = e
        pass
        self.setup_world = self.elem.find("setup/world")
        self.worldvol = self.setup_world.attrib["ref"] if self.setup_world is not None else None

        vv = self.volumes.values()

        vvs = filter(lambda v:hasattr(v,'solid'), vv)

        log.info("vv %s vvs %s " % (len(vv),len(vvs)))
        #for v in vvs:print repr(v)

        self.lv2so = dict([(v.idx, v.solid.idx) for v in vvs])








def oneplot(obj):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    obj.plot(ax)
    plt.show()

def multiplot(fig, objs, nx=4, ny=4):
    for i in range(len(objs)):
        ax = fig.add_subplot(nx,ny,i+1, aspect='equal')
        objs[i].plot(ax)
    pass




def test_children(t):

    lvn = "/dd/Geometry/PMT/lvPmtHemi0x"
 
    l = t.filternodes_lv(lvn)  # all nodes 
    assert len(l) == 672

    n = l[0] 
    sibs = n.siblings 
    assert len(sibs) == 192
   
    cn = n.lv.solid.as_ncsg()
    cn.dump()

    v = n.children[0]
    for c in v.children:
        print c.pv.transform


def test_polycone(g):
    pcs = g.findall_("solids//polycone")
    for i,pc in enumerate(pcs):
        print "pcs[%2d].zp : %s " % (i, pc.zp)
        cn = pc.as_ncsg()

def test_gdml(g):
    print g.world
    g.world.rdump()
    g.volumes["/dd/Geometry/PMT/lvPmtHemi0xc133740"].rdump()

def test_trd(g):
    trs = g.findall_("solids//trd")
    cns = []
    for i,tr in enumerate(trs):
        cn = tr.as_ncsg()
        cns.append(cn)
 

def test_gdml_fromstring():
    gg = GDML.fromstring(r"""<gdml><solids><tube aunit="deg" deltaphi="360" lunit="mm" name="AdPmtCollar0xc2c5260" rmax="106" rmin="105" startphi="0" z="12.7"/></solids></gdml>""")    
    so = gg.solids(0)
    assert type(so) is Tube
    cn = so.as_ncsg()
    assert cn.typ == cn.DIFFERENCE

def test_primitive_fromstring():
    so = Primitive.fromstring(r"""<tube aunit="deg" deltaphi="360" lunit="mm" name="AdPmtCollar0xc2c5260" rmax="106" rmin="105" startphi="0" z="12.7"/>""")
    assert type(so) is Tube
    cn = so.as_ncsg()
    assert cn.typ == cn.DIFFERENCE




if __name__ == '__main__':

    args = opticks_main()


    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    g = GDML.parse()

    from opticks.analytic.treebase import Tree
    t = Tree(g.world) 

    test_children(t)
    test_polycone(g)
    test_trd(g)

    test_gdml_fromstring()
    test_primitive_fromstring()



