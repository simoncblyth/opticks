#!/usr/bin/env python
"""

See tboolean-pmt for testing of the conversion of 
detdesc XML analytic geometry into NCSG input serialization
via: $(tboolean-dir)/tboolean_pmt.py  (tboolean-pmt-edit)


"""

import os, logging, sys, math, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG 

from ddbase import Dddb, Sphere, Tubs, Intersection, Union, Difference
from treebase import Tree


class NCSGConverter(object):
    """
    Translate single volume detdesc primitives and CSG operations
    into an NCSG style node tree
    """
    @classmethod
    def ConvertLV(cls, lv ):
        """
        :param lv: Elem
        :return cn: CSG node instance 
        """
        lvgeom = lv.geometry()
        assert len(lvgeom) == 1, "expecting single CSG operator or primitive Elem within LV"

        cn = cls.convert(lvgeom[0]) 

        if lv.posXYZ is not None:
            assert cn.transform is None, cn.transform 
            translate  = "%s,%s,%s" % (lv.xyz[0], lv.xyz[1], lv.xyz[2])
            cn.translate = translate 
            log.info("TranslateLV posXYZ:%r -> translate %s  " % (lv.posXYZ, translate) )
        pass
        return cn 

 
    @classmethod
    def convert(cls, node):
        """
        :param node: instance of ddbase.Elem subclass 
        :return cn: CSG node
        """
        assert node.is_operator ^ node.is_primitive, "node must either be operator or primitive "
        cn = cls.convert_primitive(node) if node.is_primitive else cls.convert_operator(node) 
        return cn 

    @classmethod
    def convert_Sphere(cls, en, only_inner=False):
        """
        :param en: source element node
        :param use_slab: alternative approach using intersect with a slab rather than the nascent zsphere  
        :param only_inner: used to control/distinguish internal recursive call handling the inner sphere
   
        Prior to implementing zsphere with caps, tried using infinite slab 
        in boolean intersection with the sphere to effect the zslice.
        But this approach unavoidably yields a cap, as switching off the 
        slab caps causes the intersection with the slab to yield nothing.  
        Hence proceeded to implement zsphere with cap handling.

        * z-slice sphere primitive OR intersect with a slab ?
        * r-range sphere primitive OR difference two spheres ? 

        * doing z and r both at once is problematic for param layout 
        """
        outerRadius = en.outerRadius.value 
        innerRadius = en.innerRadius.value 
        x = en.xyz[0]
        y = en.xyz[1]
        z = en.xyz[2]
  
        has_inner = not only_inner and innerRadius is not None 
        if has_inner:
            inner = cls.convert_Sphere(en, only_inner=True)  # recursive call to make inner sphere
        pass

        radius = innerRadius if only_inner else outerRadius 
        assert radius, (radius, innerRadius, outerRadius, only_inner)


        startThetaAngle = en.startThetaAngle.value 
        deltaThetaAngle = en.deltaThetaAngle.value 

        log.info("convert_Sphere outerRadius:%s innerRadius:%s radius:%s only_inner:%s  has_inner:%s " % (outerRadius,innerRadius,radius, only_inner, has_inner)) 

        zslice = startThetaAngle is not None or deltaThetaAngle is not None

        if zslice:
            if startThetaAngle is None:
                startThetaAngle = 0.

            if deltaThetaAngle is None:
                deltaThetaAngle = 180.



            # z to the right, theta   0 -> z=r, theta 180 -> z=-r
            rTheta = startThetaAngle
            lTheta = startThetaAngle + deltaThetaAngle

            assert rTheta >= 0. and rTheta <= 180.
            assert lTheta >= 0. and lTheta <= 180.
            zmin = radius*math.cos(lTheta*math.pi/180.)
            zmax = radius*math.cos(rTheta*math.pi/180.)
            assert zmax > zmin, (startThetaAngle, deltaThetaAngle, rTheta, lTheta, zmin, zmax )

            log.info("convert_Sphere rTheta:%5.2f lTheta:%5.2f zmin:%5.2f zmax:%5.2f azmin:%5.2f azmax:%5.2f " % (rTheta, lTheta, zmin, zmax, z+zmin, z+zmax ))

            cn = CSG("zsphere", name=en.name, param=[x,y,z,radius], param1=[zmin,zmax,0,0], param2=[0,0,0,0]  )

            ZSPHERE_QCAP = 0x1 << 1   # zmax
            ZSPHERE_PCAP = 0x1 << 0   # zmin
            flags = ZSPHERE_QCAP | ZSPHERE_PCAP 

            cn.param2.view(np.uint32)[0] = flags 
            pass
        else:
            cn = CSG("sphere", name=en.name)
            cn.param[0] = x
            cn.param[1] = y
            cn.param[2] = z
            cn.param[3] = radius
        pass 
        if has_inner:
            ret = CSG("difference", left=cn, right=inner )
            #ret = inner
            #ret = cn 
        else: 
            ret = cn 
        pass
        return ret



    @classmethod
    def convert_Tubs(cls, en):
        cn = CSG("cylinder", name=en.name)
        cn.param[0] = en.xyz[0] 
        cn.param[1] = en.xyz[1] 
        cn.param[2] = en.xyz[2]
        cn.param[3] = en.outerRadius.value 
        cn.param1[0] = en.sizeZ.value

        PCAP = 0x1 << 0 
        QCAP = 0x1 << 1 
        flags = PCAP | QCAP 
        cn.param1.view(np.uint32)[1] = flags  

        return cn


    @classmethod
    def convert_primitive(cls, en):
        convert_method_name = "convert_%s" % en.__class__.__name__ 
        convert_method = getattr(cls, convert_method_name, None )
        assert convert_method, "missing convert method: %s " % convert_method_name  
        #log.info("convert_primitive with %s " % convert_method_name )
        cn = convert_method(en)
        cn.elem = en   # <-- temporary during dev, not used downstream
        return cn 

    @classmethod
    def convert_operator(cls, en):
        """
        Source Elem xml tree CSG operator nodes with three children 
        have to be divided up to fit into binary CSG tree::

                                
                   1
                  / \ 
                 10  11
                /  \
               100 101

        """
        op = en.__class__.__name__.lower()
        assert op in ["intersection", "union", "difference"]
 
        children = en.geometry()
        nchild = len(children)

        if nchild == 2:

            cn = CSG(op, name=en.name)
            cn.left = cls.convert(children[0])  
            cn.right = cls.convert(children[1])  

        elif nchild == 3:

            cn = CSG(op, name=en.name)

            ln = CSG(op, name=en.name + "_split3")
            ln.left = cls.convert(children[0])
            ln.right = cls.convert(children[1])

            cn.left = ln
            cn.right = cls.convert(children[2])

        else:
            assert 0, "CSG operator nodes must have 2 or 3 children" 
        pass
        return cn

        



if __name__ == '__main__':

    args = opticks_main(apmtidx=2)

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    ncsgnode = NCSGConverter.ConvertLV( tr.root.lv )

    CSG.Dump(ncsgnode)












