#!/usr/bin/env python
"""


::

    In [1]: run ncsg.py

    In [4]: n
    Out[4]: Node  0 : dig afa2 pig d41d : LV lvPmtHemi                           Pyrex None : None  : None 

    In [5]: n.lv
    Out[5]: LV lvPmtHemi                           Pyrex None : None 

    In [6]: n.lv.comps()
    Out[6]: 
    [Union             pmt-hemi  ,
     PV pvPmtHemiVacuum      /dd/Geometry/PMT/lvPmtHemiVacuum ]

    In [7]: n.lv.comps()[0]
    Out[7]: Union             pmt-hemi  

    In [8]: n.lv.comps()[0].comps()
    Out[8]: 
    [Intersection  pmt-hemi-glass-bulb  ,
     Tubs        pmt-hemi-base : outerRadius PmtHemiGlassBaseRadius : 42.25   sizeZ PmtHemiGlassBaseLength : 169.0   :  None ,
     PosXYZ  -0.5*PmtHemiGlassBaseLength : -84.5   ]

    In [9]: n.lv.comps()[0].comps()[0]
    Out[9]: Intersection  pmt-hemi-glass-bulb  

    In [10]: n.lv.comps()[0].comps()[0].comps()
    Out[10]: 
    [sphere  pmt-hemi-face-glass : PmtHemiFaceROC : 131.0  :  None ,
     sphere   pmt-hemi-top-glass : PmtHemiBellyROC : 102.0  :  None ,
     PosXYZ  PmtHemiFaceOff-PmtHemiBellyOff : 43.0   ,
     sphere   pmt-hemi-bot-glass : PmtHemiBellyROC : 102.0  :  None ,
     PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0   ]



::

    In [18]: tr.get(0).lv.geometry()
    Out[18]: [Union             pmt-hemi  ]

    In [19]: tr.get(1).lv.geometry()
    Out[19]: [Union         pmt-hemi-vac  ]

    In [20]: tr.get(2).lv.geometry()
    Out[20]: [Union     pmt-hemi-cathode  ]

    In [21]: tr.get(3).lv.geometry()
    Out[21]: [sphere         pmt-hemi-bot : PmtHemiBellyROCvac : 99.0  :  None ]

    In [22]: tr.get(4).lv.geometry()
    Out[22]: [Tubs      pmt-hemi-dynode : outerRadius PmtHemiDynodeRadius : 27.5   sizeZ PmtHemiGlassBaseLength-PmtHemiGlassThickness : 166.0   :  None ]



Cathode is union of two spherical shells, using outerRadius/innerRadius to specify 
shell dimensions, can translate this into a union of two differences:: 

                Union
               /      \    
              /        \
             Diff      Diff 
            /   \     /    \
           s     s   s      s


Additionally the spheres are z-sliced in theta, 
handle this by using zsphere primitive which has z-range restriction.

* startThetaAngle (default is 0)
* deltaThetaAngle (default is pi)



::

    120   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    121     <union name="pmt-hemi-cathode">
    122       <sphere name="pmt-hemi-cathode-face"
    123           outerRadius="PmtHemiFaceROCvac"
    124           innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    125           deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    126       <sphere name="pmt-hemi-cathode-belly"
    127           outerRadius="PmtHemiBellyROCvac"
    128           innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"
    129           startThetaAngle="PmtHemiBellyCathodeAngleStart"
    130           deltaThetaAngle="PmtHemiBellyCathodeAngleDelta"/>
    131       <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
    132     </union>
    133   </logvol>




"""

import os, logging, sys, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.dev.csg.csg import CSG 

from ddbase import Dddb, Sphere, Tubs, Intersection, Union, Difference
from treebase import Tree


class NCSGTranslator(object):
    """
    Translate single volume detdesc primitives and CSG operations
    into an NCSG style node tree
    """
    kls2csgname = {
      Sphere:"sphere",
      Tubs:"box",
      Intersection:"intersection",
      Union:"union",
      Difference:"difference",
    }

    @classmethod
    def TranslateLV(cls, lv ):
        lvgeom = lv.geometry()
        assert len(lvgeom) == 1, "expecting single CSG operator or primitive Elem within LV"
        return cls.translate(lvgeom[0]) 
 
    @classmethod
    def translate(cls, node):
        assert node.is_operator ^ node.is_primitive, "node must either be operator or primitive "
        return cls.translate_primitive(node) if node.is_primitive else cls.translate_operator(node) 

    @classmethod
    def translate_primitive(cls, en):
        pr = cls.kls2csgname.get(type(en), None)
        assert pr, ("primitive Elem node type not handled", en, type(en))

        cn = CSG(pr, name=en.name)

        cn.param[0] = en.xyz[0] 
        cn.param[1] = en.xyz[1] 
        cn.param[2] = en.xyz[2]

        if type(en) is Sphere:
            cn.param[3] = en.outerRadius.value 
        elif type(en) is Tubs:
            cn.param[3] = en.outerRadius.value 
            cn.param1[0] = en.sizeZ.value
        else:
            pass
        pass              

        cn.elem = en   # <-- temporary during dev, not used downstream
        return cn 

    @classmethod
    def translate_operator(cls, en):
        """
        Source Elem xml tree CSG operator nodes with three children 
        have to be divided up to fit into binary CSG tree::

                                
                   1
                  / \ 
                 10  11
                /  \
               100 101

        """
        op = cls.kls2csgname.get(type(en), None)
        assert op, ("operator node type not handled", en, type(en))
 
        children = en.geometry()
        nchild = len(children)

        if nchild == 2:

            cn = CSG(op, name=en.name)
            cn.left = cls.translate(children[0])  
            cn.right = cls.translate(children[1])  

        elif nchild == 3:

            cn = CSG(op, name=en.name)

            ln = CSG(op, name=en.name + "_split3")
            ln.left = cls.translate(children[0])
            ln.right = cls.translate(children[1])

            cn.left = ln
            cn.right = cls.translate(children[2])

        else:
            assert 0, "CSG operator nodes must have 2 or 3 children" 
        pass
        return cn

        



if __name__ == '__main__':

    args = opticks_main(apmtidx=2)

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    ncsgnode = NCSGTranslator.TranslateLV( tr.root.lv )

    CSG.Dump(ncsgnode)





