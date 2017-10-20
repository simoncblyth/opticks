#!/usr/bin/env python
"""
GCSG
======

* THIS IS SCHEDULED FOR ERADICATION,
  ONCE NCSG BINARY-TREE APPROACH HAS PERCOLATED THRU FULL CHAIN

* GCSG uses a first child, last child approach which is
  not convenient CSG evaluation on GPU

* Renamed from "CSG" to "GCSG" to avoid confusion with "NCSG" and dev.csg.csg:CSG 
  and associate this with corresponding C++ side class ggeo/GCSG.


Where/how GCSG is used
-------------------------

ddpart.py
    Parts lists retain association to the basis shapes
    they came from via gcsg attributes created
    within the Elem.partition methods 

    :: 

            350     def partition_intersection(self, material=None):
            ... 
            372         if len(spheres) == 3:
            373             pts = self.partition_intersection_3spheres(spheres, material=material)
            374         elif len(spheres) == 2:
            375             pts = self.partition_intersection_2spheres(spheres, material=material)
            376         else:
            ...
            380         pts.gcsg = GCSG(self, spheres)  # so pts keep hold of reference to the basis Elem shapes list they came from
            382         return pts

            385     def partition_union_intersection_tubs(self, comps, material=None, verbose=False):
            ...
            390         ipts = comps[0].partition_intersection()
            391         sparts = Part.ascending_bbox_zmin(ipts)
            ...
            428         rparts = Parts()
            429         rparts.extend(sparts)
            430         rparts.extend([tpart])
            ...
            436         rparts.gcsg = GCSG(self, [ipts.gcsg, comps[1]] )
            437 
            438         return rparts


    As the parts lists get formed and combined the gcsg attributes
    are passed along for the ride.

    This complicated way of doing things was needed 
    in order to retain connection between the parts and the source 
    volumes (for correct material/boundary/surface assignment for example) 
    but without having a tree structure to place the info upon. As the 
    partlist was the primary structure, with no node tree to work with.



analytic.py
    at the tail of analytic.py main GPmt.save
    invokes GCSG.serialize_list if the Buf instance 
    has a .gcsg attribute 


Buffer Layout 
---------------

::

    In [12]: p.csgbuf.view(np.int32)[:,2:]
    Out[12]: 

                                      typecode, nodeindex, parentindex, base        
                                      csgindex, nchild,  , firstchild,  lastchild_inc_recursive


    array([[[10,  1,  0,  0],        10:Union  
            [ 0,  2,  1,  5]],       0:idx   2:nc  1:fc 5:lcir         lcir - fc + 1 = 4 != 2 nc

               [[20,  0,  0,  0],        20:Intersection
                [ 1,  3,  2,  4]],       1:idx   3:nc  2:fc  4:lcir         lcir - fc + 1 = 3 = nc  (means no recursion, ie immediate primitives)           
             
                   [[ 3,  0,  0,  0],        3:sph
                    [ 2,  0,  0,  0]],       2:idx

                   [[ 3,  0,  0,  0],        3:sph
                    [ 3,  0,  0,  0]],       3:idx

                   [[ 3,  0,  0,  0],        3:sph
                    [ 4,  0,  0,  0]],       4:idx

           [[ 4,  0,  0,  0],        4:tubs
            [ 5,  0,  0,  0]],       5:idx   (lc of 0:idx)              


                                      typecode, nodeindex, parentindex, base        
                                      csgindex, nchild,  , firstchild,  lastchild_inc_recursive

           [[10,  2,  1,  0],        10:Union
            [ 6,  2,  7, 11]],        6:idx  2:nc 7:fc 11:lcir 

               [[20,  0,  0,  0],        20:Intersection
                [ 7,  3,  8, 10]],       7:idx fc of 6:idx

                   [[ 3,  0,  0,  0],        3:sph
                    [ 8,  0,  0,  0]],       8: fc of 7:idx

                   [[ 3,  0,  0,  0],        3:sph
                    [ 9,  0,  0,  0]],
         
                   [[ 3,  0,  0,  0],        3:sph
                    [10,  0,  0,  0]],       10: lc of 7:idx

           [[ 4,  0,  0,  0],        4:tubs
            [11,  0,  0,  0]],       11: lc of 6:idx



               [[10,  3,  2,  0],      10:Union         2:pidx
                [12,  2, 13, 14]],

                   [[ 3,  0,  0,  0],      3:sph
                    [13,  0,  0,  0]],

                   [[ 3,  0,  0,  0],      3:sph
                    [14,  0,  0,  0]],


               [[ 3,  4,  2,  0],      3:sph           2:pidx
                [15,  0,  0,  0]],

               [[ 4,  5,  2,  0],      4:tubs          2:pidx
                [16,  0,  0,  0]]], 

            dtype=int32)






"""
import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 
from opticks.ana.base import Buf 

log = logging.getLogger(__name__)


class Progeny(list):
    pass


TYPCODE = {'Union':10, 'Intersection':20, 'Sphere':3, 'Tubs':4 }


class GCSG(object):
    """
    """
    lvnodes = []
    def __init__(self, ele, children=[]):
        """
        :param ele: dd.py Elem instance wrapping an lxml one 
        :param children: comprised of either base elements or other CSG nodes 

        Hmm need to retain original node association for material assignment
        """
        self.ele = ele
        self.children = children
        self.typ = self.ele.__class__.__name__ 
        self.lv = None
        self.node = None


    def progeny_count(self):
        n = 0 
        for c in self.children:
            if type(c) is GCSG:
                n += c.progeny_count()
            else:
                n += 1 
            pass
        return n 

    def as_gcsg(self):
        return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]


    @classmethod
    def serialize_list(cls, gcsg):
        """
        :param gcsg: list of GCSG nodes
        :return gcsgbuf: Buf instance allowing attributes

        Hmm this was an early attempt to serialize a GCSG tree 
        prior to having the ability to render on GPU.

        Is it actually used from C++ side ?  
        YES, used for Geant4 test geometry construction in cfg4/CMaker etc..

        (2,0) typecode 
        (2,1) nodeindex 

        (3,0) current offset, ie index of the serialized record  
        (3,1) number of children
        (3,2) index corresponding to first child
        (3,3) index corresponding to last child (including recursion)   

        Formerly Tree.csg_serialize

        * array csg nodes and their progeny into flat array, with n elements
        * allocate numpy buffer (n,4,4)
        * huh, looks like flat is just to find the length

        """
        flat = []
        for cn in gcsg:
            flat.extend([cn])
            pr = cn.progeny()
            flat.extend(pr)
        pass
        for k,p in enumerate(flat):
            log.debug(" %s:%s " % (k, repr(p))) 

        data = np.zeros([len(flat),4,4],dtype=np.float32)
        offset = 0 
        for cn in gcsg:
            assert type(cn) is GCSG, (cn, type(cn))
            offset = cls.serialize_r(data, offset, cn)
        pass
        log.info("GCSG.serialize tot flattened %s final offset %s " % (len(flat), offset))
        assert offset == len(flat)
        buf = data.view(Buf) 
        return buf


    @classmethod
    def serialize_r(cls, data, offset, obj):
        """
        :param data: npy buffer to fill
        :param offset: primary index of buffer at which to start writing 
        :param obj: to serialize
        :return: offset updated to next place to write
        """
        pass
        nchild = 0 
        nodeindex = 0
        parentindex = 0 

        if type(obj) is GCSG:
            log.debug("**serialize offset %s typ %s [%s] (%s)" % (offset,obj.typ,repr(obj),repr(obj.node)))
            nchild = len(obj.children)
            payload = obj.ele if nchild == 0 else obj    # CSG nodes wrapping single elem, kinda different 
            if obj.lv is not None:
               cls.lvnodes.append(obj.lv)
               nodeindex = cls.lvnodes.index(obj.lv) + 1            
               if obj.node.parent is not None:
                   parentindex = cls.lvnodes.index(obj.node.parent.lv) + 1            
               pass
               log.info("serialize nodeindex %s parentindex %s " % (nodeindex, parentindex))
               log.debug("serialize nodeindex %s lv %s no %s " % (nodeindex, repr(obj.lv), repr(obj.node)))
               log.debug("serialize parentindex %s parent %s " % (parentindex, repr(obj.node.parent))) 
        else:
            payload = obj
        pass
        base = offset

        data[base] = payload.as_gcsg()
        data[base].view(np.int32)[2,0] = TYPCODE.get(payload.typ, -1 )   # name -> code using above dict
        data[base].view(np.int32)[2,1] = nodeindex
        data[base].view(np.int32)[2,2] = parentindex
        data[base].view(np.int32)[3,0] = base
        offset += 1 

        if nchild > 0:
            data[base].view(np.int32)[3,1] = nchild
            data[base].view(np.int32)[3,2] = base + 1
            for c in obj.children:
                offset = cls.serialize_r(data, offset, c)
            pass
            data[base].view(np.int32)[3,3] = offset - 1    # after the recursion
        pass
        return offset 


    def progeny(self):
        pgs = Progeny()
        nchild = len(self.children)

        if nchild == 0:
            pgs.extend([self.ele])
        else:
            for c in self.children:
                if type(c) is GCSG:
                    pg = c.progeny()
                    pgs.extend(pg)
                else:
                    pgs.extend([c])
                pass
            pass
        pass
        return pgs



    def __repr__(self):
        pc = self.progeny_count()
        return "GCSG node, lv %s, %s children, %s progeny [%s] " % ( repr(self.lv),len(self.children), pc, repr(self.ele)) + "\n" + "\n".join(map(lambda _:"    " + repr(_), self.children))



if __name__ == '__main__':
    pass 

