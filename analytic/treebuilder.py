#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.analytic.csg import CSG 


class TreeBuilder(object):
    @classmethod 
    def balance(cls, tree):
        """
        Note that positivization is done inplace whereas
        the balanced tree is created separately 
        """
        if not tree.is_positive_form():
            log.fatal("cannot balance tree that is not in positive form")
            assert 0 
        pass
        ops = tree.operators_()
        hops = tree.operators_(minsubdepth=2)   # operators above the bileaf operators  

        if len(ops) == 1:
            op = ops[0]
            prims = tree.primitives()
            balanced = cls.commontree(op, prims, tree.name+"_prim_balanced" )
        elif len(hops) == 1: 
            op = hops[0]
            bileafs = tree.subtrees_(subdepth=1)
            balanced = cls.bileaftree(op, bileafs, tree.name+"_bileaf_balanced" )
        else:
            log.warning("balancing trees of this structure not implemented, tree %r " % tree)
            tree.meta.update(err="TreeBuilder.balance fail")
            balanced = tree
        pass
        return balanced

    @classmethod
    def bileaftree(cls, operator, bileafs, name):
        tb = TreeBuilder(bileafs, operator=operator, bileaf=True)
        root = tb.root 
        root.name = name
        root.analyse()
        return root

    @classmethod
    def commontree(cls, operator, primitives, name):
        """
        :param operator: CSG enum int 
        :param primitives: list of CSG instances 
        :param name:
        :return root: CSG instance for root node
        """
        tb = TreeBuilder(primitives, operator=operator)
        root = tb.root 
        root.name = name
        root.analyse()
        return root

    @classmethod
    def uniontree(cls, primitives, name):
        return cls.commontree(CSG.UNION, primitives, name )

    @classmethod
    def intersectiontree(cls, primitives, name):
        return cls.commontree(CSG.INTERSECTION, primitives, name )


    def __init__(self, subs, operator=CSG.UNION, bileaf=False):
        """
        :param subs: list of CSG instance primitives or bileafs when bileaf is True
        """
        self.operator = operator 

        if not bileaf:
            nprim = len(list(map(lambda n:n.is_primitive, subs)))
            assert nprim == len(subs) and nprim > 0
        else:
            nbileaf = len(list(map(lambda n:n.is_bileaf, subs)))
            assert nbileaf == len(subs) and nbileaf > 0
            nprim = 2*nbileaf
            log.debug("TreeBuilder bileaf mode, nbileaf: %s nprim:%s " % (nbileaf, nprim))
        pass
        self.nprim = nprim
        self.subs = subs

        # find complete binary tree height sufficient for nprim leaves
        #
        #     height: 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10, 
        #     tprim : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 
        #
        height = -1
        for h in range(10):
            tprim = 1 << h    
            if tprim >= nprim:
                height = h
                break
            pass
        pass
        assert height > -1 

        log.debug("TreeBuilder nprim:%d required height:%d " % (nprim, height))
        self.height = height
        
        if not bileaf:
            if height == 0:
                assert len(subs) == 1
                root = subs[0]
            else: 
                root = self.build(height, placeholder=CSG.ZERO)
                self.populate(root, subs[::-1]) 
                self.prune(root)
            pass
        else:
            assert height > 0
            root = self.build(height-1, placeholder=CSG.ZERO)   # bileaf are 2 levels, so height - 1
            self.populate(root, subs[::-1]) 
            self.prune(root)
        pass
        self.root = root 

    def build(self, height, placeholder=CSG.ZERO):
        """
        Build complete binary tree with all operators the same
        and CSG.ZERO placeholders for elevation 0
        """
        def build_r(elevation, operator):
            if elevation > 1:
                node = CSG(operator, name="treebuilder_midop")
                node.left  = build_r(elevation-1, operator) 
                node.right = build_r(elevation-1, operator)
            else:
                node = CSG(operator, name="treebuilder_bileaf")
                node.left = CSG(placeholder, name="treebuilder_bileaf_left")
                node.right = CSG(placeholder, name="treebuilder_bilead_right")
            pass
            return node  
        pass
        root = build_r(height, self.operator ) 
        return root

    def populate(self, root, subs):
        """
        :param root: CSG operator instance for root node
        :param subs: reversed list of the CSG instance primitive baubles to be hung on the tree

        During an inorder traverse of the complete binary tree, 
        the placeholder CSG.ZERO leaves are replaced with primitives
        popped off the subs list.
        """
        inorder = root.inorder_() 
        log.debug("populate filling tree of %d nodes with %d subs " % (len(inorder),len(subs)))
 
        for node in inorder:
            if node.is_operator:
                try:
                    if node.left.is_zero:
                        node.left = subs.pop()
                    if node.right.is_zero:
                        node.right = subs.pop()
                    pass
                except IndexError:
                    pass
                pass
            pass
        pass

    def prune(self, root):
        """
        Pulling leaves partnered with CSG.ZERO up to a higher elevation. 
        """
        def prune_r(node):
            if node is None:return
            if node.is_operator:
                prune_r(node.left)
                prune_r(node.right)

                # postorder visit
                if node.left.is_lrzero:
                    node.left = CSG(CSG.ZERO)
                elif node.left.is_rzero:
                    ll = node.left.left
                    node.left = ll
                pass
                if node.right.is_lrzero:
                    node.right = CSG(CSG.ZERO)
                elif node.right.is_rzero:
                    rl = node.right.left
                    node.right = rl
                pass
            pass
        pass
        prune_r(root)




def test_treebuilder():
    log.info("test_treebuilder")
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split() + sprim.split() )
    nprim = len(primitives)

    for n in range(0,nprim):
        tb = TreeBuilder(primitives[0:n+1])
        tb.root.analyse()
        print(tb.root.txt) 

def test_uniontree():
    log.info("test_uniontree")
 
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split())

    for n in range(0,len(primitives)):
        root = TreeBuilder.uniontree(primitives[0:n+1], name="test_uniontree")
        print("\n%s" % root.txt)





def test_balance():
    log.info("test_balance")
 
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split())

    sp,bo,co,zs,cy,tr = primitives


    root = sp - bo - co - zs - cy - tr

    root.analyse()    
    #root.subdepth_()

    print(root.txt)

    prims = root.primitives()
    print("prims : %s " % repr(prims))

    root.positivize() 

    balanced = TreeBuilder.balance(root)
    print(balanced.txt)


def make_bileaf(name):
    outer = CSG("cylinder", name=name+"_outer")
    inner = CSG("cylinder", name=name+"_inner")
    tube = outer - inner
    return  tube

def test_balance_bileaf():
    log.info("test_balance_bileaf")
    
    a = make_bileaf("a") 
    b = make_bileaf("b") 
    c = make_bileaf("c") 
    d = make_bileaf("d") 
    e = make_bileaf("e") 
    f = make_bileaf("f") 
    g = make_bileaf("g") 

    roots = []
    roots.append(a)
    roots.append(a+b)
    roots.append(a+b+c)
    roots.append(a+b+c+d)
    roots.append(a+b+c+d+e)
    roots.append(a+b+c+d+e+f)
    roots.append(a+b+c+d+e+f+g)

    for root in roots:
        root.analyse()
        root.positivize()

        print(root.txt)
        balanced = TreeBuilder.balance(root)

        print(balanced.txt)
    pass



if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    #test_treebuilder()
    #test_uniontree()
    test_balance()
    #test_balance_bileaf()


