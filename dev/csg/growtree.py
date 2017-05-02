#!/usr/bin/env python
"""
FAILED APPEMPT TO GROW BALANCED TREES, 
INSTEAD ADOPTED SIMPLER CSG.uniontree approach
which builds complete tree and then prunes it 


If ever need to get this workin sometime...  perhaps:
 
* http://www.geeksforgeeks.org/linked-complete-binary-tree-its-creation/

"""
import logging 
log = logging.getLogger(__name__)

from csg import CSG


class HomogeneousTree(object):
    def __init__(self):
        self.root = CSG(self.operator)
        self.curr = self.root

    def add(self, primitive):
        self.add_(primitive)
        self.root.analyse()
        log.info("root.height after add %r " % self.root.height )

    def _get_curr(self):
        return self._curr
    def _set_curr(self, node):
        if hasattr(self, '_curr') and hasattr(self._curr, 'textra'):
             del self._curr.textra
        pass
        self._curr = node
        self._curr.textra = "*"
        pass
    curr = property(_get_curr, _set_curr)

    def add_(self, primitive):
        assert primitive.is_primitive 
        assert self.curr.is_operator

        if self.curr.left is None:
            self.curr.left = primitive
        elif self.curr.right is None:
            self.curr.right = primitive
        else:
            # need to grow a new operator... either growing upwards or downwards
            log.info("grow new operator from curr %r depth %d root.height %d elev %d " % (self.curr, self.curr.depth, self.root.height, self.curr.elevation ) )           
            elev = self.curr.elevation
            upwards = elev <= 1
            if upwards: 
                oldroot = self.root

                self.root = CSG(self.operator)
                self.root.left = oldroot

                newop = CSG(self.operator)
                newop.left = primitive

                self.root.right = newop
                self.curr = newop   
            else:
                # growing downwards add intermediary opnode
                oldcurr = self.curr

                newop = CSG(self.operator)
                newop.left = oldcurr
                newop.right = primitive

                assert oldcurr.parent.right == oldcurr
                oldcurr.parent.right = newop

                self.curr = newop
            pass
        pass
    pass
            
class UnionTree(HomogeneousTree):
    operator = "union"


if __name__ == '__main__':


    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split() + sprim.split() )
    nprim = len(primitives)


    ut = UnionTree()

    for p in primitives:
        print "-----------------add %s----" % p.dsc
        ut.add(p)

        print ut.root.txt

   

  
