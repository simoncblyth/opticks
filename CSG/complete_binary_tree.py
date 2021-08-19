#!/usr/bin/env python
"""
"""
from opticks.analytic.textgrid import TextGrid

def inorder_(nds):
    """Return inorder sequence of nodes"""
    inorder = []
    def inorder_r(n):
        if n is 0:return
        nd = nds[n-1]
        inorder_r(nd.l)
        inorder.append(nd.o)
        inorder_r(nd.r)
    pass
    inorder_r(1)
    return inorder


# https://codegolf.stackexchange.com/questions/177271/find-the-number-of-leading-zeroes-in-a-64-bit-integer
f=lambda n:-1<n<2**63and-~f(2*n|1)   # clz : count leading zeros   

def ffs(x):
    """Returns the index, counting from 0, of the least significant set bit in `x`."""
    return (x&-x).bit_length()-1

def tree_height(numNodes):
    """
    #define TREE_HEIGHT(numNodes) ( __ffs((numNodes) + 1) - 2)
    """
    return ffs(numNodes+1) - 1    ## maybe __ffs is off-by-1 from above ffs

class Nd(object):
    def __init__(self, o, numNode):
        l = (o << 1) 
        r = (o << 1) + 1 
        p = o >> 1
        d = 63-f(o)  # tree depth 

        self.o = o
        self.l = 0 if l > numNode else l 
        self.r = 0 if r > numNode else r
        self.p = p
        self.d = d

    def __repr__(self):
        return "Nd  p:{p:4b}    o:{o:4b} l:{l:4b} r:{r:4b}".format(o=self.o, l=self.l, r=self.r, p=self.p)    

def inorder_tree(numNodes):
    """
    :param numNode:
    :return nn: array of 1-based level-order indices returned in-order

    [8, 4, 9, 2, 10, 5, 11, 1, 12, 6, 13, 3, 14, 7, 15]

                                    1
                      10                          11
                100       101               110         111
             1000 1001 1100 1101        1100 1101     1110 1111

              8  4 9  2 10 5  11    1    12 6 13   3 14  7  15

    """
    nds = []
    for i in range(numNodes):
        nd = Nd(i+1, numNodes) 
        nds.append(nd) 
    pass
    nn = inorder_(nds)
    pass
    return nn

def layout_tree(nds):
    """
    :param nds: array of nodes in level-order (the usual serialization order)
    :return tree: TextGrid laying out the repr of the nodes 
    """
    nn = inorder_tree(len(nds)) 
    width = len(nds)
    height = tree_height(len(nds))

    tree = TextGrid(2*(height+1), width*2 )
    for i,n in enumerate(nn):
        nd = nds[n-1]
        depth = 63-f(n) 
        tree.a[depth*2, i*2] = str(nd)
    pass 
    return tree


class N(object):
    def __init__(self, i):
        self.i = i 
    def __repr__(self):
        return "N{obj.i:b}".format(obj=self)


if __name__ == '__main__':
    #it = inorder_tree(15)
    #print(it)
    nds = list(map(lambda i:N(i+1), range(15)))
    tree = layout_tree(nds)
    print(tree)


