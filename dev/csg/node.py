#!/usr/bin/env python
"""
"""
import logging, copy
log = logging.getLogger(__name__)
import numpy as np

EMPTY = 0 
SPHERE = 1
BOX = 2 

_desc_sh = { EMPTY:"e", SPHERE:"s", BOX:"b" }
def desc_sh(sh):
    return _desc_sh[sh]


from opticks.optixrap.cu.boolean_h import desc_op, UNION, INTERSECTION, DIFFERENCE


class T(np.ndarray):
    """
    An array with a text grid representation::

        In [223]: a = np.empty((3,3),dtype=np.object)

        In [224]: t = T.init(a)


        In [230]: t[0,2] = "02"

        In [231]: t[2,0] = "20"

        In [232]: t
        Out[232]: 
           00    01    02
        
                 
           20            

        In [233]: t[1,0] = "10"

        In [234]: t
        Out[234]: 
           00    01    02

           10            

           20            

    """
    @classmethod
    def init(cls, a):  
        assert len(a.shape) == 2, a
        t = a.view(cls)
        return t

    def __repr__(self):
        row_ = lambda r:" ".join(map(lambda _:"%3s" % (_ if _ is not None else "") ,r))
        tab_ = lambda a:"\n".join(map(row_, a))
        return tab_(self)



class Node(object):
    def __init__(self, idx=0, l=None, r=None, **kwa):
        """
        :param idx: 1-based levelorder (aka breadth first) tree index, root at 1
        """
        self.idx = idx
        self.l = copy.deepcopy(l)   # for independence
        self.r = copy.deepcopy(r)

        self.next_ = None

        # below needed for CSG 
        self.shape = None 
        self.operation = None
        self._param = None
        self.parent = None
        self.name = "unnamed"
        self.ok = False
 
        self.apply_(**kwa)

    def _get_param(self):
        return self._param 
    def _set_param(self, v):
        self._param = np.asarray(v) if v is not None else None 
    param = property(_get_param, _set_param)

    def apply_(self, **kwa):
        for k,v in kwa.items():
            if k == "shape":
                self.shape = v
            elif k == "operation":
                self.operation = v
            elif k == "param":
                self.param = v
            elif k == "name":
                self.name = v 
            elif k == "ok":
                self.ok = v 
            else:
                log.warning("ignored Node param %s : %s " % (k,v))
            pass 

    def annotate(self):
        """
        * call this only on the root node
        * hmm should make tree complete first to get expected levelorder indices
        """
        # tree labelling
        self.maxidx = "WHERE IS THIS USED"
        self.lheight = Node.lheight_(self)

        Node.levelorder_label_i(self)
        self.maxdepth = Node.depth_r(self)

        Node.parenting_r(self)
        Node.postorder_threading_r(self)

        if self.name in ["root1","root2","root3","root4"]:
            Node.dress(self)
        pass

    @classmethod
    def dress(cls, root):
        """
        Assumes perfect trees 
        """
        nno = Node.nodecount_r(root)    
        lmo = Node.leftmost_leaf(root) 
        rmo = Node.rightmost_leaf(root) 
        nle = rmo.idx - lmo.idx + 1

        leftop = Node.leftmost(root)
        node = leftop
        while node is not None:
            if node.is_leaf:
                assert 0, "not expecting leaves" 
                node.apply_(shape=SPHERE, param=[0,0,0,100])
            elif node.is_bileaf:
                #node.apply_(operation=DIFFERENCE)
                node.apply_(operation=INTERSECTION) ## causes infinite tranche loop for iterative
                #node.apply_(operation=UNION)
                node.l.apply_(shape=SPHERE, param=[0,node.l.side*30+1,0,100] )             
                node.r.apply_(shape=SPHERE, param=[0,node.r.side*30+1,0,100] )             
                #node.l.apply_(shape=SPHERE, param=[node.l.side*10,0,0,100] )             
                #node.r.apply_(shape=SPHERE, param=[node.r.side*10,0,0,100] )             
            else:
                node.apply_(operation=UNION)
            pass
            #log.info(" dress %r " % node )
            node = node.next_ 


    @classmethod
    def traverse(cls, leftop, label="traverse"):
        """
        """
        print "%s : following thread..." % label
        node = leftop
        while node is not None:
            print node
            node = node.next_ 

    def _get_textgrid(self):
        if not hasattr(self, 'maxdepth'):
            self.annotate()

        maxdepth = self.maxdepth
       
        nodes = Node.inorder_r(self, [])
        #sides = map(lambda _:_.side, nodes)
        depths = map(lambda _:_.depth, nodes)
        #hw = max(map(abs,sides))

        #print "sides %r hw %d " % (sides, hw)


        ni = 2*(maxdepth+1+1)
        nj = len(nodes) + 1

        a = np.empty((ni,nj), dtype=np.object)

        if self.name is not None:
            a[0,0] = self.name
        pass

        for inorder_idx, node in enumerate(nodes):

            i = 2*node.depth + 2
            j = inorder_idx
     
            try:
                a[i-1,j] = node.tag 
                a[i,j]   = "o" # node.tag 
            except IndexError:
                print "IndexError depth:%2d inorder_idx:%2d i:%2d j:%2d : %s " % (node.depth, inorder_idx, i,j,node)

        pass
        return T.init(a) 
    txt = property(_get_textgrid)

    #def __str__(self):
    #    tg = self.textgrid
    #    return T.__str__(tg)


    def __repr__(self):
        if self.is_bare:
            if self.l is not None and self.r is not None:
                return "Node(%d,l=%r,r=%r)" % (self.idx, self.l, self.r)
            elif self.l is None and self.r is None:
                return "Node(%d)" % (self.idx)
            else:
                assert 0
        else:
            if self.is_primitive:
                return "%s.%s" % (self.tag, desc_sh(self.shape))
            else:
                return "%s.%s(%r,%r)" % ( self.tag, desc_op(self.operation),self.l, self.r )

    is_primitive = property(lambda self:self.shape is not None)
    is_operation = property(lambda self:self.operation is not None)
    is_bare = property(lambda self:self.operation is None and self.shape is None)

    is_leaf = property(lambda self:self.l is None and self.r is None)

    # bileaf is an operation applied to two leaf nodes, another name is a triple
    is_bileaf = property(lambda self:not self.is_leaf and self.l.is_leaf and self.r.is_leaf)

    is_left_requiring_levelorder = property(lambda self:self.idx % 2 == 0) # convention using 1-based levelorder index
    is_left = property(lambda self:self.parent is not None and self.parent.l is self)
    is_root = property(lambda self:self.parent is None)

    def _get_tag(self):
        if self.operation is not None:
            if self.operation in [UNION,INTERSECTION,DIFFERENCE]:
                ty = desc_op(self.operation)[0]
            else:
                assert 0
            pass
        elif self.shape is not None:
            if self.shape in [SPHERE, BOX, EMPTY]:
                ty = desc_sh(self.shape)
            else:
                assert 0
            pass
        elif self.is_leaf:
            ty = "p"
        else:
            ty = "o"
        pass
        return "%s%d" % (ty, self.idx)
    tag = property(_get_tag)

    @classmethod
    def nodecount_r(cls, root):
        if root is None:
            return 0
        return 1 + cls.nodecount_r(root.l) + cls.nodecount_r(root.r)

    @classmethod
    def is_complete_r(cls, root, index=1, nodecount=None, debug=False):
        if nodecount is None:
            nodecount = cls.nodecount_r(root)

        if index >= nodecount: 
            if debug:
               print "index %d >= nodecount %d -> not complete for node %r" % (index, nodecount, root)
            return False

        return cls.is_complete_r(root.l, index=2*index, nodecount=nodecount, debug=debug ) and cls.is_complete_r(root.r, index=2*index+1, nodecount=nodecount, debug=debug)


    @classmethod
    def lheight_(cls, root):
        level = 0
        p = root
        while p is not None:
           p = p.l
           level += 1
        pass
        return level


    @classmethod
    def is_perfect_i(cls, root):
        """
        Definition of a perfect binary tree:

        * all leaves at same depth, same as maxdepth
        * all non-leaves have both left and right children

        Perfect binary tree with 1-based levelorder index::

             1

             2                  3

             4        5         6         7

             8   9   10   11   12  13    14  15    

        
        For node i, where i in range 1 to n

        * root, i=1
        * left child, 2*i (if <= n)
        * right child, 2*i + 1 (if <= n)
        * parent, i/2 (if i != 1)
        * contiguous leaves come last in levelorder

        A perfect binary tree levelorder serialized into an array
        is navigable without tree reconstruction direct from the  
        array using the above index manipulations.

        The downside is that normally will need to 
        pad a tree with EMPTY primitives and UNION operations 
        to make it perfect.

        """
        maxdepth = cls.depth_r(root)  # label nodes with depth, 0 at root

        leafdepth = None
        q = []
        q.append(root)
        while len(q) > 0:
            node = q.pop(0)  # fifo
            if node.is_leaf: 
                if leafdepth is None:
                    leafdepth = node.depth
                    if leafdepth != maxdepth:
                        return False
                    pass
                pass
                if node.depth != leafdepth:
                    return False
                pass
            else:
                if node.l is None or node.r is None:
                    return False
                pass
            pass
            if node.l is not None:q.append(node.l)
            if node.r is not None:q.append(node.r)
        pass
        return True
 
    @classmethod
    def make_perfect_i(cls, root):
        """
        """
        while not cls.is_perfect_i(root): 
            maxdepth = cls.depth_r(root)  # label nodes with depth, 0 at root
            q = []
            q.append(root)
            while len(q) > 0:
                node = q.pop(0)   # bottom of q (ie fifo)
                assert node.depth <= maxdepth

                if node.is_leaf and node.depth < maxdepth:
                    l = Node()
                    r = Node()
                    l.depth = node.depth+1
                    r.depth = node.depth+1
                    node.l = l
                    node.r = r
                pass
                if node.l is not None:q.append(node.l)
                if node.r is not None:q.append(node.r)
            pass
        pass

    @classmethod
    def is_almost_complete_i(cls, root):
        """
        levelorder traverse of complete binary tree should see all the 
        leaves together at the last level. 

        Hmm consider pruing 6,7 in the below so 3 becomes a leaf...
        then all leaves are together 3,4,5
        but 3 is at higher level ?

        ::

             1
 
             2            3

             4      5     6       7

        """

        if root is None:return True
        q = []
        q.append(root)

        leaves = False
    
        idx = 1 
        while len(q) > 0:
           node = q.pop(0)   # bottom of q (ie fifo)

           if node.l is not None:
               if leaves:return False
               q.append(node.l)
           else:
               leaves = True
           pass

           if node.r is not None:
               if leaves:return False
               q.append(node.r)
           else:
               leaves = True
           pass
        pass
        return True


    @classmethod
    def levelorder_label_i(cls,root,idxbase=1):
        """
        Assign 1-based binary tree levelorder indices, eg for height 3 complete tree::

             1
 
             2            3

             4      5     6       7
 
             8   9  10 11 12  13  14   15

        """
        idx = idxbase 
        for node in Node.levelorder_i(root):
            node.idx = idx
            idx += 1
        pass


    @classmethod
    def depth_r(cls, node, depth=0, side=0, height=None):
        """
        Marking up the tree with depth and side
        """
        if node is None:
            return 

        if height is None:
            height = Node.lheight_(node)

        node.depth = depth
        node.side = side

        delta = 1 << (height - depth - 1)  # smaller side shifts as go away from root towards the leaves

        #print "depth_r depth %d side %d delta %d height %d height-depth-1 %s " % (depth, side, delta, height, height-depth-1)

        ldepth = depth
        rdepth = depth

        if node.l is not None: ldepth = cls.depth_r(node.l, depth+1, side-delta, height=height)
        if node.r is not None: rdepth = cls.depth_r(node.r, depth+1, side+delta, height=height)
    
        return max(ldepth, rdepth)
       
 

    @classmethod
    def progeny_i(cls,root):
        """

        1

        2          3

        4    5     6      7

        8 9  10 11 12 13 14 15

        """

        nodes = []
        q = []
        q.append(root)

        while len(q) > 0:
           node = q.pop(0)   # bottom of q (ie fifo)
           nodes.append(node)
           if not node.l is None:q.append(node.l)
           if not node.r is None:q.append(node.r)
        pass
        return nodes


    @classmethod
    def label_r(cls, node, idx, label):
        if node.idx == idx:
            log.info("label_r setting label %s idx %d node %r  " % (label, idx, node))
            setattr(node, label, 1)

        if node.l is not None:cls.label_r(node.l, idx, label)
        if node.r is not None:cls.label_r(node.l, idx, label)

    @classmethod
    def leftmost_leaf(cls, root):
        n = root  
        while n.l is not None:
            n = n.l
        return n

    @classmethod
    def rightmost_leaf(cls, root):
        n = root  
        while n.r is not None:
            n = n.r
        return n
 
    @classmethod
    def leftmost(cls, root):
        """
        :return: leftmost internal or operation node 
        """
        n = root 
        while n.l is not None:
            if n.l.is_leaf:
                break
            else:
                n = n.l
            pass
        return n

    @classmethod
    def rightmost(cls, root):
        """
        :return: rightmost internal or operation node 
        """
        n = root 
        while n.r is not None:
            if n.r.is_leaf:
                break
            else:
                n = n.r
            pass
        return n


    @classmethod
    def postorder_r(cls, root, nodes=[],leaf=True):
        """ 
        :param root:
        :param nodes: list 
        :param leaf: bool control of inclusion of leaf nodes with internal nodes

        Recursive postorder traversal
        """
        if root.l is not None: cls.postorder_r(root.l, nodes, leaf=leaf) 
        if root.r is not None: cls.postorder_r(root.r, nodes, leaf=leaf)
 
        if not leaf and root.is_leaf:
            pass  # skip leaves when leaf = False
        else: 
            nodes.append(root)
        pass
        return nodes


    @classmethod
    def inorder_r(cls, root, nodes=[], leaf=True, internal=True):
        """ 
        :param root:
        :param nodes: list 
        :param leaf: include leaf nodes
        :param internal: include non-leaf nodes, including root node

        Recursive inorder traversal
        """
        if root.l is not None: cls.inorder_r(root.l, nodes, leaf=leaf, internal=internal) 

        if root.is_leaf:
            if leaf:
                nodes.append(root)
            else:
                pass
            pass
        else:
            if internal:
                nodes.append(root)
            else:
                pass
            pass
        pass
        if root.r is not None: cls.inorder_r(root.r, nodes, leaf=leaf, internal=internal)
        return nodes


    @classmethod
    def levelorder_i(cls,root):
        """
        Why is levelorder easier iteratively than recursively ?
        """
        nodes = []
        q = []
        q.append(root)
        while len(q) > 0:
           node = q.pop(0) # bottom of q (ie fifo)
           nodes.append(node) 
           if node.l is not None:q.append(node.l)
           if node.r is not None:q.append(node.r)
        pass
        return nodes

    @classmethod
    def postorder_threading_r(cls, root):
        nodes = cls.postorder_r(root, nodes=[], leaf=False)
        for i in range(len(nodes)):
            node = nodes[i]
            next_ = nodes[i+1] if i < len(nodes)-1 else None
            node.next_ = next_
        pass


    @classmethod
    def parenting_r(cls, root, parent=None):
        if root.l is not None:
            cls.parenting_r(root.l, parent=root)
        pass
        if root.r is not None:
            cls.parenting_r(root.r, parent=root)
        pass
        root.parent = parent 


    @classmethod
    def postOrderSequence(cls,root): 
        """
        Pack the postorder sequence levelorder indices (1-based)
        into a 64 bit integer. 

        ::

            In [48]: seq4 = Node.postOrderSequence(root4)

            In [51]: seq3 = Node.postOrderSequence(root3)

            In [54]: seq2 = Node.postOrderSequence(root2)

            In [56]: for seq in [seq2,seq3,seq4]:
                print "%20s %16x " % (seq,seq)
               ....:     
                             306              132 
                        20406868          1376254 
               87818465122690200  137fe6dc25ba498 

        """
        nodes = Node.postorder_r(root, nodes=[], leaf=False)
        assert len(nodes) < 16  
        seq = np.uint64(0)
        iseq = np.uint64(0)
        for i, node in enumerate(nodes):
            j = np.uint64(node.idx)
            assert j <= 0xF
            seq | = ((j & np.uint64(0xF)) << np.uint64(i*4) )
            iseq |= ((i & np.uint64(0xF)) << np.uint64(j*4) )
        pass
        return seq, iseq

    @classmethod
    def dumpSequence(cls, seq, iseq):
        n = 0 
        idx = seq & np.uint64(0xF) ;
        while idx > 0:
            print n, idx
            n += 1 ; 
            idx = (seq & np.uint64(0xF << n*4)) >> np.uint64(n*4) 
        pass


    @classmethod
    def postOrderIterative(cls,root): 
        """
        # iterative postorder traversal using
        # two stacks : nodes processed 

        ::

              1

             [2]                 3

             [4]     [5]         6     7

             [8] [9] [10] [11]  12 13  14 15

        ::

            In [25]: Node.postOrderIterative(root3.l)
            Out[25]: 
            [Node(8),
             Node(9),
             Node(4,l=Node(8),r=Node(9)),
             Node(10),
             Node(11),
             Node(5,l=Node(10),r=Node(11)),
             Node(2,l=Node(4,l=Node(8),r=Node(9)),r=Node(5,l=Node(10),r=Node(11)))]

            In [26]: Node.postOrderIterative(root3.l.l)
            Out[26]: [Node(8), Node(9), Node(4,l=Node(8),r=Node(9))]

        """ 
        if root is None:
            return         
         
        nodes = []
        s = []
         
        nodes.append(root)
         
        while len(nodes) > 0:
             
            node = nodes.pop()
            s.append(node)
         
            if node.l is not None:
                nodes.append(node.l)
            if node.r is not None :
                nodes.append(node.r)
     
        #while len(s) > 0:
        #    node = s.pop()
        #    print node.d,
     
        return list(reversed(s))






root0 = Node(1)
root0.name = "root0"

root1 = Node(1, l=Node(2), r=Node(3))
root1.name = "root1"

root2 = Node(1, 
                l=Node(2, 
                          l=Node(4), 
                          r=Node(5)
                      ), 
                r=Node(3, l=Node(6), 
                          r=Node(7)
                      ) 
            )
root2.name = "root2"


root3 = Node(1, 
                l=Node(2, 
                          l=Node(4,
                                    l=Node(8),
                                    r=Node(9)
                                ), 
                          r=Node(5,
                                    l=Node(10),
                                    r=Node(11),
                                )
                      ), 
                r=Node(3, l=Node(6,
                                    l=Node(12),
                                    r=Node(13),
                                ), 
                          r=Node(7,
                                    l=Node(14),
                                    r=Node(15)
                                )
                      ) 
            )
root3.name = "root3"


root4 = Node(1,   
                l=Node(2, 
                          l=Node(4,
                                    l=Node(8, 
                                              l=Node(16),
                                              r=Node(17)
                                          ),
                                    r=Node(9,
                                              l=Node(18),
                                              r=Node(19)
                                          )
                                ), 
                          r=Node(5,
                                    l=Node(10,
                                              l=Node(20),
                                              r=Node(21)
                                          ),
                                    r=Node(11,
                                              l=Node(22),
                                              r=Node(23)
                                          ),
                                )
                      ), 
                r=Node(3, l=Node(6,
                                    l=Node(12,
                                              l=Node(24),
                                              r=Node(25)
                                          ),
                                    r=Node(13,
                                              l=Node(26),
                                              r=Node(27)
                                          ),
                                ), 
                          r=Node(7,
                                    l=Node(14,
                                              l=Node(28),
                                              r=Node(29)
                                          ),
                                    r=Node(15,
                                              l=Node(30),
                                              r=Node(31)
                                          )
                                )
                      ) 
            )
root4.name = "root4"





cbox = Node(shape=BOX, param=[0,0,0,200], name="cbox")
lbox = Node(shape=BOX, param=[-50,50,0,100], name="lbox")
rbox = Node(shape=BOX, param=[ 50,-50,0,100], name="rbox")



csph = Node(shape=SPHERE, param=[0,0,0,250], name="csph")

lsph = Node(shape=SPHERE, param=[-50,50,0,100], name="lsph")
rsph = Node(shape=SPHERE, param=[50,-50,0,100], name="rsph")

empty = Node(shape=EMPTY, param=[0,0,0,0], name="empty")


trees = []

lbox_ue = Node(operation=UNION, l=lbox, r=empty, name="lbox_ue")
trees += [lbox_ue]


lrbox_u = Node(operation=UNION, l=lbox,  r=rbox, name="lrbox_u", ok=True) 
lrbox_i = Node(operation=INTERSECTION, l=lbox,  r=rbox, name="lrbox_i", ok=True) 
lrbox_d1 = Node(operation=DIFFERENCE, l=lbox,  r=rbox, name="lrbox_d1", ok=True) 
lrbox_d2 = Node(operation=DIFFERENCE, l=rbox,  r=lbox, name="lrbox_d2", ok=True) 
trees += [lrbox_u, lrbox_i, lrbox_d1, lrbox_d2]


bms = Node(name="bms",operation=DIFFERENCE, l=cbox,  r=csph, ok=True )
smb = Node(name="smb",operation=DIFFERENCE, l=csph,  r=cbox, ok=True)

ubo = Node(name="ubo",operation=UNION,      l=bms,   r=lrbox_u , ok=False)

trees += [bms, smb, ubo ]

#u_lrbox_d1 = Node(name="u_lrbox_d1", operation=UNION, l=lrbox_d1, r=lrbox_d1 )
#trees += [u_lrbox_d1]


bms_rbox = Node(name="bms_rbox", operation=UNION, l=bms, r=rbox, ok=False)
bms_lbox = Node(name="bms_lbox", operation=UNION, l=bms, r=lbox, ok=False)
smb_lbox = Node(name="smb_lbox", operation=UNION, l=smb, r=lbox, ok=False)
smb_lbox_ue = Node(name="smb_lbox_ue", operation=UNION, l=smb, r=lbox_ue, ok=False)

bms_rbox_lbox = Node(name="bms_rbox_lbox", operation=UNION, l=bms_rbox,r=lbox, ok=False) 
bms_lbox_rbox = Node( name="bms_lbox_rbox", operation=UNION, l=bms_lbox, r=rbox, ok=False ) 

trees += [bms_rbox,bms_lbox,smb_lbox,smb_lbox_ue, bms_rbox_lbox,bms_lbox_rbox] 


lrsph_u = Node(operation=UNION,        l=lsph, r=rsph, name="lrsph_u", ok=True)
lrsph_i = Node(operation=INTERSECTION, l=lsph, r=rsph, name="lrsph_i", ok=True)
lrsph_d1 = Node(operation=DIFFERENCE,   l=lsph, r=rsph, name="lrsph_d1", ok=True)
lrsph_d2 = Node(operation=DIFFERENCE,   l=rsph, r=lsph, name="lrsph_d2", ok=True)

trees += [lrsph_u, lrsph_i, lrsph_d1, lrsph_d2 ]


trees += [root1, root2, root3, root4 ]



def test_is_complete():
    """
    ::

         Node(1,l=Node(2,l=Node(4),r=Node(5)),r=Node(3,l=Node(6),r=Node(7)))
         Node(1,l=Node(2,l=Node(4),r=Node(5)),r=Node(3))

    """
    root2c = copy.deepcopy(root2)    
    assert Node.is_complete_i(root2c) 
    assert root2c.r.is_bileaf and root2c.r.idx == 3
    assert root2c.r.l.is_leaf and root2c.r.l.idx == 6
    assert root2c.r.r.is_leaf and root2c.r.r.idx == 7

    # prune two right most leaves 
    root2c.r.l = None
    root2c.r.r = None         
    assert Node.is_complete_i(root2c) == True
    assert Node.is_complete_r(root2c) == False   # different defn 


def test_make_perfect():
    """
    """
    root2c = copy.deepcopy(root2)    
    assert Node.is_perfect_i(root2) == True
    assert Node.is_perfect_i(root2c) == True

    # prune two right most leaves 
    root2c.r.l = None
    root2c.r.r = None         

    assert Node.is_perfect_i(root2c) == False

    print "root2: %s " % root2
    print "root2c: %s (pruned)" % root2c

    Node.make_perfect_i(root2c)

    print "root2c: %s (restored)" % root2c

    assert Node.is_perfect_i(root2c) == True







if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    #test_is_complete()
    #test_make_perfect()


    root2.annotate()
    r2 = copy.deepcopy(root2)    
  
    seq = Node.postOrderSequence(root4)
    Node.dumpSequence(seq)
    
    


if 0:
    trees = [bmsrlbox]

    for tree in trees:
        Node.levelorder_label_i(tree)
        print "%20s : %s " % (tree.name, tree)
        Node.complete_and_full_r(tree)
        
if 0:
    for tree in [root0, root1, root2, root3, root4]:
        print "%20s : %s " % (tree.name, tree)

        nodes = Node.postorder_r(tree,nodes=[], leaf=False)
        print "Node.postorder_r(%s, leaf=False)\n" % tree.name + "\n".join(map(repr,nodes)) 

        tree.annotate() 

        lpr = Node.leftmost_leaf(tree)
        lop = Node.leftmost(tree)

        print "Node.leftmost(%s) : %s " % (tree.name, lpr )
        print "Node.leftmost_nonleaf(%s) : %s " % (tree.name, lop )

        Node.traverse(lpr, "left prim")
        Node.traverse(lop, "left operation")

        Node.dress(tree)






