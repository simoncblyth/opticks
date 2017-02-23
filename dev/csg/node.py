#!/usr/bin/env python

class Node(object):
    def __init__(self, d, l=None, r=None):
        self.d = d
        self.l = l
        self.r = r
        self.next_ = None

    def __repr__(self):
        if self.l is not None and self.r is not None:
            return "Node(%d,l=%r,r=%r)" % (self.d, self.l, self.r)
        elif self.l is None and self.r is None:
            return "Node(%d)" % (self.d)
        else:
            assert 0

    is_leaf = property(lambda self:self.l is None and self.r is None)
    is_left = property(lambda self:self.d % 2 == 0)

    @classmethod
    def leftmost(cls, root):
        n = root.l  
        while n is not None:
            n = n.l
        return n


    @classmethod
    def depth_r(cls, node, depth=0):
        """
        Marking up the tree with depth, can be done CPU side 
        during conversion : so recursive is fine
        """
        if node is None:
            return 

        #print node
        node.depth = depth

        maxd = depth
        if node.l is not None: maxd = cls.depth_r(node.l, depth+1)
        if node.r is not None: maxd = cls.depth_r(node.r, depth+1)
    
        return maxd

    @classmethod
    def progeny_i(cls,root):
        """

        1

        2          3

        4    5     6      7

        8 9  10 11 12 13 14 

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
    def levelorder_i(cls,root):
        q = []
        q.append(root)

        idx = 1 
        while len(q) > 0:
           node = q.pop(0)   # bottom of q (ie fifo)

           assert node.d == idx
           idx += 1

           if not node.l is None:q.append(node.l)
           if not node.r is None:q.append(node.r)
        pass
        return idx - 1


    @classmethod
    def label_r(cls, node, idx, label):
        if node.d == idx:
            setattr(node, label, 1)

        if node.l is not None:cls.label_r(node.l, idx, label)
        if node.r is not None:cls.label_r(node.l, idx, label)

 
    @classmethod
    def leftmost_nonleaf(cls, root):
        l = root 
        while l.l is not None:
            if l.l.is_leaf:
                break
            else:
                l = l.l
            pass
        return l

    @classmethod
    def postorder_r(cls, root, nodes=[],leaf=True):
        """ 
        :param root:
        :param nodes: list 
        :param leaf: bool control of inclusion of leaf nodes with internal nodes

        Recursive postorder traversal
        """
        if root.l is not None:
            cls.postorder_r(root.l,nodes, leaf=leaf) 
        if root.r is not None:
            cls.postorder_r(root.r,nodes, leaf=leaf)
 
        if not leaf and root.is_leaf:
            pass
        else: 
            nodes.append(root)

        return nodes


    @classmethod
    def postorder_threading_r(cls, root):
        nodes = cls.postorder_r(root, nodes=[], leaf=False)
        for i in range(len(nodes)):
            node = nodes[i]
            next_ = nodes[i+1] if i < len(nodes)-1 else None
            node.next_ = next_


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


if __name__ == '__main__':

    for tree in [root0, root1, root2, root3, root4]:
        print "%20s : %s " % (tree.name, tree)

        nodes = Node.postorder_r(tree,nodes=[], leaf=False)
        print "Node.postorder_r(%s, leaf=False)\n" % tree.name + "\n".join(map(repr,nodes)) 

        Node.postorder_threading_r(tree)
        leftop = Node.leftmost_nonleaf(tree)
        print "Node.leftmost_nonleaf(%s) : %s " % (tree.name, leftop )

        print "following thread..."
        node = leftop
        while node is not None:
            print node
            node = node.next_ 







