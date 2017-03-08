#!/usr/bin/env python
"""
* http://stackoverflow.com/questions/30149007/calculator-algorithm-using-iteration-instead-of-recursion-on-binary-search-tre

      +
  *       /
3   4   4   2

       n1 
  n2       n3
n4  n5   n6  n7



"""
import operator

class N(object):
    def __init__(self, idx, o=None, l=None, r=None, v=None):
        self.idx = idx
        self.o = o
        self.l = l 
        self.r = r
        self.v = v

    def evaluate_arb(self, sc=1):
        """
        Abitrary binary evaluation rule to mimic tmin advancement
        If the result for this node is exceeds the limit, then 
        the larger subexpression needs to be revaluated with the limit halved
        """
        if self.o is not None:
            vl = self.l.evaluate_arb(sc)
            vr = self.r.evaluate_arb(sc)
            loop = 0
            while loop < 10:
               loop += 1 
               vn = self.o(vl, vr)

               print "(%d.0) %s %s %s -> %s  " % (loop, vl, self.o.__name__, vr, vn )
               if vn > 10: 
                   vl = self.l.evaluate_arb(sc*2)
                   pass
                   vn = self.o(vl, vr)
                   print "(%d.1) %s %s %s -> %s   lim %s  " % (loop, vl, self.o.__name__, vr, vn, lim )
               else:
                   break 
               pass
            pass
            return vn
        else:
            return self.v


    def evaluate(self):
        if self.o is not None:
            vl = self.l.evaluate()
            vr = self.r.evaluate()
            return self.o(vl, vr)
        else:
            return self.v

    def __repr__(self):
        return "(%d) val:%s " % (self.idx, self.v)
        #return " ".join([ self.o.__name__ if self.o is not None else "", repr(self.l), repr(self.r), repr(self.v) ])


def postorder_nodes(n, leaf=True):
    def postorder_r(n):
        if n.l is not None:postorder_r(n.l, leaf=leaf)
        if n.r is not None:postorder_r(n.r, leaf=leaf)
        if n.l is None and n.r is None and not leaf:
            pass
        else:
            nodes.append(n)
        pass
    pass
    nodes = []
    postorder_r(n) 
    return nodes


def make_tree():
     n4 = N(4,v=3)
     n5 = N(5,v=4)
     n2 = N(2,o=operator.mul, l=n4, r=n5)

     n6 = N(6,v=4)
     n7 = N(7,v=2)
     n3 = N(3,o=operator.truediv, l=n6, r=n7 )

     n1 = N(1,o=operator.add, l=n2, r=n3 )
     return n1
 

def test_postorder_evaluation_0():
     """
     Note that using postorder node sequence 
     makes evaluation of binary expression tree very simple
    
     CSG Analogy  

     * evaluation of a primitive returns an intersect
     * binary evaluation of two primitives or op modes
       makes classification and via an action table returns 
       left/right/rightflip/miss intersect 
       
     """
     n1 = make_tree()
     nn = postorder_nodes(n1)

     s = []
     for i in range(len(nn)):
         n = nn[i]
         if n.v is not None:   # if node can be directly evaluated, like a primitive push as operand 
             s.append(n.v)  
         else:                 # apply operator to last two nodes with values
             vr = s.pop()      # note reversed pop order 
             vl = s.pop()
             vn = n.o(vl, vr )             
             print "n.o %s %s %s ->  %s" % (n.o.__name__, vr, vl, vn )
             s.append(vn)
         pass
     pass
     assert len(s) == 1
     v = s[0]
     assert n1.evaluate() == v 
     print v


def test_postorder_evaluation_0op():
     """
     Hmm flying above leaves means need to tree navigate 
     to find them from the operators, so its kinda a mixed world
     """
     n1 = make_tree()
     nn = postorder_nodes(n1, leaf=False)



 
def test_postorder_evaluation_1():
     """
     Differences to CSG tree

     * input tmin
     * binary evaluation can be forced to be redone with a changed tmin 
       input on one side, this corresponds to repeating some nodes :
       finding the correct tranche of the postorder to redo

     * no left/right distinction

     """
     n1 = make_tree()
     nn = postorder_nodes(n1)

     s = []
     tr = []
     tr.append(slice(0,len(nn)))

     while len(tr)>0:
         t = tr.pop()
         for i in range(t.start,t.stop):
             n = nn[i]
             if n.v is not None:   # if node can be directly evaluated, like a primitive push as operand 
                 s.append(n.v)  
             else:                 # apply operator to last two nodes with values
                 vr = s.pop()      # note reversed pop order 
                 vl = s.pop()
                 vn = n.o(vl, vr )             


                 print "n.o %s %s %s ->  %s" % (n.o.__name__, vr, vl, vn )
                 s.append(vn)
             pass
         pass
     pass

     assert len(s) == 1
     v = s[0]
     assert n1.evaluate() == v 
     print v



if __name__ == '__main__':
     test_postorder_evaluation_0()
     test_postorder_evaluation_1()

     n1 = make_tree()    

