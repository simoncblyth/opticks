#!/usr/bin/env python
"""

* https://stackoverflow.com/questions/5181320/under-what-circumstances-are-rmul-called


(mul) a * b 
(rmul) c * 2 
(rmul) d * 6 
(rmul) e * 24 
120


* https://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/
* http://www.lfd.uci.edu/~gohlke/code/transformations.py.html





"""

from operator import mul

class A(object):

   @classmethod
   def prod(cls, a, b):
       return a * b 
   
   def __init__(self, name, val):
       self.name = name
       self.val = val
   def __repr__(self):
       return self.name

   def __mul__(self, other):
       print "(mul) %r * %r " % (self, other)
       return self.val * other.val 

   def __rmul__(self, other):
       print "(rmul) %r * %r " % (self, other)
       return self.val * other





if __name__ == '__main__':


   a = A("a",1) 
   b = A("b",2) 
   c = A("c",3) 
   d = A("d",4) 
   e = A("e",5) 


   res = reduce(mul, [a,b,c,d,e] )

   print res 


      
