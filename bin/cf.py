#!/usr/bin/env python

import sys

class CF(object):
   """
   Compare lines in two files 
   """
   def __init__(self, a_, b_):
       a = set(open(a_).read().splitlines())
       b = set(open(b_).read().splitlines())
       a_b = a - b 
       b_a = b - a

       self.a_ = a_
       self.b_ = b_
       self.a = a 
       self.b = b
       self.a_b = a_b 
       self.b_a = b_a 

   def __repr__(self):
       lines = []
       lines.append("A")
       lines.append(self.a_)
       lines.append("B")
       lines.append(self.b_)
       lines.append("\nA_B : in A but not in B\n")
       lines.extend(list(self.a_b))
       lines.append("\nB_A : in B but not in A\n")
       lines.extend(list(self.b_a))
       return "\n".join(lines)

if __name__ == '__main__':
    cf = CF(sys.argv[1], sys.argv[2])
    print(cf)
