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
Postorder for binary trees by Morton inspired bit twiddling
===============================================================


::

    In [172]: run postorder.py
    test_postorder height 0 
    0 [1]
    test_postorder height 1 
    1 [2, 3, 1]
    test_postorder height 2 
    2 [4, 5, 2, 6, 7, 3, 1]
    test_postorder height 3 
    3 [8, 9, 4, 10, 11, 5, 2, 12, 13, 6, 14, 15, 7, 3, 1]
    test_postorder height 4 
    4 [16, 17, 8, 18, 19, 9, 4, 20, 21, 10, 22, 23, 11, 5, 2, 24, 25, 12, 26, 27, 13, 6, 28, 29, 14, 30, 31, 15, 7, 3, 1]
    test_postorder height 5 
    5 [32, 33, 16, 34, 35, 17, 8, 36, 37, 18, 38, 39, 19, 9, 4, 40, 41, 20, 42, 43, 21, 10, 44, 45, 22, 46, 47, 23, 11, 5, 2, 48, 49, 24, 50, 51, 25, 12, 52, 53, 26, 54, 55, 27, 13, 6, 56, 57, 28, 58, 59, 29, 14, 60, 61, 30, 62, 63, 31, 15, 7, 3, 1]



Morton 1d and binary trees ?
--------------------------------

::

   const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;


::

                    1                     

             2            3 

         4     5       6      7
     
       8  9   10 11  12 13  14 15


                            1

                 10                    11
        
           100        101        110        111

       1000  1001  1010 1011  1100 1101  1110  1111 


Note use of 1-based level order indices, as
shown above this is the natural indexing scheme 
for binary trees.



Can morton magic yield the post order sequence for deep trees::

    8 9 4 10 11 5 2 12 13 6 14 15 7 3 1 
    
    i = np.fromstring("8 9 4 10 11 5 2 12 13 6 14 15 7 3 1", sep=" ", dtype=np.int32)

::

    In [130]: map(bin, i)
    Out[130]: 

              ## when lsb is a 1 next code is c >> 1

            The pattern that when lsb is a 1 next code is bit shifted c >> 1 
            gives half the sequence...  reason is that are on a right child, 
            so next node is parent at the index/2
  
            When lsb is 0, what comes next depends on bit depth 

                 when depth is height,   elevation 0,        code += 1
                 when depth is height-l, elevation elev,     code = (code<<elev) + (1<<elev)


    ['0b1000',       8
     '0b1001', #     9
     '0b100',  #     4
                          (4<<1) + (1<<1) = 10
     '0b1010',      10
     '0b1011', #    11 
     '0b101',  #     5
     '0b10',   #     2
                          (2<<2) + (1<<2) = 12 
     '0b1100',       12 
     '0b1101', #     13
     '0b110',  #      6
                          (6<<1) + (1<<1) = 14  

     '0b1110',       14
     '0b1111', #     15
      '0b111', #      7
       '0b11', #      3
        '0b1'] #      1


"""

import numpy as np
from opticks.bin.ffs import clz_

bin_ = lambda _:"{0:08b}".format(_)
len_ = lambda code:32 - clz_(code) 


class Node(object):
   """
   Used to check the no-tree postorder sequence 
   """
   def __init__(self, idx):
       self.idx = idx 
       self.left = None
       self.right = None 

   @classmethod
   def make_tree(cls, height):
       root = cls.make_tree_r(1, height, 0)
       return root

   @classmethod
   def make_tree_r(cls, idx, height, depth):
       if depth > height: return None
       node = Node(idx) 
       node.left = cls.make_tree_r( idx*2, height, depth+1)
       node.right = cls.make_tree_r( idx*2+1, height, depth+1)
       return node

   @classmethod
   def postorder(cls, height):
       idxs = []
       def postorder_r(node):
           if node.left and node.right:
               postorder_r(node.left)
               postorder_r(node.right)
           pass
           idxs.append(node.idx)
       pass
       root = cls.make_tree(height)
       postorder_r(root)
       return idxs


def postorder_next( code, height ):
    """
    Use pattern in the bits to yield the postorder idx sequence, 
    allowing to do postorder traverses on serialized binary trees
    without creating a node tree.

    :param code:  level order complete binary tree index
    :param height: of the complete tree, heigh 0 is one node, height 1 is 3 nodes, 
    """
    length = len_(code)
    depth = length - 1
    elev = height - depth 
    assert elev >= 0

    if code & 1: 
        nextcode = code >> 1
    else:
        nextcode = (code << elev) + (1 << elev)
    pass
    #print " height %d code %4d length %d depth %d elev %d nextcode %d " % ( height, code, length, depth, elev, nextcode )
    return nextcode 
    
def postorder_(height):
    code = 1 << height 
    postorder = []
    while code:
        postorder.append(code) 
        code = postorder_next(code, height) 
    pass
    return postorder

def test_postorder():
    for height in range(10):
        print "test_postorder height %d " % height 
        postorder = postorder_(height)
        node_postorder = Node.postorder(height)
        print height, postorder
        assert postorder == node_postorder
    pass

if __name__ == '__main__':
    test_postorder()


