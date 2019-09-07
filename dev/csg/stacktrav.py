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

* https://en.wikipedia.org/wiki/Tree_traversal


            1
          /   \
         2     3
       /  \
      4    5   


   preorder   1 2 4 5 3    # visit before navigating to other node 
   inorder    4 2 5 1 3    # keep going left until run out of nodes then visit then go right 
   postorder  4 5 2 3 1 
   levelorder 1 2 3 4 5 


* http://www.geeksforgeeks.org/serialize-deserialize-binary-tree/



"""

def visit(node, label=""):
    print label, node.data



def preorder(node):
    if node is None: return
    visit(node, " preorder")
    preorder(node.left)
    preorder(node.right)

def inorder(node):
    if node is None: return
    inorder(node.left)
    visit(node, " inorder")
    inorder(node.right)

def postorder(node):
    if node is None: return
    postorder(node.left)
    postorder(node.right)
    visit(node, " postorder")


def iterativeLevelorder(root):
    q = []
    q.append(root)
    while len(q) > 0:
       node = q.pop(0)   # bottom of q (ie fifo)
       visit(node, " iterativeLevelorder")
       if not node.left is None:q.append(node.left)
       if not node.right is None:q.append(node.right)


def iterativePreorder(node):
    if node is None: return 
    s = []
    s.append(node)
    while len(s)>0:
       node = s.pop()
       visit(node, " iterativePreorder")
       if not node.right is None:s.append(node.right)
       if not node.left  is None:s.append(node.left)

def iterativeInorder(node):
    s = []
    while len(s) > 0 or (not node is None): 
       if not node is None:
           s.append(node)
           node = node.left
       else:
           node = s.pop()
           visit(node, " iterativeInorder")
           node = node.right

def iterativePostorder(node):
    s = []
    lastNodeVisited = None
    while len(s) > 0 or (not node is None): 
       if not node is None:
           s.append(node)
           node = node.left
       else:
           peekNode = s[-1]
      
           # if right child exists and traversing node from left child, then move right
           if (not peekNode.right is None) and lastNodeVisited != peekNode.right:
               node = peekNode.right
           else:
               visit(peekNode, " iterativePostorder" )
               lastNodeVisited = s.pop()




class Node(object):
    @classmethod
    def traverse(cls, root):
        s = []
        s.append(root)

        while len(s)>0:
            current = s.pop(0)   # fifo queue  -> 1 2 3 4 5 
            #current = s.pop()     # lifo stack  -> 1 3 2 5 4 
            if current is not None:
                print current.data,
                s.extend([current.left,current.right])


    def __init__(self, data, left=None, right=None):
        self.data = data 
        self.left = left
        self.right = right
 








 
""" 
            1
          /   \
         2     3
       /  \
      4    5   

"""

two = Node(2, left=Node(4), right=Node(5) )
root = Node(1, left=two, right=Node(3) )

 
preorder(root)
iterativePreorder(root)

inorder(root)
iterativeInorder(root)

postorder(root)
iterativePostorder(root)

iterativeLevelorder(root)

