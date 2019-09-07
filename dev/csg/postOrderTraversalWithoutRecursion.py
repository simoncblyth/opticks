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

    going down tree L : 2 : None -> (0)30 
    going down tree L : 3 : (0)30 -> (1)20 
 going down tree leaf : 2 : (1)20 -> (2)15 
 up from left child R : 3 : (2)15 -> (1)20 
 going down tree leaf : 2 : (1)20 -> (2)25 
  up from right child : 1 : (2)25 -> (1)20 
 up from left child R : 2 : (1)20 -> (0)30 
    going down tree L : 3 : (0)30 -> (1)40 
 going down tree leaf : 2 : (1)40 -> (2)37 
 up from left child R : 3 : (2)37 -> (1)40 
 going down tree leaf : 2 : (1)40 -> (2)45 
  up from right child : 1 : (2)45 -> (1)40 
  up from right child : 0 : (1)40 -> (0)30 

iterativeLevelOrder
(0)30 
(1)20 (1)40 
(2)15 (2)25 (2)37 (2)45


"""

from intersect import intersect_primitive, Node, Ray, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE, EMPTY, desc


class N(object):
    def __init__(self, value, left=None, right=None, shape=None, operation=None, depth=0):
        self.value = value
        self.left = left
        self.right = right 
        self.shape = shape
        self.operation = operation 
        self.depth = depth

    def __repr__(self):
        return "%s" % (self.value)




class Node(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right 

    def __repr__(self):
        return "(%d)%s" % (self.depth, self.value)

     
    @classmethod
    def add(cls, node, value, depth=0):
        if node is None:
            node = Node(value)
            node.depth = depth 
        else:
            if value > node.value:
                node.right = cls.add(node.right, value, depth=depth+1)
            else: 
                node.left = cls.add(node.left, value, depth=depth+1)
            pass
        return node


def postorder(root):
    if root is None:return
    postorder(root.left) 
    postorder(root.right)
    print "%d " % root.value 


def postorderTraversalWithoutRecursion(root):
    """

    * http://algorithmsandme.in/2015/03/postorder-traversal-without-recursion/

    1. Start with the root node and push the node onto stack.
    2. Repeat all steps till stack is not empty. 
    3. Peek the top of the stack. 

      3.1 If previous node is parent of current node : ( When we are moving down the tree)

         3.1.1 If left child is present, push left child onto stack. 
         3.1.2 Else if right child is present, push right child onto stack
         3.1.3 If left and right children are not present, print the node. 

      3.2 If previous node is left child of current node ( When moving up after visiting left node)

         3.2.1 If right child is not present, print current node
         3,2.2 If right child is present, push it onto stack. 

      3.3 If previous node is right child of current node ( When moving up after visiting right child )

         3.3.1 Print the node. 
         3.3.2 Pop node from stack.



    * pushing a node as navigate across non-leaf nodes preps it for traversal
    * popping a left node, communicates that it has been processed 

    """
    pass
    prev = None
    stackPush(root)

    while not stackEmpty():
        node = stackPeek()

        stage = "other"
        if (prev is None) or (node is prev.left) or (node is prev.right): 
            stage = "going down tree"
            if node.left is not None:
                stackPush(node.left)
                stage += " L"  
            elif node.right is not None:
                stackPush(node.right)
                stage += " R"  
            else:
                stage += " leaf"
                stackPop() 
            pass
        pass

        if prev is node.left:
            stage = "up from left child"
            if node.right is not None:
                stackPush(node.right)
                stage += " R"
            else:
                stage += " leaf"
                stackPop() 
            pass
        pass

        if prev is node.right:
            stage = "up from right child"
            stackPop() 
        pass

        print " %20s : %d : %s -> %s   l:%s r:%s  " % (stage, stackCount(), prev, node, node.left, node.right)

        prev = node




def iterativeLevelorder(root):
    q = []
    q.append(root)

    prev = None
    while len(q) > 0:
       node = q.pop(0)   # bottom of q (ie fifo)

       if prev and node.depth > prev.depth:print "\n",
       print node, 

       if not node.left is None:q.append(node.left)
       if not node.right is None:q.append(node.right)

       prev = node




stack = []
def stackPeek():
    global stack
    if len(stack) > 0:
        return stack[-1]
    return None

def stackPop():
    global stack
    if len(stack) == 0:
        return None
    return stack.pop()
    
def stackPush(obj):
    global stack
    stack.append(obj)
 
def stackEmpty():
    global stack
    return len(stack) == 0

def stackCount():
    global stack
    return len(stack)



if __name__ == '__main__':

    root0 = None
    for val in [30,20,15,25,40,37,45]:
        root0 = Node.add(root0, val)


    l = N(20, left=N(15,depth=2), right=N(25,depth=2), depth=1)
    r = N(40, left=N(37,depth=2), right=N(45,depth=2), depth=1)
    root1 = N(30, left=l, right=r, depth=0  )

    postorder(root0)
    print "\n"
    postorder(root1)
    print "\n"


    postorderTraversalWithoutRecursion(root1)

    print "\niterativeLevelOrder"
    iterativeLevelorder(root1)
      

 
