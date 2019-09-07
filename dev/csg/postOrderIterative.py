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

     1

     2       3

     4   5   6   7



outer 1 len(stack) 5 : 3,1,5,2,4 
outer 2 len(stack) 4 : 3,1,5,2 
outer 3 len(stack) 4 : 3,1,2,5 
outer 4 len(stack) 3 : 3,1,2 
outer 5 len(stack) 2 : 3,1 
outer 6 len(stack) 4 : 1,7,3,6 
outer 7 len(stack) 3 : 1,7,3 
outer 8 len(stack) 3 : 1,3,7 
outer 9 len(stack) 2 : 1,3 
outer 10 len(stack) 1 : 1 




[4, 5, 2, 6, 7, 3, 1]

# http://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/

# Python program for iterative postorder traversal
# using one stack






* http://algorithmsandme.in/2015/03/postorder-traversal-without-recursion/

Looking at code, it is clear that parent node is visited twice, once coming up
from left sub tree and second time when coming up from right sub tree. However,
parent node is to be printed when we have already printed left as well as right
child. So, we need to keep track of the previous visited node.  There are three
values possible for previous node:

prev == node.parent
      traversing tree downwards.  No need to do anything with the current node.

prev == node.left  
      have already visited left child, but still not have visited right child, 
      hence we move to right child of current node.

prev == node.right
      left and right child of current node are already visited, 
      hence all we need to do is to print current node.




"""

# Stores the answer
ans = []

class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        return "%s" % self.data


def peek(stack):
    if len(stack) > 0:
        return stack[-1]
    return None


def postOrderIterative(root):
        
    if root is None:
        return 

    stack = []
    
    outer = 0 

    while(True):
        

        while (root):
            if root.right is not None:
                stack.append([root.right,"right"])
            stack.append([root,"left"])
            root = root.left
        pass
        # right/left/right/left/... stack until run out 

        outer += 1 
       
        print "outer %d len(stack) %d : %s " % (outer, len(stack), ",".join(map(str,stack)))  
        
        root,label = stack.pop()

        # If the popped item has a right child and the
        # right child is not processed yet, then make sure
        # right child is processed before root
        if (root.right is not None and peek(stack) == root.right):

            print "has right"
            stack.pop()        # Remove right child from stack 
            stack.append([root,"root again"]) # Push root back to stack
            root = root.right  # change root so that the 
                               # righ childis processed next
        else:
            ans.append(root.data) 
            root = None
            print "no right"


        if (len(stack) <= 0):
                break

# Driver pogram to test above function
root = Node(1, left=Node(2, left=Node(4), right=Node(5)), right=Node(3, left=Node(6), right=Node(7)))

print "Post Order traversal of binary tree is"
postOrderIterative(root)
print ans


