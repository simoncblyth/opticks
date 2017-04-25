CSG Algorithm Prototyping
=============================

"Production" code
--------------------

Production code meaning that it is used from elsewhere, and 
thus cannot be changed without regard to its users.

csg.py
    python CSG description, with serialization that is deserialized by npy/NCSG,
    used by tboolean- 

glm.py
    translation into python of some GLM matrix transform manipulations

GParts.py
    indexed access to the partlist of a primitive for oldstyle CGS_FLAGPARTLIST 



Development Exploration code
------------------------------

Ulyanov Excursion into Heart of Darkness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Failed attempt to implement pseudocode from 
a computer science paper that purported to implement iterative CSG. 

boolean.py
    bugged boolean tables 

csgtree.py
    heart of darkness


Potentially Useful machinery developed whilst debugging in the Heart of Darkness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

intersect.py
    numpy based CPU side handling of lots of intersects, some 
    potentially useful techniques that came out of the Heart of Darkness

intersectTest.py
    2d matplotlib plotting of intersects 

ray.py
    numpy based collections of rays, used for CPU side debugging 
    of CSG algorithms (2d intersections used with matplotlib plotting of intersects) 

nodeRenderer.py
    matplotlib based rendering of geometry in 2d, used for intersect debugging  

Learning : binary tree gymnastics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stacktrav.py
    basics of binary tree traversal in preorder/postorder/inorder/levelorder 

bintree.py
    roundtrip binary tree serialization/deserialization testing, and 
    multi binary tree concatenation/de-concatentation testing

node.py
    binary tree gymnastics and serialization dev

postOrderIterative.py
    alt iterative approach

postOrderTraversalWithoutRecursion.py
    exploring how to traverse binary trees without using recursion


Learning : Morton magic bit twiddling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

morton.py
    playing with morton 2d codes

zorder.py
    quadtree exploration, 2d morton codes, multires z-order curve plotting


Learning : Converting Recursive to Iterative 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

factorial.py
    very simple example of converting recursive into iterative

hilbert.py
    more realistic example of converting recursive into iterative, 
    with Hilbert curve plotting with matplotlib
 

GPU CSG Design
~~~~~~~~~~~~~~~~~~

iterativeExpression.py
    Provided the seed for GPU CSG implementation, with the realization that 
    externalizing the postorder traversal makes binary tree evaluation very simple
    and tractable without recursion.  Also the direct parallels between 
    CSG node tree evaluation to binary expression evaluation was realized. 
   
iterativeTreeEvaluation.py
    starting point for the design of GPU CSG, handling the node traversal 
    and reiteration techniques, lots of design notes : Heart of Design 

ctrl.py
    enum used by iterativeTreeEvaluation*

iterativeTreeEvaluationFake.py
    comparing iterative and recursive imps

iterativeTreeEvaluationCSG.py
    applying the nascent algorithm

postorder.py
    devise bit twiddling postorder for binary trees, inspired by 1d Morton codes, 
    postorder_next allows doing bare naked postorder traverses of 
    serialized binary trees
    
    * without reconstructing the node tree 
    * without using recursion
    * without storing postorder sequence indices


slavish.py
    slavish translation of CUDA CSG ray trace back to python, used for debugging 






