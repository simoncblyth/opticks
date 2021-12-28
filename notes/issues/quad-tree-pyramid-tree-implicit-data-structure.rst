quad-tree-pyramid-tree-implicit-data-structure
================================================


Issue
-------

cxs 2D intersect use of a regular 2D grid becomes
horrendously inefficient for large areas, also small scale 
features even in small areas are inefficient to capture.
Need a better approach.

* want genstep source locations to follow the geometry (where the intersects are) 
  rather than being blind to the geometry  

* so need a CUDA 2D histogram (or maybe Quadtree)  


Ideas for solution
--------------------

Based around quadtree division of a rectangular 2D region of space. 
Even though might not have to construct the quadtree (can just get the Z-order)
it is there in the mind to inform the approach.

Reading around makes be think that using the Z-order curve approach would
allow the 2D (x,y) positions of intersects at one depth level of the quadtree 
to be expressed in 1D as (morton_xy,) and to be transformed by bit shifts 
into a coarser form at a deeper level.

The aim is to use the morton_xy distribution of the intersects
at one level to steer the positions of gensteps at the next level
of detail.  Can just calculate the morton_xy of the potential 
genstep position.  



Using Morton
---------------


* https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
* https://graphics.cs.kuleuven.be/publications/BLD13OCCSVO/

* https://john.cs.olemiss.edu/~rhodes/papers/Nocentino10.pdf


Morton Order Progressive Refinement
--------------------------------------

* morton order progressive refinement


::

    00    1

     2    3 




     00*  1      4*  5 

      2   3      6   7

      8*  9     12*  13 

     10   11    14   15


Bit shifts to the right get from one level of fine index to coarse index::

     00 >> 2 = 0 
      4 >> 2 = 1
      8 >> 2 = 2
     12 >> 2 = 3 


::


       00*  1    4   5*  16* 17   20  21*

       02   3    6   7   18  19   22  23

       08   9   12  13   24  25   28  29

       10* 11   14  15*  26  27   30  31

       32* 33   36  37   40* 41   44  45*

       34  35   38  39   42  43   46  46

       48  49   52  53   56  57   60  61

       50  51   54  55   58  59   62  63*


Trying to relate the indices between resolution levels, it doesnt work out simply

* maybe that is not the correct Z-order 


* ~/opticks_refs/morton_ahnetafel_indices_10.1.1.118.7720.pdf


::


       00 >> 2 = 0 
       21 >> 2 = 5 
       50 >> 2  = 12

        5 >> 2 = 1 
       10 >> 2 = 2
       15 >> 2 = 3
       16 >> 2 = 4 
       63 >> 2 = 15 


       40 >> 2 = 10   ## not 12 
       41 >> 2 = 10        

       45 >> 2 = 11   ## not 13, seems only the corners are really at same position 






Two bits shift gets from one level to the next::

    In [4]: for i in range(16): print(" %4d : %8s : %8s  " % (i,bin(i),bin(i >> 2) )
       ...: )
        0 :      0b0 :      0b0  
        1 :      0b1 :      0b0  
        2 :     0b10 :      0b0  
        3 :     0b11 :      0b0  
        4 :    0b100 :      0b1  
        5 :    0b101 :      0b1  
        6 :    0b110 :      0b1  
        7 :    0b111 :      0b1  
        8 :   0b1000 :     0b10  
        9 :   0b1001 :     0b10  
       10 :   0b1010 :     0b10  
       11 :   0b1011 :     0b10  
       12 :   0b1100 :     0b11  
       13 :   0b1101 :     0b11  
       14 :   0b1110 :     0b11  
       15 :   0b1111 :     0b11  




Morton Integrals for High Speed Geometry Simplification
---------------------------------------------------------

* https://perso.telecom-paristech.fr/boubek/papers/HSGS/HSGS.pdf


Global Static Indexing for Real-time Exploration of Very Large Regular Grids
------------------------------------------------------------------------------

* https://www.osti.gov/servlets/purl/15006198

Hierarchical indexing schemes


* ~/opticks_refs/osti_morton_zorder_hierarchical_reindexing_scheme_15006198.pdf




Tree-pyramid
-------------

* https://en.wikipedia.org/wiki/Quadtree

A tree-pyramid (T-pyramid) is a "complete" tree; every node of the T-pyramid
has four child nodes except leaf nodes; all leaves are on the same level, the
level that corresponds to individual pixels in the image. The data in a
tree-pyramid can be stored compactly in an array as an implicit data structure
similar to the way a complete binary tree can be stored compactly in an
array.[2]


* http://lumetta.web.engr.illinois.edu/220-F18/slides/581-pyramid-tree-I-O-example.pdf 


Region Quadtree
-----------------

* https://en.wikipedia.org/wiki/Quadtree

The region quadtree represents a partition of space in two dimensions by
decomposing the region into four equal quadrants, subquadrants, and so on with
each leaf node containing data corresponding to a specific subregion. Each node
in the tree either has exactly four children, or has no children (a leaf node).
The height of quadtrees that follow this decomposition strategy (i.e.
subdividing subquadrants as long as there is interesting data in the
subquadrant for which more refinement is desired) is sensitive to and dependent
on the spatial distribution of interesting areas in the space being decomposed.
The region quadtree is a type of trie.

A region quadtree with a depth of n may be used to represent an image
consisting of 2^n × 2^n pixels, where each pixel value is 0 or 1. The root node
represents the entire image region. If the pixels in any region are not
entirely 0s or 1s, it is subdivided. In this application, each leaf node
represents a block of pixels that are all 0s or all 1s. Note the potential
savings in terms of space when these trees are used for storing images; images
often have many regions of considerable size that have the same colour value
throughout. Rather than store a big 2-D array of every pixel in the image, a
quadtree can capture the same information potentially many divisive levels
higher than the pixel-resolution sized cells that we would otherwise require.
The tree resolution and overall size is bounded by the pixel and image sizes.

A region quadtree may also be used as a variable resolution representation of a
data field. For example, the temperatures in an area may be stored as a
quadtree, with each leaf node storing the average temperature over the
subregion it represents.

If a region quadtree is used to represent a set of point data (such as the
latitude and longitude of a set of cities), regions are subdivided until each
leaf contains at most a single point.


Z-order curve and efficiently building Quadtrees
--------------------------------------------------- 

* https://en.wikipedia.org/wiki/Z-order_curve

Interleaving the bits of (x,y) provides a 1D ordering for 2D data.

The Z-ordering can be used to efficiently build a quadtree (2D) or octree (3D)
for a set of points.[4][5] The basic idea is to sort the input set according to
Z-order. Once sorted, the points can either be stored in a binary search tree
and used directly, which is called a linear quadtree,[6] or they can be used to
build a pointer based quadtree.


linear quadtree
------------------

* http://www.sigapp.org/sac/sac2000/Proceed/FinalPapers/DB-27/node3.html

* https://icaci.org/files/documents/ICC_proceedings/ICC2001/icc2001/file/f11020.pdf



CUDA linear quadtree
----------------------

* https://www.sccs.swarthmore.edu/users/10/mkelly1/quadtrees.pdf
* ~/opticks_refs/swarthmore_mkelly_quadtrees.pdf



Can OptiX ray tracing somehow be tricked into doing this ?
------------------------------------------------------------

* https://www.highperformancegraphics.org/wp-content/uploads/2019/session2/rtx_for_tet-mesh.pdf

* Tet-mesh point location 

Abuse OptiX by ray tracing "zero"-length rays to implement sampling of points




