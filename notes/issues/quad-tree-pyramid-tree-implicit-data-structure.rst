quad-tree-pyramid-tree-implicit-data-structure
================================================

cxs 2D intersect use of a regular 2D grid becomes
horrendously inefficient for large areas, also small scale 
features even in small areas are inefficient to capture.
Need a better approach.


* want genstep source locations to follow the geometry (where the intersects are) 
  rather than being blind to the geometry  

* so need a CUDA 2D histogram (or maybe Quadtree)  



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




