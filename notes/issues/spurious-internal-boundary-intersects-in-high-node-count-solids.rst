spurious-internal-boundary-intersects-in-high-node-count-solids
================================================================


Issue was first noticed with JUNO AltXJfixtureConstruction_XY
The Alt fixed the coincident faces but was still left with 
some spurios intersects on parts of the cylinder within 
some of the boxes visible in XY cross section.

Actually looking back the first hint was with the 
unexpected vertical (+Z) cylinder edge spurious 
intersects from XJfixtureConstruction.

To provide a more systematic way of checking all intersects I started
filling out the distance_node functions to csg_intersect_node.h
and implemented the postorder traverse combined CSG calculation
in csg_intersect_tree.h::distance_tree

Using AnnulusFourBoxUnion AnnulusThreeBoxUnion noted that 
only high node count solids are afflicted putting 
suspicion on balancing. 

Extensive CSG debugging using DEBUG_RECORD and CSGRecord to 
follow the micro-steps of the CSG intersection implementation 
revealed the cause to be disjoint unions. Working through the
CSG algorithm manually for various CSG trees demonstrates how.

Cause has been confirmed to be due to tree balancing 
managing to sometimes yield disjoint unions which 
mess up the CSG algorithm preventing it from getting 
all the way to the last exit. 

Initially I suspected that the primitive traversal order was being 
messed up by the balancing : but in the GeoChain conversion 
comparing X4SolidTree::Draw of the initial tree and CSGGeometry::Draw of the final
balanced tree shows that even when balanced the primitive traversal order does stay the same.

Note in NTreeBalanceTest that the balancing actually can mess up the primitive 
ordering, but that does not happen with the simple BoxFourBoxUnion so 
primitive order is not the cause of the problem.

Possibly::

    When the CSG algorithm loops one side with tmin advanced it only covers part 
    of the balanced tree (which then has the discontiguous problem).  
    Perhaps looping over the full tree might avoid the problem.

    This is why the unbalanced tree doesnt have the issue. 



::

    c
    ./csg_geochain.sh 

    SPURIOUS=1 IXIYIZ=-6,0,0 IW=17 GEOM=AnnulusFourBoxUnion_YX  ./csg_geochain.sh
    SPURIOUS=1 IXIYIZ=-6,0,0 IW=17 GEOM=BoxFourBoxUnion_YX      ./csg_geochain.sh


Switch off balancing with a GeoChain run
---------------------------------------------

Switching off balancing avoids the problem but causes
a performance issue for solids with many primitives.

::

    gc ; NTREEPROCESS_MAXHEIGHT0=10 GEOM=BoxFourBoxUnion ./run.sh 

    c ; GEOM=BoxFourBoxUnion_YX ./csg_geochain.sh 

* now no spurious    


::

   SPURIOUS=1 GEOM=BoxFourBoxUnion_YX ./csg_geochain.sh 


TODO
------

1. check how balancing changes the ordering, expt to see 
   if disjointness at the end of the postorder traverse 
   is also precluded 


2. experiment with a new compound "CSG_BLOBUNION" that 
   does something a bit similar to G4MultiUnion perhaps with restrictions
   on the topology to make finding compound enter and exit points 
   performant   

   * much of the demand for handling high node count solids could 
     be mopped up into blobs 

3. improve presentation of CSG/tests/CSGIntersectSolidTest.py so can explain the issue 





Check tree changes from balancing
------------------------------------




::

    GeoChain::convertSolid original G4VSolid tree [-1] nameprefix   NODE:9 PRIM:5 UNDEFINED:9 EXCLUDE:0 INCLUDE:0 MIXED:0 Order:IN

                                                            Uni                     
                                                            U                       
                                                            7                       
                                                                                    
                                                                                    
                                            Uni                     Box             
                                            U                       U               
                                            5                       8               
                                                                                    
                                                                                    
                            Uni                     Box                             
                            U                       U                               
                            3                       6                               
                                                                                    
                                                                                    
            Uni                     Box                                             
            U                       U                                               
            1                       4                                               
                                                                                    
                                                                                    
    Box             Box                                                             
    U               U                                                               
    0               2                                                               
                                                                                    
                                                                                    
       0.00            0.00            0.00           50.00          -50.00 zdelta  
                                                                                    
      45.00           11.50           11.50           65.00          -35.00 az1     
     -45.00          -11.50          -11.50           35.00          -65.00 az0     
    cbo0    cpx1    bpx2    cnx3    bnx4    cpy5    bpy6    cny7    bny8            


    2022-02-01 20:20:43.573 INFO  [23809019] [CSGDraw::draw@27] GeoChain::convertSolid converted CSGNode tree axis Y

                                                                           un                                                                                       
                                                                          1                                                                                         
                                                                             0.00                                                                                   
                                                                            -0.00                                                                                   
                                                                                                                                                                    
                                   un                                                bo                                                                             
                                  2                                                 3                                                                               
                                     0.00                                            -35.00                                                                         
                                    -0.00                                            -65.00                                                                         
                                                                                                                                                                    
               un                                      un                                                                                                           
              4                                       5                                                                                                             
                 0.00                                    0.00                                                                                                       
                -0.00                                   -0.00                                                                                                       
                                                                                                                                                                    
     bo                  bo                  bo                  bo                                                                                                 
    8                   9                   10                  11                                                                                                  
      45.00               11.50               11.50               65.00                                                                                             
     -45.00              -11.50              -11.50               35.00                                                                                             

      big box             +X                  -X                  +Y                 -Y


Tracing the unbalanced tree, never 




    2022-02-01 20:33:00.198 INFO  [23823644] [CSGDraw::draw@27] CSGGeometry::centerExtentGenstepIntersect axis Y

                                                                           un                  
                                                                          1                     
                                                                             0.00                
                                                                            -0.00                
                                                                                                 
                                                       un                            bo          
                                                      2                             3            
                                                         0.00                        -35.00      
                                                        -0.00                        -65.00      
                                                                                                 
                                   un                            bo                              
                                  4                             5                                
                                     0.00                         65.00                          
                                    -0.00                         35.00                          
                                                                                                 
               un                            bo                                                  
              8                             9                                                    
                 0.00                         11.50                                              
                -0.00                        -11.50                                              
                                                                                                 
     bo                  bo                                                                      
    16                  17                                                                       
      45.00               11.50                                                                  
     -45.00              -11.50                                                                  



     big box              +X                  -X                  +Y                 -Y       






                                           +---------------+
                                           |               |
                                           |               |
                                           |               |
                             +-------------|---------------|-----------+
                             |             |               |           |
                             |             |               |           |
                             |             |               |           |
                             |             +---------------+           |
                             |                                         |
                             |                                         |
                      +-------------+                            +--------------+
                      |      |      |                            |     |        |
                      |      |      |                            |     |        |
                      |      |      |                            |     |        |
                      |      |      |                            |     |        |
                      |      |      |                            |     |        |
                      |      |      |                            |     |        |
                      +-------------+                            +--------------+
                             |                                         |
                             |                                         |
                             |                                         |
                             |                                         |
                             |             +---------------+           |
                             |             |               |           |
                             |             |               |           |
                             |             |               |           |
                             +-------------|---------------|-----------+
                                           |               |
                                           |               |
                                           |               |
                                           +---------------+





Complete binary tree of height 4 (31 nodes) with 1-based nodeIdx in binary:: 
                                                                                                                                          depth    elevation
                                                                      1                                                                      0         4
 
                                      10                                                            11                                       1         3

                          100                        101                            110                           [111]                       2         2

                 1000            1001          1010         1011             1100          1101            *1110*           1111              3         1
 
             10000  10001    10010 10011    10100 10101   10110 10111     11000 11001   11010  11011   *11100* *11101*   11110   11111          4         0
                                                                                                     


CSG looping in the below implementation has been using the below complete binary tree slices(tranche)::

    unsigned fullTree  = PACK4(  0,  0,  1 << height, 0 ) ;    

    unsigned leftIdx = 2*nodeIdx  ;    // left child of nodeIdx
    unsigned rightIdx = leftIdx + 1 ; // right child of nodeIdx  

    unsigned endTree   = PACK4(  0,  0,  nodeIdx,  endIdx  );
    unsigned leftTree  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
    unsigned rightTree = PACK4(  0,  0,  rightIdx << (elevation-1), nodeIdx );


1 << height 
    leftmost, eg 10000
0 = 1 >> 1 
    one beyond root(1) in the sequence
 
nodeIdx
     node reached in the current slice of postorder sequence  
endIdx 
     one beyond the last node in the current sequence (for fulltree that is 0)

leftTree 
     consider example nodeIdx 111 which has elevation 2 in a height 4 tree
     
     nodeIdx  :  111
     leftIdx  : 1110  
     rightIdx : 1111

     leftTree.start : leftIdx << (2-1)  : 11100
     leftTree.end   : rightIdx << (2-1) : 11110    one beyond the leftIdx subtree of three nodes in the postorder sequence 

rightTree
    again consider nodeIdx 111

    nodeIdx   :  111
    rightIdx  : 1111

    rightTree.start : rightIdx << (2-1) : 11110     same one beyond end of leftTree is the start of the rightTree slice 
    rightTree.end   :nodeIdx 


Now consider how different things would be with an unbalanced tree : the number of nodes traversed in a leftTree traverse
of an unbalanced tree would be much more... the leftTree  would encompass the entirety of the postorder sequence 
up until the same end points as above.  The rightTree would not change.

Perhaps leftTreeOld should be replaced with leftTreeNew starting all the way from leftmode 
beginning of the postorder sequence::

    unsigned leftTreeOld  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
    unsigned leftTreeNew  = PACK4(  0,  0,  1 << height , rightIdx << (elevation-1)) ; 


I suspect that when using balanced trees the below leftTree can cause spurious intersects
due to discontiguity from incomplete geometry as a result of not looping over the 
full prior postorder sequence. 

Tried using leftTreeNew with a balanced tree and it still gives spurious intersects on internal boundariues,
so it looks like tree balanching and the CSG algorithm as it stands are not compatible.  

Tree balancing is a bandaid to allow greater node count trees to be used without 
replacing use of complete binary tree storage. 

Implementing CSG_CONTIGUOUS and CSG_DISCONTIGUOUS to handle multiunions of lots of nodes 
looks to be straightforward (especially CSG_CONTIGUOUS) and would remove most of the need for 


tree balancing by being able to mop up large numbers of nodes into these new compound primitives.

An intuitive way of understanding the issue::

    CSG geometries are grown in a sequence of combinations
    The CSG intersect implementation makes binary decisions between intersects, this
    works with unblanced trees which preserve the original intent of the shape. 

HMM: so why did leftTreeNew give spurious with balanced trees  ? Need to check the detailed causes

TODO: improve CSGRecord visualization and use to compare 

1. unbalanced tree running 
2. balanced tree running with leftTreeOld 
3. balanced tree running with leftTreeNew 




