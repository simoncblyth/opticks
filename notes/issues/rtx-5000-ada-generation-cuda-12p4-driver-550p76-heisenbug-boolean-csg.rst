rtx-5000-ada-generation-cuda-12p4-driver-550p76-heisenbug-boolean-csg
=========================================================================


Using GEOM DifferenceBoxSphere and ~/o/cxd.sh::

    #!/bin/bash
    ...

    moi=EXTENT:200
    pidxyz=MIDDLE
    sleep_break=1

    export MOI=${MOI:-$moi}
    export PIDXYZ=${PIDXYZ:-$pidxyz}
    export SLEEP_BREAK=${SLEEP_BREAK:-$sleep_break}

    ~/o/cxr_min.sh



::

    [blyth@localhost heisenbug]$ diff heisenbug_cxd_750_11p7.txt heisenbug_cxd_800_12p4.txt
    29c29
    < //intersect_tree  nodeIdx 2 node_or_leaf 1 nd_isect (    0.0000     0.0000     0.0000    -0.0000) nd_valid_isect 0 
    ---
    > //intersect_tree  nodeIdx 2 node_or_leaf 1 nd_isect (   -0.0000     0.0000     0.0000    -0.0000) nd_valid_isect 0 
    34,35c34,36
    < //intersect_tree  nodeIdx 1 l/r_complement 0/1 l/r_unbounded 0/0 l/r_promote_miss 0/0 
    < //__intersection__is  idx(1280,720,0) dim(2560,1440,1) dump:1 valid_isect:0
    ---
    > //intersect_tree  nodeIdx 1 l/r_complement 1/1 l/r_unbounded 0/0 l/r_promote_miss 1/0 
    > //__intersection__is  idx(1280,720,0) dim(2560,1440,1) dump:1 valid_isect:1
    > 
    [blyth@localhost heisenbug]$ 




Somehow an extraneous nd_isect x-flip (missed complement) happens with heisenbug_cxd_800_12p4.txt::

    229     if(valid_isect)
    230     {
    231         if(q) q->left_multiply_inplace( isect, 0.f ) ;
    232         // normals transform differently : with inverse-transform-transposed 
    233         // so left_multiply the normal by the inverse-transform rather than the right_multiply 
    234         // done above to get the inverse transformed origin and direction
    235         //const unsigned boundary = node->boundary();  ???
    236 
    237         if(complement)  // flip normal for complement 
    238         {
    239             isect.x = -isect.x ;
    240             isect.y = -isect.y ;
    241             isect.z = -isect.z ;
    242         }
    243     }
    244     else
    245     {
    246          // even for miss need to signal the complement with a -0.f in isect.x
    247          if(complement) isect.x = -isect.x ;
    248          // note that isect.y is also flipped for unbounded exit : for consumption by intersect_tree
    249     }


Using the magic printf and eliding its output gives exactly the same as optix 750, restoring expected intersects::

    [blyth@localhost heisenbug]$ diff heisenbug_cxd_750_11p7.txt heisenbug_cxd_800_12p4_with_magic_printf_elided.txt
    [blyth@localhost heisenbug]$ 


