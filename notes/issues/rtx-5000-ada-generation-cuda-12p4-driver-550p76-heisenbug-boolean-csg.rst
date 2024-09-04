rtx-5000-ada-generation-cuda-12p4-driver-550p76-heisenbug-boolean-csg
=========================================================================

Prior

* :doc:`rtx-5000-ada-generation-cuda-12p4-driver-550p76-some-analytic-geometry-skipped`



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





FIXED IN ACCEPTABLE WAY : WITHOUT MAGIC PRINTF : BY RE-EXPRESSING SOME CODE IN csg_intersect_leaf.h
----------------------------------------------------------------------------------------------------

::

    172 LEAF_FUNC
    173 void intersect_leaf(bool& valid_isect, float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction, bool dumpxyz )
    174 {
    175     valid_isect = false ;
    176     isect.x = 0.f ;
    177     isect.y = 0.f ;
    178     isect.z = 0.f ;
    179     isect.w = 0.f ;
    180 
    181     const unsigned typecode = node->typecode() ;
    182     const unsigned gtransformIdx = node->gtransformIdx() ;
    183     const bool complement = node->is_complement();
    184 
    185     const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    186 
    187     float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin ;
    188     float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction ;
    189 
    190 #if !defined(PRODUCTION) && defined(DEBUG_RECORD)
    191     printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx );
    192 #endif
    193 
    194 #if !defined(PRODUCTION) && defined(DEBUG)
    195     //printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    196     //printf("//intersect_leaf ray_origin (%10.4f,%10.4f,%10.4f) \n",  ray_origin.x, ray_origin.y, ray_origin.z ); 
    197     //printf("//intersect_leaf ray_direction (%10.4f,%10.4f,%10.4f) \n",  ray_direction.x, ray_direction.y, ray_direction.z ); 
    198     /*
    199     if(q) 
    200     {
    201         printf("//intersect_leaf q.q0.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q0.f.x,q->q0.f.y,q->q0.f.z,q->q0.f.w  ); 
    202         printf("//intersect_leaf q.q1.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q1.f.x,q->q1.f.y,q->q1.f.z,q->q1.f.w  ); 
    203         printf("//intersect_leaf q.q2.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q2.f.x,q->q2.f.y,q->q2.f.z,q->q2.f.w  ); 
    204         printf("//intersect_leaf q.q3.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q3.f.x,q->q3.f.y,q->q3.f.z,q->q3.f.w  ); 
    205         printf("//intersect_leaf origin (%10.4f,%10.4f,%10.4f) \n",  origin.x, origin.y, origin.z ); 
    206         printf("//intersect_leaf direction (%10.4f,%10.4f,%10.4f) \n",  direction.x, direction.y, direction.z ); 
    207     }
    208     */
    209 #endif
    210 
    211     switch(typecode)
    212     {
    213         case CSG_SPHERE:           intersect_leaf_sphere(           valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    214         case CSG_ZSPHERE:          intersect_leaf_zsphere(          valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    215         case CSG_CYLINDER:         intersect_leaf_cylinder(         valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    216         case CSG_BOX3:             intersect_leaf_box3(             valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    217         case CSG_CONE:             intersect_leaf_newcone(          valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    218         case CSG_CONVEXPOLYHEDRON: intersect_leaf_convexpolyhedron( valid_isect, isect, node, plan,             t_min, origin, direction ) ; break ;
    219         case CSG_HYPERBOLOID:      intersect_leaf_hyperboloid(      valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    220 #if !defined(PRODUCTION) && defined(CSG_EXTRA)
    221         case CSG_PLANE:            intersect_leaf_plane(            valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    222         case CSG_SLAB:             intersect_leaf_slab(             valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    223         case CSG_OLDCYLINDER:      intersect_leaf_oldcylinder(      valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    224         case CSG_PHICUT:           intersect_leaf_phicut(           valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    225         case CSG_THETACUT:         intersect_leaf_thetacut(         valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    226         case CSG_OLDCONE:          intersect_leaf_oldcone(          valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
    227         case CSG_INFCYLINDER:      intersect_leaf_infcylinder(      valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    228         case CSG_DISC:             intersect_leaf_disc(             valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
    229 #endif
    230     }
    231     // NB: changing typecode->imp mapping is a handy way to use old imp with current geometry 
    232 
    233 
    234 
    235 #if defined(DEBUG_PIDXYZ)
    236     // HMM MAGIC ACTIVE HERE TOO
    237     //if(dumpxyz) printf("//[intersect_leaf.MAGIC typecode %d valid_isect %d isect (%10.4f %10.4f %10.4f %10.4f) complement %d \n",  typecode, valid_isect, isect.x, isect.y, isect.z, isect.w, complement ); 
    238     //if(dumpxyz) printf("//[intersect_leaf.MAGIC typecode %d                isect (%10.4f %10.4f %10.4f %10.4f) complement %d \n",  typecode,              isect.x, isect.y, isect.z, isect.w, complement ); 
    239     //if(dumpxyz) printf("//[intersect_leaf.MAGIC typecode %d \n",  typecode ); 
    240     //if(dumpxyz) printf("//[intersect_leaf.MAGIC \n"); 
    241 
    242     /**
    243     when applying the MAGIC here just a string will do 
    244     **/
    245 #endif
    ...
    248 //#define WITH_HEISENBUG 1
    249 #if !defined(WITH_HEISENBUG)
    250 
    251    if(valid_isect && q ) q->left_multiply_inplace( isect, 0.f ) ;
    252     // normals transform differently : with inverse-transform-transposed 
    253     // so left_multiply the normal by the inverse-transform rather than the right_multiply 
    254     // done above to get the inverse transformed origin and direction
    255 
    256 
    257     /// BIZARRO : RE-EXPRESSING THE MISS-COMPLEMENT-SIGNALLING IMPL FROM THE ABOVE TO A 
    258     /// TERSE FORM BELOW (WHICH SHOULD DO EFFECTIVELY THE SAME THING)
    259     /// AVOIDS THE HEISENBUG : NO NEED FOR MAGIC PRINTF IN intersect_leaf
    260 
    261     if(complement)
    262     {
    263         if(dumpxyz) printf("// intersect_leaf complement %d valid_isect %d \n", complement, valid_isect );
    264 
    265         // flip normal for hit, signal complement for miss 
    266         isect.x = valid_isect ? -isect.x : -0.f ;    // miss needs to signal complement with -0.f signbit 
    267         isect.y = valid_isect ? -isect.y : isect.y ; // miss unbounded exit signalled in isect.y for intersect_tree
    268         isect.z = valid_isect ? -isect.z : isect.z ;
    269     }
    270 
    271 #else
    272     /// CAUTION : SOMETHING ABOUT THE BELOW MISS-COMPLEMENT-SIGNALLING 
    273     /// CODE CAUSES OPTIX 7.5 AND 8.0 HEISENBUG WITH CUDA 12.4 AS REVEALED 
    274     /// WITH RTX 5000 Ada GENERATION GPU
    275 
    276      if(valid_isect)
    277      {
    278          if(q) q->left_multiply_inplace( isect, 0.f );
    279 
    280          if(complement)  // flip normal for complement 
    281          {
    282             isect.x = -isect.x ;
    283             isect.y = -isect.y ;
    284             isect.z = -isect.z ;
    285         }
    286     }
    287     else
    288     {
    289         // even for miss need to signal the complement with a -0.f in isect.x
    290         if(complement) isect.x = -0.f ;
    291         // note that isect.y is also flipped for unbounded exit : for consumption by intersect_tree
    292     }
    293 #endif
    294 
    295 
    296     // NOTICE THAT "VALID_ISECT" IS A BIT MIS-NAMED : AS FALSE JUST MEANS A GEOMETRY MISS 
    297 
    ...
    302 #if defined(DEBUG_PIDXYZ)
    303     // BIZARRELY WITH OptiX 7.5.0 CUDA 12.4 "RTX 5000 Ada Generation" : commenting the below line breaks boolean intersects 
    304     // NOPE SAME WITH OptiX 8.0.0 CUDA 12.4 "RTX 5000 Ada Generation" 
    305 
    306     //if(dumpxyz) printf("%d\n", valid_isect );  // HUH : NEED THIS LINE WITH OPTIX 7.5 CUDA 12.4 RTX 5000 ADA
    307     //if(dumpxyz) printf("//]intersect_leaf typecode %d valid_isect %d isect (%10.4f %10.4f %10.4f %10.4f) complement %d \n",  typecode, valid_isect, isect.x, isect.y, isect.z, isect.w, complement ); 
    308 
    309     //if(dumpxyz) printf("//hello\n"); 
    310     //if(dumpxyz) printf("//]intersect_leaf typecode %d \n", typecode );
    311     //if(dumpxyz) printf("//]intersect_leaf isect (%10.4f %10.4f %10.4f %10.4f) \n", isect.x, isect.y, isect.z, isect.w ); 
    312     //if(dumpxyz) printf("//]intersect_leaf complement %d \n", complement );
    313 
    314     /**
    315     Seems have to potentially dump valid_isect here for the restorative sorcery to work 
    316     **/
    317 
    318 #endif







