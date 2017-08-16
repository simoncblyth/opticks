nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched 
=======================================================================================

Actions
---------

* split NGLMCF from NGLMExt for easier control 
* spill nglmext::invert_trs up into NScene::import_r for debug collection of the transforms


Issue
---------

* j1707 gives huge amounts of warning output re mismatched invert_trs
* lots of the noise from nans comparing zeros, some real discrep 

::


    op --j1707 --tracer --gltf 3


    2017-08-16 19:27:09.525 WARN  [98406] [NScene::load_asset_extras@275] NScene::load_asset_extras verbosity increase from scene gltf  extras_verbosity 1 m_verbosity 0
    2017-08-16 19:27:09.525 INFO  [98406] [NScene::init@177] NScene::init START age(s) 1135221 days  13.139
    2017-08-16 19:27:09.526 INFO  [98406] [NScene::load_csg_metadata@310] NScene::load_csg_metadata verbosity 1 num_meshes 35
    nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  epsilon 1e-05 diff 0.00195312 diff2 0.00195312 diffFractional 2 diffFractionalMax 0.001
           trs -0.847  -0.489   0.208   0.000 
               -0.500   0.866  -0.000   0.000 
               -0.180  -0.104  -0.978   0.000 
              3352.658 1935.658 18213.107   1.000 

        isirit -0.847  -0.500  -0.180   0.000 
               -0.489   0.866  -0.104   0.000 
                0.208  -0.000  -0.978   0.000 
               -0.000   0.001 18619.998   1.000 

        i_trs  -0.847  -0.500  -0.180   0.000 
               -0.489   0.866  -0.104   0.000 
                0.208  -0.000  -0.978  -0.000 
               -0.000   0.001 18620.000   1.000 

    ...


    nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  epsilon 1e-05 diff 4.05361e-05 diff2 0.00100265 diffFractional 2 diffFractionalMax 0.001
           trs  0.600  -0.032   0.800   0.000 
               -0.053  -0.999  -0.000   0.000 
                0.798  -0.042  -0.601   0.000 
              -15529.219 823.015 11681.974   1.000 

        isirit  0.600  -0.053   0.798   0.000 
               -0.032  -0.999  -0.042   0.000 
                0.800  -0.000  -0.601   0.000 
                0.001   0.001 19450.000   1.000 

        i_trs   0.600  -0.053   0.798  -0.000 
               -0.032  -0.999  -0.042   0.000 
                0.800  -0.000  -0.601  -0.000 
                0.001   0.001 19450.000   1.000 

    [  0.599774:  0.599774:         0:         0][-0.0529236:-0.0529236:         0:        -0][  0.798417:  0.798417:         0:         0][**         0:        -0:         0:       nan**]
    [-0.0317868:-0.0317868:         0:        -0][ -0.998599: -0.998599:         0:        -0][-0.0423144:-0.0423144:3.72529e-09:-8.80385e-08][**         0:         0:         0:       nan**]
    [  0.799538:  0.799538:         0:         0][**-6.33299e-08:-6.70552e-08:         0:       nan**][ -0.600616: -0.600616:         0:        -0][**         0:        -0:         0:       nan**]
    [**0.000976562:0.00100265:2.60881e-05: 0.0263621**][**0.000678783:0.00071932:4.05361e-05: 0.0579873**][     19450:     19450:         0:         0][         1:         1:         0:         0]
    nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  epsilon 1e


     0]
    2017-08-16 19:27:20.854 INFO  [98406] [NScene::postimportnd@558] NScene::postimportnd numNd 290276 num_selected 290276 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-16 19:27:24.579 INFO  [98406] [NScene::count_progeny_digests@932] NScene::count_progeny_digests verbosity 1 node_count 290276 digest_size 35



::


     520 nd* NScene::import_r(int idx,  nd* parent, int depth)
     521 {
     522     ygltf::node_t* ynode = getNode(idx);
     523     auto extras = ynode->extras ;
     524     std::string boundary = extras["boundary"] ;
     525     std::string pvname = extras["pvname"] ;
     526     unsigned selected = extras["selected"] ;
     527 
     528     nd* n = new nd ;   // NB these are structural nodes, not CSG tree nodes
     529 
     530     n->idx = idx ;
     531     n->repeatIdx = 0 ;
     532     n->mesh = ynode->mesh ;
     533     n->parent = parent ;
     534     n->depth = depth ;
     535     n->boundary = boundary ;
     536     n->pvname = pvname ;
     537     n->selected = selected ;  // TODO: get rid of this, are now doing selection in GScene 
     538     n->containment = 0 ;
     539     n->transform = new nmat4triple( ynode->matrix.data() );
     540     n->gtransform = nd::make_global_transform(n) ;
     541 
     542     if(selected) m_num_selected++ ;
     543 
     544 
     545     for(int child : ynode->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call
     546 
     547     m_nd[idx] = n ;
     548 
     549     return n ;
     550 }





::

     45 struct NPY_API nmat4triple
     46 {
     ..
     74     nmat4triple( const glm::mat4& transform );
     75     nmat4triple( const float* data );
     76     nmat4triple( const glm::mat4& transform, const glm::mat4& inverse, const glm::mat4& inverse_T )
     77          :
     78             t(transform),
     79             v(inverse),
     80             q(inverse_T)
     81          {} ;
     82 

::

    491 nmat4triple::nmat4triple(const float* data )
    492      :
    493      t(glm::make_mat4(data)),
    494      v(nglmext::invert_trs(t)),
    495      q(glm::transpose(v))
    496 {   
    497 }



::

     63 const nmat4triple* nd::make_global_transform(const nd* n) // static
     64 {
     65     std::vector<const nmat4triple*> tvq ;
     66     while(n)
     67     {
     68         if(n->transform) tvq.push_back(n->transform);
     69         n = n->parent ;
     70     }
     71     bool reverse = true ; // as tvq in leaf-to-root order
     72     return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ;
     73 }




