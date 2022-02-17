compound_nodes_within_tree_how
================================

* in the CSGFoundry model list nodes simply laydown the sub nodes after the header, 
  but cannot to that ay GGeo/GParts level because the machinery expects complete binary
  tree num nodes 



::

     344 void NCSG::init()
     345 {
     346     setBoundary( m_root->boundary );  // boundary spec
     347 
     348     unsigned num_serialization_nodes = m_root->num_serialization_nodes();
     349     LOG(LEVEL) << "[ init csgdata : num_serialization_nodes " << num_serialization_nodes << " height " << m_height << "(-1 for lists)"   ;
     350     m_csgdata->init_node_buffers(num_serialization_nodes) ;
     351     LOG(LEVEL) << "] init csgdata " ;
     352 }

     467 /**
     468 nnode::num_serialization_nodes
     469 ---------------------------------
     470 
     471 TODO: this is not handling having list nodes within the tree
     472 
     473 **/
     474 
     475 unsigned nnode::num_serialization_nodes() const
     476 {
     477     assert( is_root() );
     478     unsigned num_nodes = 0 ;
     479     if( is_list() )
     480     {
     481         unsigned num_subs = subs.size() ;
     482         num_nodes = 1 + num_subs ;  
     483     }   
     484     else if( is_tree() )
     485     {
     486         unsigned height = maxdepth() ;
     487         num_nodes = TREE_NODES(height); // number of nodes for a complete binary tree of the needed height, with no balancing 
     488     }   
     489     else if( is_leaf() )
     490     {
     491         num_nodes = 1 ;
     492     }   
     493     else
     494     {
     495         LOG(fatal) << " m_root node type is not list/tree/leaf ABORT " ;
     496         std::raise(SIGINT); 
     497     }
     498     return num_nodes ;
     499 }


Instead of assuming all nodes are from the tree record the number of tree nodes in the root, using the subNum::

    2139 void nnode::prepareTree()
    2140 {
    2141     nnode* root = this ;
    2142     unsigned num_subs = root->subs.size() ;
    2143     assert( num_subs == 0 );
    2144 
    2145     nnode::Set_parent_links_r(root, NULL);
    2146     root->check_tree( FEATURE_PARENT_LINKS );
    2147     root->update_gtransforms() ;  // sets node->gtransform (not gtransform_idx) parent links are required 
    2148     root->check_tree( FEATURE_GTRANSFORMS );
    2149     root->check_tree( FEATURE_PARENT_LINKS | FEATURE_GTRANSFORMS );
    2150 
    2151     unsigned tree_nodes = num_tree_nodes();
    2152     LOG(LEVEL) << " tree_nodes " << tree_nodes ;
    2153     root->setSubNum( tree_nodes );
    2154 }   


For lists within trees the subs do not immediately follow the header, they follow the tree nodes : so need to 
record an offset at which to find the subs in the list header node.

Yes, but intersect_node gets called on the list node, at which point do not have access to the tree root. 
So need to calculate the number of intermediate nodes between the list hdr where it appears in the 
tree serialization and the start of the subs.  Hmm maybe simpler to use node0 ? No not simpler node0 is 
the first of all nodes in entire geometry.  Hmm but you do have the nodeIdx of the list node within the tree, 
so if also have the absolute offset of the start of the subs.

nd
    compound list node, which carries (subNum, subOffset) 

    While traversing the tree, where node is the root::

        const CSGNode* nd = node + nodeIdx - 1 ;

nd - (nodeIdx - 1)
    back to node : the root of the tree 

node - nodeIdx + subOffset  
    start of the subs for the list 

HMM: could just pass node=root when call intersect_node, then subOffset can be relative to root 

  

::

    183 INTERSECT_FUNC
    184 bool intersect_node_contiguous( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
    185 {
    186 #ifdef DEBUG
    187      printf("//intersect_node_contiguous \n");
    188 #endif
    189 
    190     float sd = distance_node_list( CSG_CONTIGUOUS, ray_origin + t_min*ray_direction, node, plan, itra );
    191     bool inside_or_surface = sd <= 0.f ;
    192     const unsigned num_sub = node->subNum() ;
    193     
    194 #ifdef DEBUG 
    195      printf("//intersect_node_contiguous sd %10.4f inside_or_surface %d num_sub %d \n", sd, inside_or_surface, num_sub);
    196 #endif
    197     
    198     float4 nearest_enter = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ;
    199     float4 farthest_exit = make_float4( 0.f, 0.f, 0.f, t_min ) ;
    200         
    201     float4 sub_isect_0 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;
    202     float4 sub_isect_1 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;
    203     // HMM: are both these sub_isect needed ? seems not : there is no comparison between them, 
    204     // HMM: same with sub_state_0, sub_state_1  
    205 
    206     unsigned enter_count = 0 ; 
    207     unsigned exit_count = 0 ; 
    208     float propagate_epsilon = 0.0001f ; 
    209 
    210     for(unsigned isub=0 ; isub < num_sub ; isub++)
    211     {   
    212         const CSGNode* sub_node = node+1u+isub ; 
    213         // hmm for lists within trees the subs do not immediately follow the header, they follow the tree nodes 
    214  


