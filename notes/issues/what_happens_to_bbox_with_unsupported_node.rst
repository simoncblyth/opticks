what_happens_to_bbox_with_unsupported_node
============================================


The sn bbox ends up in LeafFrame::

    5300 inline void sn::setAABB_LeafFrame_All()
    5301 {
    5302     std::vector<const sn*> prim ;
    5303     collect_prim(prim);
    5304     int num_prim = prim.size() ;
    5305     for(int i=0 ; i < num_prim ; i++)
    5306     {
    5307         const sn* p = prim[i] ;
    5308         sn* _p = const_cast<sn*>(p) ;
    5309         _p->setAABB_LeafFrame() ;
    5310     }
    5311 }



    5154 inline void sn::setAABB_LeafFrame()
    5155 {
    5156     if(typecode == CSG_SPHERE)
    5157     {
    5158         double cx, cy, cz, r, a, b ;
    5159         getParam_(cx, cy, cz, r, a, b );
    5160         assert( cx == 0. && cy == 0. && cz == 0. );
    5161         assert( a == 0. && b == 0. );
    5162         setBB(  -r, -r, -r,  r, r, r  );
    5163     }
    ....
    5253     else if( typecode == CSG_CONTIGUOUS || typecode == CSG_DISCONTIGUOUS )
    5254     {
    5255         // cannot define bbox of list header nodes without combining bbox of all the subs
    5256         // so have to defer setting the bbox until all the subs are converted
    5257         setBB( 0. );
    5258     }
    5259     else if( typecode == CSG_NOTSUPPORTED )
    5260     {
    5261         setBB( 0. );
    5262     }
    5263     else if( typecode == CSG_ZERO )
    5264     {
    5265         setBB( UNBOUNDED_DEFAULT_EXTENT );
    5266     }



    2830 inline void sn::setBB( double x0 )
    2831 {
    2832     if( aabb == nullptr ) aabb = new s_bb ;
    2833     aabb->x0 = -x0 ;
    2834     aabb->y0 = -x0 ;
    2835     aabb->z0 = -x0 ;
    2836     aabb->x1 = +x0 ;
    2837     aabb->y1 = +x0 ;
    2838     aabb->z1 = +x0 ;
    2839 }





    3178 /**
    3179 stree::get_combined_tran_and_aabb
    3180 --------------------------------------
    3181 
    3182 Critical usage of this from CSGImport::importNode
    3183 
    3184 0. early exits returning nullptr for non leaf nodes
    3185 1. gets combined structural(snode.h) and CSG tree(sn.h) transform
    3186 2. collects that combined transform and its inverse (t,v) into Tran instance
    3187 3. copies leaf frame bbox values from the CSG nd into callers aabb array
    3188 4. transforms the bbox of the callers aabb array using the combined structural node
    3189    + tree node transform
    3190 
    3191 
    3192 Note that sn::uncoincide needs CSG tree frame AABB but whereas this needs leaf
    3193 frame AABB. These two demands are met by changing the AABB frame
    3194 within sn::postconvert
    3195 
    3196 **/
    3197 
    3198 inline const Tran<double>* stree::get_combined_tran_and_aabb(
    3199     double* aabb,
    3200     const snode& node,
    3201     const sn* nd,
    3202     std::ostream* out
    3203     ) const
    3204 {
    3205     assert( nd );
    3206     if(!CSG::IsLeaf(nd->typecode)) return nullptr ;
    3207 
    3208     glm::tmat4x4<double> t(1.) ;
    3209     glm::tmat4x4<double> v(1.) ;
    3210     get_combined_transform(t, v, node, nd, out );
    3211         
    3212     // NB ridx:0 full stack of transforms from root down to CSG constituent nodes
    3213     //    ridx>0 only within the instance and within constituent CSG tree
    3214         
    3215     const Tran<double>* tv = new Tran<double>(t, v);
    3216     
    3217     nd->copyBB_data( aabb );
    3218     stra<double>::Transform_AABB_Inplace(aabb, t);
    3219         
    3220     return tv ; 
    3221 }       



    664 template<typename T>
    665 inline void stra<T>::Transform_AABB_Inplace( T* aabb, const glm::tmat4x4<T>& t )
    666 {   
    667     Transform_AABB( aabb, aabb, t ); 
    668 }   





