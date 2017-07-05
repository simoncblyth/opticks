#include "NGLMExt.hpp"
#include "NTreeTools.hpp"
#include "NOctools.hpp"


#include "DualContouringSample/FGLite.h"

#include "NBBox.hpp"
#include "NGrid3.hpp"
#include "NField3.hpp"
#include "NFieldGrid3.hpp"

#include "NTrianglesNPY.hpp"


#include "Timer.hpp"

#include "PLOG.hh"


typedef NField<glm::vec3,glm::ivec3,3> FI ; 
typedef NFieldGrid3<glm::vec3,glm::ivec3> FG3 ; 



template <typename T>
const glm::ivec3 NConstructor<T>::_CHILD_MIN_OFFSETS[] =
{
	{ 0, 0, 0 },
	{ 0, 0, 1 },
	{ 0, 1, 0 },
	{ 0, 1, 1 },
	{ 1, 0, 0 },
	{ 1, 0, 1 },
	{ 1, 1, 0 },
	{ 1, 1, 1 },
};


template <typename T>
NConstructor<T>::NConstructor(FG3* fieldgrid, FGLite* fglite,  const nvec4& ce, const nbbox& bb, int nominal, int coarse, int verbosity )
   :
   m_fieldgrid(fieldgrid),  
   m_field(fieldgrid->field),  
   m_func(m_field->f),
   m_fglite(fglite),
   m_ce(ce),
   m_bb(bb),
   m_nominal(fieldgrid->grid),  
   m_coarse( m_mgrid.grid[coarse] ),  
   m_verbosity(verbosity),
   m_subtile(NULL),  
   m_dgrid(NULL),  
   m_nominal_min(-m_nominal->size/2, -m_nominal->size/2, -m_nominal->size/2),
   m_upscale_factor(0),
   m_root(NULL),
   m_num_leaf(0),
   m_num_from_cache(0),
   m_num_into_cache(0),
   m_coarse_corners(0),
   m_nominal_corners(0)

{

   bool level_match = m_nominal->level == nominal ;

   if(!level_match) LOG(fatal) << "level_match FAIL "
                               << " nominal level " << nominal 
                               << " nominal " << m_nominal->desc()
                               ;

   assert( level_match ); 

   assert( coarse <= nominal && nominal < maxlevel ); 

   m_subtile = m_mgrid.grid[nominal-coarse] ;  
   m_upscale_factor = m_nominal->upscale_factor( *m_coarse );
   assert( m_upscale_factor == m_subtile->size );

   std::cout << "NConstructor"
              << " upscale_factor " << m_upscale_factor
              << " verbosity " << m_verbosity
              << std::endl 
              << " nominal " << m_nominal->desc()
              << std::endl  
              << " coarse  " << m_coarse->desc()
              << std::endl  
              << " subtile " << m_subtile->desc()
              << std::endl ; 

   if(m_verbosity > 1)
   {
       dump_domain(); 
   }
   if(m_verbosity > 2)
   {
       for(int d=1 ; d <= nominal ; d++) scan("scan", d, 16); 
       for(int d=1 ; d <= nominal ; d++) corner_scan("corner_scan", d, 16); 
   } 
}


template <typename T>
glm::vec3 NConstructor<T>::position_ce(const glm::ivec3& offset_ijk, int depth) const 
{
    assert( depth >= 0 && depth <= m_nominal->level  ); 
    G3* dgrid = m_mgrid.grid[depth] ; 
    float scale = m_ce.w*m_nominal->size/dgrid->size ;   // unjiggery poke

    glm::vec3 xyz ; 
    xyz.x = m_ce.x + offset_ijk.x*scale ; 
    xyz.y = m_ce.y + offset_ijk.y*scale ; 
    xyz.z = m_ce.z + offset_ijk.z*scale ; 
    return xyz ; 
}

template <typename T>
float NConstructor<T>::density_ce(const glm::ivec3& offset_ijk, int depth) const 
{
    glm::vec3 world_pos = position_ce(offset_ijk, depth);
    return (*m_func)(world_pos.x, world_pos.y, world_pos.z);
}



template <typename T>
glm::vec3 NConstructor<T>::position_bb(const glm::ivec3& ijk, int depth) const 
{
    assert( depth >= 0 && depth <= m_nominal->level  ); 
    G3* dgrid = m_mgrid.grid[depth] ; 
     
    assert( ijk.x >= 0 && ijk.x <= dgrid->size );
    assert( ijk.y >= 0 && ijk.y <= dgrid->size );
    assert( ijk.z >= 0 && ijk.z <= dgrid->size );

    glm::vec3 frac_pos = dgrid->fpos(ijk);

    nvec3 bb_side = m_bb.side(); 

    glm::vec3 world_pos ; 
    world_pos.x = m_bb.min.x + frac_pos.x*bb_side.x ; 
    world_pos.y = m_bb.min.y + frac_pos.y*bb_side.y ; 
    world_pos.z = m_bb.min.z + frac_pos.z*bb_side.z ; 

    return world_pos ; 
}


template <typename T>
float NConstructor<T>::density_bb(const glm::ivec3& ijk, int depth) const 
{
    glm::vec3 world_pos = position_bb(ijk, depth);
    return (*m_func)(world_pos.x, world_pos.y, world_pos.z);
}




template <typename T>
void NConstructor<T>::dump()
{
    std::cout << "ConstructOctreeBottomUp"
              << " num_leaf " << m_num_leaf 
              << " num_into_cache " << m_num_into_cache 
              << " num_from_cache " << m_num_from_cache 
              << " num_leaf/nominal.nloc " << float(m_num_leaf)/float(m_nominal->nloc) 
              << std::endl ;

     report("NConstructor::dump");
}


template <typename T>
void NConstructor<T>::report(const char* msg)
{
    LOG(info) << msg ;
    std::cout  
              << " coarse_level   " << std::setw(7) << m_coarse->level
              << " coarse_nloc    " << std::setw(7) << m_coarse->nloc 
              << " coarse_corners " << std::setw(7) << m_coarse_corners 
              << " coarse_frac    " << std::setw(7) << float(m_coarse_corners)/float(m_coarse->nloc)
              << std::endl ; 
    std::cout  
              << " nominal_level   " << std::setw(7) << m_nominal->level
              << " nominal_nloc    " << std::setw(7) << m_nominal->nloc 
              << " nominal_corners " << std::setw(7) << m_nominal_corners 
              << " nominal_frac    " << std::setw(7) << float(m_nominal_corners)/float(m_nominal->nloc)
              << std::endl ; 
}



template <typename T>
void NConstructor<T>::buildBottomUpFromLeaf(int leaf_loc, T* leaf)
{
/*
Due to the z-order : 

* there are only ever up to 8 nodes waiting around at each level 
  to be parented : so could use static arrays and avoid finding from 
  big caches
              
* also perhaps could do parent hookup in batches, on passing last child slot 

BUT, minimal time is spent doing this compared to leaf scanning and creation
so no point optimizing here.

*/

    T* node = leaf ; 
    T* dnode = NULL ; 

    int depth = m_nominal->level  ; // start from nominal level, with the leaves 
    int dloc = leaf_loc ; 
    unsigned dsize = 1 ; 
    unsigned dchild = dloc & 7 ;    // lowest 3 bits, gives child index in immediate parent

    // at each turn : decrement depth, right-shift morton code to that of parent, left shift size doubling it 
    while(depth >= 1)
    {
        depth-- ; 
        dloc >>= 3 ;     
        dsize <<= 1 ;     
        m_dgrid = m_mgrid.grid[depth] ; 

        typename UMAP::const_iterator it = cache[depth].find(dloc);
        if(it == cache[depth].end())
        {
            m_num_into_cache++ ; 
            glm::ivec3 d_ijk = m_dgrid->ijk(dloc); 

            d_ijk *= dsize ;      // scale coordinates to nominal 

            d_ijk += m_nominal_min ;      //OFF


            dnode = new T ; 
            dnode->size = dsize ; 
            dnode->min = glm::ivec3(d_ijk.x, d_ijk.y, d_ijk.z) ;  
            dnode->type = T::Node_Internal;

            cache[depth].emplace(dloc, dnode)  ;

            if(m_num_into_cache < 10)
            std::cout << "into_cache " 
                      << " num_into_cache " << m_num_into_cache
                      << " dloc " << std::setw(6) << dloc
                      << " d_ijk " << glm::to_string(d_ijk)
                      << " m_nominal_min " << glm::to_string(m_nominal_min)
                      << " dsize " << dsize
                      << std::endl ; 
        }
        else
        {
            m_num_from_cache++ ; 
            dnode = it->second ;     
        }

        dnode->children[dchild] = node ;  
        node = dnode ; 
        dchild = dloc & 7 ;  // child index for next round
    }              // up the heirarchy from each leaf to root
}




template <typename T>
T* NConstructor<T>::create()
{
    T* root = NULL ; 
    if( m_coarse->level == m_nominal->level )
    {
        root = create_nominal() ;
    }
    else
    {
        root = create_coarse_nominal() ;
    }
    return root ; 
}



template <typename T>
T* NConstructor<T>::create_coarse_nominal()
{
    int leaf_size = 1 ; 
    for(int c=0 ; c < m_coarse->nloc ; c++) 
    {
        glm::ivec3 c_ijk = m_coarse->ijk( c );
        c_ijk *= m_subtile->size ;    // scale coarse coordinates up to nominal 
        c_ijk += m_nominal_min ;     //OFF

        int corners = T::Corners( c_ijk , m_fglite, 8, m_upscale_factor ); 
        if(corners == 0 || corners == 255) continue ;   
        m_coarse_corners++ ; 
 
        for(int s=0 ; s < m_subtile->nloc ; s++)  // over nominal(at level) voxels in coarse tile
        {
            glm::ivec3 s_ijk = m_subtile->ijk( s );
            s_ijk += c_ijk ; 
 
            int corners = T::Corners( s_ijk, m_fglite, 8, leaf_size ); 
            if(corners == 0 || corners == 255) continue ;  
            m_nominal_corners++ ; 

            glm::ivec3 a_ijk = s_ijk - m_nominal_min ;   // take out the offset, need 0:128 range

            T* leaf = T::MakeLeaf( s_ijk, corners, m_fglite, leaf_size ); 

            int leaf_loc = m_nominal->loc( a_ijk );

            buildBottomUpFromLeaf(leaf_loc, leaf); // negligible time doing this


        }   // over nominal voxels within coarse tile
    }       // over coarse tiles

    typename UMAP::const_iterator it0 = cache[0].find(0);
    T* root = it0 == cache[0].end() ? NULL : it0->second ; 
    if(!root) LOG(fatal) << "FAILED TO FIND ROOT" ; 
    assert(root);

    return root ; 
}

template <typename T>
T* NConstructor<T>::create_nominal()
{
    int leaf_size = 1 ; 

    for(int c=0 ; c < m_nominal->nloc ; c++) 
    {
        glm::ivec3 ijk = m_nominal->ijk( c );
        glm::ivec3 offset_ijk = ijk + m_nominal_min ;  //OFF

        int corners = T::Corners( offset_ijk , m_fglite, 8, leaf_size ); 
        if(corners == 0 || corners == 255) continue ;   
        m_nominal_corners++ ; 
 
        T* leaf = T::MakeLeaf( offset_ijk, corners, m_fglite, leaf_size ); 

        buildBottomUpFromLeaf( c, leaf);
    }   

    LOG(info) << "NConstructor<T>::create_nominal"
              << " nloc " << m_nominal->nloc
              << " nominal_corners " << m_nominal_corners
               ;
 

    typename UMAP::const_iterator it0 = cache[0].find(0);
    T* root = it0 == cache[0].end() ? NULL : it0->second ; 
    assert(root);
    return root ; 
}








template <typename T>
void NConstructor<T>::dump_domain(const char* msg) const 
{
    assert( *m_fieldgrid->grid == *m_nominal ) ;  // <-- make sure start at nominal
    bool fg_offset = m_fieldgrid->offset  ;  

    LOG(info) <<  msg 
              << " nominal_min " << glm::to_string(m_nominal_min)
              << " fg_offset " << ( fg_offset ? "YES" : "NO" )
              << " ce  " << m_ce.desc()
              << " bb "  << m_bb.desc()
              ;

    LOG(info) << "dump_domain : positions and SDF values at domain corners at all resolutions, they should all match " ; 

    // skip depth 0 singularity, as 8-corners not meaningful there
    for(int depth = 1 ; depth <= m_nominal->level ; depth++ )
    {
        int size = 1 << depth ; 
        G3* dgrid = m_mgrid.grid[depth] ; 
        assert(size == dgrid->size);
 
        std::cout << " depth " << depth
                  << " size " << size
                  << " grid " << dgrid->desc()
                  << std::endl ; 

        glm::ivec3 dmin(-size/2);               //OFF
        assert( dgrid->half_min == dmin );

        m_fieldgrid->grid = dgrid ;  // <---- OOHH SURGERY ON THE FIELDGRID

        for (int i = 0; i < 8; i++)
        {
             const glm::ivec3 ijk = _CHILD_MIN_OFFSETS[i] * dgrid->size ;   // <-- not scaling to different rez, this is within dgrid rez

             const glm::ivec3 offset_ijk = dmin + ijk ;    //OFF
             glm::vec3 pce = position_ce(offset_ijk, depth);
             float vce = density_ce(offset_ijk, depth) ;

             glm::vec3 fpos = dgrid->fpos(ijk); 
             float vfi = (*m_field)(fpos);
             glm::vec3 pfi = m_field->position(fpos);

             glm::vec3 pfg = m_fieldgrid->position( fg_offset ? offset_ijk : ijk );
             float vfg = m_fieldgrid->value(    fg_offset ? offset_ijk : ijk );

             assert( pfi == pfg );
             assert( vfi == vfg );


             if(depth < m_nominal->level )
             {
                 std::cout << " i " << std::setw(3) << i 
                           << " ijk " << ijk
                           << " offset_ijk " << offset_ijk
                           << " pce " << pce
                           << " pfg " << pfg
                           << " vce " << vce
                           << " vfg " << vfg
                           << std::endl ; 

             }
             else
             {
                 glm::vec3 pfgl = m_fglite->position_f( offset_ijk ) ; 
                 float     vfgl = m_fglite->value_f( offset_ijk ) ; 
                 std::cout << " i " << std::setw(3) << i 
                           << " ijk " << ijk
                           << " offset_ijk " << offset_ijk
                           << " pce " << pce
                           << " pfg " << pfg
                           << " vce " << vce
                           << " vfg " << vfg
                           << " pfgl " << pfgl 
                           << " vfgl " << vfgl
                           << std::endl ; 

             }
             



        }
    } // over depth

    assert( *m_fieldgrid->grid == *m_nominal ) ;  // <-- make sure end at nominal
}





template <typename T>
void NConstructor<T>::scan(const char* msg, int depth, int limit ) const 
{
    G3* dgrid = m_mgrid.grid[depth] ; 

    std::cout << " dgrid   " << dgrid->desc()  << std::endl ; 
    std::cout << " nominal " << m_nominal->desc() << std::endl ; 

    int scale_to_nominal = 1 << (m_nominal->level - dgrid->level ) ; 
    int upscale_factor = m_nominal->upscale_factor( *dgrid ); 

    LOG(info) << msg 
              << " nominal level " << m_nominal->level
              << " dgrid level " << dgrid->level
              << " scale_to_nominal " << scale_to_nominal 
              << " upscale_factor " << upscale_factor
              << " limit " << limit 
              ;

    assert( scale_to_nominal == upscale_factor );
    LOG(info) << "first and last z-order locations, at higher rez positions should be close to min and max " ; 

    int nloc = dgrid->nloc ;  

    bool fg_offset = m_fieldgrid->offset ;  


    typedef std::vector<int> VI ;
    VI locs ; 
    for(int c=0                   ; c < nmini(limit,nloc)     ; c++) locs.push_back(c) ; 
    for(int c=nmaxi(nloc-limit,0) ; c < nloc                  ; c++) locs.push_back(c) ; 

    for(VI::const_iterator it=locs.begin() ; it != locs.end() ; it++)
    {
        int c = *it ; 
        glm::ivec3 raw = dgrid->ijk( c );

        glm::vec3 fpos = dgrid->fpos(raw);   // can also get direct from c, also no fiddly scaling or offsetting 
        glm::vec3 pfi = m_field->position(fpos);
        float vfi = (*m_field)(fpos);

        glm::ivec3 ijk = raw * scale_to_nominal ; 
        glm::ivec3 offset_ijk = ijk + m_nominal->half_min ; // <-- after scaling to nominal, must use nominal offset 


        glm::vec3 pfg = m_fieldgrid->position( fg_offset ? offset_ijk : ijk );
        float vfg = m_fieldgrid->value(    fg_offset ? offset_ijk : ijk );


        glm::vec3 pbb = position_bb(ijk, m_nominal->level);
        float vbb = density_bb(ijk, m_nominal->level); 

        assert( pbb == pfi );
        assert( vbb == vfi );
        assert( pfg == pfi );
        assert( vfg == vfi );

        glm::vec3 pce = position_ce(offset_ijk, m_nominal->level);
        float vce = density_ce(offset_ijk, m_nominal->level); 


        std::cout << " c " << std::setw(10) << c 
                  << " raw " << raw
                  << " ijk " << ijk
                  << " offset_ijk " << offset_ijk 
                  << " pce "  << pce
                  << " pfg "  << pfg
                  << " vce "  << vce
                  << " vfg "  << vfg
                  << std::endl 
                  ; 
    }
}


template <typename T>
void NConstructor<T>::corner_scan(const char* msg, int depth, int limit) const 
{
    G3* dgrid = m_mgrid.grid[depth] ; 
    int upscale = m_nominal->upscale_factor( *dgrid ); 
    LOG(info) << msg 
              << " depth " << depth
              << " limit " << limit 
              << " upscale " << upscale
              << " dgrid.elem " << dgrid->elem 
              ;

    // elem is 1/size, which is correct fdelta as fpos range is 0:1

    std::cout << " dgrid   " << dgrid->desc()  << std::endl ; 
    std::cout << " nominal " << m_nominal->desc() << std::endl ; 

    int count0 = 0 ; 
    for(int c=0 ; c < dgrid->nloc ; c++) 
    {
        glm::vec3 fpos = dgrid->fpos(c); 
        int corners = m_field->zcorners(fpos, dgrid->elem ) ;  

        if(corners == 0 || corners == 255) continue ; 

        count0++ ; 
        if( count0 > limit ) break ; 
        
        glm::vec3 pfi = m_field->position(fpos);
        float vfi = (*m_field)(fpos);

        std::cout 
              << " count0 " << std::setw(5) << count0
              << " c " << std::setw(10) << c
              << " fpos " << glm::to_string(fpos)
              << " cnrs " << std::bitset<8>(corners) 
              << " pfi " << std::setw(20) << glm::to_string(pfi)
              << " vfi " << std::setw(10) << vfi 
              << std::endl ; 
    }

    // even though the fpos (fractional position on the grid) 
    // are the same the real world positions are different between
    // the ce and bb grids : as those are in different places
    // ... they should however be in same ballpark as the geometry sdf
    // is in the same place

    int count1 = 0 ; 
    for(int c=0 ; c < dgrid->nloc ; c++) 
    {
        glm::vec3 fpos = dgrid->fpos(c); 
        glm::ivec3 dijk = dgrid->ijk( c );
        dijk *= upscale ;
        dijk += m_nominal_min ;  //OFF

        int corners = T::Corners( dijk , m_fglite, 8, upscale ); 
        if(corners == 0 || corners == 255) continue ; 

        count1++ ; 
        if( count1 > limit ) break ; 
        
        glm::vec3 pce = position_ce(dijk, m_nominal->level);
        float vce = density_ce(dijk, m_nominal->level); 

        std::cout 
              << " count1 " << std::setw(5) << count1
              << " c " << std::setw(10) << c
              << " fpos " << glm::to_string(fpos)
              << " cnrs " << std::bitset<8>(corners) 
              << " pce " << std::setw(20) << glm::to_string(pce)
              << " vce " << std::setw(10) << vce
              << std::endl ; 
    }
} 










template<typename T>
NManager<T>::NManager( const unsigned ctrl,  const int nominal, const int coarse, const int verbosity, const float threshold, FG3* fieldgrid, FGLite* fglite, const nbbox& bb, Timer* timer )
    :
    m_ctrl(ctrl),
    m_nominal_size( 1 << nominal ),
    m_verbosity(verbosity),
    m_threshold(threshold), 
    m_fieldgrid(fieldgrid),
    m_fglite(fglite),
    m_bb(bb), 
    m_timer(timer),
    m_ctor(NULL),
    m_bottom_up(NULL),
    m_top_down(NULL),
    m_raw(NULL),
    m_simplified(NULL)
{
    nvec4     bbce = m_bb.center_extent();

    float ijkExtent = m_nominal_size/2 ;      // eg 64.f
    float xyzExtent = bbce.w  ;
    float ijk2xyz = xyzExtent/ijkExtent ;     // octree -> real world coordinates

    m_ce = make_nvec4(bbce.x, bbce.y, bbce.z, ijk2xyz );

    m_ctor = new NConstructor<T>(m_fieldgrid, m_fglite, m_ce, m_bb, nominal, coarse, m_verbosity );

    if(m_verbosity > 10 ) assert(0 && "hari kari for verbosity > 10");

    LOG(info) << "Manager::Manager"
              << " xyzExtent " << xyzExtent
              << " ijkExtent " << ijkExtent
              << " bbce " << bbce.desc()
              << " ce " << m_ce.desc()
              << " bb.min " << bb.min.desc() 
              << " bb.max " << bb.max.desc() 
              << " bb.side " << bb.side().desc() 
              ;
}



template <typename T>
T* NManager<T>::getRaw()
{
    return m_raw ; 
}

template <typename T>
T* NManager<T>::getSimplified()
{
    return m_simplified ; 
}





template <typename T>
void NManager<T>::buildOctree()
{
    if( m_ctrl & BUILD_BOTTOM_UP ) LOG(info) << "BUILD_BOTTOM_UP" ; 
    if( m_ctrl & BUILD_TOP_DOWN )  LOG(info) << "BUILD_TOP_DOWN" ;
    if( m_ctrl & BUILD_BOTH )      LOG(info) << "BUILD_BOTH" ; 
    int verbosity = 0 ; 

    if( m_ctrl & BUILD_BOTTOM_UP )
    {
        m_timer->stamp("_ConstructOctreeBottomUp");

        m_bottom_up = m_ctor->create();
        m_ctor->dump();

        //assert(m_bottom_up);
        m_timer->stamp("ConstructOctreeBottomUp");

        if(m_bottom_up)
        NTraverser<T,8>(m_bottom_up, "bottom_up", verbosity, 30 );
    }

    if( m_ctrl & BUILD_TOP_DOWN )
    {
  	    T* root0 = new T;
  	    root0->min = glm::ivec3(-m_nominal_size/2);   // <--- cause of great pain TODO: get from fieldgrid ?
	    root0->size = m_nominal_size;
	    root0->type = T::Node_Internal;

        m_timer->stamp("_ConstructOctreeNodes");
        int count = 0 ; 
	    m_top_down = T::ConstructOctreeNodes(root0, m_fglite, count);
        m_timer->stamp("ConstructOctreeNodes");
        std::cout << "ConstructOctreeNodes count " << count << std::endl ; 
        NTraverser<T,8>(m_top_down, "top_down", verbosity, 30);
    }

    if( (m_ctrl & BUILD_BOTTOM_UP) && (m_ctrl & BUILD_TOP_DOWN)  )
    {
        m_timer->stamp("_Comparer");
        NComparer<T,8> cmpr(m_bottom_up, m_top_down);
        cmpr.dump("Comparer result");
        m_timer->stamp("Comparer");
    }
    
    m_raw  = m_ctrl & USE_BOTTOM_UP ? m_bottom_up : m_top_down ; 
    assert(m_raw);

}


template <typename T>
void NManager<T>::simplifyOctree()
{
    assert(m_raw);

    m_timer->stamp("_SimplifyOctree");
	m_simplified = T::SimplifyOctree(m_raw, m_threshold);
    m_timer->stamp("SimplifyOctree");
}
 




template <typename T>
void NManager<T>::generateMeshFromOctree()
{
    assert(m_simplified);

	m_vertices.clear();
	m_normals.clear();
	m_indices.clear();

	T::GenerateVertexIndices(m_simplified, m_vertices,m_normals, m_fglite );
	T::ContourCellProc(m_simplified, m_indices);
}


template <typename T>
void NManager<T>::meshReport(const char* msg)
{
    LOG(info) << msg ; 

    unsigned npol = m_indices.size() ; 

    assert( npol % 3 == 0) ;
    unsigned ntri = npol / 3 ; 

    LOG(info) << " npol " << npol 
              << " ntri " << ntri 
              << " vertices " << m_vertices.size() 
              << " normals " << m_normals.size() 
              << " indices  " << m_indices.size() 
              ;


    typedef std::vector<int> VI ; 

    VI::iterator  pmin = std::min_element(std::begin(m_indices), std::end(m_indices));
    VI::iterator  pmax = std::max_element(std::begin(m_indices), std::end(m_indices));

    size_t imin = std::distance(std::begin(m_indices), pmin) ;
    size_t imax = std::distance(std::begin(m_indices), pmax) ;

    LOG(debug) << "min element at: " << imin << " " << m_indices[imin] ; 
    LOG(debug) << "max element at: " << imax << " " << m_indices[imax] ;

}


template <typename T>
NTrianglesNPY* NManager<T>::collectTriangles()
{
    m_timer->stamp("_CollectTriangles");

    unsigned npol = m_indices.size() ; 
    assert( npol % 3 == 0) ;
    unsigned ntri = npol / 3 ; 

    assert( m_vertices.size() == m_normals.size() );


    NTrianglesNPY* tris = new NTrianglesNPY();
    for(unsigned t=0 ; t < ntri ; t++)
    {
         assert( t*3+2 < npol );

         unsigned i0 = m_indices[t*3 + 0];
         unsigned i1 = m_indices[t*3 + 1];
         unsigned i2 = m_indices[t*3 + 2];
          
         glm::vec3& v0 = m_vertices[i0] ;
         glm::vec3& v1 = m_vertices[i1] ;
         glm::vec3& v2 = m_vertices[i2] ;

         glm::vec3& n0 = m_normals[i0] ;
         glm::vec3& n1 = m_normals[i1] ;
         glm::vec3& n2 = m_normals[i2] ;

         tris->add( v0, v1, v2 );
         tris->addNormal( n0, n1, n2 );
    }
    m_timer->stamp("CollectTriangles");
    return tris ; 
}





#include "DualContouringSample/octree.h"
template class NConstructor<OctreeNode> ; 
template class NManager<OctreeNode> ; 

/*
#include "NOct.hpp"
template class NConstructor<NOct> ; 
template class NManager<NOct> ; 
*/



